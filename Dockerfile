# One image, two entrypoints (task T-602).
#
# The API and the dashboard are the same code with the same dependencies, so one build serves
# both and they cannot drift apart: a response shape the dashboard expects is the shape the
# API in the same image produces.
#
# NOT BUILT LOCALLY. Docker is not installed on the owner's machine (docs/13, OQ-3), so this
# file is written to the requirements in docs/12 and is proved by CI, which builds it on every
# push and polls /health inside the running container. Until that has run, treat it as
# reviewed rather than tested.
#
#   docker build -t leadscore:local .
#   docker run --rm -p 8000:8000 leadscore:local
#   docker run --rm -p 8501:8501 leadscore:local dashboard

# Pinned by digest, not by tag: `3.11-slim` moves, and a base image that changes underneath a
# build is how a green pipeline starts producing a different artifact (the same class of
# problem as D-017 and 019).
#
# Digest resolved from Docker Hub on 2026-11-19 for python:3.11-slim. To move to a newer
# base, resolve the new digest deliberately and record the change in docs/14 - never paste a
# digest from memory, and never replace it with a floating tag to make a build pass.
FROM python:3.11-slim@sha256:07d4824036af339fa1b6ef769cc99d9c10a141769538b52b3b597c84f7a7b535 AS build

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /build

# The lockfile first, so a code change does not invalidate the dependency layer.
COPY requirements.lock ./
RUN python -m venv /opt/venv \
    && /opt/venv/bin/python -m pip install --upgrade pip \
    && /opt/venv/bin/python -m pip install -r requirements.lock

# The package itself, installed without dependencies: the lockfile already resolved them.
COPY pyproject.toml README.md ./
COPY src/ ./src/
RUN /opt/venv/bin/python -m pip install --no-deps .


# --- the image that actually ships -----------------------------------------------------
FROM python:3.11-slim@sha256:07d4824036af339fa1b6ef769cc99d9c10a141769538b52b3b597c84f7a7b535 AS runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/opt/venv/bin:$PATH" \
    LEADSCORE_MODEL_PATH=/app/models/model.joblib \
    LEADSCORE_MODEL_CARD_PATH=/app/models/model_card.json \
    LEADSCORE_DB_PATH=/app/data/predictions.db \
    LEADSCORE_API_BASE_URL=http://127.0.0.1:8000

# Two packages, both deliberate:
#   curl      - for HEALTHCHECK, and nothing else.
#   libgomp1  - the OpenMP runtime that LightGBM's Linux wheel links against dynamically.
#               `python:*-slim` does not ship it, so without this line the image builds
#               perfectly and then dies on `import lightgbm` with "libgomp.so.1: cannot open
#               shared object file". That is exactly how CI run #2 failed (D-032), and it is
#               invisible on Windows, where the wheel carries its own OpenMP.
# Still no compiler and no build tools: whatever a final image contains is attack surface,
# and this one needs neither to run.
RUN apt-get update \
    && apt-get install --no-install-recommends -y curl libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 10001 leadscore

COPY --from=build /opt/venv /opt/venv

WORKDIR /app
COPY --chown=leadscore:leadscore dashboard/ ./dashboard/
# The pinned Streamlit theme travels with the image. Without it the deployed demo
# takes its text colour from each viewer's own dark-mode setting, which is how the
# page became unreadable locally (D-035) - and a reviewer opening the public URL on a
# dark-mode laptop is exactly the case that must not break.
COPY --chown=leadscore:leadscore .streamlit/ ./.streamlit/
COPY --chown=leadscore:leadscore docker/entrypoint.sh ./entrypoint.sh
COPY --chown=leadscore:leadscore reports/figures/ ./reports/figures/

# The model is baked in, so the image is self-contained and its version is fixed. `models/`
# is git-ignored, so CI trains before it builds; a local build needs `make train` first.
COPY --chown=leadscore:leadscore models/ ./models/

RUN chmod +x ./entrypoint.sh && mkdir -p /app/data && chown leadscore:leadscore /app/data

USER leadscore
EXPOSE 8000 8501

# Hugging Face Spaces routes to 7860 by default; `app_port: 8501` in the Space's README
# points it at the dashboard instead. See docs/12.
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
    CMD curl -fsS http://127.0.0.1:8000/health || exit 1

ENTRYPOINT ["./entrypoint.sh"]
CMD ["api"]
