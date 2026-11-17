# 12 — Deployment Guide

Written in advance of Phase 6. Anything in here not yet executed is marked **[planned]**; nothing claims to work until it has been run.

## Environments

| Environment | Where | Purpose | Data | Model |
| --- | --- | --- | --- | --- |
| Local dev | the laptop | build and train | real `Leads.csv` in `data/raw/` | trained locally by `make train` |
| CI | GitHub Actions runner | lint, test, build, gate | 200-row fixture; the real CSV only on `main` | fixture model; a real one on `main` |
| Demo (production) | Hugging Face Spaces (Docker) · alt. Render | the public demo URL | none — the image carries no dataset | the CI-built artifact, baked into the image |

There is no staging environment, and that is deliberate: the deployable is a stateless container with an immutable model, so "staging" would differ from production only by URL. The capstone, which writes to real systems, gets one.

## Configuration

Every setting is an environment variable with a documented default in `.env.example`. Nothing environment-specific is hardcoded.

| Variable | Local | CI | Demo |
| --- | --- | --- | --- |
| `LEADSCORE_DATA_RAW` | `data/raw/Leads.csv` | fixture path | unset — no training in the demo |
| `LEADSCORE_MODEL_PATH` | `models/model.joblib` | fixture model | baked image path |
| `LEADSCORE_DB_PATH` | `data/predictions.db` | tmp | a mounted volume, or ephemeral — see below |
| `LEADSCORE_API_BASE_URL` | `http://localhost:8000` | — | the internal service URL |
| `LEADSCORE_MAX_BATCH` | 1000 | 1000 | 1000, possibly lower if abused |
| `LEADSCORE_API_KEY` | empty | empty | set only if the demo is abused |
| `LEADSCORE_LOG_LEVEL` | `INFO` | `WARNING` | `INFO` |
| `MLFLOW_TRACKING_URI` | `sqlite:///mlflow.db` | `sqlite:///mlflow.db` | unset — no tracking in the demo |
| `KAGGLE_USERNAME` / `KAGGLE_KEY` | `.env.local` | **GitHub secrets** | unset |

**Secrets rule:** the Kaggle token is the only real secret, and it exists in exactly two places — the owner's machine (`.env.local`, git-ignored) and GitHub Actions secrets. Never in the image, never in a log, never in a doc. Full rules in doc 09.

## Build

One image, two entrypoints — the API and the dashboard are the same code and the same dependencies, so one build serves both and they cannot drift apart.

**[planned]** `Dockerfile` requirements:

- multi-stage: a build stage installs from the lockfile, the final stage copies only the virtualenv, `src/`, `dashboard/` and `models/`
- pinned base image **by digest**, not a floating tag
- runs as a **non-root** user
- no compilers or build tools in the final layer
- `models/model.joblib` and `model_card.json` copied in at build time, so the image is self-contained and its version is fixed
- `HEALTHCHECK` hitting `/health`
- target: image < 1.5 GB (NFR-5); LightGBM, SHAP and matplotlib are the weight — no notebook dependencies in the image

**[planned]** `docker-compose.yml`: two services from the one image (`api` on 8000, `dashboard` on 8501), the dashboard depending on the API's healthcheck, the prediction DB on a named volume, and a `.env` file for local settings.

## Local run

```bash
# first time
cp .env.example .env.local      # then fill in nothing — defaults are fine locally
make install                    # dependency install
# download Leads.csv to data/raw/ (steps in the README)
make train                      # writes models/model.joblib + model_card.json

# run both services
docker compose up
# API       http://localhost:8000/docs
# Dashboard http://localhost:8501
```

Without Docker (current state of this machine — see OQ-3): `make serve` runs Uvicorn and `make dashboard` runs Streamlit, in two terminals. The Makefile provides both paths so a missing Docker install does not block Phases 1–5.

## CI/CD

**[planned]** `.github/workflows/ci.yml`, as specified in doc 10:

- **every push:** install from lockfile → `ruff` → `pytest` (fixtures only) → `docker build` → run the image and poll `/health`
- **`main` only:** download the dataset with the Kaggle secret → `make train` → the model quality gate → upload `model.joblib` and `model_card.json` as build artifacts

Deployment stays **manual** in P1. A push that passes CI does not ship a model; a human looks at the gate's numbers and the reliability curve and then deploys. Automatic model promotion needs monitoring that P1 does not have, and pretending otherwise would be the kind of shortcut the roadmap exists to avoid.

## Deploying the demo

**[planned]** Hugging Face Spaces, Docker SDK:

1. verify the free tier's current CPU, RAM and sleep policy — limits change, so they are checked at deploy time rather than trusted from memory
2. create a Space, SDK = Docker
3. push the repo; the Space builds the Dockerfile
4. expose the **dashboard** as the public surface, with the API internal to the container
5. set any non-default env var in the Space's settings, secrets in its secret store
6. confirm `/health` returns the expected `model_version`
7. record the URL in the README

Fallback: Render Web Service from the same Dockerfile — one build command, one port. The image is portable on purpose, so the host is replaceable.

**Expected limits, to be verified not assumed:** a free tier sleeps when idle, so the first request after a sleep is slow; the README says so beside the demo link rather than letting a reviewer think the model is slow.

## State and the prediction log in production

The container is stateless apart from the SQLite file. Two options, and the choice is deliberate:

- **Ephemeral** (default for a public demo): the log resets on restart. Acceptable — the demo's log is not evidence of anything, and there is nothing to lose.
- **Persistent volume**: needed only if live accuracy from demo traffic is ever reported. If that is wanted, it comes with the retention window from OQ-7.

Either way the model is never mutable at runtime; `models/` is read-only in the container.

## Rollback

1. redeploy the previous image tag — the model is inside it, so rolling back the image rolls back the model
2. confirm `/health` reports the expected older `model_version`
3. record what happened and why in doc 14

No migration, no data to unwind, no coordination between instances. That property is the reason the training and serving lanes are split by a file.

## Monitoring, honestly

P1 has no tracing stack — Langfuse arrives in P3 with the first LLM project. What exists:

- `/health` for liveness, used by the host and by CI
- per-request latency and model version in the prediction log (NFR-1 measured from real traffic)
- `/feedback` making live accuracy computable once outcomes exist
- the host's own logs and uptime view

What is missing and known: no alerting, no error aggregation, no dashboard of drift. For a public demo of a scoring pipeline that is the right amount; the capstone adds a cost dashboard and an error-rate alert, where real traffic justifies them.

## Release checklist

- [ ] CI green on `main`, quality gate passed, numbers recorded in doc 14
- [ ] `model_card.json` metrics filled from a real run, no `null`s
- [ ] README results table matches the card
- [ ] image builds, runs as non-root, `/health` returns the expected version
- [ ] demo URL live; cold-start caveat noted beside the link
- [ ] no dataset, model, DB or secret in git history (T-702)
- [ ] docs 01–14 reconciled with what shipped (T-703)
- [ ] roadmap statuses honest — nothing marked `Done (verified)` that was not run
