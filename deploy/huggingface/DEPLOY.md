# Deploying the demo to Hugging Face Spaces (task T-604) - SUPERSEDED

> **Hugging Face made Docker Spaces a paid feature in July 2026.** Creating one now requires
> a PRO subscription ($9/month); only Static Spaces remain free, and a static page cannot run
> this API. Verified on the owner's account on 2026-11-19: the create-Space screen labels both
> Docker and Gradio **Paid**.
>
> **The live target is Google Cloud Run instead - see [../cloudrun/DEPLOY.md](../cloudrun/DEPLOY.md).**
>
> This file is kept, not deleted, for two reasons: the steps still work if the owner ever
> takes a PRO subscription, and the episode is the clearest example in this project of why
> docs/12 says to verify a host's limits at deploy time rather than trust them from memory.
> That instruction was written in Phase 0 and paid for itself here.

Written to be followed by the repo owner, because every step needs an account this project
does not have. Nothing here has been executed yet — the Space does not exist until you create
it, so treat this as instructions rather than a record.

## The one real complication

`models/model.joblib` and `models/model_card.json` are **git-ignored** in the GitHub repo, on
purpose: a trained model is a build artifact and the rule against committing it is the same
rule that keeps the dataset out (CLAUDE.md). But the image bakes the model in, so the Space's
build needs those two files present.

So the Space repo is **not** a mirror of the GitHub repo. It carries the model; GitHub does
not. That is the one deliberate difference, and it is why step 4 copies them in by hand.

If you would rather not keep a model in a second git repo, the alternative is to have the
Space download it at build time from a CI artifact or a model registry — more moving parts
than a P1 demo needs, and it is written down in docs/13 as OQ-13 rather than built.

## Steps

**1. Train a model locally, and check the gate passed.**

```bash
.venv/Scripts/python -m leadscore.train
```

The run must end with `quality gate PASSED` and must have written both files under `models/`.
A failing gate means there is nothing worth deploying.

**2. Create the Space.**

On huggingface.co: **New** → **Space**. Owner = your account, name = `lead-scoring-pipeline`,
**SDK = Docker**, hardware = the free CPU tier, visibility = public. Do not pick a template.

**3. Clone the Space and copy the project in.**

```bash
git clone https://huggingface.co/spaces/<your-username>/lead-scoring-pipeline hf-space
cd hf-space

# Everything the image needs to build:
cp -r ../lead-scoring-pipeline/src .
cp -r ../lead-scoring-pipeline/dashboard .
cp -r ../lead-scoring-pipeline/docker .
cp ../lead-scoring-pipeline/Dockerfile .
cp ../lead-scoring-pipeline/pyproject.toml .
cp ../lead-scoring-pipeline/requirements.lock .

# The Space's own README carries the front matter Hugging Face reads:
cp ../lead-scoring-pipeline/deploy/huggingface/README.md ./README.md

# The charts the model card page shows:
mkdir -p reports/figures
cp ../lead-scoring-pipeline/reports/figures/*.png reports/figures/
```

**4. Add the model — the step GitHub does not do for you.**

```bash
mkdir -p models
cp ../lead-scoring-pipeline/models/model.joblib models/
cp ../lead-scoring-pipeline/models/model_card.json models/
```

`model.joblib` is a few megabytes, so plain git is fine; if it ever grows past 10 MB, Hugging
Face will ask you to use Git LFS and tell you the command.

**5. Tell the container to run both services.**

The Space runs **one** container, so the dashboard is the public surface and the API stays
internal on loopback. No file to edit: in the Space, go to **Settings -> Variables and
secrets -> New variable** and add

| Name | Value |
| --- | --- |
| `LEADSCORE_MODE` | `both` |

`docker/entrypoint.sh` reads that variable ahead of the Dockerfile's `CMD`, starts Uvicorn on
127.0.0.1:8000, waits for `/health` to answer, then starts Streamlit on 8501 - which is the
port `app_port: 8501` in the README front matter tells Hugging Face to route. It is a
variable, not a secret: nothing about it is sensitive.

**6. Push.**

```bash
git add -A
git commit -m "Deploy lead scoring demo"
git push
```

The Space builds the Dockerfile. Watch the build log in the browser; a first build takes
several minutes because LightGBM, SHAP and Streamlit are large.

**7. Check it, then record the URL.**

- the three pages load, and **Rank leads** scores the sample CSV
- the model card page shows the version that matches `models/model_card.json`
- the footer names that same version

Then put the URL in the project README beside the cold-start note, and in `docs/12`.

## What to expect

- **It sleeps.** The free tier suspends an idle Space; the first request afterwards waits for
  the container and the model load. The README says so beside the link, so a reviewer does not
  read a cold start as a slow model.
- **The prediction log is ephemeral.** The container's SQLite file disappears on restart.
  Fine: the demo's log is evidence of nothing. Making it persistent means a volume and a
  retention window (OQ-7), which a public demo does not need.
- **No secrets.** The Space needs none. The Kaggle token stays on your machine and in GitHub
  Actions; the demo never trains.

## If it fails to build

| Symptom | Cause | Fix |
| --- | --- | --- |
| `no model at /app/models/model.joblib` at startup | step 4 was skipped | copy both model files, commit, push |
| build dies pulling the base image | the pinned digest is not available for the Space's architecture | resolve the current digest for `python:3.11-slim` and update both `FROM` lines together |
| the Space shows a blank page | Streamlit is listening on the wrong port | `app_port` in the README must match the port `entrypoint.sh` gives Streamlit |
| build times out | the free tier's limit | build locally once to warm nothing — or move to Render, which docs/12 keeps as the fallback |
