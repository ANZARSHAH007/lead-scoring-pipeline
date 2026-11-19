# Deploying to Google Cloud Run (task T-604)

The production-shaped path: the image goes to a registry, Cloud Run runs an immutable
revision of it, and traffic is pointed at that revision. Nothing about this is specific to a
demo — it is the same sequence a team uses to ship a service, which is why it is worth
learning on a project you already understand.

Written to be run by the repo owner. Nothing here has been executed; the Space, project and
service do not exist until you make them.

## What you are actually doing, in four moves

| Move | Command | What it means |
| --- | --- | --- |
| 1. Build | `gcloud builds submit` | your source is uploaded, Docker builds it **on Google's machines**, the image lands in Artifact Registry |
| 2. Store | Artifact Registry | a private registry of versioned images; the tag is the thing you deploy |
| 3. Deploy | `gcloud run deploy` | creates an **immutable revision** of the service from that image |
| 4. Route | automatic, by default | 100% of traffic is sent to the new revision; the old one stays and can be rolled back to |

The part worth internalising: **a revision is immutable**. You never edit a running service —
you build a new image, deploy a new revision, and move traffic. A rollback is therefore just
pointing traffic at an older revision, which takes seconds and cannot half-fail. That is the
same property the training lane has, where a model is a file rather than something mutated in
place.

## Before you start

- A Google account.
- A **billing account** attached to the project. Cloud Run's free allowance (180,000
  vCPU-seconds, 360,000 GiB-seconds, 2M requests per month) covers a demo like this many
  times over, but Google requires a card on file to enable the APIs. You should owe nothing;
  set a budget alert at step 2 so you find out immediately if that ever changes.
- A trained model on disk. Check it:

  ```bash
  cd e:/AI_Projects/lead-scoring-pipeline
  ls -la models/
  ```

  You want `model.joblib` and `model_card.json`. If they are missing, run
  `.venv/Scripts/python -m leadscore.train` and confirm it ends with `quality gate PASSED`.

## 1. Install the gcloud CLI

Download the Google Cloud CLI installer for Windows from
<https://cloud.google.com/sdk/docs/install> and run it. It is a per-user install, not a
system service, and it does not need Docker.

Then, in a **new** terminal:

```bash
gcloud --version
gcloud auth login
```

A browser opens; sign in and allow access. `gcloud auth login` stores a credential for the
CLI itself.

## 2. Create the project and turn on billing

```bash
# A globally unique id. Lower case, digits and dashes.
gcloud projects create leadscore-demo-001 --name="Lead Scoring Demo"
gcloud config set project leadscore-demo-001
```

Then in the console at <https://console.cloud.google.com/billing> link the project to your
billing account, and while you are there set a **budget alert** at a small amount — $1 is
fine. It costs nothing and it means a surprise shows up as an email rather than as a bill.

Confirm billing is on:

```bash
gcloud beta billing projects describe leadscore-demo-001
```

`billingEnabled: true` is what you want.

## 3. Enable the three services this needs

```bash
gcloud services enable run.googleapis.com artifactregistry.googleapis.com cloudbuild.googleapis.com
```

- **run** — the runtime.
- **artifactregistry** — where images live.
- **cloudbuild** — builds the image on Google's machines, so you do not need Docker locally.

That last one is why this path works for you today: the build happens server-side.

## 4. Create the image registry

```bash
gcloud artifacts repositories create leadscore \
  --repository-format=docker \
  --location=asia-south1 \
  --description="Lead scoring pipeline images"
```

`asia-south1` is Mumbai — the closest region to you, so cold starts and latency are lower.
Any region works; use the same one everywhere below.

## 5. Build the image

```bash
cd e:/AI_Projects/lead-scoring-pipeline

gcloud builds submit \
  --tag asia-south1-docker.pkg.dev/leadscore-demo-001/leadscore/api:v1 \
  --timeout=20m
```

This uploads the directory (minus everything in `.gcloudignore`), builds the Dockerfile, and
pushes the result. First build takes several minutes: LightGBM, SHAP and Streamlit are large.

**`.gcloudignore` is doing something important here.** Without it, gcloud falls back to
`.gitignore`, which excludes `models/` — the build would succeed and the container would die
at startup with `no model at /app/models/model.joblib`. The file in this repo deliberately
uploads `models/` while still excluding the dataset, the prediction log, `.venv` and secrets.

Check what was uploaded if a build ever behaves oddly:

```bash
gcloud builds list --limit=3
gcloud builds log <BUILD_ID>
```

## 6. Deploy

```bash
gcloud run deploy leadscore \
  --image asia-south1-docker.pkg.dev/leadscore-demo-001/leadscore/api:v1 \
  --region asia-south1 \
  --platform managed \
  --allow-unauthenticated \
  --memory 1Gi \
  --cpu 1 \
  --cpu-boost \
  --min-instances 0 \
  --max-instances 2 \
  --timeout 120 \
  --set-env-vars LEADSCORE_MODE=both
```

Every flag is a decision, so here is why each one:

| Flag | Why |
| --- | --- |
| `--allow-unauthenticated` | it is a public demo; without this, every request needs a Google identity |
| `--memory 1Gi` | measured need is ~400 MB (API 259 MB + dashboard 137 MB); 1 GiB leaves real headroom and still fits the free allowance |
| `--cpu-boost` | extra CPU during startup, which is when the model loads — this is the cold-start fix |
| `--min-instances 0` | scale to zero. An idle demo then costs nothing, at the price of a cold start |
| `--max-instances 2` | a cap so a traffic spike cannot run up a bill |
| `--timeout 120` | a 1,000-lead batch is slower than a default-length request |
| `LEADSCORE_MODE=both` | one container runs the API on loopback and the dashboard on `$PORT`, which Cloud Run sets |

The command prints a URL ending in `.run.app`. That is your demo link.

## 7. Check it

```bash
SERVICE_URL=$(gcloud run services describe leadscore --region asia-south1 --format='value(status.url)')
echo "$SERVICE_URL"

curl -fsS "$SERVICE_URL/_stcore/health"     # the dashboard answers
```

Then open the URL in a browser and check the four things:

1. three pages load, **Rank leads** is the default;
2. the sample CSV downloads, and uploading it back scores and sorts;
3. the **Model card** page shows the version that matches `models/model_card.json`;
4. the footer names that version on every page.

The API is internal in this mode, so there is no public `/docs`. If you want the API public
too, deploy a second service from the same image with `LEADSCORE_MODE=api` — same image, a
different revision, which is the whole point of building it this way.

## 8. Watch it, and know what it costs

```bash
# structured logs, as the service emits them
gcloud run services logs read leadscore --region asia-south1 --limit 50

# every revision, and which one has traffic
gcloud run revisions list --service leadscore --region asia-south1
```

Cost: with `--min-instances 0`, you are billed only while a request is being served. The free
allowance is 360,000 GiB-seconds a month; at 1 GiB that is 100 hours of *active* time. A
reviewer clicking around uses minutes.

## Shipping a new model

This is the loop worth remembering, because it is the real one:

```bash
.venv/Scripts/python -m leadscore.train          # gate must pass
gcloud builds submit --tag asia-south1-docker.pkg.dev/leadscore-demo-001/leadscore/api:v2
gcloud run deploy leadscore --image asia-south1-docker.pkg.dev/leadscore-demo-001/leadscore/api:v2 --region asia-south1
```

A new tag, a new revision, traffic moves. To roll back:

```bash
gcloud run services update-traffic leadscore --region asia-south1 --to-revisions leadscore-00001-abc=100
```

The model is inside the image, so rolling back the revision rolls back the model. No
migration, nothing to unwind — the property docs/12 described, now actually true.

## If it fails

| Symptom | Cause | Fix |
| --- | --- | --- |
| `no model at /app/models/model.joblib` | `models/` not uploaded | confirm `.gcloudignore` exists and does **not** list `models/` |
| `Container failed to start and listen on PORT` | the dashboard is not on `$PORT` | `LEADSCORE_MODE=both` must be set; the entrypoint reads `$PORT` for Streamlit |
| build fails on the base image digest | the pinned digest is unavailable | resolve the current `python:3.11-slim` digest deliberately and update both `FROM` lines |
| `PERMISSION_DENIED` enabling services | billing not linked | step 2 |
| first request takes 20+ seconds | cold start loading the model | expected with `--min-instances 0`; say so beside the link, or set `--min-instances 1` and accept the cost |
| build times out | 20 minutes was not enough | raise `--timeout`, e.g. `--timeout=30m` |

## Tearing it down

If you stop wanting the demo, delete the project and everything inside it goes with it:

```bash
gcloud projects delete leadscore-demo-001
```

Worth knowing *before* you create things, which is why it is written here rather than left to
be discovered.
