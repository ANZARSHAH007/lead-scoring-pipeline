# 13 — Assumptions and Open Questions

Updated 2026-11-13. Three separate things, kept separate on purpose: what is **confirmed**, what is **assumed** (and would change the build if wrong), and what is **open** (and needs an answer from the owner).

## Confirmed requirements

From the owner's P1 design doc and roadmap. Treated as decided; changing one of these is a scope change, not a clarification.

| # | Confirmed |
| --- | --- |
| C-1 | Target delivery 2026-11-29, ten sessions of 1.5–2 h |
| C-2 | Dataset: X Education `Leads.csv`, 9,240 × 37, target `Converted`, 38.54% positive |
| C-3 | Headline metric is **Capture@20%** |
| C-4 | Leakage rule, and the published leakage row in the results table |
| C-5 | Three baselines: random, one-rule (time on site), logistic regression |
| C-6 | LightGBM challenger, Optuna tuning, calibration |
| C-7 | Grades A = top 20%, B = next 30%, C = the rest |
| C-8 | Five API endpoints; SQLite prediction log; Streamlit dashboard with three pages |
| C-9 | Docker Compose, GitHub Actions, CI quality gate, live demo link, README results table |
| C-10 | Cut order under time pressure: Optuna, then `/feedback`; never the leakage audit, baselines or README |
| C-11 | Repo name `lead-scoring-pipeline`, **public** from the first commit |
| C-12 | The deliverable is the **pipeline**; the dashboard is only its demo surface |

## Assumptions

Each is a decision made to keep moving. If one is wrong, the listed consequence follows — so each is worth a glance before coding starts.

| # | Assumption | If wrong |
| --- | --- | --- |
| A-1 | ~~The Kaggle version matches doc 02's figures~~ — **confirmed 2026-11-13**: 9,240 x 37, conversion rate 0.3854 exactly, every column present | one correction was needed: doc 02 mixed blank-only and blank-plus-`Select` rates. Both bases are now tabulated, and `City` (39.7% effectively missing) was added |
| A-2 | ~~`Last Activity` / `Last Notable Activity` partly encode sales contact~~ — **confirmed 2026-11-17** | they do: `SMS Sent` (62.9% / 69.4%) and `Unreachable` (68.8%) are company actions. Both columns excluded; the variant is reported as a labelled comparison (D-016) |
| A-3 | Website-behaviour columns are the strongest admissible signals | if not, the one-rule baseline is weak and the model's lift over it looks larger than it is; error analysis (T-406) catches this |
| A-4 | ~~LightGBM will beat logistic regression~~ — **confirmed 2026-11-17, narrowly** | untuned it won by +0.004 (inside fold noise); tuned, by **+0.015** (above it). The margin is real but small, so Phase 4 decides on calibration and Capture@20% too, not ROC-AUC alone |
| A-5 | Calibration will be needed (boosted trees are usually overconfident) | if the reliability curve is already diagonal, skip `CalibratedClassifierCV` and record that it was not needed |
| A-6 | 9,240 rows train in seconds on a laptop CPU | if tuning is slow, cut the Optuna budget (it is already first on the cut list) |
| A-7 | A free tier can host the image within 2 GB RAM | drop SHAP from the serving path and precompute reasons, or move to Render; decided at Phase 6 |
| A-8 | SQLite is sufficient for demo traffic | the log is isolated behind `store.py`; swapping it is a one-file change |
| A-9 | No PII in P1, so `input_json` can be stored whole | the controls in doc 09 for the carry-over case are applied immediately |
| A-10 | Streamlit is acceptable for the demo surface | see OQ-5 |
| A-11 | The owner has a Kaggle account and can generate an API token | CI cannot retrain on `main`; the gate then runs on the fixture only and the real run stays local |
| A-12 | ~~University exams will consume some sessions~~ — **resolved 2026-11-13: mid-terms are done, the owner is free until finals** | capacity is now 3–4 h a day. **Finals start 2027-01-01**, so the free window ends around 2026-12-20 once revision time is allowed for |

## Open questions

Ordered by what blocks soonest. **OQ-2, OQ-3 and OQ-4 block Phase 1 and need an answer before any code is written.**

### OQ-1 — Real-estate theme decision, due 2026-11-25
Does P1 stay on the X Education dataset, or re-theme to property inquiries?
**Current plan:** P1 stays on X Education **either way** — the re-theme is a carry-over (new `config.py`, `schema.py`, API fields), not a rewrite. No action needed now; noted so the Phase-5 timing is not a surprise.
**Blocks:** nothing in P1. **Owner decision due:** 2026-11-25.

### OQ-2 — ~~Dependency manager~~ **RESOLVED 2026-11-13: `pip` + `venv`**
`uv` is in the P1 design doc but **is not installed on this machine** (`uv: command not found`). Installing it is a machine-wide change, which needs your approval.
**Options:** (a) approve `uv` — fast, lockfile, matches the design doc, one global install; (b) `pip` + `venv` + `requirements.txt` — nothing to install, slower, weaker lockfile; (c) `uv` in CI only, `pip` locally — divergence between CI and laptop, which is the thing a lockfile exists to prevent.
**Resolved:** (b), `pip` + `venv`. The owner said to start Phase 1 without answering this explicitly, and installing `uv` is a machine-wide change that needs a clear yes, so the path that needs no install was taken. `pyproject.toml` is uv-compatible, so `uv sync` works later with no repo change. Revisit if install speed becomes a real cost.

### OQ-3 — Docker: ~~install now~~ **DEFERRED to Phase 6 (2026-11-27)** — still open then
Docker is **not installed** (`docker: command not found`, no Docker Desktop directory). Phases 1–5 do not need it; Phase 6's `docker compose up` and the CI image build do, and it is an acceptance criterion (FR-11).
**Options:** (a) install Docker Desktop now; (b) install it at Phase 6; (c) skip local Docker and let CI build the image — the image then ships untested locally, which contradicts "a stranger can run it".
**Recommendation:** (b) — it is not needed until 2026-11-27, so it should not block Session 1. **Owner decision needed by Phase 6.**

### OQ-4 — ~~Python 3.11 or 3.12?~~ **RESOLVED 2026-11-13: 3.11.9**
Installed: **3.11.9** (Microsoft Store build). The design doc assumed 3.12.
**Options:** (a) target 3.11 — nothing to install, all dependencies support it; (b) install 3.12 — matches the doc, another machine-wide change for no functional gain.
**Resolved:** (a). The Store build's virtualenv quirk did **not** materialise — `.venv` was created and the suite runs from it. `requires-python = ">=3.11"`, and CI pins 3.11.

### OQ-5 — Does the demo surface stay Streamlit?
Streamlit gives a three-page UI in one file but cannot carry the generated design system beyond theme tokens (doc 08).
**Options:** (a) keep Streamlit, apply tokens via `.streamlit/config.toml`, accept its look — cheapest, and P1 is a pipeline; (b) a small static HTML + Tailwind page against the API — full design control, roughly one extra session; (c) defer the polished UI to P3, where React is on the roadmap anyway.
**Recommendation:** (a) now, (c) later. **Owner decision needed by Phase 6.**

### OQ-6 — Timezone for displayed timestamps?
Storage is UTC, settled. Display is not: the owner is in **UTC+5**.
**Options:** (a) show UTC everywhere with a `Z` suffix; (b) show local time with the offset labelled.
**Recommendation:** (a) in the API, (b) in the dashboard with the offset shown. Low impact, needs one answer.

### OQ-7 — Prediction-log retention?
How long do prediction rows live, and does the demo's log persist across restarts (doc 12)?
**Options:** (a) ephemeral in the demo, keep everything locally — simplest, nothing to manage; (b) a persistent volume with a 90-day window — needed only if live accuracy from demo traffic is ever published.
**Recommendation:** (a) for P1. Becomes mandatory to revisit if real inquiry data ever flows through (A-9).

### OQ-8 — Host: Hugging Face Spaces or Render?
**Recommendation:** Spaces (Docker SDK, free, no card), Render as the fallback. Either way the current free-tier limits are verified at deploy time, not assumed. **Decide at Phase 6.**

### OQ-9 — Does the public demo need the API key enabled?
A public `/score/batch` accepting 1,000 leads is cheap to abuse.
**Recommendation:** ship open, watch it, set `LEADSCORE_API_KEY` and add per-IP rate limiting only if abused. **No action now.**

### OQ-10 — Should CI download the dataset on `main`?
It needs the Kaggle token as a repo secret (A-11), in a **public** repo. GitHub secrets are not exposed to forked-PR builds, so this is safe, but it is worth a conscious yes.
**Options:** (a) yes — CI retrains and runs the real gate; (b) no — CI gates on the fixture only, the real gate runs locally and its numbers are pasted into the PR.
**Recommendation:** (a). **Owner decision needed by Phase 6.**

### OQ-11 — How is the GitHub repo created? `gh` is not installed.
**Options:** (a) create it in the browser, then `git remote add` — no install; (b) install the GitHub CLI.
**Still owner action.** The local repo is initialised and committed (`872d629`); create `lead-scoring-pipeline` as a **public** repo on GitHub, then `git remote add origin <url>` and `git push -u origin main`.

### OQ-12 — Who is the "friend" for the T-701 README run-through?
The Phase-1 gate in the roadmap requires someone else to run the repo unaided. Needs a real person named before 2026-11-29. **Owner action.**

## How this doc is used

A task that depends on an open question is marked `Blocked` in doc 11 and is not started on a guess. When a question is answered, the answer moves to **Confirmed** here, the decision and its reasoning go into doc 14, and the blocked task's status changes in the same session.
