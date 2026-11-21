# 11 — Development Roadmap

Eight phases over ten working sessions, **2026-11-13 → 2026-11-29**.

**Capacity update, 2026-11-13: mid-term exams are finished and the owner is free until finals.** Sessions were planned at 1.5–2 h; budget 3–4 h instead, which means two sessions can land in one sitting and the windows below can compress by roughly a week. **Finals start 2027-01-01**, so the free window is 2026-11-13 → roughly 2026-12-20, leaving the last week-and-a-half of December for revision. Both P1 (due 2026-11-29) and P2 (due 2026-12-20) fit inside it, but P2 has no slack against finals prep — bank it early. The dates stay as written because they are the commitment; finishing early is the upside. The extra hours go into the work that is already planned — a fuller leakage audit (T-203), the error analysis (T-406), and the README — not into new scope. Phases 2–4 happen in notebooks; 5–6 turn them into a product. Every session ends at a commit.

Status values: `Not started` · `In progress` · `Blocked` · `Done (verified)`. A task becomes `Done (verified)` only when its acceptance criterion has actually been run — not when the code merely exists.

## Phase overview

| Phase | Session | Window | Goal | Exit condition |
| --- | --- | --- | --- | --- |
| 0 Planning | — | 2026-11-13 | docs + structure, no code | owner has read docs 01–14 and approved |
| 1 Foundation | 1 | ~~2026-11-14~~ **done 2026-11-13** | repo runs, config exists, tests execute | met: ruff clean, 32 tests passing |
| 2 Data + leakage | 2–3 | 2026-11-15/17 | the data is understood and the column decisions are recorded | leakage audit notebook complete, schema enforced |
| 3 Baselines + model | 4–5 | 2026-11-18/20 | a challenger beats three baselines | MLflow shows all four, LightGBM wins on CV |
| 4 Calibrate, grade, explain | 6–7 | 2026-11-21/23 | trustworthy scores with reasons | Capture@20% measured once on the test set |
| 5 Serving | 8–9 | 2026-11-24/26 | the pipeline is an API | `/score` returns score, grade, reasons; log writes |
| 6 Package + demo | 10 | 2026-11-27/28 | anyone can run it, anyone can see it | compose up works, CI green, public URL live |
| 7 Hardening | buffer | 2026-11-29 | the repo survives a stranger | a friend runs it from the README unaided |

## Tasks

Dependencies use task IDs. Priority: **P0** = P1 fails without it, **P1** = planned, **P2** = cut first.

### Phase 1 — Foundation (Session 1)

| ID | Task | Feature | Depends | Pri | Expected outcome | Acceptance criterion | Status |
| --- | --- | --- | --- | --- | --- | --- | --- |
| T-101 | Decide and install the toolchain (uv or pip+venv; Docker; gh) | — | — | P0 | approved tool list, installed | `uv --version` or `python -m venv` both documented and working | **Done (verified)** — pip + venv; nothing installed machine-wide |
| T-102 | `pyproject.toml` with pinned deps, ruff and pytest config | M1 | T-101 | P0 | one install command sets up the project | fresh env installs with no manual step | **Done (verified)** |
| T-103 | `src/leadscore/config.py` — paths, seed, thresholds, feature lists | M1 | T-102 | P0 | no magic numbers anywhere else | `test_config.py` asserts defaults and env override | **Done (verified)** — 32 tests |
| T-104 | `Makefile` targets: `install train serve test lint docker` | — | T-102 | P1 | one entry point per action | every target runs or explains why it cannot | **Done (untested)** — `make` is not installed on this machine; the README lists each target's raw command, and CI runs on Linux |
| T-105 | `tests/` scaffold + first real test | — | T-103 | P0 | the suite exists and runs | `make test` exits 0 with ≥ 1 passing test | **Done (verified)** — ruff clean, 32 passed |
| T-106 | `git init`, first commit, push to the public repo | — | T-102 | P0 | history starts clean | remote shows the commit; no data/model/secret in it | **In progress** — local commit `872d629`, 34 files, verified clean; remote not created yet (owner, OQ-11) |
| T-107 | Download `Leads.csv` to `data/raw/` and confirm 9,240 × 37 and 38.54% positive | M2 | T-106 | P0 | the real data is on disk, untracked | shape and target rate printed and matching | **Done (verified)** 2026-11-13 — 9,240 x 37, conversion 0.3854, drift 0.0000, all 37 config columns present, 12/12 checks pass |
| T-108 | `scripts/doctor.py` — a two-second preflight: Python, editable install, imports, lockfile, settings, MLflow store, dataset, write targets | — | T-103 | P0 | environment faults are named up front, not discovered mid-run | a broken install, a missing package or a retired MLflow store each produce one named line and exit 1 | **Done (verified)** 2026-11-17 — 10 checks, exit 0 with 2 warnings on this machine |
| T-109 | `src/leadscore/fixtures.py` + `tests/test_smoke.py` — the whole chain on 300 synthetic rows, no real data | M1 | T-305 | P0 | CI and tests can run the real pipeline without the dataset | the fixture satisfies the contract; the chain validates, splits, cross-validates, tracks and reports in a temp dir | **Done (verified)** 2026-11-17 — fixture 37 cols, 15 tests, chain green in ~10 s |
| T-110 | `plotting.metric_axis` — the honest-axis rule enforced in code | — | T-305 | P1 | a truncated metric axis cannot hide the baseline | the axis helper raises when the chance line falls outside the limits | **Done (verified)** 2026-11-17 — 8 tests; report 03's chart now goes through it |
| T-111 | `requirements.lock` — pin the resolution that actually works | — | T-102 | P0 | a second machine and CI install what this machine ran | 111 packages pinned; `mlflow>=3.1` floor and `sqlalchemy>=2.0,<2.1` in pyproject, so neither a 2.x file store nor a 2.1 deprecation can come back | **Done (verified)** 2026-11-18 — strict warning filter needs no exceptions |
| T-112 | `config.resolve()` — every output path configured, relative to the project root | M1 | T-103 | P1 | no module derives an output path from `__file__` | reports and figures land in the same place from any working directory; an absolute override passes through | **Done (verified)** 2026-11-18 — `train.py`, `eda.py`, `leakage_audit.py` all routed through it; both reports regenerate byte-identical |

### Phase 2 — Data contract and leakage audit (Sessions 2–3)

| ID | Task | Feature | Depends | Pri | Expected outcome | Acceptance criterion | Status |
| --- | --- | --- | --- | --- | --- | --- | --- |
| T-201 | `scripts/eda.py` → `reports/01_eda.md` — missingness, near-constant columns, numeric profile, conversion by origin and source | M2 | T-107 | P0 | the data is understood, 4 charts saved | **Done (verified)** 2026-11-13 — report + 4 figures generated; 3 findings changed the plan |
| T-202 | `src/leadscore/schema.py` — pandera contract | M2 | T-201 | P0 | bad data fails loudly | three corruption tests produce three distinct errors | **Done (verified)** — 20 schema tests; the real file validates clean |
| T-203 | `notebooks/02_leakage_audit.ipynb` — a yes/no decision per column, with reason | M2 | T-201 | **P0** | the audit record doc 02 promises | every one of the 37 columns is classified | **Done (verified)** — `reports/02` carries the full decision record; `classify()` raises if any column is unclassified |
| T-204 | Quantify the leakage gap: logistic regression with vs without sales columns | M2 | T-203 | P0 | the number the README's last row reports | both ROC-AUCs reported in `reports/02` | **Done (verified)** — 0.866 shipping vs 0.982 with leakage, **+0.117**. MLflow deferred to Phase 3 (T-303) where it is introduced; the numbers here come from `cross_val_score` with the config seed |
| T-205 | Freeze the feature sets in config | M1 | T-203 | P0 | `shipping` (13), `with_activity`, `audit_with_leakage` | **Done (verified)** — `test_config` asserts the shipping set is 13 columns and contains no banned, flag or activity column |

### Phase 3 — Baselines and challenger (Sessions 4–5)

| ID | Task | Feature | Depends | Pri | Expected outcome | Acceptance criterion | Status |
| --- | --- | --- | --- | --- | --- | --- | --- |
| T-301 | `features.py` — ColumnTransformer, rare-level grouping, unknown-safe one-hot | M3 | T-205 | P0 | identical preprocessing both lanes | an unseen category scores without error | **Done (verified)** — 16 feature tests, including unseen categories, all-null rows and a joblib round trip |
| T-302 | Single stratified 80/20 split, seeded, in `train.py` | M4 | T-301 | P0 | one split for the whole project | split is reproducible across runs | **Done (verified)** — 7,392 train / 1,848 test; a test spies on `cross_val_score` and asserts no held-out index reaches it |
| T-303 | Three baselines with 5-fold CV, logged to MLflow | M4 | T-302 | P0 | the bar to beat | MLflow lists random, one-rule, logistic regression | **Done (verified)** — 0.5000 / 0.6517 / 0.8679 CV ROC-AUC, tracked in `sqlite:///mlflow.db` |
| T-304 | LightGBM + Optuna (budget from config) | M4 | T-303 | P1 | the challenger | CV ROC-AUC beats logistic regression | **Done (verified)** — 0.8825 ± 0.0045 tuned, vs 0.8679. Untuned it was 0.8718, inside the noise; tuning earned the margin |
| T-305 | Pick the winner on CV only; test set still untouched | M4 | T-304 | P0 | an honest selection | the run log states the test set was not read | **Done (verified)** — selection ran on CV only; `evaluate.py` opens the test set in Phase 4 |

### Phase 4 — Calibration, grades, explanations (Sessions 6–7)

| ID | Task | Feature | Depends | Pri | Expected outcome | Acceptance criterion | Status |
| --- | --- | --- | --- | --- | --- | --- | --- |
| T-401 | Reliability curve + Brier; calibrate if off-diagonal | M5 | T-305 | P0 | a score that means a probability | curve plotted; Brier recorded before and after | **Done (verified)** 2026-11-18 — out-of-fold Brier: raw 0.13119, sigmoid 0.13145, isotonic 0.13195. **No calibrator won**, so none was applied (D-020); the reliability curve is in report 04 |
| T-402 | Grade cut-offs A/B/C from validation data → model card | M5 | T-401 | P0 | bands nobody hardcodes | `/model-info` later serves these exact numbers | **Done (verified)** 2026-11-18 — A from score 76, B from 26, set on out-of-fold validation scores and stored in the card, never hardcoded |
| T-403 | `evaluate.py` — ROC-AUC, PR-AUC, Brier, Capture@20%, decile lift | M5 | T-401 | P0 | the published metrics | all four computed on the single test-set pass | **Done (verified)** 2026-11-18 — one pass over 1,848 held-out rows for all five models; the card records `test_set_openings: 1` and a fingerprint of the row ids |
| T-404 | Quality gate function + thresholds written into config | M5 | T-403 | P0 | a worse model cannot ship | a deliberately crippled model is rejected | **Done (verified)** 2026-11-18 — 5 checks, all passed; a label-shuffled model is rejected in `test_model_quality.py`, and a failing gate stops the artifact being written at all |
| T-405 | `explain.py` — SHAP top-3 + phrase map | M6 | T-403 | P0 | three readable reasons | every admissible feature has both directions mapped | **Done (verified)** 2026-11-18 — 13/13 admissible columns mapped in both directions, 19 tests, including one that checks the attribution sign follows the data |
| T-406 | Error analysis: 20 high-score non-converters, 20 low-score converters | M5 | T-403 | P1 | written list of what the model misses | findings recorded in the notebook and doc 14 | **Done (verified)** 2026-11-18 — both groups in report 04 with computed findings; the referral-channel pattern is recorded in docs/14 |
| T-407 | Write `models/model_card.json` + the README results table | M4 | T-403 | P0 | the headline number published | five rows, four columns, all from real runs | **Done (verified)** 2026-11-18 — the README block is **generated** from the card by `scripts/publish_results.py`, with `--check` for CI, so the two cannot disagree |

### Phase 5 — Serving (Sessions 8–9)

| ID | Task | Feature | Depends | Pri | Expected outcome | Acceptance criterion | Status |
| --- | --- | --- | --- | --- | --- | --- | --- |
| T-501 | Move notebook code into `src/leadscore/`; `make train` end to end | M4 | T-407 | P0 | one command trains and writes artifacts | clean clone + CSV → both artifacts | **Done (verified)** 2026-11-19 — nothing was ever in a notebook (D-015); `python -m leadscore.train` runs load → validate → split → select → evaluate once → gate → write `models/` and `reports/` end to end |
| T-502 | `api/schemas.py` — Pydantic request/response models | S1 | T-501 | P0 | typed boundary | 422 on a negative visit count | **Done (verified)** 2026-11-19 — 13 fields, one per admissible column, `extra="forbid"`; a negative `total_visits` returns 422 naming the field, and `last_activity` is refused rather than ignored (D-025) |
| T-503 | `api/main.py` — five endpoints, model loaded in lifespan | S1 | T-502 | P0 | the service | `/health` returns the model version | **Done (verified)** 2026-11-19 — five endpoints, model and explainer loaded in the lifespan hook; `/health` returned `20261118-6ae06cc8` from a live server, and the app refuses to start without a model |
| T-504 | `api/store.py` + SQLite DDL; log every score | M7 | T-503 | P0 | measurable in production | 1,000 scores → 1,000 rows | **Done (verified)** 2026-11-19 — SQLite log with WAL, predictions and feedback in separate tables; a 20-lead batch writes 20 rows under one `request_id`, and a failed write still returns a score |
| T-505 | `/feedback` endpoint | M7 | T-504 | **P2** | the retraining loop's inlet | unknown `prospect_id` → 404, nothing written | **Done (verified)** 2026-11-19 — kept despite being P2, since the hours were there; 404 for an unscored lead with nothing written, 409 for a duplicate, and `live_accuracy()` becomes computable once outcomes arrive |
| T-506 | `test_api.py` — contract, validation, batch cap, latency | S1 | T-503 | P0 | the API cannot regress silently | all pass; p95 latency recorded | **Done (verified)** 2026-11-19 — 37 contract tests on a fixture-trained model; p95 latency is read back out of the prediction log and asserted under the 300 ms target |

### Phase 6 — Package, demo, CI (Session 10)

| ID | Task | Feature | Depends | Pri | Expected outcome | Acceptance criterion | Status |
| --- | --- | --- | --- | --- | --- | --- | --- |
| T-601 | `dashboard/app.py` — three pages per doc 04, styled per doc 08 | S2 | T-506 | P1 | the demo surface | all three pages work against the API | **Done (verified)** 2026-11-19 — three pages, logic split into `dashboard/client.py` so it is testable; **37 tests**, and all three pages rendered against the live API with zero errors via Streamlit's `AppTest` |
| T-602 | `Dockerfile` + `docker-compose.yml` (api + dashboard) | — | T-601 | P0 | one command runs everything | `docker compose up` serves both ports | **Done (untested locally)** 2026-11-19 — multi-stage, base pinned by a digest resolved from Docker Hub, non-root uid 10001, `HEALTHCHECK`, one image with `api` / `dashboard` / `both` modes. Docker is not installed here (OQ-3), so **CI proves it**: it builds the image, polls `/health` inside it and scores a lead. **Proved for real on run #4, 2026-11-21** — built, imported, served and scored on a Linux runner. Two bugs had to be fixed first and neither was findable on this machine: a Windows-only pin in the lockfile (D-031) and a missing `libgomp1` that LightGBM needs at import (D-032). Still never run locally, which the README's *What is not here* section continues to say |
| T-603 | `.github/workflows/ci.yml` — ruff, pytest, docker build, gate on main | — | T-602 | P0 | every push is checked | badge green on a fresh push | **Done (untested)** 2026-11-19 — two jobs: `check` on every push (doctor, ruff, tests, docker build, container health, a real `/score` call, image size against NFR-5) and `model` on `main` (Kaggle download, retrain, quality gate, README drift check, artifact upload). **Run #1 on 2026-11-20 failed in 16 s** at the install step — the lockfile pinned `pywin32`, which has no Linux distribution (D-031). Fixed by `scripts/lock.py`, which restores the environment markers `pip freeze` drops, and CI now checks portability before installing. **Run #2 got through lint, 266 tests and the Docker build (124 s), then failed at the container health poll** — the runtime image had no `libgomp1`, so `import lightgbm` could not resolve `libgomp.so.1` (D-032). Fixed in the Dockerfile, plus an import check inside the image so the next failure of this kind names itself. **Run #3: the `check` job is fully green** — image built, container healthy in 4 s, a lead scored through it, image size reported. The `model` job also ran for the first time: the Kaggle download, the dataset verification and **the retrain and quality gate all passed**, and it failed only on the README check, which compares a date-stamped `model_version` and so could never pass (D-033). Replaced with `publish_results.py --drift`. **Run #4 on 2026-11-21 is fully green: both jobs, every step.** The image built in 100 s, the container answered `/health` in 2 s and scored a lead; the `model` job downloaded the real dataset, retrained, passed the quality gate in 111 s, and `--drift` confirmed the retrained numbers match the published ones **with an identical test-set fingerprint** — the split reproduces across operating systems. Status: **Done (verified)** |
| T-604 | Deploy the container to a public URL (Cloud Run; Render as fallback) | — | T-602 | P0 | a public URL | a stranger can score a lead in the browser | **Not started** - **target changed 2026-11-19**: Hugging Face made Docker Spaces PRO-only in July 2026 (D-029), so the demo goes to **Google Cloud Run** (`asia-south1`, built by Cloud Build, so no local Docker is needed). Steps in `deploy/cloudrun/DEPLOY.md`; needs a Google billing account. Render stays the no-card fallback and takes the same image |
| T-605 | README: results table, architecture diagram, 60-second demo GIF | — | T-604 | P0 | the reviewer's first minute | all three present and accurate | **Partly done** 2026-11-19 — results table generated from the card, architecture diagram and a *What is not here* section added. The 60-second GIF records the deployed demo, so it waits on T-604 |

### Phase 7 — Hardening (buffer, 2026-11-29)

| ID | Task | Feature | Depends | Pri | Expected outcome | Acceptance criterion | Status |
| --- | --- | --- | --- | --- | --- | --- | --- |
| T-701 | A friend runs the repo from the README, unaided | — | T-605 | P0 | the README is true | they succeed; every stumble is fixed | Not started |
| T-702 | Secret and data sweep of the full git history | — | — | P0 | nothing leaked | grep for the data filename and key patterns finds nothing | **Done** 2026-11-21 — swept all 13 commits across every ref. 89 paths have ever existed and all are source, docs, reports or figures: no `Leads.csv`, no `.db`, no `.joblib`, no `.env`, no `kaggle.json`. Content search for `KGAT_`, `AKIA`, `ghp_`, `github_pat_`, `xoxb-`, `AIzaSy` and PEM private-key headers: the only hits are the documentation placeholder `KGAT_...`, and zero blobs match a token-shaped `KGAT_[A-Za-z0-9_-]{8,}`. Largest object in history is a 79 KB figure, so no dataset was ever staged and squashed away. Ran ahead of T-605 — it never depended on the deploy |
| T-703 | Reconcile docs 01–14 with what was actually built | — | — | P0 | docs that can be trusted | every doc's claims match the code | **In progress** 2026-11-21 — started ahead of T-605, which it never depended on. Done so far: README status line (it still said "the pipeline itself is not built yet" after six phases) and the CI badge added; **docs/06** reconciled with `store.py`, which had drifted in five places, with the four dropped `CHECK` constraints raised as OQ-13 rather than quietly deleted from the doc; **docs/07** had a misplaced code fence rendering a paragraph as JSON, and its endpoint list and request example both verified against the live app. `tests/test_docs.py` now holds seven checks so this class of drift fails the suite instead of waiting to be read. Verified-correct already: the 5 endpoints, the 2 tables, and docs/04's three surface flows against the dashboard's three pages. Still to review: docs 01–03, 05, 08–10 |
| T-704 | LinkedIn post led by the Capture@20% number | — | T-605 | P1 | the portfolio signal | posted with the demo link | Not started |

## Milestones

| Date | Milestone | Proof |
| --- | --- | --- |
| 2026-11-14 | Phase 1 done | public repo with a running test suite |
| 2026-11-17 | Leakage audit done | every column classified, gap quantified |
| 2026-11-20 | Challenger beats baselines | MLflow comparison |
| 2026-11-23 | Capture@20% measured | model card written |
| 2026-11-26 | API serving | `/score` live locally with reasons |
| 2026-11-28 | Public demo + green CI | URL and badge |
| 2026-11-29 | **P1 delivered** | README results table, demo GIF, doc reconciliation |

## Risks

| Risk | Likelihood | Impact | Response |
| --- | --- | --- | --- |
| Exams consume the sessions | High | slips past 2026-11-29 | cut in the doc 03 order; the leakage audit and baselines never get cut |
| Toolchain install blocked (no uv/Docker) | Confirmed today | Phase 1 cannot complete | fall back to `pip` + `venv`; defer Docker to Phase 6 and decide then (OQ-2/3) |
| The real-estate decision on 2026-11-25 lands mid-Phase-5 | Medium | a mid-flight theme change | P1 stays on X Education data either way; the re-theme is P1's *carry-over*, not a rewrite |
| LightGBM barely beats logistic regression | Medium | the headline is unimpressive | that is a publishable finding — report it and ship the simpler model |
| Free-tier host changes its limits | Low | no demo URL | Render as the fallback; the Docker image is portable |
| Dataset has no true holdout population | Certain | measured lift may not survive on new data | state the limit on the model card; this is why `/feedback` exists |
