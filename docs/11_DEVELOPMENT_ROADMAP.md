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

### Phase 2 — Data contract and leakage audit (Sessions 2–3)

| ID | Task | Feature | Depends | Pri | Expected outcome | Acceptance criterion | Status |
| --- | --- | --- | --- | --- | --- | --- | --- |
| T-201 | `notebooks/01_eda.ipynb` — missing values, `"Select"`, distributions, conversion by source | M2 | T-107 | P0 | the data is understood, charts saved | missing-value rates match doc 02's figures | Not started |
| T-202 | `src/leadscore/schema.py` — pandera contract | M2 | T-201 | P0 | bad data fails loudly | three corruption tests produce three distinct errors | Not started |
| T-203 | `notebooks/02_leakage_audit.ipynb` — a yes/no decision per column, with reason | M2 | T-201 | **P0** | the audit record doc 02 promises | every one of the 37 columns is classified | Not started |
| T-204 | Quantify the leakage gap: logistic regression with vs without sales columns | M2 | T-203 | P0 | the number the README's last row reports | both ROC-AUCs logged to MLflow | Not started |
| T-205 | Freeze the two feature sets in config | M1 | T-203 | P0 | `shipping` and `audit_with_leakage` | `test_features` asserts banned columns absent from `shipping` | Not started |

### Phase 3 — Baselines and challenger (Sessions 4–5)

| ID | Task | Feature | Depends | Pri | Expected outcome | Acceptance criterion | Status |
| --- | --- | --- | --- | --- | --- | --- | --- |
| T-301 | `features.py` — ColumnTransformer, rare-level grouping, unknown-safe one-hot | M3 | T-205 | P0 | identical preprocessing both lanes | an unseen category scores without error | Not started |
| T-302 | Single stratified 80/20 split, seeded, in `train.py` | M4 | T-301 | P0 | one split for the whole project | split is reproducible across runs | Not started |
| T-303 | Three baselines with 5-fold CV, logged to MLflow | M4 | T-302 | P0 | the bar to beat | MLflow lists random, one-rule, logistic regression | Not started |
| T-304 | LightGBM + Optuna (budget from config) | M4 | T-303 | P1 | the challenger | CV ROC-AUC beats logistic regression | Not started |
| T-305 | Pick the winner on CV only; test set still untouched | M4 | T-304 | P0 | an honest selection | the run log states the test set was not read | Not started |

### Phase 4 — Calibration, grades, explanations (Sessions 6–7)

| ID | Task | Feature | Depends | Pri | Expected outcome | Acceptance criterion | Status |
| --- | --- | --- | --- | --- | --- | --- | --- |
| T-401 | Reliability curve + Brier; calibrate if off-diagonal | M5 | T-305 | P0 | a score that means a probability | curve plotted; Brier recorded before and after | Not started |
| T-402 | Grade cut-offs A/B/C from validation data → model card | M5 | T-401 | P0 | bands nobody hardcodes | `/model-info` later serves these exact numbers | Not started |
| T-403 | `evaluate.py` — ROC-AUC, PR-AUC, Brier, Capture@20%, decile lift | M5 | T-401 | P0 | the published metrics | all four computed on the single test-set pass | Not started |
| T-404 | Quality gate function + thresholds written into config | M5 | T-403 | P0 | a worse model cannot ship | a deliberately crippled model is rejected | Not started |
| T-405 | `explain.py` — SHAP top-3 + phrase map | M6 | T-403 | P0 | three readable reasons | every admissible feature has both directions mapped | Not started |
| T-406 | Error analysis: 20 high-score non-converters, 20 low-score converters | M5 | T-403 | P1 | written list of what the model misses | findings recorded in the notebook and doc 14 | Not started |
| T-407 | Write `models/model_card.json` + the README results table | M4 | T-403 | P0 | the headline number published | five rows, four columns, all from real runs | Not started |

### Phase 5 — Serving (Sessions 8–9)

| ID | Task | Feature | Depends | Pri | Expected outcome | Acceptance criterion | Status |
| --- | --- | --- | --- | --- | --- | --- | --- |
| T-501 | Move notebook code into `src/leadscore/`; `make train` end to end | M4 | T-407 | P0 | one command trains and writes artifacts | clean clone + CSV → both artifacts | Not started |
| T-502 | `api/schemas.py` — Pydantic request/response models | S1 | T-501 | P0 | typed boundary | 422 on a negative visit count | Not started |
| T-503 | `api/main.py` — five endpoints, model loaded in lifespan | S1 | T-502 | P0 | the service | `/health` returns the model version | Not started |
| T-504 | `api/store.py` + SQLite DDL; log every score | M7 | T-503 | P0 | measurable in production | 1,000 scores → 1,000 rows | Not started |
| T-505 | `/feedback` endpoint | M7 | T-504 | **P2** | the retraining loop's inlet | unknown `prospect_id` → 404, nothing written | Not started |
| T-506 | `test_api.py` — contract, validation, batch cap, latency | S1 | T-503 | P0 | the API cannot regress silently | all pass; p95 latency recorded | Not started |

### Phase 6 — Package, demo, CI (Session 10)

| ID | Task | Feature | Depends | Pri | Expected outcome | Acceptance criterion | Status |
| --- | --- | --- | --- | --- | --- | --- | --- |
| T-601 | `dashboard/app.py` — three pages per doc 04, styled per doc 08 | S2 | T-506 | P1 | the demo surface | all three pages work against the API | Not started |
| T-602 | `Dockerfile` + `docker-compose.yml` (api + dashboard) | — | T-601 | P0 | one command runs everything | `docker compose up` serves both ports | Not started |
| T-603 | `.github/workflows/ci.yml` — ruff, pytest, docker build, gate on main | — | T-602 | P0 | every push is checked | badge green on a fresh push | Not started |
| T-604 | Deploy to Hugging Face Spaces (or Render) | — | T-602 | P0 | a public URL | a stranger can score a lead in the browser | Not started |
| T-605 | README: results table, architecture diagram, 60-second demo GIF | — | T-604 | P0 | the reviewer's first minute | all three present and accurate | Not started |

### Phase 7 — Hardening (buffer, 2026-11-29)

| ID | Task | Feature | Depends | Pri | Expected outcome | Acceptance criterion | Status |
| --- | --- | --- | --- | --- | --- | --- | --- |
| T-701 | A friend runs the repo from the README, unaided | — | T-605 | P0 | the README is true | they succeed; every stumble is fixed | Not started |
| T-702 | Secret and data sweep of the full git history | — | T-605 | P0 | nothing leaked | grep for the data filename and key patterns finds nothing | Not started |
| T-703 | Reconcile docs 01–14 with what was actually built | — | T-605 | P0 | docs that can be trusted | every doc's claims match the code | Not started |
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
