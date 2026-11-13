# 02 — Product Requirements

Status key: **[C]** confirmed by the owner's P1 design doc · **[R]** recommendation by the architect · **[A]** assumption awaiting confirmation (also in doc 13).

## Functional requirements

### FR-1 Data contract and validation **[C]**

The pipeline reads `data/raw/Leads.csv`: 9,240 rows × 37 columns, target `Converted` (1 = became a customer), 38.54% positive. Before any processing a `pandera` schema checks that expected columns exist with expected types and ranges. A missing or retyped column fails the run with a readable error.

The string `"Select"` is a form default, not an answer. It is treated as missing everywhere it appears. **[C]**

### FR-2 Leakage control **[C]** — the project's defining requirement

A feature is admissible only if its value is known at scoring time.

| Column group | Columns | Admissible | Reason |
| --- | --- | --- | --- |
| IDs | `Prospect ID`, `Lead Number` | **No** — keep `Prospect ID` to join results only | identifiers carry no signal |
| Source | `Lead Origin`, `Lead Source` | **Yes** | known on arrival |
| Website behaviour | `TotalVisits`, `Total Time Spent on Website`, `Page Views Per Visit` | **Yes** | pre-contact; expected strongest signals |
| Form answers | `Specialization`, `What is your current occupation`, `What matters most to you in choosing a course`, `City`, `Country`, `How did you hear about X Education` | **Yes, with care** | self-reported at submission; `"Select"` → missing, rare values grouped |
| Contact preferences | `Do Not Email`, `Do Not Call` | **Yes** | known on arrival |
| Ad / opt-in flags | `Search`, `Magazine`, `Newspaper Article`, `X Education Forums`, `Newspaper`, `Digital Advertisement`, `Through Recommendations`, `Receive More Updates About Our Courses`, `Update me on Supply Chain Content`, `Get updates on DM Content`, `I agree to pay the amount through cheque`, `A free copy of Mastering The Interview` | **Check first** | drop any that is near-constant — no variance, no signal |
| Activity | `Last Activity`, `Last Notable Activity` | **Test both ways** | some values may encode sales contact; train with and without, report both |
| Sales-filled | `Tags`, `Lead Quality`, `Asymmetrique Activity Index`, `Asymmetrique Profile Index`, `Asymmetrique Activity Score`, `Asymmetrique Profile Score` | **No** | written after a rep works the lead; they leak the outcome |
| Mostly empty | `Lead Profile` (74.2% empty counting `"Select"`) | **Drop** | too sparse to learn from |

**Missing values, measured on the real file 2026-11-13** by `scripts/check_data.py`. The basis
matters: `Select` is a form default, so a column can look populated and be empty in practice.

| Column | Blank | `Select` | Effectively missing |
| --- | --- | --- | --- |
| `How did you hear about X Education` | 23.9% | 54.6% | **78.5%** |
| `Lead Profile` | 29.3% | 44.9% | **74.2%** |
| `Lead Quality` | 51.6% | — | **51.6%** |
| `Asymmetrique` columns (4) | 45.7% | — | **45.7%** |
| `City` | 15.4% | 24.3% | **39.7%** |
| `Specialization` | 15.6% | 21.0% | **36.6%** |
| `Tags` | 36.3% | — | **36.3%** |
| `What is your current occupation` | 29.1% | — | **29.1%** |
| `Country` | 26.6% | — | **26.6%** |

`City` was not in the original figures and is **39.7% effectively missing** — higher than
`Specialization`. It is an admissible feature, so Phase 2 decides whether it earns its place
(T-201). Treat the 15.4% blank figure as misleading on its own.

**Acceptance:** a leakage-audit notebook records a yes/no decision per column with its reason, and the results table contains a LightGBM row trained *with* the sales-filled columns, labelled unshippable, so the inflated gap is visible.

### FR-3 Training pipeline **[C]**

One command (`make train`): load → validate → split → cross-validated model selection → tune → calibrate → evaluate on the test set once → write `models/model.joblib` and `models/model_card.json` → log everything to MLflow.

Split: 20% held out, stratified on `Converted`, fixed seed. Model selection by 5-fold stratified CV on the remaining 80%. **The test set is evaluated exactly once.**

### FR-4 Baselines **[C]**

Three, all logged and published:

1. **Random** — ROC-AUC 0.5, Capture@20% = 20%.
2. **One-rule** — rank by `Total Time Spent on Website` alone. This is what a sales manager does by hand, and it is the bar that matters.
3. **Logistic regression** — one-hot encoding and scaling inside a scikit-learn `Pipeline`.

### FR-5 Challenger model **[C]**

LightGBM, tuned with Optuna for 30–50 trials on cross-validated ROC-AUC. Wrapped in `CalibratedClassifierCV` if the reliability curve or Brier score shows mis-confidence.

### FR-6 Score, grade and reasons **[C]**

- `score = round(100 × calibrated_probability)`.
- Grades from cut-offs set on validation data and stored in the model card: **A** = top 20% of scores, **B** = next 30%, **C** = the rest.
- Exactly three reasons per lead, from SHAP `TreeExplainer`, each with the feature, a direction (`up` / `down`) and a plain-English phrase. No raw SHAP values in the response text. **[R]**

### FR-7 Serving API **[C]**

`GET /health`, `GET /model-info`, `POST /score`, `POST /score/batch` (≤ 1,000 leads), `POST /feedback`. Full schemas in `07_API_DOCUMENTATION.md`. The model loads once at startup, not per request.

### FR-8 Prediction log **[C]**

Every scoring call writes one row per lead to SQLite: timestamp, model version, input JSON, score, grade, latency. `/feedback` later attaches the real outcome to a `prospect_id`, which makes live accuracy measurable and supplies future training rows.

### FR-9 Dashboard **[C]**

Streamlit, three pages: **Rank leads** (upload CSV → sorted table with grades and reasons, downloadable), **Try one lead** (form → single score), **Model card** (test metrics, calibration plot, feature importance, stated limits). The dashboard calls the API over HTTP; it never imports the model.

### FR-10 Quality gate in CI **[C]**

CI fails if lint or tests fail, or if a retrained model's test ROC-AUC does not beat logistic regression, or its Brier score is worse than the current model's. Exact floors are set after the first full run and recorded in config.

### FR-11 Packaging and demo **[C]**

`docker compose up` runs API + dashboard. A public demo URL is live. The README carries the results table, an architecture diagram and a demo GIF.

## Non-functional requirements

| ID | Requirement | Target | How it is verified |
| --- | --- | --- | --- |
| NFR-1 | Single-lead latency | p95 < 300 ms server-side, model already loaded | timing recorded per request in the prediction log |
| NFR-2 | Batch throughput | 1,000 leads in < 10 s | integration test with a 1,000-row fixture |
| NFR-3 | Reproducibility | same data + same seed → identical metrics | `make train` twice on the fixture in CI; metrics compared |
| NFR-4 | Cold start | API ready < 15 s after container start | `/health` polled in the Docker smoke test |
| NFR-5 | Resource ceiling | ≤ 1 vCPU, ≤ 2 GB RAM, image < 1.5 GB | measured at build; `docker stats` under batch load |
| NFR-6 | Input robustness | an unknown category or a null never 500s | property tests over malformed payloads |
| NFR-7 | Explanation cost | SHAP reasons add < 100 ms per lead | benchmark in the API test |
| NFR-8 | Repo runnability | a stranger runs it from the README with no help | a friend attempts it at the Phase-1 gate |
| NFR-9 | Demo accessibility | text contrast ≥ 4.5:1, keyboard-navigable, usable at 1024 px | checklist in `08_UI_UX_DESIGN_SYSTEM.md` |
| NFR-10 | No secret or dataset in git history | zero findings | `.gitignore` plus a pre-push grep for the data filename and key patterns |

## Constraints

- **Time:** ten sessions of 1.5–2 h, ending 2026-11-29, during exam weeks. Each session must end committable.
- **Cost:** free tiers only. No paid API, no managed database, no GPU. No LLM, so no token cost.
- **Data licence:** the dataset is third-party and is never redistributed in the repo.
- **Platform:** developed on Windows 11; must run in Linux containers in CI and in production.

## Acceptance criteria for P1 as a whole

1. `make train` runs end to end from a clean clone plus the downloaded CSV, and writes a model and a model card.
2. The published results table has five rows (random, one-rule, logistic regression, LightGBM, LightGBM-with-leakage) and four metric columns, all from real runs.
3. Capture@20% for the shipped model beats the one-rule baseline, and the reliability curve is reported.
4. `docker compose up` serves API and dashboard; `/health` returns the model version.
5. A public demo URL responds, and a 60-second GIF in the README shows a CSV being ranked.
6. CI runs lint, tests and the model quality gate on every push; the badge is green.
7. `tests/` covers features, API contract and model quality; all pass.
8. Docs 01–14 reflect what was actually built, and every roadmap task carries its real status.
9. The leakage audit exists, with a written decision per column.
10. No dataset, model, database or secret in git history.
