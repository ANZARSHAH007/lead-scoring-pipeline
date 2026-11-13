# 03 — Feature Breakdown

The pipeline is six modules plus two surfaces. Each row below is a unit of work with an owner file, an input, an output and a dependency. Nothing here is a screen except where stated — this is a pipeline.

## Module map

| # | Module | File | Input | Output | Depends on |
| --- | --- | --- | --- | --- | --- |
| M1 | Config | `src/leadscore/config.py` | env vars, defaults | one frozen settings object | — |
| M2 | Data contract | `src/leadscore/schema.py` | `data/raw/Leads.csv` | validated DataFrame, or a loud failure | M1 |
| M3 | Features | `src/leadscore/features.py` | validated DataFrame | fitted `ColumnTransformer` + feature names | M1, M2 |
| M4 | Training | `src/leadscore/train.py` | validated DataFrame | `model.joblib`, MLflow run | M1–M3 |
| M5 | Evaluation | `src/leadscore/evaluate.py` | model + test set | metrics dict, plots, gate verdict | M1, M3 |
| M6 | Explanation | `src/leadscore/explain.py` | model + one row | 3 reasons in plain English | M1, M3 |
| S1 | API | `src/leadscore/api/` | JSON request | score, grade, reasons | M1, M3, M6, M7 |
| M7 | Prediction log | `src/leadscore/api/store.py` | score event, feedback event | SQLite rows | M1 |
| S2 | Dashboard | `dashboard/app.py` | CSV upload or form | ranked table, single score, model card | S1 over HTTP |

## M1 — Config

**Purpose:** one place for every path, threshold, feature list and the random seed, so nothing is a magic number.

Holds: raw data path, model path, DB path, random seed, test size, CV folds, Optuna trial budget, gate floors (`min_roc_auc`, `max_brier`), grade band edges, max batch size, API base URL, log level.

**Requirements:** immutable once loaded; read from environment with documented defaults; never contains a secret value, only the name of one.
**Acceptance:** importing config twice yields equal objects; an unset env var falls back to the documented default; a bad value fails at import, not at use.

## M2 — Data contract

**Purpose:** fail loudly when the data is not what the pipeline was built for.

Sub-features:
1. **Schema check** — expected columns, dtypes, allowed ranges (`TotalVisits` ≥ 0, `Converted` ∈ {0,1}).
2. **`"Select"` → missing** — applied before any other transform, across all object columns.
3. **Column admissibility** — the feature list comes from config, derived from the leakage table in doc 02. Any column not classified raises.
4. **Row-count and target-rate sanity** — logs 9,240 rows / 38.54% positive as expected values and warns on drift beyond a tolerance.

**Dependency:** the CSV must exist locally; the module never downloads it.
**Acceptance:** a dropped column, a renamed column and a string in a numeric column each produce a distinct readable error.

## M3 — Features

**Purpose:** identical preprocessing in training and serving, by construction.

Sub-features:
1. Numeric branch — median impute, then scale (scaling matters for logistic regression, is harmless for LightGBM).
2. Categorical branch — `"Select"`/null → `"Missing"`, rare levels (< 1% frequency) → `"Other"`, then one-hot with `handle_unknown="infrequent_if_exist"` so an unseen category at serving time cannot crash the request.
3. Feature-name export — the ordered list that SHAP and `/model-info` both report.
4. Two feature sets, selected by config flag: `shipping` (admissible columns only) and `audit_with_leakage` (adds the sales-filled columns) — used once, for the leakage row.

**Acceptance:** a category never seen in training scores without error; both feature sets build; feature names round-trip through save/load.

## M4 — Training

**Purpose:** one command produces a versioned, tracked, calibrated model.

Ordered steps: load → validate → single stratified 80/20 split on the config seed → 5-fold stratified CV for model selection → the three baselines → LightGBM with Optuna (budget from config) → calibration check → fit the winner on the full training split → evaluate the test set **once** → write `models/model.joblib` and `models/model_card.json` → log params, metrics, plots and the model to MLflow.

**Requirements:** reruns with the same seed and data reproduce the metrics; the test set is read exactly once per run; the run aborts before writing if the gate fails.
**Acceptance:** `make train` on a clean clone writes both artifacts; a second run reports identical metrics; MLflow lists the run with its parameters.

## M5 — Evaluation

**Purpose:** compute the numbers the README publishes, and decide whether a model may ship.

Metrics: ROC-AUC, PR-AUC, Brier, **Capture@20%** (share of all conversions falling in the top 20% of scores), plus a lift/gain table by decile.
Plots: ROC curve, reliability (calibration) curve, score distribution by outcome, permutation or gain-based feature importance.
Grade bands: cut-offs computed on validation data, written into the model card.
**Quality gate:** test ROC-AUC must beat the logistic-regression baseline; Brier must not exceed the configured ceiling. Verdict is returned, logged and enforced by both `make train` and CI.

**Acceptance:** a deliberately crippled model fails the gate and is not written; the gate's numbers appear in the run log.

## M6 — Explanation

**Purpose:** three reasons a rep can argue with.

Sub-features: SHAP `TreeExplainer` over the fitted model; top three absolute contributions per row; a phrase map from feature name + direction to plain English ("Spent a long time on the website"); a fallback phrase for any feature without a mapping, so a new feature never produces an empty reason.

**Requirements:** no raw SHAP numbers in the response text; explanation adds < 100 ms per lead (NFR-7); the explainer is built once at startup, not per request.
**Acceptance:** every admissible feature has a mapped phrase for both directions; reasons for a known-hot and a known-cold lead read sensibly in the notebook check.

## M7 — Prediction log

**Purpose:** make the pipeline measurable after deployment, and collect future training rows.

Sub-features: write one row per scored lead (timestamp, model version, input JSON, score, grade, latency ms); `/feedback` attaches an outcome to a `prospect_id`; a read helper for the model-card page. Schema and DDL in doc 06.

**Acceptance:** scoring 1,000 leads writes 1,000 rows; feedback for an unknown `prospect_id` returns 404 and writes nothing; the DB file is git-ignored.

## S1 — API

Five endpoints, specified in doc 07. Behaviour requirements: model and explainer loaded once in the lifespan hook; Pydantic validation at the boundary; unknown category tolerated, negative number rejected with 422, oversized batch rejected with 413; every response carries `model_version`; errors are JSON, never an HTML traceback.

## S2 — Dashboard

Three pages, specified in doc 04 and styled per doc 08. It is a demo surface over the API, so it holds no model logic and no business rules — grade bands and metrics are read from `/model-info`.

## Build order and dependencies

```
M1 ──> M2 ──> M3 ──> M4 ──> M5
                │      │
                └──────┴──> M6 ──> S1 ──> S2
                                    │
                                    └──> M7
```

M1–M3 block everything. M5's gate blocks shipping. S2 cannot be built before S1 answers.

## Cut list, in order

If the 2026-11-29 date is at risk, drop in this order and record it in doc 14:

1. Optuna tuning → sensible LightGBM defaults.
2. `/feedback` endpoint and its table.
3. The lift/gain decile table (keep Capture@20%).
4. The "Try one lead" dashboard page (keep Rank leads and Model card).

Never cut: the leakage audit, the three baselines, the calibration check, the README results table.
