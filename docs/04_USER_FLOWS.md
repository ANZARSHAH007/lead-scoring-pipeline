# 04 — Flows

Two kinds of flow: **pipeline flows** (what the system does, no human watching) and **surface flows** (the three dashboard journeys). Exceptional paths are listed with each.

## Pipeline flow 1 — Train a model

```mermaid
flowchart TD
    A[make train] --> B[load data/raw/Leads.csv]
    B --> C{pandera schema valid?}
    C -- no --> C1[abort: name the bad column] 
    C -- yes --> D["Select" to missing, drop banned columns]
    D --> E[stratified 80/20 split, fixed seed]
    E --> F[5-fold CV: random, one-rule, logistic regression]
    F --> G[LightGBM + Optuna on CV ROC-AUC]
    G --> H{reliability curve off-diagonal?}
    H -- yes --> I[wrap in CalibratedClassifierCV]
    H -- no --> J[keep raw probabilities]
    I --> K[fit winner on full train split]
    J --> K
    K --> L[evaluate test set ONCE]
    L --> M{quality gate passes?}
    M -- no --> M1[abort: report numbers, write nothing]
    M -- yes --> N[write model.joblib + model_card.json]
    N --> O[log params, metrics, plots, model to MLflow]
```

**Exceptions:** CSV absent → message with the Kaggle download steps, exit 1. Schema invalid → the failing column and expectation, exit 1. Gate failed → both models' numbers side by side, exit 1, previous model untouched.

## Pipeline flow 2 — Score a lead (request path)

```mermaid
flowchart LR
    A[client POST /score] --> B[Pydantic validation]
    B -- invalid --> B1[422 with field errors]
    B -- valid --> C[ColumnTransformer from the loaded model]
    C --> D[calibrated probability]
    D --> E[score = round 100x p; grade from card bands]
    E --> F[SHAP top-3 to phrases]
    F --> G[write prediction log row]
    G --> H[200: score, grade, reasons, model_version]
```

Unknown category → mapped to `Other`, scored normally, no error. Model file missing at startup → the app refuses to start rather than serving a fake score. Log write fails → the score is still returned; the failure is logged at `ERROR` (scoring must not depend on the log).

## Pipeline flow 3 — Outcome feedback and retraining

```mermaid
flowchart LR
    A[lead was called] --> B[POST /feedback: prospect_id, converted]
    B --> C{prospect_id in log?}
    C -- no --> C1[404, nothing written]
    C -- yes --> D[attach outcome + timestamp]
    D --> E[live accuracy measurable on logged rows]
    E --> F[manual: export outcomes as new training rows]
    F --> G[make train again, gate decides]
```

Retraining is never automatic in P1. The loop exists so the capstone can close it.

## Surface flow 1 — Rank leads (the main journey)

1. Rep opens the dashboard, lands on **Rank leads**.
2. Empty state explains what to upload and offers a sample CSV.
3. Rep uploads a CSV.
4. Client-side check: required columns present, row count ≤ batch limit. Missing column → inline error naming it, nothing sent.
5. Spinner while `POST /score/batch` runs.
6. Table renders, sorted by score descending: `score`, grade chip (A/B/C), `prospect_id`, the three reasons, then the input columns.
7. Rep filters to grade A, scans the reasons, downloads the ranked CSV.

**Exceptions:** over the batch limit → offer to score the first 1,000 and say so. API unreachable → "scoring service is not responding", keep the uploaded file in session so nothing is lost. All rows fail validation → show the first three field errors, not a traceback. Empty file → the empty state, unchanged.

## Surface flow 2 — Try one lead

1. Rep opens **Try one lead**; form fields are exactly the admissible features, each labelled in business language with its unit.
2. Defaults are the dataset medians/modes so the form is never blank.
3. Submit → `POST /score` → score, grade and the three reasons, with the probability shown as a percentage.
4. Changing one field and resubmitting shows how the score moves — the cheapest way to build trust in the model.

**Exceptions:** negative number → field-level 422 message, form state kept. Service down → same message as flow 1.

## Surface flow 3 — Model card

Read-only. Shows: model version and training date; the results table (all five rows from doc 02); the reliability curve; score distribution by outcome; feature importance; the grade bands with their cut-offs; and a plain **"What this model must not be used for"** section — no automated discard of leads, no use on a population unlike X Education's web leads, no inference about individuals beyond ranking.

All figures come from `/model-info` and the model card file. The page never recomputes a metric.

## Navigation

Three pages in a left sidebar, in the order above. **Rank leads** is the default. No nested navigation, no modal dialogs, no login. The model version is visible on every page, in the footer, so a screenshot is always attributable to a model.
