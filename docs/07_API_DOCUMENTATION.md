# 07 — API Documentation

FastAPI, JSON over HTTP, no authentication in P1 (see doc 09). Interactive docs at `/docs` are part of the demo, so schemas and examples are written to be read by a human.

Base URL, local: `http://localhost:8000`

All payload values below are **shapes, not results** — real values come from the dataset.

## Conventions

- Field names are `snake_case`, lower-cased versions of the CSV columns.
- Every response that involves a model carries `model_version`.
- Times are ISO 8601 with a `Z` suffix (UTC).
- Errors are JSON with a stable shape; never an HTML traceback.
- A request body is validated by Pydantic before any model code runs.

## Endpoints

| Method | Path | Purpose | Auth |
| --- | --- | --- | --- |
| GET | `/health` | liveness + which model is loaded | none |
| GET | `/model-info` | version, training date, metrics, features, grade bands | none |
| POST | `/score` | score one lead | none |
| POST | `/score/batch` | score up to 1,000 leads | none |
| POST | `/feedback` | report the real outcome for a scored lead | none in P1 |

---

### GET `/health`

Cheap, no model inference. Used by Docker, CI smoke tests and the host's health check.

```json
{
  "status": "ok",
  "model_version": "lgbm-2026-11-26",
  "model_loaded": true,
  "uptime_seconds": 4213
}
```

`503` with `"model_loaded": false` if the app is up but the model is unusable. The app refuses to start at all if `models/model.joblib` is missing — serving a fake score would be worse than being down.

---

### GET `/model-info`

Everything the dashboard's model card page needs, read from `model_card.json`. Nothing is recomputed per request.

```json
{
  "model_version": "lgbm-2026-11-26",
  "trained_at": "2026-11-26T18:40:00Z",
  "algorithm": "LightGBM + CalibratedClassifierCV(sigmoid)",
  "features": ["lead_origin", "lead_source", "total_visits", "..."],
  "excluded_as_leakage": ["Tags", "Lead Quality", "Asymmetrique Activity Index", "..."],
  "metrics": {"roc_auc": 0.0, "pr_auc": 0.0, "brier": 0.0, "capture_at_20": 0.0},
  "baselines": {"one_rule": {"roc_auc": 0.0, "capture_at_20": 0.0}},
  "grade_bands": {"A": {"min_score": 0}, "B": {"min_score": 0}, "C": {"min_score": 0}},
  "limits": ["Ranks leads; it does not decide whether to contact one.", "..."]
}
```

---

### POST `/score`

Scores one lead using only fields known at arrival.

**Request**

```json
{
  "prospect_id": "7927b2df-8bba-4d29-b9a2-b6e0beafe620",
  "lead_origin": "Landing Page Submission",
  "lead_source": "Google",
  "do_not_email": false,
  "do_not_call": false,
  "total_visits": 5,
  "total_time_spent_on_website": 1532,
  "page_views_per_visit": 2.5,
  "specialization": "Finance Management",
  "current_occupation": "Working Professional",
  "what_matters_most": "Better Career Prospects",
  "city": "Mumbai",
  "country": "India",
  "last_activity": "SMS Sent"
}
```

| Field | Type | Required | Rules |
| --- | --- | --- | --- |
| `prospect_id` | string | yes | 1–64 chars; echoed back and written to the log |
| `lead_origin` | string | yes | unknown value → `Other`, no error |
| `lead_source` | string | no | null allowed |
| `do_not_email`, `do_not_call` | bool | no | default `false` |
| `total_visits` | int | no | **≥ 0**; negative → 422 |
| `total_time_spent_on_website` | int | yes | **≥ 0**, seconds |
| `page_views_per_visit` | float | no | **≥ 0** |
| `specialization`, `current_occupation`, `what_matters_most`, `city`, `country`, `last_activity` | string | no | `"Select"` is accepted and treated as missing |

Unknown extra fields are rejected (`extra="forbid"`), so a typo'd field name fails loudly instead of being silently dropped and scored wrong.

**Response `200`**

```json
{
  "prospect_id": "7927b2df-8bba-4d29-b9a2-b6e0beafe620",
  "score": 82,
  "probability": 0.82,
  "grade": "A",
  "reasons": [
    {"feature": "total_time_spent_on_website", "direction": "up",   "text": "Spent a long time on the website"},
    {"feature": "current_occupation",          "direction": "up",   "text": "Working professional"},
    {"feature": "lead_source",                 "direction": "down", "text": "Came from a Google search"}
  ],
  "model_version": "lgbm-2026-11-26",
  "scored_at": "2026-11-27T09:14:02Z"
}
```

Always exactly three reasons, ordered by absolute contribution. `direction` is `up` if the feature raised the score.

---

### POST `/score/batch`

```json
{ "leads": [ { "prospect_id": "...", "total_time_spent_on_website": 1532 } ] }
```

- `leads`: 1–1,000 items, each identical to the `/score` body. Over the cap → `413`.
- Response: `{"results": [...], "count": 2, "model_version": "...", "request_id": "..."}`, same order as the request, one object per lead.
- `request_id` groups the batch in the prediction log.
- **Partial failure:** the whole batch is validated first. If any lead is invalid the call returns `422` listing the failing indexes and nothing is scored — a half-scored batch would be worse than none.

---

### POST `/feedback`

```json
{
  "prospect_id": "7927b2df-8bba-4d29-b9a2-b6e0beafe620",
  "converted": true,
  "note": "signed up after the second call"
}
```

`200` → `{"recorded": true, "prediction_id": 4821}`. `404` if that `prospect_id` was never scored — nothing is written. `409` if the same outcome was already reported for that prediction.

This endpoint is **P2** on the cut list: it is the first thing dropped if the 2026-11-29 date is at risk.

## Errors

```json
{
  "detail": "total_visits must be greater than or equal to 0",
  "code": "validation_error",
  "field": "total_visits"
}
```

| Status | `code` | When |
| --- | --- | --- |
| 400 | `bad_request` | malformed JSON |
| 404 | `prospect_not_found` | feedback for an unscored lead |
| 409 | `already_reported` | duplicate feedback |
| 413 | `batch_too_large` | more than `LEADSCORE_MAX_BATCH` leads |
| 422 | `validation_error` | a field failed its rule; `field` names it |
| 500 | `internal_error` | unexpected; the stack trace stays server-side |
| 503 | `model_unavailable` | the model is not loaded |

## Behaviour guarantees

- The model and the SHAP explainer load **once**, in the FastAPI lifespan hook. No per-request loading.
- An unseen category value never errors — it maps to `Other` inside the fitted transformer.
- A failed prediction-log write never fails the response.
- No endpoint mutates the model. Retraining happens offline, out of band.
- Latency targets: p95 < 300 ms for `/score`, 1,000 leads in < 10 s for `/score/batch` (NFR-1, NFR-2), measured from the log's `latency_ms`.

## What the API deliberately does not do

No pagination (no list endpoint), no deletes, no partial updates, no CRM write-back, no auth/accounts, no CORS wildcard in production, no model management over HTTP. Each of those belongs to a later project; adding them here would be scope nobody asked for.
