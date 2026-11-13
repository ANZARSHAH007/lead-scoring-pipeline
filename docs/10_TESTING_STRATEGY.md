# 10 — Testing Strategy

The thing that separates this repo from a notebook is that **CI can fail it**. Two kinds of failure matter: the code broke, or the model got worse. Both must be automatic.

## Layers

| Layer | Tool | Runs on | Needs the dataset? | Gate |
| --- | --- | --- | --- | --- |
| Lint / format | `ruff` | every push | no | blocking |
| Unit — config | `pytest` | every push | no | blocking |
| Unit — schema | `pytest` | every push | no (synthetic frames) | blocking |
| Unit — features | `pytest` | every push | no (small fixture) | blocking |
| Contract — API | `pytest` + `TestClient` | every push | no (fixture model) | blocking |
| Model quality gate | `pytest` | main branch, after a retrain | **yes** | blocking |
| Docker smoke | `docker build` + `/health` | every push | no | blocking |
| Reproducibility | `make train` twice on the fixture | main branch | fixture only | blocking |
| Manual acceptance | a human | before delivery | yes | blocking |

**Fixtures, not the real CSV, for everything but the quality gate.** A ~200-row fixture with the same columns and a deliberately learnable signal, plus a tiny fixture model, keeps the suite under a minute and lets CI run without a Kaggle token on pull requests.

## Unit tests

### `test_config.py`
- documented defaults load when no env var is set
- an env var overrides its default, with the right type
- a bad value (negative seed, `test_size` of 2.0) fails at import, not at use
- the `shipping` feature list contains **none** of the banned leakage columns — the single most important assertion in the suite
- grade bands are ordered and cover 0–100 without a gap

### `test_schema.py`
Three corruptions, three distinct readable errors:
- a required column dropped
- `Converted` containing `2`
- `TotalVisits` containing a string

Plus: `"Select"` becomes null in every object column; a negative `TotalVisits` is rejected; the row-count/target-rate warning fires on a drifted frame.

### `test_features.py`
- a category value **never seen in training** transforms without raising, and lands in `Other`
- a null in every nullable column still produces a complete feature row
- the fitted transformer's feature names are stable across save → load
- `shipping` and `audit_with_leakage` both build, and the leakage set is strictly larger
- transforming the same frame twice gives identical output (no fitted state mutating on transform)

## API contract tests

- `/health` → 200 with a `model_version`
- `/model-info` → 200, and its `grade_bands` match the model card byte for byte
- `/score` valid → 200, response matches the schema, `score` ∈ 0–100, `grade` ∈ {A,B,C}, **exactly three reasons**
- `/score` negative `total_visits` → 422, with `field: "total_visits"`
- `/score` unknown extra field → 422 (not silently ignored)
- `/score` unknown category value → **200**, not an error
- `/score` missing a required field → 422
- `/score/batch` 1,001 leads → 413
- `/score/batch` one invalid lead among valid ones → 422, nothing scored, log unchanged
- `/score/batch` preserves input order
- `/feedback` unknown `prospect_id` → 404 and no row written
- `/feedback` twice for the same prediction → 409
- a scoring call writes exactly one prediction row per lead
- a failing log write still returns 200 (log failure is simulated)
- the response carries no stack trace on a forced internal error

## Model quality gate

Run as a test so it fails CI like anything else, in `test_model_quality.py`:

1. test **ROC-AUC must beat the logistic-regression baseline** — if the gradient-boosted model cannot beat a linear baseline, it has no reason to exist
2. test **Brier must not exceed `LEADSCORE_GATE_MAX_BRIER`** — a sharper ranking that lies about probabilities is a regression
3. **Capture@20% must beat 0.20** (the random floor) and must beat the one-rule baseline
4. the fitted model exposes every feature named in the card, and no banned column
5. a **deliberately crippled model** (labels shuffled) must be **rejected** — this tests the gate itself, which is otherwise never exercised

Floors are set after the first real run and recorded in config. The rule in `CLAUDE.md`: report the numbers, never raise the gate to make a model pass.

## Edge cases to cover explicitly

| Case | Expected |
| --- | --- |
| Lead with every optional field null | scores; reasons fall back to available features |
| `total_time_spent_on_website` = 0 | scores low, no error |
| Extreme value (999,999 seconds) | scores; no crash; flagged in the reliability check |
| All-`"Select"` categorical row | treated as all-missing, scores |
| Duplicate `prospect_id` in one batch | both scored, two log rows, same id — allowed, documented |
| CSV with columns in a different order | works (keyed by name, never by position) |
| CSV with extra unrelated columns | ignored with a notice, not an error |
| Empty CSV / header only | the empty state, not a traceback |
| Unicode in a category value | round-trips through transform, log and CSV download |
| Model file missing at startup | the app refuses to start |
| Model card missing but model present | 503 `model_unavailable`; no scoring with unknown grade bands |
| Disk full on log write | score still returned, `ERROR` logged |

## Reproducibility test

`make train` on the fixture, twice, same seed → identical metrics to 6 decimal places. This catches the whole class of bug where an unseeded split or a dict-ordering dependency makes results drift between runs, which would make every published number unverifiable.

## CI pipeline

On every push:

1. checkout, set up Python 3.11, install from the lockfile
2. `ruff check`
3. `pytest` (everything except the quality gate), coverage printed
4. `docker build`, then run the image and poll `/health`

On `main` only:

5. download the dataset with the Kaggle token from secrets
6. `make train`
7. the model quality gate
8. publish `model.joblib` + `model_card.json` as a build artifact

A red build blocks the merge. CI does not auto-deploy a model that failed the gate.

## Manual acceptance, before delivery

| # | Check | Pass condition |
| --- | --- | --- |
| 1 | Clean clone → README steps → `make train` | both artifacts written, no undocumented step |
| 2 | `docker compose up` | API on 8000, dashboard on 8501 |
| 3 | Rank 200 real leads in the dashboard | sorted table, grades, reasons, download works |
| 4 | Change one field in "Try one lead" | the score moves in the direction the reasons imply |
| 5 | Kill the API, use the dashboard | the service-down message, upload preserved |
| 6 | Read the model card as a sceptic | every number traceable to a run; limits present |
| 7 | **A friend runs it from the README with no help** (T-701) | they succeed; every stumble becomes a README fix |
| 8 | Secret and data sweep of git history (T-702) | zero findings |

## What is not tested, and why

No load testing (a demo on a free tier; the batch cap is the control), no browser automation of Streamlit (its own DOM is not a contract worth testing at this scale; the manual checks cover it), no fairness audit (stated as a limit on the model card rather than claimed), no security pen-test (no auth surface, no user data). Each of these is a deliberate omission, not an oversight — and each is written on the model card or in doc 09 where a reader would look for it.
