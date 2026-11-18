# Lead Scoring Pipeline

Scores a sales lead 0–100 the moment it arrives, and says why in three plain-English reasons, so a sales team calls the most likely converters first.

**Status:** Phase 1 complete — foundation, config and tests in place; the pipeline itself is not built yet (as of 2026-11-13). `ruff` clean, 32 tests passing. Target delivery **2026-11-29**. Progress by task: [docs/11](docs/11_DEVELOPMENT_ROADMAP.md).

This is project P1 of a seven-project AI-engineering portfolio. The deliverable is an **AI pipeline** — data contract, training, evaluation, calibration, explanation, serving and a retraining loop — not an end-user application. The Streamlit dashboard exists only to demonstrate the pipeline.

## The headline number

<!-- headline:start -->
> If sales calls only the top 20% of scored leads, they reach **44.1%** of all conversions.

Measured once on 1,848 held-out leads the model never saw (fingerprint `e30abe9c3fe596c1`), by model version `20261118-6ae06cc8`. Quality gate: **passed**.
<!-- headline:end -->

**Capture@20%** is the headline metric, and every other metric supports it. The block above is generated from `models/model_card.json` by `scripts/publish_results.py`, so the README and the card cannot disagree.

## What it does

| Piece | What it produces |
| --- | --- |
| Training pipeline | one command → a versioned model, its metrics and a model card |
| Evaluation + quality gate | ROC-AUC, PR-AUC, Brier, Capture@20%; blocks a worse model from shipping |
| Scoring API (FastAPI) | `/score`, `/score/batch` → score, grade A/B/C, 3 SHAP reasons |
| Prediction log (SQLite) | every score stored, with `/feedback` for real outcomes |
| Dashboard (Streamlit) | rank an uploaded CSV, try one lead, read the model card |

## Results

Generated from the model card. Every row is one pass over the same 1,848 held-out leads; full detail in [reports/04_evaluation.md](reports/04_evaluation.md).

<!-- results:start -->
| Model | ROC-AUC | PR-AUC | Brier | Capture@20% |
| --- | ---: | ---: | ---: | ---: |
| Random baseline | 0.500 | 0.385 | 0.250 | 20.2% |
| One-rule (time on site) | 0.646 | 0.569 | 0.241 | 35.3% |
| Logistic regression | 0.850 | 0.777 | 0.152 | 42.6% |
| **LightGBM (tuned, no calibrator needed)** | 0.865 | 0.805 | 0.142 | 44.1% |
| LightGBM + sales columns *(leakage, not shippable)* | 0.984 | 0.977 | 0.043 | 51.5% |
<!-- results:end -->

The last row is deliberate: it shows how much accuracy comes from columns that only exist *after* sales has already called the lead. See [docs/02_PRODUCT_REQUIREMENTS.md](docs/02_PRODUCT_REQUIREMENTS.md).

## Quickstart

Python 3.11 or newer. No other tool is required.

```bash
# 1. environment
python -m venv .venv
.venv/Scripts/python -m pip install -e ".[dev]"     # Windows
# .venv/bin/python -m pip install -e ".[dev]"       # macOS / Linux

# 2. check it works — doctor first: it names an environment fault in two seconds
.venv/Scripts/python scripts/doctor.py
.venv/Scripts/ruff check src tests scripts
.venv/Scripts/python -m pytest

# 3. data — not in this repo, see docs/06_DATABASE_DESIGN.md for why
#    download the X Education lead dataset from Kaggle into data/raw/Leads.csv
#    steps: data/raw/README.md

# 4. train (writes models/model.joblib + models/model_card.json)   [from Phase 3]
.venv/Scripts/python -m leadscore.train

# 5. serve the API                                                  [from Phase 5]
.venv/Scripts/python -m uvicorn leadscore.api.main:app --port 8000
#    interactive docs  http://localhost:8000/docs

# 6. both services in containers                                    [from Phase 6]
docker compose up
#    API       http://localhost:8000/docs
#    Dashboard http://localhost:8501
```

### Scoring a lead

The API serves whatever model `models/model.joblib` holds, and reads its grade cut-offs from
the model card, so the numbers below move when the model does rather than when someone edits
serving code.

```bash
curl -X POST http://localhost:8000/score \
  -H "content-type: application/json" \
  -d '{
    "prospect_id": "demo-1",
    "lead_origin": "Lead Add Form",
    "lead_source": "Reference",
    "total_time_spent_on_website": 1532,
    "total_visits": 5,
    "current_occupation": "Working Professional"
  }'
```

```json
{
  "prospect_id": "demo-1",
  "score": 92,
  "grade": "A",
  "reasons": [
    {"feature": "current_occupation", "direction": "up", "text": "Is working or self-employed, which converts well"},
    {"feature": "total_time_spent_on_website", "direction": "up", "text": "Spent a long time on the website"},
    {"feature": "lead_origin", "direction": "up", "text": "Arrived through a channel that converts well"}
  ],
  "model_version": "20261118-6ae06cc8"
}
```

That is a real response from a real run, trimmed for width. A lead with nothing known and no
time on the site scores **4, grade C** from the same model. The request accepts only the 13
columns known when a lead arrives — send `last_activity` and you get a `422`, because that
column is filled in *after* sales has made contact and the model is not allowed to see it.
Full contract in [docs/07_API_DOCUMENTATION.md](docs/07_API_DOCUMENTATION.md).

`uv` works too, with no change to this repo: `uv sync` in place of step 1.

`requirements.lock` pins the exact resolution this project was built and tested against. `pyproject.toml` states what the project *needs*; the lockfile states what actually worked. Install from the lockfile when you want the tested environment rather than the newest one — a floating dependency is how MLflow retired a storage backend under this project mid-phase (docs/14, D-019).

### If you have `make`

Each target is one command; `make help` lists them. `make` is not installed on the author's machine, so the raw equivalents above are the tested path:

| Target | Raw command |
| --- | --- |
| `make venv` | `python -m venv .venv` |
| `make install` | `.venv/Scripts/python -m pip install -e ".[dev]"` |
| `make doctor` | `.venv/Scripts/python scripts/doctor.py` |
| `make lint` | `.venv/Scripts/ruff check src tests scripts` |
| `make smoke` | `.venv/Scripts/python -m pytest tests/test_smoke.py -v` |
| `make lock` | `.venv/Scripts/python -m pip freeze --exclude-editable > requirements.lock` |
| `make test` | `.venv/Scripts/python -m pytest` |
| `make train` | `.venv/Scripts/python -m leadscore.train` |
| `make publish` | `.venv/Scripts/python scripts/publish_results.py` |
| `make serve` | `.venv/Scripts/python -m uvicorn leadscore.api.main:app --port 8000` |

## Documentation

Read these before changing anything. They are the project's source of truth.

| Doc | Contents |
| --- | --- |
| [01_PROJECT_OVERVIEW](docs/01_PROJECT_OVERVIEW.md) | problem, purpose, users, objectives |
| [02_PRODUCT_REQUIREMENTS](docs/02_PRODUCT_REQUIREMENTS.md) | functional + non-functional requirements, acceptance criteria |
| [03_FEATURE_BREAKDOWN](docs/03_FEATURE_BREAKDOWN.md) | every pipeline stage, its inputs, outputs and dependencies |
| [04_USER_FLOWS](docs/04_USER_FLOWS.md) | pipeline flows and the three dashboard journeys |
| [05_TECHNICAL_ARCHITECTURE](docs/05_TECHNICAL_ARCHITECTURE.md) | stack, modules, data flow, diagrams |
| [06_DATABASE_DESIGN](docs/06_DATABASE_DESIGN.md) | raw data contract, prediction log DDL, model registry |
| [07_API_DOCUMENTATION](docs/07_API_DOCUMENTATION.md) | endpoints, schemas, validation, errors |
| [08_UI_UX_DESIGN_SYSTEM](docs/08_UI_UX_DESIGN_SYSTEM.md) | design tokens, components, charts, accessibility |
| [09_SECURITY_AND_PERMISSIONS](docs/09_SECURITY_AND_PERMISSIONS.md) | secrets, input trust, PII, rate limits |
| [10_TESTING_STRATEGY](docs/10_TESTING_STRATEGY.md) | test layers, the model quality gate, edge cases |
| [11_DEVELOPMENT_ROADMAP](docs/11_DEVELOPMENT_ROADMAP.md) | phases, task IDs, acceptance criteria, status |
| [12_DEPLOYMENT_GUIDE](docs/12_DEPLOYMENT_GUIDE.md) | environments, config, CI, hosting |
| [13_ASSUMPTIONS_AND_OPEN_QUESTIONS](docs/13_ASSUMPTIONS_AND_OPEN_QUESTIONS.md) | what is confirmed, assumed, and still open |
| [14_CHANGELOG_AND_DECISIONS](docs/14_CHANGELOG_AND_DECISIONS.md) | architectural decisions and their reasons |

Design rules live in [design-system/lead-scoring-pipeline/MASTER.md](design-system/lead-scoring-pipeline/MASTER.md).

Working rules for AI coding sessions live in [CLAUDE.md](CLAUDE.md).

## Licence and data

Code: MIT (to be added). The X Education lead dataset is third-party; it is downloaded by the user, never redistributed in this repo.
