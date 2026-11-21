"""Contract tests for the scoring API (task T-506).

The API is the part of this project other people touch, so these tests are about the
promises docs/07 makes to a caller: the shapes, the status codes, the batch cap, the error
bodies, and the rule that a logging failure must not become the caller's problem.

Everything runs against a model trained on the synthetic fixture, written to a temporary
directory. No real data, no network, a few seconds - so CI can hold the API to its contract
on every push.
"""

from __future__ import annotations

import json
import re
import sqlite3
import warnings
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from leadscore import config as cfg
from leadscore import evaluate as ev
from leadscore import explain as ex
from leadscore import fixtures, schema
from leadscore import train as tr
from leadscore.api import main as api
from leadscore.api import schemas as sc
from leadscore.api import store
from leadscore.api.store import PredictionLog

VALID_LEAD: dict[str, Any] = {
    "prospect_id": "lead-0001",
    "lead_origin": "Landing Page Submission",
    "lead_source": "Google",
    "do_not_email": False,
    "do_not_call": False,
    "total_visits": 5,
    "total_time_spent_on_website": 1532,
    "page_views_per_visit": 2.5,
    "specialization": "Finance Management",
    "current_occupation": "Working Professional",
    "what_matters_most": "Better Career Prospects",
    "city": "Mumbai",
    "country": "India",
    "how_did_you_hear": "Online Search",
}


@pytest.fixture(scope="module")
def trained(tmp_path_factory) -> cfg.Settings:
    """Train a small model on the fixture and write a real bundle and card to a temp dir.

    Module-scoped: the model is read-only to every test, and fitting it once keeps the file
    under ten seconds.
    """
    tmp_path = tmp_path_factory.mktemp("api")
    settings = cfg.Settings(
        _env_file=None,
        cv_folds=3,
        optuna_trials=0,
        model_path=tmp_path / "models" / "model.joblib",
        model_card_path=tmp_path / "models" / "model_card.json",
        db_path=tmp_path / "predictions.db",
        reports_dir=tmp_path / "reports",
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", schema.DataDriftWarning)
        frame = schema.validate(fixtures.synthetic_leads(rows=400), settings)
        split = tr.make_split(frame, "shipping", settings)
        model = tr.build_lightgbm(tr.DEFAULT_LGBM, settings).fit(split.x_train, split.y_train)
        explainer_model = tr.build_lightgbm(tr.DEFAULT_LGBM, settings).fit(
            split.x_train, split.y_train
        )

    card = {
        "model_version": "test-20261119-abcdef12",
        "created_at": "2026-11-19T10:00:00+00:00",
        "calibration": {"method": "none"},
        "grades": {"A_from_score": 70, "B_from_score": 30, "C_from_score": 0},
        "features": {
            "admissible_source_columns": list(cfg.shipping_features()),
            "excluded": {"banned_for_leakage": list(cfg.BANNED_LEAKAGE_COLUMNS)},
        },
        "test_metrics": [
            {"name": "lightgbm_shipped", "roc_auc": 0.86, "pr_auc": 0.80,
             "brier": 0.14, "capture_at_20": 0.44},
            {"name": "one_rule", "roc_auc": 0.65, "pr_auc": 0.57,
             "brier": 0.24, "capture_at_20": 0.35},
            {"name": "logistic_regression", "roc_auc": 0.85, "pr_auc": 0.78,
             "brier": 0.15, "capture_at_20": 0.43},
        ],
        "limitations": ["A fixture model. No number from it means anything."],
    }
    ev.save_model_bundle(
        model=model, explainer_model=explainer_model, card=card, settings=settings
    )
    ev.write_model_card(card, settings)
    return settings


@pytest.fixture
def client(trained: cfg.Settings, tmp_path: Path):
    """A client with its own prediction log, so tests cannot see each other's rows."""
    settings = trained.model_copy(update={"db_path": tmp_path / "predictions.db"})
    api.app.state.settings = settings
    with TestClient(api.app) as test_client:
        yield test_client


# ---------------------------------------------------------------------------
# the field mapping
# ---------------------------------------------------------------------------


def test_the_api_accepts_exactly_the_admissible_columns() -> None:
    """One field per shipping feature, and nothing the model was not fitted on."""
    assert set(sc.FIELD_TO_COLUMN.values()) == set(cfg.shipping_features())


def test_no_api_field_maps_to_a_banned_column() -> None:
    """The leakage rule, enforced at the boundary as well as in training."""
    mapped = set(sc.FIELD_TO_COLUMN.values())
    assert not mapped & set(cfg.BANNED_LEAKAGE_COLUMNS)
    assert not mapped & set(cfg.ACTIVITY_FEATURES)


def test_last_activity_is_rejected_rather_than_quietly_ignored(client) -> None:
    """docs/07's first draft had this field; D-016 excluded the activity columns.

    Accepting it would tell a caller it affects the score. It does not, and cannot.
    """
    response = client.post("/score", json={**VALID_LEAD, "last_activity": "SMS Sent"})
    assert response.status_code == 422
    assert response.json()["code"] == "validation_error"


# ---------------------------------------------------------------------------
# /health and /model-info
# ---------------------------------------------------------------------------


def test_health_reports_the_loaded_model(client) -> None:
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True
    assert body["model_version"] == "test-20261119-abcdef12"
    assert body["uptime_seconds"] >= 0


def test_model_info_serves_the_card_not_a_recomputation(client) -> None:
    body = client.get("/model-info").json()
    assert body["model_version"] == "test-20261119-abcdef12"
    assert body["metrics"]["capture_at_20"] == pytest.approx(0.44)
    assert body["baselines"]["one_rule"]["capture_at_20"] == pytest.approx(0.35)
    assert "Tags" in body["excluded_as_leakage"]
    assert body["limits"]


def test_model_info_serves_the_grade_bands_from_the_card(client) -> None:
    """The dashboard and the API must agree because both read the same numbers."""
    bands = client.get("/model-info").json()["grade_bands"]
    assert bands["A"]["min_score"] == 70
    assert bands["B"]["min_score"] == 30
    assert bands["C"]["min_score"] == 0


def test_model_info_names_the_features_in_the_api_vocabulary(client) -> None:
    features = client.get("/model-info").json()["features"]
    assert "total_time_spent_on_website" in features
    assert "Total Time Spent on Website" not in features
    assert len(features) == len(cfg.shipping_features())


# ---------------------------------------------------------------------------
# /score
# ---------------------------------------------------------------------------


def test_scoring_one_lead_returns_the_documented_shape(client) -> None:
    body = client.post("/score", json=VALID_LEAD).json()
    assert set(body) == {
        "prospect_id", "score", "probability", "grade", "reasons",
        "model_version", "scored_at",
    }
    assert body["prospect_id"] == "lead-0001"
    assert 0 <= body["score"] <= 100
    assert 0.0 <= body["probability"] <= 1.0
    assert body["grade"] in {"A", "B", "C"}


def test_the_score_is_the_rounded_probability(client) -> None:
    body = client.post("/score", json=VALID_LEAD).json()
    assert body["score"] == round(body["probability"] * 100)


def test_the_grade_follows_the_cards_cutoffs(client) -> None:
    body = client.post("/score", json=VALID_LEAD).json()
    expected = "A" if body["score"] >= 70 else "B" if body["score"] >= 30 else "C"
    assert body["grade"] == expected


def test_exactly_three_reasons_in_the_callers_vocabulary(client) -> None:
    reasons = client.post("/score", json=VALID_LEAD).json()["reasons"]
    assert len(reasons) == 3
    for reason in reasons:
        assert set(reason) == {"feature", "direction", "text"}
        assert reason["feature"] in sc.FIELD_TO_COLUMN
        assert reason["direction"] in {"up", "down"}
        assert reason["text"] and not any(ch.isdigit() for ch in reason["text"])


def test_only_prospect_id_origin_and_time_are_required(client) -> None:
    minimal = {
        "prospect_id": "lead-min",
        "lead_origin": "API",
        "total_time_spent_on_website": 0,
    }
    assert client.post("/score", json=minimal).status_code == 200


def test_an_unknown_category_scores_instead_of_failing(client) -> None:
    """docs/07: an unseen value maps to 'Other' inside the fitted transformer."""
    payload = {**VALID_LEAD, "lead_source": "a source invented after training"}
    assert client.post("/score", json=payload).status_code == 200


def test_select_is_treated_as_missing(client) -> None:
    payload = {**VALID_LEAD, "city": "Select", "specialization": "Select"}
    assert client.post("/score", json=payload).status_code == 200


def test_an_all_missing_lead_still_scores(client) -> None:
    minimal = {
        "prospect_id": "lead-empty",
        "lead_origin": "API",
        "total_time_spent_on_website": 0,
        "lead_source": None,
        "total_visits": None,
        "page_views_per_visit": None,
        "specialization": None,
        "current_occupation": None,
        "what_matters_most": None,
        "city": None,
        "country": None,
        "how_did_you_hear": None,
    }
    body = client.post("/score", json=minimal).json()
    assert body["grade"] in {"A", "B", "C"}
    assert len(body["reasons"]) == 3


def test_an_extreme_value_scores_without_crashing(client) -> None:
    payload = {**VALID_LEAD, "total_time_spent_on_website": 999_999}
    assert client.post("/score", json=payload).status_code == 200


# ---------------------------------------------------------------------------
# validation: the 422s docs/07 promises
# ---------------------------------------------------------------------------


def test_a_negative_visit_count_is_rejected_with_the_field_named(client) -> None:
    """T-502's acceptance criterion."""
    response = client.post("/score", json={**VALID_LEAD, "total_visits": -1})
    assert response.status_code == 422
    body = response.json()
    assert body["code"] == "validation_error"
    assert body["field"] == "total_visits"
    assert "detail" in body


def test_negative_time_on_site_is_rejected(client) -> None:
    response = client.post("/score", json={**VALID_LEAD, "total_time_spent_on_website": -5})
    assert response.status_code == 422
    assert response.json()["field"] == "total_time_spent_on_website"


def test_a_misspelled_field_fails_loudly(client) -> None:
    """`extra="forbid"`: a typo must not be dropped and scored as a missing value."""
    payload = {**VALID_LEAD, "total_visitss": 4}
    assert client.post("/score", json=payload).status_code == 422


def test_a_missing_required_field_names_itself(client) -> None:
    payload = {key: value for key, value in VALID_LEAD.items() if key != "lead_origin"}
    response = client.post("/score", json=payload)
    assert response.status_code == 422
    assert response.json()["field"] == "lead_origin"


def test_a_blank_prospect_id_is_rejected(client) -> None:
    assert client.post("/score", json={**VALID_LEAD, "prospect_id": ""}).status_code == 422


def test_malformed_json_does_not_return_a_traceback(client) -> None:
    response = client.post(
        "/score", content=b"{not json", headers={"content-type": "application/json"}
    )
    assert response.status_code in {400, 422}
    assert "Traceback" not in response.text


# ---------------------------------------------------------------------------
# /score/batch
# ---------------------------------------------------------------------------


def test_a_batch_returns_one_result_per_lead_in_order(client) -> None:
    leads = [{**VALID_LEAD, "prospect_id": f"lead-{index}"} for index in range(5)]
    body = client.post("/score/batch", json={"leads": leads}).json()
    assert body["count"] == 5
    assert [result["prospect_id"] for result in body["results"]] == [
        f"lead-{index}" for index in range(5)
    ]
    assert body["request_id"]


def test_a_batch_over_the_cap_is_refused(client, trained) -> None:
    """docs/07: more than LEADSCORE_MAX_BATCH leads is a 413, not a slow success."""
    settings = client.app.state.settings
    leads = [
        {**VALID_LEAD, "prospect_id": f"lead-{index}"}
        for index in range(settings.max_batch + 1)
    ]
    response = client.post("/score/batch", json={"leads": leads})
    assert response.status_code == 413
    assert response.json()["code"] == "batch_too_large"


def test_an_invalid_lead_fails_the_whole_batch(client) -> None:
    """A half-scored batch would be worse than none, so nothing is scored."""
    leads = [VALID_LEAD, {**VALID_LEAD, "prospect_id": "bad", "total_visits": -3}]
    response = client.post("/score/batch", json={"leads": leads})
    assert response.status_code == 422
    assert "1" in str(response.json()["field"])


def test_an_empty_batch_is_rejected(client) -> None:
    assert client.post("/score/batch", json={"leads": []}).status_code == 422


def test_a_batch_and_a_single_call_agree(client) -> None:
    single = client.post("/score", json=VALID_LEAD).json()
    batched = client.post("/score/batch", json={"leads": [VALID_LEAD]}).json()["results"][0]
    assert single["score"] == batched["score"]
    assert single["grade"] == batched["grade"]


# ---------------------------------------------------------------------------
# the prediction log
# ---------------------------------------------------------------------------


def test_every_score_is_logged_once(client) -> None:
    log = PredictionLog(settings=client.app.state.settings)
    before = log.count()
    client.post("/score", json={**VALID_LEAD, "prospect_id": "logged-1"})
    assert log.count() == before + 1

    row = log.recent(1)[0]
    assert row["prospect_id"] == "logged-1"
    assert row["model_version"] == "test-20261119-abcdef12"
    assert row["latency_ms"] > 0
    assert json.loads(row["reasons_json"])
    assert json.loads(row["input_json"])["lead_origin"] == VALID_LEAD["lead_origin"]


def test_a_batch_writes_one_row_per_lead_under_one_request_id(client) -> None:
    leads = [{**VALID_LEAD, "prospect_id": f"batch-{index}"} for index in range(20)]
    request_id = client.post("/score/batch", json={"leads": leads}).json()["request_id"]

    log = PredictionLog(settings=client.app.state.settings)
    rows = log.by_request(request_id)
    assert len(rows) == 20
    assert [row["prospect_id"] for row in rows] == [f"batch-{i}" for i in range(20)]


def test_a_failed_log_write_still_returns_a_score(client, monkeypatch) -> None:
    """docs/07's behaviour guarantee: bookkeeping failure is not the caller's problem."""
    monkeypatch.setattr(
        client.app.state.service.log, "log_prediction", lambda **_: None
    )
    response = client.post("/score", json={**VALID_LEAD, "prospect_id": "no-log"})
    assert response.status_code == 200
    assert response.json()["score"] >= 0


# ---------------------------------------------------------------------------
# /feedback
# ---------------------------------------------------------------------------


def test_feedback_attaches_an_outcome_to_a_scored_lead(client) -> None:
    client.post("/score", json={**VALID_LEAD, "prospect_id": "fb-1"})
    response = client.post("/feedback", json={"prospect_id": "fb-1", "converted": True})
    assert response.status_code == 200
    body = response.json()
    assert body["recorded"] is True
    assert body["prediction_id"] > 0


def test_feedback_for_an_unscored_lead_is_a_404_and_writes_nothing(client) -> None:
    log = PredictionLog(settings=client.app.state.settings)
    before = log.live_accuracy()
    response = client.post("/feedback", json={"prospect_id": "never-scored", "converted": True})
    assert response.status_code == 404
    assert response.json()["code"] == "prospect_not_found"
    assert log.live_accuracy() == before


def test_reporting_the_same_outcome_twice_is_a_409(client) -> None:
    client.post("/score", json={**VALID_LEAD, "prospect_id": "fb-dup"})
    client.post("/feedback", json={"prospect_id": "fb-dup", "converted": True})
    response = client.post("/feedback", json={"prospect_id": "fb-dup", "converted": False})
    assert response.status_code == 409
    assert response.json()["code"] == "already_reported"


def test_feedback_makes_live_accuracy_computable(client) -> None:
    for index in range(4):
        client.post("/score", json={**VALID_LEAD, "prospect_id": f"acc-{index}"})
        client.post(
            "/feedback", json={"prospect_id": f"acc-{index}", "converted": index % 2 == 0}
        )
    accuracy = PredictionLog(settings=client.app.state.settings).live_accuracy()
    assert accuracy is not None
    assert accuracy["leads_with_outcomes"] >= 4
    assert 0.0 <= accuracy["conversion_rate"] <= 1.0


# ---------------------------------------------------------------------------
# startup and latency
# ---------------------------------------------------------------------------


def test_the_app_refuses_to_start_without_a_model(tmp_path) -> None:
    """Serving an invented score would be worse than being down."""
    settings = cfg.Settings(
        _env_file=None,
        model_path=tmp_path / "missing" / "model.joblib",
        model_card_path=tmp_path / "missing" / "card.json",
    )
    with pytest.raises(FileNotFoundError, match=re.escape("leadscore.train")):
        api.load_service(settings)


def test_the_model_is_loaded_once_not_per_request(client, monkeypatch) -> None:
    calls: list[int] = []
    original = ex.Explainer.__init__

    def counting_init(self, model):
        calls.append(1)
        original(self, model)

    monkeypatch.setattr(ex.Explainer, "__init__", counting_init)
    for _ in range(3):
        client.post("/score", json=VALID_LEAD)
    assert calls == [], "an explainer was built during a request"


def test_single_score_latency_is_within_the_target(client) -> None:
    """NFR-1: p95 under 300 ms, measured from the log rather than asserted."""
    for index in range(12):
        client.post("/score", json={**VALID_LEAD, "prospect_id": f"lat-{index}"})
    p95 = PredictionLog(settings=client.app.state.settings).latency_percentile(0.95)
    assert p95 is not None
    assert p95 < 300, f"p95 latency {p95:.1f} ms exceeds the 300 ms target"


# ---------------------------------------------------------------------------
# the prediction log's CHECK constraints (OQ-13, D-034)
#
# The schema promised four of these from Phase 0 and shipped with none of them; T-703 found
# the gap. They are defence in a second layer: the API validates every field before it
# writes, so nothing invalid reaches the table through the front door. These tests cover the
# other doors - a future model with an off-by-one score, a migration, someone with sqlite3
# and good intentions - and, more importantly, pin the rule that a rejected *log* row must
# never cost the caller their *score*.
# ---------------------------------------------------------------------------


def _valid_row(**overrides: object) -> dict:
    row = {
        "prospect_id": "constraint-test",
        "model_version": "test",
        "score": 50,
        "probability": 0.5,
        "grade": "B",
        "reasons": [],
        "request_payload": {},
        "latency_ms": 1.0,
    }
    row.update(overrides)
    return row


@pytest.mark.parametrize(
    ("field", "bad_value"),
    [
        ("score", 137),       # outside 0-100
        ("score", -1),
        ("probability", 1.5),  # outside 0-1
        ("grade", "F"),        # not a real grade
    ],
)
def test_the_log_refuses_an_impossible_row(tmp_path: Path, field: str, bad_value: object) -> None:
    log = PredictionLog(path=tmp_path / "constrained.db")
    assert log.log_prediction(**_valid_row(**{field: bad_value})) is None
    assert log.count() == 0, f"a row with {field}={bad_value!r} reached the table"


def test_a_refused_row_returns_none_rather_than_raising(tmp_path: Path) -> None:
    """The rule the whole design turns on: bookkeeping never breaks the product."""
    log = PredictionLog(path=tmp_path / "constrained.db")
    assert log.log_prediction(**_valid_row(score=500)) is None  # must not raise
    assert log.log_prediction(**_valid_row(score=50)) == 1, "the log still works afterwards"


def test_one_bad_row_does_not_discard_the_rest_of_a_batch(tmp_path: Path) -> None:
    log = PredictionLog(path=tmp_path / "constrained.db")
    written = log.log_batch(
        [_valid_row(prospect_id="a"), _valid_row(prospect_id="b", score=9999), _valid_row(prospect_id="c")]
    )
    assert len(written) == 2, "the two valid rows should survive the one invalid row"
    assert log.count() == 2


def test_a_valid_row_is_unaffected_by_the_constraints(tmp_path: Path) -> None:
    log = PredictionLog(path=tmp_path / "constrained.db")
    for score, grade in ((0, "C"), (100, "A"), (76, "A")):
        assert log.log_prediction(**_valid_row(score=score, grade=grade)) is not None
    assert log.count() == 3


def test_a_legacy_log_without_constraints_is_reported_not_silently_accepted(
    tmp_path: Path, caplog
) -> None:
    """SQLite cannot add a CHECK to an existing table, so the old file must announce itself."""
    path = tmp_path / "legacy.db"
    connection = sqlite3.connect(path)
    connection.executescript(
        store.SCHEMA.replace(" CHECK (score BETWEEN 0 AND 100)", "")
        .replace(" CHECK (probability BETWEEN 0 AND 1)", "")
        .replace(" CHECK (grade IN ('A','B','C'))", "")
    )
    connection.commit()
    connection.close()

    with caplog.at_level("WARNING"):
        PredictionLog(path=path)
    assert any("predates the CHECK constraints" in r.message for r in caplog.records)


def test_scoring_still_succeeds_when_the_log_cannot_write(client, monkeypatch) -> None:
    """The end the user sees: a broken log degrades observability, not the service."""

    def refuse(*args, **kwargs):
        return None

    monkeypatch.setattr(PredictionLog, "log_prediction", refuse)
    response = client.post("/score", json={**VALID_LEAD, "prospect_id": "log-down"})
    assert response.status_code == 200
    assert 0 <= response.json()["score"] <= 100
