"""Tests for the dashboard's logic (task T-601).

A Streamlit script renders as it runs, which makes `app.py` awkward to test and is exactly
why the parts that can be wrong on their own live in `dashboard/client.py`. What is tested
here is what a rep would notice: a file that is missing a column is refused before anything
is sent, a CSV exported from somewhere else still loads, and every API failure turns into one
readable sentence rather than a traceback.
"""

from __future__ import annotations

import io
from pathlib import Path

import pandas as pd
import pytest
import requests
from dashboard import client

from leadscore.api import schemas as sc

GOOD_CSV = (
    "prospect_id,lead_origin,total_time_spent_on_website,total_visits,city\n"
    "p1,Landing Page Submission,1532,5,Mumbai\n"
    "p2,API,0,0,Select\n"
)


def _frame(text: str) -> pd.DataFrame:
    return client.read_csv(text.encode("utf-8"))


# ---------------------------------------------------------------------------
# reading and checking an upload
# ---------------------------------------------------------------------------


def test_a_good_csv_passes_the_check() -> None:
    check = client.check_frame(_frame(GOOD_CSV), max_batch=1_000)
    assert check.ok
    assert check.rows == 2
    assert check.unknown == []
    assert check.truncated_to is None


def test_a_missing_required_column_is_refused_before_anything_is_sent() -> None:
    text = "prospect_id,total_time_spent_on_website\np1,100\n"
    check = client.check_frame(_frame(text), max_batch=1_000)
    assert not check.ok
    assert check.missing == ["lead_origin"]
    assert "lead_origin" in check.message
    assert "Nothing was sent" in check.message


def test_an_empty_file_gets_the_empty_state_not_an_error() -> None:
    frame = pd.DataFrame(columns=list(client.REQUIRED_COLUMNS))
    check = client.check_frame(frame, max_batch=1_000)
    assert not check.ok
    assert check.rows == 0
    assert "no rows" in check.message


def test_an_oversized_file_offers_to_score_the_first_n() -> None:
    """docs/04: over the batch limit, score the first 1,000 and say so."""
    rows = "\n".join(f"p{index},API,100" for index in range(1_200))
    text = "prospect_id,lead_origin,total_time_spent_on_website\n" + rows + "\n"
    check = client.check_frame(_frame(text), max_batch=1_000)
    assert check.ok
    assert check.truncated_to == 1_000
    assert "first 1,000" in check.message


def test_columns_the_model_does_not_use_are_named_not_rejected() -> None:
    text = GOOD_CSV.replace("city\n", "city,tags\n").replace("Mumbai\n", "Mumbai,Will revert\n")
    text = text.replace("Select\n", "Select,Ringing\n")
    check = client.check_frame(_frame(text), max_batch=1_000)
    assert check.ok
    assert check.unknown == ["tags"]


def test_prospect_id_and_the_label_are_not_reported_as_unknown() -> None:
    """`prospect_id` is required but is not a feature; `converted` is the training label."""
    text = "prospect_id,lead_origin,total_time_spent_on_website,converted\np1,API,10,1\n"
    assert client.check_frame(_frame(text), max_batch=1_000).unknown == []


# ---------------------------------------------------------------------------
# turning a frame into API payloads
# ---------------------------------------------------------------------------


def test_payloads_carry_only_fields_the_api_accepts() -> None:
    payloads = client.to_payloads(_frame(GOOD_CSV))
    assert len(payloads) == 2
    for payload in payloads:
        extra = set(payload) - set(sc.FIELD_TO_COLUMN) - {"prospect_id"}
        assert extra == set(), f"would be rejected by extra=forbid: {extra}"


def test_every_payload_is_accepted_by_the_api_schema() -> None:
    """The strongest check available without a server: build the real request model."""
    for payload in client.to_payloads(_frame(GOOD_CSV)):
        sc.Lead.model_validate(payload)


def test_select_and_blanks_are_dropped_rather_than_sent() -> None:
    payloads = client.to_payloads(_frame(GOOD_CSV))
    assert payloads[0]["city"] == "Mumbai"
    assert "city" not in payloads[1], "'Select' should not be sent as a value"


def test_a_missing_prospect_id_gets_a_positional_one() -> None:
    text = "prospect_id,lead_origin,total_time_spent_on_website\n,API,100\n"
    assert client.to_payloads(_frame(text))[0]["prospect_id"] == "row-1"


def test_an_unknown_origin_is_sent_as_other() -> None:
    """docs/07: an unknown value scores as 'Other' rather than failing."""
    text = "prospect_id,lead_origin,total_time_spent_on_website\np1,,100\n"
    assert client.to_payloads(_frame(text))[0]["lead_origin"] == "Other"


def test_a_negative_number_is_clamped_rather_than_sent_and_rejected() -> None:
    text = "prospect_id,lead_origin,total_time_spent_on_website,total_visits\np1,API,-50,-2\n"
    payload = client.to_payloads(_frame(text))[0]
    assert payload["total_time_spent_on_website"] == 0
    assert payload["total_visits"] == 0


def test_boolean_spellings_from_other_systems_are_understood() -> None:
    text = (
        "prospect_id,lead_origin,total_time_spent_on_website,do_not_email,do_not_call\n"
        "p1,API,100,Yes,0\np2,API,100,TRUE,no\n"
    )
    payloads = client.to_payloads(_frame(text))
    assert payloads[0]["do_not_email"] is True
    assert payloads[0]["do_not_call"] is False
    assert payloads[1]["do_not_email"] is True


def test_the_limit_is_honoured() -> None:
    rows = "\n".join(f"p{index},API,100" for index in range(50))
    text = "prospect_id,lead_origin,total_time_spent_on_website\n" + rows + "\n"
    assert len(client.to_payloads(_frame(text), limit=10)) == 10


def test_a_text_value_in_a_number_column_is_dropped_not_crashed_on() -> None:
    text = "prospect_id,lead_origin,total_time_spent_on_website,page_views_per_visit\np1,API,10,lots\n"
    payload = client.to_payloads(_frame(text))[0]
    assert "page_views_per_visit" not in payload


# ---------------------------------------------------------------------------
# the sample CSV offered in the empty state
# ---------------------------------------------------------------------------


def test_the_sample_csv_is_something_the_dashboard_can_score() -> None:
    frame = client.read_csv(client.sample_csv(rows=10).encode("utf-8"))
    check = client.check_frame(frame, max_batch=1_000)
    assert check.ok and check.unknown == []
    for payload in client.to_payloads(frame):
        sc.Lead.model_validate(payload)


def test_the_sample_csv_puts_plausible_values_in_each_column() -> None:
    """A sample full of nonsense teaches a reviewer the wrong thing about the data."""
    frame = client.read_csv(client.sample_csv(rows=60).encode("utf-8"))
    cities = set(frame["city"].dropna())
    sources = set(frame["lead_source"].dropna())
    assert cities <= {"Select", "Mumbai", "Thane & Outskirts", "Other Metro Cities", "Pune"}
    assert "Mumbai" in cities
    assert not sources & {"Landing Page Submission", "Lead Add Form", "Lead Import"}


# ---------------------------------------------------------------------------
# results and errors
# ---------------------------------------------------------------------------


def test_results_come_back_sorted_by_score() -> None:
    results = [
        {"prospect_id": "a", "score": 40, "grade": "B", "probability": 0.4, "reasons": []},
        {"prospect_id": "b", "score": 91, "grade": "A", "probability": 0.91,
         "reasons": [{"feature": "f", "direction": "up", "text": "did a thing"}]},
    ]
    frame = client.results_to_frame(results)
    assert list(frame["prospect_id"]) == ["b", "a"]
    assert frame.loc[0, "reason_1"] == "did a thing"
    assert frame.loc[1, "reason_1"] == ""


def test_three_reason_columns_exist_even_with_fewer_reasons() -> None:
    frame = client.results_to_frame(
        [{"prospect_id": "a", "score": 1, "grade": "C", "probability": 0.01, "reasons": []}]
    )
    assert {"reason_1", "reason_2", "reason_3"} <= set(frame.columns)


@pytest.mark.parametrize(
    ("status", "payload", "expected"),
    [
        (413, {"code": "batch_too_large", "detail": "1001 leads exceeds the limit of 1000"},
         "Too many leads"),
        (422, {"code": "validation_error", "detail": "must be >= 0", "field": "total_visits"},
         "total_visits"),
        (404, {"code": "prospect_not_found", "detail": "no scored lead"}, "Not found"),
        (409, {"code": "already_reported", "detail": "already recorded"}, "Already recorded"),
        (500, {"code": "internal_error", "detail": "boom"}, "internal error"),
    ],
)
def test_every_api_error_becomes_one_readable_sentence(status, payload, expected) -> None:
    message = client.describe_error(status, payload)
    assert expected in message
    assert "Traceback" not in message


def test_a_500_does_not_leak_the_servers_detail() -> None:
    message = client.describe_error(500, {"code": "internal_error", "detail": "KeyError at 0x7f"})
    assert "KeyError" not in message
    assert "service log" in message


# ---------------------------------------------------------------------------
# an unreachable service
# ---------------------------------------------------------------------------


def test_an_unreachable_api_raises_the_one_exception_the_ui_handles(monkeypatch) -> None:
    def refuse(*_args, **_kwargs):
        raise requests.ConnectionError("connection refused")

    monkeypatch.setattr(requests, "post", refuse)
    with pytest.raises(client.ApiUnreachable, match="not responding"):
        client.score_batch("http://127.0.0.1:9", [{"prospect_id": "p"}])


def test_an_unreachable_health_check_raises_the_same_thing(monkeypatch) -> None:
    def refuse(*_args, **_kwargs):
        raise requests.Timeout("timed out")

    monkeypatch.setattr(requests, "get", refuse)
    with pytest.raises(client.ApiUnreachable):
        client.health("http://127.0.0.1:9")


def test_a_non_json_response_still_produces_a_sentence(monkeypatch) -> None:
    class FakeResponse:
        status_code = 502

        @staticmethod
        def json():
            raise ValueError("not json")

        text = "<html>bad gateway</html>"

    monkeypatch.setattr(requests, "post", lambda *a, **k: FakeResponse())
    status, payload = client.score_one("http://example.invalid", {"prospect_id": "p"})
    assert status == 502
    assert payload["code"] == "unexpected_response"
    assert "Traceback" not in client.describe_error(status, payload)


def test_read_csv_accepts_a_file_like_object() -> None:
    frame = client.read_csv(io.BytesIO(GOOD_CSV.encode("utf-8")))
    assert len(frame) == 2


# ---------------------------------------------------------------------------
# the pages themselves, rendered by Streamlit's own harness
# ---------------------------------------------------------------------------
#
# `AppTest` runs `app.py` the way Streamlit does and exposes what it drew, which is the only
# way to find out that a page raises - an HTTP 200 from the server proves only that the shell
# loaded. The API is stubbed here so CI needs no running service; the same pages were also
# rendered against a live API on 2026-11-19 (docs/14).

APP_PATH = str(Path(__file__).resolve().parents[1] / "dashboard" / "app.py")

FAKE_INFO = {
    "model_version": "stub-1",
    "trained_at": "2026-11-19T10:00:00+00:00",
    "algorithm": "LightGBM, Optuna-tuned",
    "features": ["total_time_spent_on_website", "lead_origin"],
    "excluded_as_leakage": ["Tags", "Lead Quality"],
    "metrics": {"roc_auc": 0.865, "pr_auc": 0.805, "brier": 0.142, "capture_at_20": 0.441},
    "baselines": {"one_rule": {"roc_auc": 0.646, "capture_at_20": 0.353}},
    "grade_bands": {"A": {"min_score": 76}, "B": {"min_score": 26}, "C": {"min_score": 0}},
    "limits": ["A stub. No number here means anything."],
}

FAKE_SCORE = {
    "prospect_id": "dashboard-try-one",
    "score": 92,
    "probability": 0.9168,
    "grade": "A",
    "reasons": [
        {"feature": "current_occupation", "direction": "up", "text": "Is working"},
        {"feature": "total_time_spent_on_website", "direction": "up", "text": "Long visit"},
        {"feature": "lead_origin", "direction": "up", "text": "Good channel"},
    ],
    "model_version": "stub-1",
    "scored_at": "2026-11-19T10:00:00Z",
}


@pytest.fixture
def app_test():
    """A fresh app, with Streamlit's data cache cleared between tests."""
    streamlit_testing = pytest.importorskip("streamlit.testing.v1")
    import streamlit as st

    st.cache_data.clear()
    return streamlit_testing.AppTest.from_file(APP_PATH, default_timeout=60)


@pytest.fixture
def stub_api(monkeypatch):
    monkeypatch.setattr(client, "health", lambda _url: {"status": "ok", "model_version": "stub-1"})
    monkeypatch.setattr(client, "model_info", lambda _url: FAKE_INFO)
    monkeypatch.setattr(client, "score_one", lambda _url, _payload: (200, FAKE_SCORE))
    monkeypatch.setattr(
        client,
        "score_batch",
        lambda _url, payloads: (
            200,
            {
                "results": [{**FAKE_SCORE, "prospect_id": p["prospect_id"]} for p in payloads],
                "count": len(payloads),
                "model_version": "stub-1",
                "request_id": "stub-request",
            },
        ),
    )


@pytest.mark.parametrize("page", ["Rank leads", "Try one lead", "Model card"])
def test_every_page_renders_without_raising(app_test, stub_api, page) -> None:
    app_test.run()
    assert not app_test.exception, app_test.exception
    app_test.sidebar.radio[0].set_value(page).run()
    assert not app_test.exception, app_test.exception
    assert [element.value for element in app_test.title] == [page]
    assert not app_test.error


def test_rank_leads_opens_on_the_empty_state(app_test, stub_api) -> None:
    """docs/04: the empty state explains what to upload and offers a sample."""
    app_test.run()
    assert not app_test.exception
    info = " ".join(str(element.value) for element in app_test.info)
    assert "Required columns" in info
    for column in client.REQUIRED_COLUMNS:
        assert column in info


def test_submitting_the_form_shows_a_score_and_three_reasons(app_test, stub_api) -> None:
    app_test.run()
    app_test.sidebar.radio[0].set_value("Try one lead").run()
    app_test.button[0].click().run()

    assert not app_test.exception, app_test.exception
    assert [element.value for element in app_test.subheader] == ["Why"]
    drawn = " ".join(str(element.value) for element in app_test.markdown)
    assert "92" in drawn
    for reason in FAKE_SCORE["reasons"]:
        assert reason["text"] in drawn


def test_the_model_card_page_shows_the_served_metrics(app_test, stub_api) -> None:
    app_test.run()
    app_test.sidebar.radio[0].set_value("Model card").run()
    assert not app_test.exception, app_test.exception
    drawn = " ".join(str(element.value) for element in app_test.markdown)
    assert "44.1%" in drawn
    assert "stub-1" in drawn
    assert "must not be used for" in " ".join(
        str(element.value) for element in app_test.subheader
    )


def test_an_unreachable_api_is_reported_without_an_exception(app_test, monkeypatch) -> None:
    """docs/04: one message, and nothing the rep typed is lost."""
    def unreachable(*_args, **_kwargs):
        raise client.ApiUnreachable("connection refused")

    monkeypatch.setattr(client, "health", unreachable)
    monkeypatch.setattr(client, "model_info", unreachable)
    app_test.run()
    app_test.sidebar.radio[0].set_value("Model card").run()

    assert not app_test.exception, app_test.exception
    assert any("not responding" in str(element.value) for element in app_test.error)
    assert any("unreachable" in str(element.value).lower() for element in app_test.warning)


def test_the_footer_names_the_model_on_every_page(app_test, stub_api) -> None:
    """A screenshot is always attributable to a model version (docs/04)."""
    for page in ("Rank leads", "Try one lead", "Model card"):
        app_test.run()
        app_test.sidebar.radio[0].set_value(page).run()
        drawn = " ".join(str(element.value) for element in app_test.markdown)
        assert "stub-1" in drawn, f"{page} does not name the model"
