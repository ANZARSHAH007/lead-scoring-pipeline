"""Tests for the three reasons (task T-405).

The reasons are the part of the output a salesperson actually reads, so the tests are about
readability as much as correctness: three of them, always; a real sentence for every
direction of every admissible column; and no raw SHAP number anywhere in the text.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from leadscore import config as cfg
from leadscore import evaluate as ev
from leadscore import explain as ex
from leadscore import features as ft
from leadscore import fixtures, schema
from leadscore import train as tr


@pytest.fixture
def settings() -> cfg.Settings:
    return cfg.Settings(_env_file=None, cv_folds=3, optuna_trials=0)


@pytest.fixture
def split(settings: cfg.Settings):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", schema.DataDriftWarning)
        frame = schema.validate(fixtures.synthetic_leads(rows=400), settings)
    return tr.make_split(frame, "shipping", settings)


@pytest.fixture
def fitted(split, settings):
    return tr.build_lightgbm(tr.DEFAULT_LGBM, settings).fit(split.x_train, split.y_train)


@pytest.fixture
def explainer(fitted) -> ex.Explainer:
    return ex.Explainer(fitted)


# ---------------------------------------------------------------------------
# the phrase map (the acceptance criterion for T-405)
# ---------------------------------------------------------------------------


def test_every_admissible_column_has_a_phrase_in_both_directions() -> None:
    """T-405's acceptance criterion, as one assertion."""
    assert ex.missing_phrases() == []
    assert len(ex.PHRASES) == len(cfg.shipping_features())


def test_no_phrase_describes_a_banned_column() -> None:
    assert not set(ex.PHRASES) & set(cfg.BANNED_LEAKAGE_COLUMNS)
    assert not set(ex.PHRASES) & set(cfg.ACTIVITY_FEATURES)


def test_a_phrase_reads_as_a_sentence_fragment_not_a_number() -> None:
    for column, directions in ex.PHRASES.items():
        for direction, phrase in directions.items():
            assert len(phrase.split()) >= 3, f"{column}/{direction} is not a phrase"
            assert not any(character.isdigit() for character in phrase)


def test_an_unmapped_column_falls_back_and_says_so(caplog) -> None:
    phrase = ex.phrase_for("Tags", "up")
    assert "tags" in phrase.lower()
    assert "add one to explain.PHRASES" in caplog.text


# ---------------------------------------------------------------------------
# reasons for one lead
# ---------------------------------------------------------------------------


def test_exactly_three_reasons_come_back(explainer, split) -> None:
    reasons = explainer.reasons(split.x_test.iloc[[0]])
    assert len(reasons) == ex.REASON_COUNT == 3


def test_reasons_are_ordered_by_strength(explainer, split) -> None:
    reasons = explainer.reasons(split.x_test.iloc[[1]])
    weights = [abs(reason.weight) for reason in reasons]
    assert weights == sorted(weights, reverse=True)


def test_every_reason_names_a_real_column_and_a_direction(explainer, split) -> None:
    for reason in explainer.reasons(split.x_test.iloc[[2]]):
        assert reason.feature in cfg.shipping_features()
        assert reason.direction in {"up", "down"}
        assert reason.phrase == ex.PHRASES[reason.feature][reason.direction]


def test_the_response_shape_carries_no_shap_value(explainer, split) -> None:
    """FR-6: no raw SHAP values in the response text."""
    payload = explainer.reasons(split.x_test.iloc[[3]])[0].as_dict()
    assert set(payload) == {"feature", "direction", "phrase"}
    assert "weight" not in payload


def test_contributions_are_grouped_back_onto_source_columns(explainer, split) -> None:
    grouped = explainer.contributions(split.x_test.iloc[[4]])
    assert set(grouped) <= set(cfg.shipping_features())
    assert len(grouped) == len(cfg.shipping_features())


def test_explaining_more_than_one_lead_at_a_time_is_refused(explainer, split) -> None:
    with pytest.raises(ValueError, match="one lead at a time"):
        explainer.reasons(split.x_test.iloc[:2])


def test_a_lead_with_everything_missing_still_gets_three_reasons(explainer, split) -> None:
    """An edge case from docs/10: reasons must fall back to whatever is available."""
    empty = split.x_test.iloc[[0]].copy()
    for column in empty.columns:
        empty[column] = pd.NA
    reasons = explainer.reasons(empty)
    assert len(reasons) == 3
    assert all(reason.phrase for reason in reasons)


def test_an_unseen_category_still_scores_and_explains(explainer, split) -> None:
    row = split.x_test.iloc[[0]].copy()
    row["Lead Source"] = "a source that did not exist at training time"
    assert len(explainer.reasons(row)) == 3


def test_explain_frame_returns_reasons_for_every_row(fitted, split) -> None:
    batch = ex.explain_frame(fitted, split.x_test.head(5))
    assert len(batch) == 5
    assert all(len(reasons) == 3 for reasons in batch)


# ---------------------------------------------------------------------------
# finding the tree inside whatever it was wrapped in
# ---------------------------------------------------------------------------


def test_the_tree_is_found_inside_a_calibrated_wrapper(split, settings) -> None:
    calibrated = ev.apply_calibration(
        tr.build_lightgbm(tr.DEFAULT_LGBM, settings), "isotonic", settings
    ).fit(split.x_train, split.y_train)
    pipeline = ex._tree_pipeline(calibrated)
    assert "prep" in pipeline.named_steps


def test_the_tree_is_found_inside_a_saved_bundle(fitted, split, settings) -> None:
    bundle = {"model": "not a model", "explainer_model": fitted, "card": {}}
    explainer = ex.Explainer(bundle)
    assert len(explainer.reasons(split.x_test.iloc[[0]])) == 3


def test_something_that_is_not_a_model_is_refused() -> None:
    with pytest.raises(TypeError, match="cannot find a tree pipeline"):
        ex._tree_pipeline(object())


# ---------------------------------------------------------------------------
# the attributions point the way the data does
# ---------------------------------------------------------------------------


def test_time_on_site_pushes_the_score_the_way_the_fixture_built_it(explainer, split) -> None:
    """The fixture makes conversion rise with time on site, so the sign must follow.

    A test of the explanation's *meaning*, not its plumbing: if attributions came back with
    the sign flipped, every reason shown to a user would be backwards and nothing else in
    the suite would notice.
    """
    column = fixtures.SIGNAL_COLUMN
    frame = split.x_test
    numeric = pd.to_numeric(frame[column], errors="coerce")
    busiest = frame.loc[[numeric.idxmax()]]
    quietest = frame.loc[[numeric.idxmin()]]

    assert explainer.contributions(busiest)[column] > 0
    assert explainer.contributions(quietest)[column] < 0


def test_grouping_sums_signed_values_onto_their_columns() -> None:
    names = ["Lead Origin_API", "Lead Origin_Lead Import", "TotalVisits"]
    grouped = ft.group_by_source(names, np.array([0.4, -0.1, 0.25]))
    assert grouped["Lead Origin"] == pytest.approx(0.3)
    assert grouped["TotalVisits"] == pytest.approx(0.25)


def test_grouping_refuses_mismatched_lengths() -> None:
    with pytest.raises(ValueError, match="names against"):
        ft.group_by_source(["a", "b"], np.array([1.0]))
