"""Tests for the published metrics, calibration, grades and the gate (T-401 to T-404).

Every number in this file is hand-computed. A metric implementation that is wrong in a
plausible direction — capture counting the wrong tail, a decile table off by one row — would
produce numbers that look reasonable and are false, and the README quotes them. So the
assertions here compare against arithmetic done by hand, never against the code's own output.
"""

from __future__ import annotations

import itertools
import json
import logging

import numpy as np
import pandas as pd
import pytest

from leadscore import config as cfg
from leadscore import evaluate as ev


@pytest.fixture
def settings() -> cfg.Settings:
    return cfg.Settings(_env_file=None, cv_folds=3, optuna_trials=0)


@pytest.fixture(autouse=True)
def clean_ledger():
    ev.reset_ledger()
    yield
    ev.reset_ledger()


# ---------------------------------------------------------------------------
# Capture@20% — the headline
# ---------------------------------------------------------------------------


def test_capture_at_counts_the_top_of_the_ranking() -> None:
    """10 leads, 4 conversions, both of the top 2 scores convert -> 2/4."""
    y = np.array([1, 1, 0, 0, 1, 1, 0, 0, 0, 0])
    scores = np.array([0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0])
    assert ev.capture_at(y, scores, 0.20) == pytest.approx(0.5)


def test_a_perfect_ranking_captures_everything_it_can_reach() -> None:
    """20 leads, 4 conversions, all 4 ranked first: the top 20% is 4 leads -> all of them."""
    y = np.array([1, 1, 1, 1] + [0] * 16)
    scores = np.linspace(1.0, 0.0, 20)
    assert ev.capture_at(y, scores, 0.20) == pytest.approx(1.0)
    assert ev.perfect_capture_at(y, 0.20) == pytest.approx(1.0)


def test_a_constant_score_captures_the_base_fraction() -> None:
    """No ranking information means the top 20% is an arbitrary 20% — and must score as one."""
    y = np.array([1, 0] * 5)
    assert ev.capture_at(y, np.full(10, 0.38), 0.20) == pytest.approx(0.2)


def test_capture_at_is_not_fooled_by_reversed_scores() -> None:
    y = np.array([1, 1, 0, 0, 1, 1, 0, 0, 0, 0])
    worst_first = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    assert ev.capture_at(y, worst_first, 0.20) == pytest.approx(0.0)


def test_capture_at_rounds_the_depth_and_never_takes_zero_rows() -> None:
    y = np.array([1, 0, 0])
    assert ev.capture_at(y, np.array([0.9, 0.5, 0.1]), 0.01) == pytest.approx(1.0)


def test_capture_at_with_no_conversions_is_zero_not_an_error() -> None:
    assert ev.capture_at(np.zeros(10), np.random.default_rng(0).random(10), 0.2) == 0.0


def test_capture_at_rejects_an_impossible_depth() -> None:
    with pytest.raises(ValueError, match="top_fraction"):
        ev.capture_at(np.array([1, 0]), np.array([0.9, 0.1]), 0.0)
    with pytest.raises(ValueError, match="top_fraction"):
        ev.capture_at(np.array([1, 0]), np.array([0.9, 0.1]), 1.5)


# ---------------------------------------------------------------------------
# decile lift — the table a sales manager reads
# ---------------------------------------------------------------------------


def test_decile_lift_puts_a_perfectly_ranked_cohort_in_the_first_decile() -> None:
    y = np.array([1] * 10 + [0] * 90)
    scores = np.linspace(1.0, 0.0, 100)
    rows = ev.decile_lift(y, scores)

    assert len(rows) == 10
    assert sum(row.leads for row in rows) == 100
    assert rows[0].conversions == 10
    assert rows[0].conversion_rate == pytest.approx(1.0)
    assert rows[0].lift == pytest.approx(10.0)  # 100% against a 10% overall rate
    assert rows[0].cumulative_capture == pytest.approx(1.0)
    assert rows[-1].conversions == 0


def test_decile_lift_cumulative_capture_reaches_one() -> None:
    rng = np.random.default_rng(3)
    y = rng.binomial(1, 0.4, 500)
    rows = ev.decile_lift(y, rng.random(500))
    assert rows[-1].cumulative_capture == pytest.approx(1.0)
    assert all(a.cumulative_capture <= b.cumulative_capture for a, b in itertools.pairwise(rows))


def test_decile_lift_of_a_random_ranking_is_flat_around_one() -> None:
    rng = np.random.default_rng(11)
    y = rng.binomial(1, 0.4, 4_000)
    lifts = [row.lift for row in ev.decile_lift(y, rng.random(4_000))]
    assert max(lifts) < 1.3
    assert min(lifts) > 0.7


# ---------------------------------------------------------------------------
# the metric bundle
# ---------------------------------------------------------------------------


def test_score_model_on_a_perfect_ranking() -> None:
    y = pd.Series([0, 0, 1, 1])
    metrics = ev.score_model("perfect", y, np.array([0.0, 0.1, 0.9, 1.0]))
    assert metrics.roc_auc == pytest.approx(1.0)
    assert metrics.pr_auc == pytest.approx(1.0)
    assert metrics.rows == 4
    assert metrics.positives == 2


def test_brier_is_the_mean_squared_error_of_the_probability() -> None:
    """Hand-computed: ((0.2-0)^2 + (0.8-1)^2) / 2 = 0.04."""
    metrics = ev.score_model("two", pd.Series([0, 1]), np.array([0.2, 0.8]))
    assert metrics.brier == pytest.approx(0.04)


def test_a_confident_wrong_model_has_a_terrible_brier() -> None:
    metrics = ev.score_model("wrong", pd.Series([0, 1]), np.array([1.0, 0.0]))
    assert metrics.brier == pytest.approx(1.0)
    assert metrics.roc_auc == pytest.approx(0.0)


def test_score_model_refuses_mismatched_lengths() -> None:
    with pytest.raises(ValueError, match="labels against"):
        ev.score_model("bad", pd.Series([0, 1, 1]), np.array([0.5, 0.5]))


def test_metrics_round_trip_through_a_dict() -> None:
    metrics = ev.score_model("m", pd.Series([0, 1, 1, 0]), np.array([0.1, 0.9, 0.8, 0.2]))
    row = metrics.as_dict()
    assert set(row) == {"name", "roc_auc", "pr_auc", "brier", "capture_at_20", "rows", "positives"}
    assert json.loads(json.dumps(row))["name"] == "m"


# ---------------------------------------------------------------------------
# scores and grades (T-402)
# ---------------------------------------------------------------------------


def test_a_score_is_a_rounded_percentage_of_the_probability() -> None:
    scores = ev.to_score(np.array([0.0, 0.004, 0.385, 0.5, 0.999, 1.0]))
    assert list(scores) == [0, 0, 38, 50, 100, 100]
    assert np.issubdtype(scores.dtype, np.integer)


def test_an_exact_half_rounds_to_even_like_pythons_round() -> None:
    """FR-6 says `round(100 * p)`, and Python's `round` is half-to-even, so this is 12 not 13.

    Pinned by a test because the alternative (half-up) is what most people assume, and a
    silent change of convention would move a lead across a grade boundary.
    """
    assert list(ev.to_score(np.array([0.125, 0.135]))) == [12, 14]


def test_a_score_is_clipped_into_zero_to_one_hundred() -> None:
    assert list(ev.to_score(np.array([-0.5, 1.5]))) == [0, 100]


def test_grade_cutoffs_sit_at_the_configured_percentiles(settings) -> None:
    """Probabilities spread evenly over 0-1: the 80th percentile score is 80."""
    cutoffs = ev.grade_cutoffs(np.linspace(0.0, 1.0, 101), settings)
    assert cutoffs.a_from == 80
    assert cutoffs.b_from == 50
    assert cutoffs.source == "out-of-fold"


def test_grades_cover_every_score_with_no_gap(settings) -> None:
    cutoffs = ev.grade_cutoffs(np.linspace(0.0, 1.0, 101), settings)
    grades = {cutoffs.grade(score) for score in range(101)}
    assert grades == {"A", "B", "C"}
    assert cutoffs.grade(cutoffs.a_from) == "A"
    assert cutoffs.grade(cutoffs.a_from - 1) == "B"
    assert cutoffs.grade(cutoffs.b_from) == "B"
    assert cutoffs.grade(cutoffs.b_from - 1) == "C"
    assert cutoffs.grade(0) == "C"


def test_a_flat_score_distribution_does_not_collapse_the_bands(settings) -> None:
    """A degenerate model must still produce three usable bands, not two identical ones."""
    cutoffs = ev.grade_cutoffs(np.full(200, 0.38), settings)
    assert cutoffs.b_from < cutoffs.a_from
    assert {cutoffs.grade(s) for s in range(101)} >= {"A", "B"}


def test_the_cutoffs_record_where_they_were_fitted(settings) -> None:
    card = ev.grade_cutoffs(np.linspace(0, 1, 50), settings, source="held-out").as_dict()
    assert card["fitted_on"] == "held-out"
    assert card["C_from_score"] == 0
    assert card["policy"]["A_from_percentile"] == pytest.approx(0.80)


# ---------------------------------------------------------------------------
# the reliability curve (T-401)
# ---------------------------------------------------------------------------


def test_a_perfectly_calibrated_model_sits_on_the_diagonal() -> None:
    rng = np.random.default_rng(5)
    probabilities = rng.uniform(0.05, 0.95, 20_000)
    outcomes = rng.binomial(1, probabilities)
    for row in ev.reliability_curve(outcomes, probabilities):
        if row["leads"] > 100:
            assert abs(row["observed_rate"] - row["mean_predicted"]) < 0.03


def test_the_reliability_curve_reports_empty_bins_instead_of_hiding_them() -> None:
    rows = ev.reliability_curve(np.array([0, 1]), np.array([0.05, 0.95]), bins=10)
    assert len(rows) == 10
    assert rows[0]["leads"] == 1 and rows[-1]["leads"] == 1
    middle = rows[5]
    assert middle["leads"] == 0 and np.isnan(middle["observed_rate"])


def test_the_top_bin_includes_a_probability_of_exactly_one() -> None:
    rows = ev.reliability_curve(np.array([1]), np.array([1.0]), bins=10)
    assert rows[-1]["leads"] == 1


# ---------------------------------------------------------------------------
# the quality gate (T-404)
# ---------------------------------------------------------------------------


def _metrics(name: str, roc: float, brier: float, capture: float) -> ev.Metrics:
    return ev.Metrics(
        name=name, roc_auc=roc, pr_auc=roc - 0.1, brier=brier,
        capture_at_20=capture, rows=1_848, positives=712,
    )


SHIPPING_NAMES = [
    "TotalVisits", "Total Time Spent on Website", "Page Views Per Visit",
    *[f"{column}_level" for column in cfg.categorical_features()],
]


def test_a_good_model_passes_every_check(settings) -> None:
    gate = ev.quality_gate(
        candidate=_metrics("lightgbm_shipped", 0.88, 0.14, 0.52),
        linear=_metrics("logistic_regression", 0.86, 0.15, 0.49),
        one_rule=_metrics("one_rule", 0.65, 0.30, 0.31),
        feature_names=SHIPPING_NAMES,
        settings=settings,
    )
    assert gate.passed, str(gate)
    assert gate.failures == []
    assert len(gate.checks) == 5


def test_a_model_that_cannot_beat_logistic_regression_is_rejected(settings) -> None:
    gate = ev.quality_gate(
        candidate=_metrics("lightgbm_shipped", 0.855, 0.14, 0.52),
        linear=_metrics("logistic_regression", 0.86, 0.15, 0.49),
        one_rule=_metrics("one_rule", 0.65, 0.30, 0.31),
        feature_names=SHIPPING_NAMES,
        settings=settings,
    )
    assert not gate.passed
    assert [check.name for check in gate.failures] == ["beats the linear baseline on ROC-AUC"]


def test_a_sharp_ranking_that_lies_about_probabilities_is_rejected(settings) -> None:
    """Brier above the ceiling fails even with an excellent ROC-AUC."""
    gate = ev.quality_gate(
        candidate=_metrics("lightgbm_shipped", 0.95, 0.25, 0.60),
        linear=_metrics("logistic_regression", 0.86, 0.15, 0.49),
        one_rule=_metrics("one_rule", 0.65, 0.30, 0.31),
        feature_names=SHIPPING_NAMES,
        settings=settings,
    )
    assert not gate.passed
    assert "Brier ceiling" in [check.name for check in gate.failures]


def test_capture_must_beat_the_spreadsheet_sort_not_just_random(settings) -> None:
    gate = ev.quality_gate(
        candidate=_metrics("lightgbm_shipped", 0.88, 0.14, 0.29),
        linear=_metrics("logistic_regression", 0.86, 0.15, 0.25),
        one_rule=_metrics("one_rule", 0.65, 0.30, 0.31),
        feature_names=SHIPPING_NAMES,
        settings=settings,
    )
    assert not gate.passed
    failure = next(c for c in gate.failures if c.name.startswith("Capture@20%"))
    assert failure.threshold == pytest.approx(0.31)


def test_a_banned_column_reaching_the_model_fails_the_gate(settings) -> None:
    """The most important check in the suite: leakage must stop the model shipping."""
    gate = ev.quality_gate(
        candidate=_metrics("lightgbm_shipped", 0.98, 0.08, 0.80),
        linear=_metrics("logistic_regression", 0.86, 0.15, 0.49),
        one_rule=_metrics("one_rule", 0.65, 0.30, 0.31),
        feature_names=[*SHIPPING_NAMES, "Tags_Will revert after reading the email"],
        settings=settings,
    )
    assert not gate.passed
    failure = next(c for c in gate.failures if "banned" in c.name)
    assert "Tags" in str(failure.value)


def test_a_missing_admissible_column_also_fails_the_gate(settings) -> None:
    gate = ev.quality_gate(
        candidate=_metrics("lightgbm_shipped", 0.88, 0.14, 0.52),
        linear=_metrics("logistic_regression", 0.86, 0.15, 0.49),
        one_rule=_metrics("one_rule", 0.65, 0.30, 0.31),
        feature_names=SHIPPING_NAMES[:-3],
        settings=settings,
    )
    assert not gate.passed
    assert any("banned" in check.name for check in gate.failures)


def test_the_gate_reports_the_numbers_it_judged_on(settings) -> None:
    """A gate that fails without saying what the value was is useless in CI output."""
    gate = ev.quality_gate(
        candidate=_metrics("lightgbm_shipped", 0.70, 0.14, 0.52),
        linear=_metrics("logistic_regression", 0.86, 0.15, 0.49),
        one_rule=_metrics("one_rule", 0.65, 0.30, 0.31),
        feature_names=SHIPPING_NAMES,
        settings=settings,
    )
    as_dict = gate.as_dict()
    assert as_dict["passed"] is False
    for check in as_dict["checks"]:
        assert check["value"] is not None and check["why"]
    assert "FAIL" in str(gate)


# ---------------------------------------------------------------------------
# the ledger and the fingerprint: the test set is opened once
# ---------------------------------------------------------------------------


def test_the_fingerprint_is_stable_under_reordering() -> None:
    assert ev.fingerprint(pd.Index([3, 1, 2])) == ev.fingerprint(pd.Index([1, 2, 3]))


def test_the_fingerprint_changes_when_the_rows_change() -> None:
    assert ev.fingerprint(pd.Index([1, 2, 3])) != ev.fingerprint(pd.Index([1, 2, 4]))


def test_the_ledger_records_each_opening() -> None:
    assert ev.ledger() == []
    ev.record_opening("evaluate_on_test", 1_848, "abc123")
    entries = ev.ledger()
    assert len(entries) == 1
    assert entries[0]["reader"] == "evaluate_on_test"
    assert entries[0]["rows"] == 1_848


def test_a_second_opening_is_recorded_and_warned_about(caplog) -> None:
    ev.record_opening("first", 10, "x")
    with caplog.at_level(logging.WARNING, logger="leadscore.evaluate"):
        ev.record_opening("second", 10, "x")
    assert len(ev.ledger()) == 2
    assert "read 2 times" in caplog.text


def test_the_ledger_cannot_be_edited_from_outside() -> None:
    ev.record_opening("one", 1, "x")
    stolen = ev.ledger()
    stolen.clear()
    stolen.append({"reader": "forged"})
    assert len(ev.ledger()) == 1


# ---------------------------------------------------------------------------
# the model card
# ---------------------------------------------------------------------------


def test_the_model_version_is_deterministic(settings) -> None:
    first = ev.model_version({"n_estimators": 400}, ["a", "b"], settings)
    second = ev.model_version({"n_estimators": 400}, ["b", "a"], settings)
    assert first == second


def test_a_different_model_gets_a_different_version(settings) -> None:
    assert ev.model_version({"n_estimators": 400}, ["a"], settings) != ev.model_version(
        {"n_estimators": 401}, ["a"], settings
    )


def test_the_seed_is_part_of_the_version(settings) -> None:
    other = cfg.Settings(_env_file=None, random_seed=7)
    assert ev.model_version({}, ["a"], settings) != ev.model_version({}, ["a"], other)


def test_the_card_is_written_as_readable_json(tmp_path, settings) -> None:
    local = cfg.Settings(_env_file=None, model_card_path=tmp_path / "models" / "card.json")
    card = {"model_version": "20261118-abcdef12", "headline": {"value": 0.52}}
    path = ev.write_model_card(card, local)
    assert path.exists()
    assert json.loads(path.read_text(encoding="utf-8"))["model_version"] == "20261118-abcdef12"


# ---------------------------------------------------------------------------
# the published row label (T-407)
# ---------------------------------------------------------------------------


def test_the_shipping_label_states_what_was_actually_done() -> None:
    """The plan said "calibrated". If no calibrator was applied, the table must not say so."""
    assert ev.shipping_label("none") == "**LightGBM (tuned, no calibrator needed)**"
    assert ev.shipping_label("isotonic") == "**LightGBM (isotonic-calibrated)**"
    assert ev.shipping_label("sigmoid") == "**LightGBM (sigmoid-calibrated)**"


def test_the_results_table_has_a_row_per_model_in_readme_order() -> None:
    metrics = [
        _metrics(name, 0.8, 0.15, 0.4)
        for name in ("lightgbm_leakage", "random", "lightgbm_shipped",
                     "one_rule", "logistic_regression")
    ]
    lines = ev.results_table(metrics, "none")
    assert lines[0].startswith("| Model |")
    body = lines[2:]
    assert len(body) == 5
    assert body[0].startswith("| Random baseline |")
    assert "no calibrator needed" in body[3]
    assert body[4].startswith("| LightGBM + sales columns")
    assert all(line.startswith("| ") and line.endswith(" |") for line in body)


def test_a_missing_model_is_left_out_rather_than_faked() -> None:
    lines = ev.results_table([_metrics("random", 0.5, 0.25, 0.2)], "none")
    assert len(lines) == 3
    assert "LightGBM" not in "\n".join(lines)
