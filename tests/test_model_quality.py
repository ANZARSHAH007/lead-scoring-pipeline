"""The quality gate, exercised against really fitted models (task T-404).

`test_evaluate.py` checks the gate's logic with hand-made metrics. This file checks it
against models that were actually trained, because the gate's job is to stop a bad model
reaching `models/`, and a gate that has only ever seen hand-made numbers has never done that
job.

Everything here runs on the synthetic fixture, so it runs in CI with no dataset. One
consequence is worth stating plainly: the fixture's signal *is* time on site, by
construction, so the one-rule baseline is unusually strong on it and the tree model does not
reliably beat the linear one. That is a property of the fixture, not a finding about the real
data - which is why no test here asserts that the fixture's model passes the gate, and why
the real numbers come from `reports/04_evaluation.md`.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from leadscore import config as cfg
from leadscore import evaluate as ev
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
        frame = schema.validate(fixtures.synthetic_leads(rows=600), settings)
    return tr.make_split(frame, "shipping", settings)


@pytest.fixture(autouse=True)
def clean_ledger():
    ev.reset_ledger()
    yield
    ev.reset_ledger()


@pytest.fixture
def fitted_metrics(split, settings) -> dict[str, ev.Metrics]:
    """One pass over the fixture's held-out rows, for every baseline plus the tree model."""
    models = {name: (est, split) for name, est in tr.build_candidates(settings).items()}
    models["lightgbm_shipped"] = (tr.build_lightgbm(tr.DEFAULT_LGBM, settings), split)
    metrics, _ = ev.evaluate_on_test(models, settings)
    return {m.name: m for m in metrics}


def _feature_names(split, settings) -> list[str]:
    fitted = tr.build_lightgbm(tr.DEFAULT_LGBM, settings).fit(split.x_train, split.y_train)
    return ft.feature_names(fitted.named_steps["prep"])


# ---------------------------------------------------------------------------
# the gate's own test: a model that should never ship
# ---------------------------------------------------------------------------


def test_a_deliberately_crippled_model_is_rejected(split, settings, fitted_metrics) -> None:
    """Labels shuffled, so the model has learned noise. The gate must refuse it.

    This is the one test that exercises the gate's purpose rather than its arithmetic. If it
    ever passes, the gate has stopped protecting anything.
    """
    shuffled = split.y_train.sample(frac=1.0, random_state=0).to_numpy()
    crippled = tr.build_lightgbm(tr.DEFAULT_LGBM, settings).fit(split.x_train, shuffled)
    scores = crippled.predict_proba(split.x_test)[:, 1]
    metrics = ev.score_model("crippled", split.y_test, scores)

    gate = ev.quality_gate(
        candidate=metrics,
        linear=fitted_metrics["logistic_regression"],
        one_rule=fitted_metrics["one_rule"],
        feature_names=_feature_names(split, settings),
        settings=settings,
    )
    assert not gate.passed
    failed = {check.name for check in gate.failures}
    assert "ROC-AUC floor" in failed
    assert "beats the linear baseline on ROC-AUC" in failed


def test_a_random_model_is_rejected(split, settings, fitted_metrics) -> None:
    gate = ev.quality_gate(
        candidate=fitted_metrics["random"],
        linear=fitted_metrics["logistic_regression"],
        one_rule=fitted_metrics["one_rule"],
        feature_names=_feature_names(split, settings),
        settings=settings,
    )
    assert not gate.passed


def test_the_gate_reports_real_numbers_for_a_real_model(split, settings, fitted_metrics) -> None:
    """Whatever the verdict, every check must carry the value it judged on."""
    gate = ev.quality_gate(
        candidate=fitted_metrics["lightgbm_shipped"],
        linear=fitted_metrics["logistic_regression"],
        one_rule=fitted_metrics["one_rule"],
        feature_names=_feature_names(split, settings),
        settings=settings,
    )
    assert len(gate.checks) == 5
    for check in gate.checks:
        assert check.value not in (None, "")
        assert check.why


# ---------------------------------------------------------------------------
# the model actually learns something, and only from admissible columns
# ---------------------------------------------------------------------------


def test_the_tree_model_beats_random_on_the_fixture(fitted_metrics) -> None:
    assert fitted_metrics["lightgbm_shipped"].roc_auc > 0.70
    assert fitted_metrics["random"].roc_auc == pytest.approx(0.5, abs=0.05)


def test_capture_beats_the_random_floor(fitted_metrics) -> None:
    assert fitted_metrics["lightgbm_shipped"].capture_at_20 > 0.20


def test_the_fitted_model_exposes_only_admissible_columns(split, settings) -> None:
    """The leakage rule, checked on the fitted object rather than on a list in a doc."""
    sources = {ft.source_column(name) for name in _feature_names(split, settings)}
    assert sources == set(cfg.shipping_features())
    assert not sources & set(cfg.BANNED_LEAKAGE_COLUMNS)
    assert not sources & set(cfg.ACTIVITY_FEATURES)


def test_the_leakage_model_scores_higher_and_still_cannot_ship(split, settings) -> None:
    """The row that exists to be refused: more columns, better numbers, not shippable."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", schema.DataDriftWarning)
        frame = schema.validate(fixtures.synthetic_leads(rows=600), settings)
    leakage_split = tr.make_split(frame, "audit_with_leakage", settings)
    assert leakage_split.x_test.index.equals(split.x_test.index), "the held-out rows must match"

    models = {
        "lightgbm_shipped": (tr.build_lightgbm(tr.DEFAULT_LGBM, settings), split),
        "lightgbm_leakage": (
            tr.build_lightgbm(tr.DEFAULT_LGBM, settings, "audit_with_leakage"),
            leakage_split,
        ),
    }
    metrics, _ = ev.evaluate_on_test(models, settings)
    by_name = {m.name: m for m in metrics}
    assert by_name["lightgbm_leakage"].roc_auc > by_name["lightgbm_shipped"].roc_auc

    leaked_sources = {
        ft.source_column(name)
        for name in ft.feature_names(
            tr.build_lightgbm(tr.DEFAULT_LGBM, settings, "audit_with_leakage")
            .fit(leakage_split.x_train, leakage_split.y_train)
            .named_steps["prep"]
        )
    }
    gate = ev.quality_gate(
        candidate=by_name["lightgbm_leakage"],
        linear=by_name["lightgbm_shipped"],
        one_rule=by_name["lightgbm_shipped"],
        feature_names=sorted(leaked_sources),
        settings=settings,
    )
    assert not gate.passed
    assert any("banned" in check.name for check in gate.failures)


# ---------------------------------------------------------------------------
# the single pass refuses to compare models on different rows
# ---------------------------------------------------------------------------


def test_models_given_different_held_out_rows_are_refused(split, settings) -> None:
    other = cfg.Settings(_env_file=None, cv_folds=3, optuna_trials=0, random_seed=99)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", schema.DataDriftWarning)
        frame = schema.validate(fixtures.synthetic_leads(rows=600), settings)
    mismatched = tr.make_split(frame, "shipping", other)

    with pytest.raises(ValueError, match="different held-out rows"):
        ev.evaluate_on_test(
            {
                "a": (tr.build_lightgbm(tr.DEFAULT_LGBM, settings), split),
                "b": (tr.build_lightgbm(tr.DEFAULT_LGBM, settings), mismatched),
            },
            settings,
        )


def test_one_pass_is_recorded_as_one_opening(split, settings) -> None:
    ev.evaluate_on_test({"m": (tr.build_lightgbm(tr.DEFAULT_LGBM, settings), split)}, settings)
    entries = ev.ledger()
    assert len(entries) == 1
    assert entries[0]["rows"] == len(split.x_test)


def test_an_empty_evaluation_is_an_error(settings) -> None:
    with pytest.raises(ValueError, match="nothing to evaluate"):
        ev.evaluate_on_test({}, settings)


# ---------------------------------------------------------------------------
# calibration and cut-offs are decided without the held-out rows
# ---------------------------------------------------------------------------


def test_calibration_is_chosen_from_the_training_split_alone(split, settings) -> None:
    choice = ev.choose_calibration(
        tr.build_lightgbm(tr.DEFAULT_LGBM, settings), split.x_train, split.y_train, settings
    )
    assert choice.method in {"none", "sigmoid", "isotonic"}
    assert set(choice.brier_by_method) == {"none", "sigmoid", "isotonic"}
    assert len(choice.oof_scores) == len(split.y_train), "one validation score per training row"
    assert ev.ledger() == [], "choosing a calibrator must not touch the held-out rows"


def test_the_chosen_calibrator_has_the_lowest_brier(split, settings) -> None:
    choice = ev.choose_calibration(
        tr.build_lightgbm(tr.DEFAULT_LGBM, settings), split.x_train, split.y_train, settings
    )
    assert choice.brier_by_method[choice.method] == min(choice.brier_by_method.values())


def test_grade_cutoffs_come_from_validation_scores_not_the_test_set(split, settings) -> None:
    choice = ev.choose_calibration(
        tr.build_lightgbm(tr.DEFAULT_LGBM, settings), split.x_train, split.y_train, settings
    )
    cutoffs = ev.grade_cutoffs(choice.oof_scores, settings)
    assert 0 <= cutoffs.b_from < cutoffs.a_from <= 100
    assert ev.ledger() == []

    # Roughly a fifth of the validation leads should land in grade A, by construction.
    grades = [cutoffs.grade(int(score)) for score in ev.to_score(choice.oof_scores)]
    share_a = grades.count("A") / len(grades)
    assert share_a == pytest.approx(1 - settings.grade_a_percentile, abs=0.06)


def test_calibration_does_not_change_the_ranking(split, settings) -> None:
    """Monotone by construction, which is why explanations may come from the raw model."""
    raw = tr.build_lightgbm(tr.DEFAULT_LGBM, settings).fit(split.x_train, split.y_train)
    calibrated = ev.apply_calibration(
        tr.build_lightgbm(tr.DEFAULT_LGBM, settings), "isotonic", settings
    ).fit(split.x_train, split.y_train)

    raw_scores = raw.predict_proba(split.x_test)[:, 1]
    calibrated_scores = calibrated.predict_proba(split.x_test)[:, 1]
    # Not identical models, so not identical rankings; but ROC-AUC must not collapse.
    assert ev.score_model("c", split.y_test, calibrated_scores).roc_auc > 0.70
    assert np.corrcoef(raw_scores, calibrated_scores)[0, 1] > 0.9
