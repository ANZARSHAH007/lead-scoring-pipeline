"""Tests for the split and the baselines (tasks T-302, T-303).

The split test is the important one: if the held-out rows ever leak into training, every
number this project publishes is wrong, and nothing else in the suite would notice.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import StratifiedKFold, cross_val_score

from leadscore import config as cfg
from leadscore import train as tr


def make_frame(rows: int = 400) -> pd.DataFrame:
    """A frame with a learnable signal, so a model can plausibly beat chance."""
    rng = np.random.default_rng(7)
    time_on_site = rng.integers(0, 2000, rows).astype(float)
    probability = 1 / (1 + np.exp(-(time_on_site - 900) / 300))
    target = rng.binomial(1, probability)

    data: dict[str, object] = {cfg.TARGET: target}
    for column in cfg.shipping_features():
        if column == tr.ONE_RULE_COLUMN:
            data[column] = time_on_site
        elif column in cfg.NUMERIC_FEATURES:
            data[column] = rng.integers(0, 20, rows).astype(float)
        else:
            data[column] = rng.choice(["a", "b", "c"], rows)
    return pd.DataFrame(data)


@pytest.fixture
def settings() -> cfg.Settings:
    return cfg.Settings(_env_file=None, cv_folds=3, optuna_trials=0)


# ---------------------------------------------------------------------------
# T-302 — the split
# ---------------------------------------------------------------------------


def test_the_split_holds_back_the_configured_share(settings) -> None:
    frame = make_frame()
    split = tr.make_split(frame, "shipping", settings)
    assert split.sizes["test"] == pytest.approx(len(frame) * settings.test_size, abs=1)
    assert split.sizes["train"] + split.sizes["test"] == len(frame)


def test_train_and_test_rows_never_overlap(settings) -> None:
    split = tr.make_split(make_frame(), "shipping", settings)
    assert not set(split.x_train.index) & set(split.x_test.index)


def test_the_split_is_stratified(settings) -> None:
    frame = make_frame()
    split = tr.make_split(frame, "shipping", settings)
    assert split.y_train.mean() == pytest.approx(frame[cfg.TARGET].mean(), abs=0.02)
    assert split.y_test.mean() == pytest.approx(frame[cfg.TARGET].mean(), abs=0.02)


def test_the_same_seed_reproduces_the_same_split(settings) -> None:
    frame = make_frame()
    first = tr.make_split(frame, "shipping", settings)
    second = tr.make_split(frame, "shipping", settings)
    assert list(first.x_test.index) == list(second.x_test.index)


def test_a_different_seed_gives_a_different_split() -> None:
    frame = make_frame()
    a = tr.make_split(frame, "shipping", cfg.Settings(_env_file=None, random_seed=1))
    b = tr.make_split(frame, "shipping", cfg.Settings(_env_file=None, random_seed=2))
    assert list(a.x_test.index) != list(b.x_test.index)


def test_the_split_carries_only_shipping_columns(settings) -> None:
    split = tr.make_split(make_frame(), "shipping", settings)
    assert list(split.x_train.columns) == list(cfg.shipping_features())
    assert cfg.TARGET not in split.x_train.columns


# ---------------------------------------------------------------------------
# T-303 — the one-rule baseline
# ---------------------------------------------------------------------------


def test_one_rule_ranks_longer_visits_higher() -> None:
    frame = make_frame()
    model = tr.OneRuleRanker().fit(frame, frame[cfg.TARGET])
    probabilities = model.predict_proba(frame)[:, 1]
    ranked = pd.Series(probabilities).corr(frame[tr.ONE_RULE_COLUMN], method="spearman")
    assert ranked > 0.99


def test_one_rule_probabilities_are_bounded() -> None:
    frame = make_frame()
    model = tr.OneRuleRanker().fit(frame, frame[cfg.TARGET])
    probabilities = model.predict_proba(frame)
    assert probabilities.min() >= 0.0
    assert probabilities.max() <= 1.0
    np.testing.assert_allclose(probabilities.sum(axis=1), 1.0)


def test_one_rule_survives_a_null_in_its_column() -> None:
    frame = make_frame()
    model = tr.OneRuleRanker().fit(frame, frame[cfg.TARGET])
    unseen = frame.head(1).copy()
    unseen.loc[unseen.index[0], tr.ONE_RULE_COLUMN] = np.nan
    assert np.isfinite(model.predict_proba(unseen)).all()


def test_one_rule_works_inside_cross_val_score(settings) -> None:
    frame = make_frame()
    folds = StratifiedKFold(n_splits=3, shuffle=True, random_state=settings.random_seed)
    scores = cross_val_score(
        tr.OneRuleRanker(), frame, frame[cfg.TARGET], cv=folds, scoring="roc_auc"
    )
    assert scores.mean() > 0.5, "on data built to reward time on site, it must beat chance"


# ---------------------------------------------------------------------------
# cross-validation wiring
# ---------------------------------------------------------------------------


def test_every_candidate_cross_validates(settings) -> None:
    split = tr.make_split(make_frame(), "shipping", settings)
    for name, estimator in tr.build_candidates(settings).items():
        result = tr.cross_validate(estimator, split, settings, name)
        assert 0.0 <= result.mean_auc <= 1.0
        assert result.std_auc >= 0.0


def test_the_random_baseline_scores_about_a_half(settings) -> None:
    split = tr.make_split(make_frame(rows=800), "shipping", settings)
    candidates = tr.build_candidates(settings)
    result = tr.cross_validate(candidates["random"], split, settings, "random")
    assert result.mean_auc == pytest.approx(0.5, abs=0.08)


def test_logistic_regression_beats_chance_on_learnable_data(settings) -> None:
    split = tr.make_split(make_frame(rows=800), "shipping", settings)
    candidates = tr.build_candidates(settings)
    result = tr.cross_validate(candidates["logistic_regression"], split, settings, "lr")
    assert result.mean_auc > 0.6


def test_cross_validation_never_sees_the_test_rows(settings, monkeypatch) -> None:
    """A guard against the single most damaging possible bug in this project."""
    split = tr.make_split(make_frame(), "shipping", settings)
    seen: list[int] = []

    original = tr.cross_val_score

    def spy(estimator, x, y, **kwargs):
        seen.extend(x.index.tolist())
        return original(estimator, x, y, **kwargs)

    monkeypatch.setattr(tr, "cross_val_score", spy)
    tr.cross_validate(tr.build_candidates(settings)["one_rule"], split, settings, "one_rule")
    assert not set(seen) & set(split.x_test.index)
