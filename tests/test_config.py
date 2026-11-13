"""Tests for the config module (docs/10, T-105).

The assertion that matters most here is :func:`test_shipping_features_contain_no_banned_column`.
Everything else in the pipeline trusts the shipping feature list to be leakage-free, so this is
the test that enforces the project's defining rule.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from leadscore import config as cfg

# ---------------------------------------------------------------------------
# the leakage rule
# ---------------------------------------------------------------------------


def test_shipping_features_contain_no_banned_column() -> None:
    shipping = set(cfg.shipping_features())
    banned = set(cfg.BANNED_LEAKAGE_COLUMNS)
    leaked = shipping & banned
    assert not leaked, f"leaking columns reached the shipping feature set: {sorted(leaked)}"


def test_shipping_features_exclude_identifiers_and_sparse_columns() -> None:
    shipping = set(cfg.shipping_features())
    assert not shipping & set(cfg.ID_COLUMNS)
    assert not shipping & set(cfg.DROPPED_SPARSE_COLUMNS)
    assert cfg.TARGET not in shipping


def test_audit_set_is_the_shipping_set_plus_exactly_the_banned_columns() -> None:
    shipping = set(cfg.shipping_features())
    audit = set(cfg.audit_features())
    assert shipping < audit
    assert audit - shipping == set(cfg.BANNED_LEAKAGE_COLUMNS)


def test_activity_columns_are_opt_outable() -> None:
    with_activity = set(cfg.shipping_features(include_activity=True))
    without = set(cfg.shipping_features(include_activity=False))
    assert with_activity - without == set(cfg.ACTIVITY_FEATURES)


def test_feature_set_resolves_known_names() -> None:
    assert cfg.feature_set("shipping") == cfg.shipping_features()
    assert cfg.feature_set("audit_with_leakage") == cfg.audit_features()


def test_feature_set_rejects_an_unknown_name() -> None:
    with pytest.raises(ValueError, match="unknown feature set"):
        cfg.feature_set("everything")  # type: ignore[arg-type]


def test_no_column_appears_in_two_groups() -> None:
    groups = (
        cfg.ID_COLUMNS,
        cfg.SOURCE_FEATURES,
        cfg.NUMERIC_FEATURES,
        cfg.FORM_FEATURES,
        cfg.CONTACT_PREFERENCE_FEATURES,
        cfg.FLAG_FEATURES,
        cfg.ACTIVITY_FEATURES,
        cfg.BANNED_LEAKAGE_COLUMNS,
        cfg.DROPPED_SPARSE_COLUMNS,
    )
    flat = [column for group in groups for column in group]
    duplicates = {name for name in flat if flat.count(name) > 1}
    assert not duplicates, f"column listed in more than one group: {sorted(duplicates)}"


def test_excluded_columns_never_overlap_the_shipping_set() -> None:
    assert not set(cfg.excluded_columns()) & set(cfg.shipping_features())


def test_column_groups_describe_the_documented_dataset() -> None:
    """The groups plus the target should account for the 37 columns docs/02 describes."""
    accounted = len(set(cfg.excluded_columns()) | set(cfg.shipping_features())) + 1  # + target
    assert accounted == cfg.EXPECTED_COLUMNS, (
        f"column groups account for {accounted} of {cfg.EXPECTED_COLUMNS} columns; "
        "update docs/02 and config together"
    )


# ---------------------------------------------------------------------------
# settings: defaults, overrides, validation
# ---------------------------------------------------------------------------


def test_documented_defaults_load_with_no_environment() -> None:
    s = cfg.Settings(_env_file=None)
    assert s.random_seed == 42
    assert s.test_size == pytest.approx(0.2)
    assert s.cv_folds == 5
    assert s.max_batch == 1_000
    assert s.log_level == "INFO"
    assert s.api_key == ""
    assert s.data_raw == Path("data/raw/Leads.csv")


def test_environment_overrides_a_default_with_the_right_type(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LEADSCORE_RANDOM_SEED", "7")
    monkeypatch.setenv("LEADSCORE_MAX_BATCH", "250")
    monkeypatch.setenv("LEADSCORE_DATA_RAW", "tests/fixtures/leads_sample.csv")
    s = cfg.Settings(_env_file=None)
    assert s.random_seed == 7
    assert isinstance(s.random_seed, int)
    assert s.max_batch == 250
    assert s.data_raw == Path("tests/fixtures/leads_sample.csv")


def test_settings_are_cached_so_two_calls_agree() -> None:
    cfg.get_settings.cache_clear()
    try:
        assert cfg.get_settings() is cfg.get_settings()
    finally:
        cfg.get_settings.cache_clear()


def test_settings_are_frozen() -> None:
    s = cfg.Settings(_env_file=None)
    with pytest.raises(ValidationError):
        s.random_seed = 1  # type: ignore[misc]


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("random_seed", -1),
        ("test_size", 0.0),
        ("test_size", 1.0),
        ("test_size", 2.0),
        ("cv_folds", 1),
        ("optuna_trials", -5),
        ("rare_level_threshold", 0.0),
        ("rare_level_threshold", 1.0),
        ("gate_min_roc_auc", 0.4),
        ("gate_min_roc_auc", 1.5),
        ("gate_max_brier", 0.0),
        ("max_batch", 0),
        ("api_port", 0),
        ("api_port", 70_000),
        ("log_level", "TRACE"),
    ],
)
def test_a_bad_value_is_rejected_on_construction(field: str, value: object) -> None:
    with pytest.raises(ValidationError):
        cfg.Settings(_env_file=None, **{field: value})


def test_grade_bands_must_be_ordered() -> None:
    with pytest.raises(ValidationError, match="grade_b_percentile must be below"):
        cfg.Settings(_env_file=None, grade_a_percentile=0.5, grade_b_percentile=0.8)
    with pytest.raises(ValidationError, match="grade_b_percentile must be below"):
        cfg.Settings(_env_file=None, grade_a_percentile=0.6, grade_b_percentile=0.6)


def test_grade_band_policy_is_ordered_and_reaches_zero() -> None:
    policy = cfg.Settings(_env_file=None).grade_band_policy
    assert policy["A_from_percentile"] > policy["B_from_percentile"] > policy["C_from_percentile"]
    assert policy["C_from_percentile"] == 0.0


def test_an_empty_path_is_rejected() -> None:
    with pytest.raises(ValidationError, match="path must not be empty"):
        cfg.Settings(_env_file=None, model_path="")


def test_default_grade_policy_matches_the_documented_twenty_thirty_split() -> None:
    s = cfg.Settings(_env_file=None)
    assert s.grade_a_percentile == pytest.approx(0.80), "grade A is the top 20% (docs/02 FR-6)"
    assert s.grade_b_percentile == pytest.approx(0.50), "grade B is the next 30%"
