"""Tests for the synthetic fixture (task T-109).

The fixture is CI's only dataset, so if it drifts out of agreement with the data contract, CI
stops testing the pipeline and starts testing a relaxed version of it. These tests fail at the
source instead.
"""

from __future__ import annotations

import warnings

import pytest

from leadscore import config as cfg
from leadscore import fixtures, schema


def test_the_fixture_satisfies_the_real_data_contract() -> None:
    frame = fixtures.synthetic_leads(rows=200)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", schema.DataDriftWarning)
        validated = schema.validate(frame)
    assert len(validated) == 200


def test_the_fixture_has_every_column_the_config_names_and_no_others() -> None:
    frame = fixtures.synthetic_leads(rows=50)
    assert schema.missing_columns(frame) == []
    assert schema.unclassified_columns(frame) == []
    assert len(frame.columns) == cfg.EXPECTED_COLUMNS


def test_the_fixture_is_deterministic() -> None:
    first = fixtures.synthetic_leads(rows=120, seed=3)
    second = fixtures.synthetic_leads(rows=120, seed=3)
    assert first.equals(second)


def test_a_different_seed_gives_different_data() -> None:
    assert not fixtures.synthetic_leads(rows=120, seed=3).equals(
        fixtures.synthetic_leads(rows=120, seed=4)
    )


def test_the_conversion_rate_matches_the_documented_one() -> None:
    rate = fixtures.synthetic_leads(rows=2_000)[cfg.TARGET].mean()
    assert rate == pytest.approx(cfg.EXPECTED_TARGET_RATE, abs=0.03)


def test_the_fixture_carries_the_defects_the_pipeline_handles() -> None:
    """Missing numerics, `Select` placeholders and near-constant flags, all present."""
    frame = fixtures.synthetic_leads(rows=500)
    assert frame["TotalVisits"].isna().any(), "no missing numerics to impute"
    assert (frame["Specialization"] == cfg.SELECT_PLACEHOLDER).any(), "no Select placeholder"
    flag = cfg.FLAG_FEATURES[0]
    assert frame[flag].value_counts(normalize=True).iloc[0] > 0.95, "flag column is not constant"


def test_the_row_count_never_matches_the_real_file() -> None:
    """A drift warning on every fixture run is the point: it can never pass for real data."""
    frame = fixtures.synthetic_leads(rows=300)
    with pytest.warns(schema.DataDriftWarning, match="rows"):
        schema.check_shape(frame)


def test_tags_is_leaky_on_purpose() -> None:
    """The fixture must be able to demonstrate the leakage the audit is built to catch."""
    frame = fixtures.synthetic_leads(rows=1_000)
    crosstab = frame.groupby("Tags")[cfg.TARGET].mean()
    assert crosstab.max() - crosstab.min() > 0.5


def test_write_fixture_produces_a_readable_csv(tmp_path) -> None:
    path = fixtures.write_fixture(tmp_path / "nested" / "Leads.csv", rows=80)
    assert path.exists()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", schema.DataDriftWarning)
        frame = schema.load_and_validate(path, cfg.Settings(_env_file=None))
    assert len(frame) == 80
