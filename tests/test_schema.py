"""Tests for the data contract (docs/10, task T-202).

Everything here runs on synthetic frames, so the suite needs no Kaggle download and stays
fast. One test is marked `needs_data` and exercises the real file when it is present.
"""

from __future__ import annotations

import re
import warnings

import pandas as pd
import pytest

from leadscore import config as cfg
from leadscore import schema as sch


def make_frame(rows: int = 6) -> pd.DataFrame:
    """A minimal frame that satisfies the contract, with every known column."""
    data: dict[str, list[object]] = {
        "Prospect ID": [f"id-{i}" for i in range(rows)],
        "Lead Number": list(range(rows)),
        cfg.TARGET: [i % 2 for i in range(rows)],
        "TotalVisits": [float(i) for i in range(rows)],
        "Total Time Spent on Website": [float(i * 100) for i in range(rows)],
        "Page Views Per Visit": [float(i) / 2 for i in range(rows)],
    }
    text_columns = (
        set(cfg.known_columns() if hasattr(cfg, "known_columns") else sch.known_columns())
        - set(data)
    )
    for column in sorted(text_columns):
        data[column] = ["Yes" if i % 2 else "No" for i in range(rows)]
    return pd.DataFrame(data)


# ---------------------------------------------------------------------------
# the three corruptions each produce their own readable error
# ---------------------------------------------------------------------------


def test_a_dropped_column_is_named_in_the_error() -> None:
    frame = make_frame().drop(columns=["Lead Source"])
    with pytest.raises(sch.DataContractError, match="Lead Source"):
        sch.validate(frame)


def test_a_non_binary_target_is_rejected() -> None:
    frame = make_frame()
    frame.loc[0, cfg.TARGET] = 2
    with pytest.raises(sch.DataContractError, match="Converted"):
        sch.validate(frame)


def test_a_string_in_a_numeric_column_is_rejected() -> None:
    frame = make_frame()
    frame["TotalVisits"] = frame["TotalVisits"].astype(object)
    frame.loc[0, "TotalVisits"] = "many"
    with pytest.raises(sch.DataContractError, match="TotalVisits"):
        sch.validate(frame)


def test_the_three_corruptions_give_three_different_messages() -> None:
    messages = []
    for corrupt in (
        lambda f: f.drop(columns=["Lead Source"]),
        lambda f: f.assign(**{cfg.TARGET: [2] + [0] * (len(f) - 1)}),
        lambda f: f.assign(TotalVisits=["many"] + [1.0] * (len(f) - 1)),
    ):
        with pytest.raises(sch.DataContractError) as caught:
            sch.validate(corrupt(make_frame()))
        messages.append(str(caught.value))
    assert len(set(messages)) == 3, "each corruption must be distinguishable from the others"


# ---------------------------------------------------------------------------
# admissibility coverage
# ---------------------------------------------------------------------------


def test_an_unclassified_column_is_rejected() -> None:
    frame = make_frame().assign(**{"Mystery Column": "x"})
    with pytest.raises(sch.DataContractError, match="Mystery Column"):
        sch.validate(frame)


def test_the_error_for_an_unclassified_column_explains_why_it_matters() -> None:
    frame = make_frame().assign(**{"Mystery Column": "x"})
    with pytest.raises(sch.DataContractError, match="admissibility"):
        sch.validate(frame)


def test_known_columns_covers_every_documented_column() -> None:
    assert len(sch.known_columns()) == cfg.EXPECTED_COLUMNS


# ---------------------------------------------------------------------------
# negative numbers
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("column", cfg.NUMERIC_FEATURES)
def test_a_negative_value_is_rejected_in_every_numeric_column(column: str) -> None:
    frame = make_frame()
    frame.loc[0, column] = -1.0
    with pytest.raises(sch.DataContractError, match=column):
        sch.validate(frame)


def test_nulls_are_allowed_in_numeric_columns() -> None:
    frame = make_frame()
    frame.loc[0, "TotalVisits"] = None
    with warnings.catch_warnings():
        # a 6-row fixture always drifts from the documented shape; that is check_shape's
        # own test to make, not this one's
        warnings.simplefilter("ignore", sch.DataDriftWarning)
        validated = sch.validate(frame)
    assert validated["TotalVisits"].isna().sum() == 1


# ---------------------------------------------------------------------------
# the Select placeholder
# ---------------------------------------------------------------------------


def test_select_becomes_missing_in_every_text_column() -> None:
    frame = make_frame()
    frame.loc[0, "Specialization"] = cfg.SELECT_PLACEHOLDER
    frame.loc[1, "City"] = cfg.SELECT_PLACEHOLDER
    out = sch.normalise_select(frame)
    assert pd.isna(out.loc[0, "Specialization"])
    assert pd.isna(out.loc[1, "City"])


def test_select_with_surrounding_whitespace_is_still_missing() -> None:
    frame = make_frame()
    frame.loc[0, "Specialization"] = f"  {cfg.SELECT_PLACEHOLDER}  "
    assert pd.isna(sch.normalise_select(frame).loc[0, "Specialization"])


def test_a_real_answer_survives_normalisation() -> None:
    frame = make_frame()
    frame.loc[0, "Specialization"] = "Finance Management"
    assert sch.normalise_select(frame).loc[0, "Specialization"] == "Finance Management"


def test_normalise_does_not_mutate_the_caller_s_frame() -> None:
    frame = make_frame()
    frame.loc[0, "Specialization"] = cfg.SELECT_PLACEHOLDER
    sch.normalise_select(frame)
    assert frame.loc[0, "Specialization"] == cfg.SELECT_PLACEHOLDER


# ---------------------------------------------------------------------------
# shape: warnings, never failures
# ---------------------------------------------------------------------------


def test_a_short_frame_warns_rather_than_fails() -> None:
    # a wide tolerance isolates the row-count warning from the target-rate one
    settings = cfg.Settings(_env_file=None, target_rate_tolerance=0.9)
    with pytest.warns(sch.DataDriftWarning, match="rows"):
        notes = sch.check_shape(make_frame(rows=6), settings)
    assert notes == [f"expected {cfg.EXPECTED_ROWS:,} rows, found 6"]


def test_a_drifted_target_rate_warns() -> None:
    # full-size frame, so the only thing that drifts is the conversion rate
    frame = make_frame(rows=cfg.EXPECTED_ROWS)
    frame[cfg.TARGET] = 1
    with pytest.warns(sch.DataDriftWarning, match="conversion rate"):
        notes = sch.check_shape(frame)
    assert len(notes) == 1
    assert "conversion rate" in notes[0]


def test_validate_still_returns_a_frame_when_the_shape_drifts() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", sch.DataDriftWarning)
        out = sch.validate(make_frame(rows=6))
    assert len(out) == 6


# ---------------------------------------------------------------------------
# a missing file is explained, not traced
# ---------------------------------------------------------------------------


def test_a_missing_file_points_at_the_download_instructions(tmp_path) -> None:
    with pytest.raises(sch.DataContractError, match=re.escape("data/raw/README.md")):
        sch.load_and_validate(tmp_path / "nope.csv")


# ---------------------------------------------------------------------------
# the real file, when it is there
# ---------------------------------------------------------------------------


@pytest.mark.needs_data
def test_the_real_dataset_satisfies_the_contract() -> None:
    settings = cfg.Settings(_env_file=None)
    if not settings.data_raw.exists():
        pytest.skip("Leads.csv not downloaded")
    frame = sch.load_and_validate(settings.data_raw, settings)
    assert len(frame) == cfg.EXPECTED_ROWS
    assert frame[cfg.TARGET].isin([0, 1]).all()
    assert abs(frame[cfg.TARGET].mean() - cfg.EXPECTED_TARGET_RATE) < 0.01
