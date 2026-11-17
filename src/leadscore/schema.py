"""The data contract for `data/raw/Leads.csv` (task T-202).

Nothing else in the pipeline touches raw data before this module has passed it. The contract
exists so that a changed upload fails here, by name, instead of surfacing as a strange metric
three stages later.

Four things are enforced, in this order:

1. **Every column the config names is present.** A renamed column is the most likely way a
   different Kaggle upload breaks the pipeline, and it is silent unless checked.
2. **No column is unclassified.** A column in the file that no config group mentions raises,
   rather than being quietly ignored — an unclassified column has had no admissibility
   decision made about it, and the leakage rule is only as good as its coverage.
3. **Types and ranges hold.** Negative visit counts, a non-binary target, a string in a
   numeric column.
4. **Shape and target rate are sane.** A warning, not an error: a few points of drift means a
   different upload, not a wrong one.

``Select`` is normalised to missing before any of this, because it is a form default rather
than an answer (docs/02 FR-1).

    from leadscore.schema import load_and_validate
    frame = load_and_validate()
"""

from __future__ import annotations

import warnings
from pathlib import Path

import pandas as pd
import pandera.pandas as pa

from leadscore import config as cfg


class DataContractError(Exception):
    """Raised when the raw data is not what the pipeline was built for."""


class DataDriftWarning(UserWarning):
    """The data is usable but differs from what the docs describe."""


# ---------------------------------------------------------------------------
# the schema
# ---------------------------------------------------------------------------


def build_schema() -> pa.DataFrameSchema:
    """The contract for the raw file, built from the column groups in config."""
    columns: dict[str, pa.Column] = {
        cfg.TARGET: pa.Column(
            int,
            checks=pa.Check.isin([0, 1], error="target must be 0 or 1"),
            nullable=False,
            coerce=True,
        ),
        "Prospect ID": pa.Column(str, unique=True, nullable=False),
        "Lead Number": pa.Column(int, unique=True, nullable=False, coerce=True),
    }

    for column in cfg.NUMERIC_FEATURES:
        columns[column] = pa.Column(
            float,
            checks=pa.Check.ge(0, error=f"{column} cannot be negative"),
            nullable=True,
            coerce=True,
        )

    # Everything else is categorical text. `Select` has already become null by this point.
    categorical = (
        cfg.SOURCE_FEATURES
        + cfg.FORM_FEATURES
        + cfg.CONTACT_PREFERENCE_FEATURES
        + cfg.FLAG_FEATURES
        + cfg.ACTIVITY_FEATURES
        + cfg.BANNED_LEAKAGE_COLUMNS
        + cfg.DROPPED_SPARSE_COLUMNS
    )
    for column in categorical:
        if column in columns:
            continue
        columns[column] = pa.Column(str, nullable=True, coerce=True)

    return pa.DataFrameSchema(
        columns,
        strict=False,  # unexpected columns are reported by `unclassified_columns`, with a reason
        coerce=True,
        name="Leads.csv",
    )


# ---------------------------------------------------------------------------
# normalisation
# ---------------------------------------------------------------------------


def normalise_select(frame: pd.DataFrame) -> pd.DataFrame:
    """Replace the `Select` form default with a real missing value, in every text column.

    Applied before validation, not after: a column that is 45% `Select` is 45% missing, and
    every rule downstream should see it that way.
    """
    out = frame.copy()
    for column in out.columns:
        series = out[column]
        # pandas 3 gives text columns the `str` dtype rather than `object`, so testing for
        # `object` alone silently skips every column on a modern pandas. Test for "not a
        # number" instead, which holds on both.
        if pd.api.types.is_numeric_dtype(series) or pd.api.types.is_bool_dtype(series):
            continue
        stripped = series.astype("string").str.strip()
        out[column] = stripped.mask(stripped == cfg.SELECT_PLACEHOLDER, other=pd.NA)
    return out


# ---------------------------------------------------------------------------
# checks
# ---------------------------------------------------------------------------


def known_columns() -> set[str]:
    """Every column name any config group mentions, plus the target."""
    return (
        set(cfg.shipping_features())
        | set(cfg.excluded_columns())
        | set(cfg.FLAG_FEATURES)
        | {cfg.TARGET}
    )


def missing_columns(frame: pd.DataFrame) -> list[str]:
    return sorted(known_columns() - set(frame.columns))


def unclassified_columns(frame: pd.DataFrame) -> list[str]:
    return sorted(set(frame.columns) - known_columns())


def check_shape(frame: pd.DataFrame, settings: cfg.Settings | None = None) -> list[str]:
    """Warn — never fail — when the shape differs from what the docs describe."""
    settings = settings or cfg.Settings()
    notes: list[str] = []

    if len(frame) != cfg.EXPECTED_ROWS:
        notes.append(f"expected {cfg.EXPECTED_ROWS:,} rows, found {len(frame):,}")
    if len(frame.columns) != cfg.EXPECTED_COLUMNS:
        notes.append(f"expected {cfg.EXPECTED_COLUMNS} columns, found {len(frame.columns)}")

    if cfg.TARGET in frame.columns and len(frame):
        rate = float(frame[cfg.TARGET].mean())
        drift = abs(rate - cfg.EXPECTED_TARGET_RATE)
        if drift > settings.target_rate_tolerance:
            notes.append(
                f"conversion rate {rate:.4f} differs from the documented "
                f"{cfg.EXPECTED_TARGET_RATE:.4f} by {drift:.4f}"
            )

    for note in notes:
        warnings.warn(note, DataDriftWarning, stacklevel=2)
    return notes


# ---------------------------------------------------------------------------
# the entry point
# ---------------------------------------------------------------------------


def validate(frame: pd.DataFrame, settings: cfg.Settings | None = None) -> pd.DataFrame:
    """Normalise, then enforce the contract. Returns the validated frame.

    Raises:
        DataContractError: with the offending column names and why they failed.
    """
    settings = settings or cfg.Settings()

    absent = missing_columns(frame)
    if absent:
        raise DataContractError(
            f"{len(absent)} column(s) named in config are missing from the data: "
            f"{', '.join(absent)}. Either the wrong file was downloaded, or config and the "
            "upload disagree — see data/raw/README.md."
        )

    extra = unclassified_columns(frame)
    if extra:
        raise DataContractError(
            f"{len(extra)} column(s) in the data are not classified in config: "
            f"{', '.join(extra)}. Every column needs an admissibility decision before it can "
            "be used or ignored (docs/02 FR-2); add it to a group in config.py."
        )

    normalised = normalise_select(frame)

    try:
        validated = build_schema().validate(normalised, lazy=True)
    except pa.errors.SchemaErrors as error:
        cases = error.failure_cases
        summary = (
            cases.groupby("column")["failure_case"]
            .agg(["count", lambda s: ", ".join(map(str, s.unique()[:3]))])
            .reset_index()
        )
        detail = "; ".join(
            f"{row['column']}: {row['count']} bad value(s) ({row['<lambda_0>']})"
            for _, row in summary.iterrows()
        )
        raise DataContractError(f"the raw data violates its contract — {detail}") from error

    check_shape(validated, settings)
    return validated


def load_and_validate(
    path: Path | None = None, settings: cfg.Settings | None = None
) -> pd.DataFrame:
    """Read the raw CSV and enforce the contract. The only supported way in."""
    settings = settings or cfg.Settings()
    path = path or settings.data_raw
    if not Path(path).exists():
        raise DataContractError(
            f"{path} not found. The dataset is not in this repo by design; "
            "see data/raw/README.md for how to get it."
        )
    return validate(pd.read_csv(path), settings)
