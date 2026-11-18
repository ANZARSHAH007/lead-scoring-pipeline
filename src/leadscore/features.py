"""Preprocessing (task T-301).

One rule governs this module: **training and serving must apply the same transforms, by
construction rather than by discipline.** So every transform lives inside a scikit-learn
`ColumnTransformer` that is fitted once and saved *with* the model. The API never touches a
raw column by hand; it hands the request to this fitted object. Training/serving skew is the
most common silent failure in deployed ML, and this is the structural defence against it.

Three behaviours matter at serving time and are tested:

* a category value never seen in training does not raise — it folds into the infrequent
  bucket, because a live lead will eventually arrive with a city nobody saw before;
* a missing value does not raise — nulls become the literal ``Missing`` category, which is
  information rather than absence, given how much of this dataset is `Select`;
* rare levels are grouped, so a one-hot column is never fitted on three examples.

Helper functions are module-level on purpose: joblib has to pickle them with the model, and
a lambda cannot be pickled.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from leadscore import config as cfg

MISSING_CATEGORY = "Missing"


def to_text(frame: pd.DataFrame) -> pd.DataFrame:
    """Cast categorical columns to plain strings, with nulls as an explicit category.

    pandas nullable dtypes (`string`, `pd.NA`) do not survive scikit-learn's encoders, and
    `Select` has already become null by the time this runs, so the fill is meaningful: a
    lead who skipped the question is a different kind of lead from one who answered.
    """
    return frame.astype(object).where(frame.notna(), MISSING_CATEGORY).astype(str)


def to_float(frame: pd.DataFrame) -> pd.DataFrame:
    """Cast numeric columns to plain float64, preserving nulls for the imputer."""
    return frame.apply(pd.to_numeric, errors="coerce").astype("float64")


def build_preprocessor(
    name: cfg.FeatureSet = "shipping", settings: cfg.Settings | None = None
) -> ColumnTransformer:
    """The fitted-once transform for a named feature set."""
    settings = settings or cfg.Settings()
    columns = list(cfg.feature_set(name))
    numeric = [c for c in columns if c in cfg.NUMERIC_FEATURES]
    categorical = [c for c in columns if c not in cfg.NUMERIC_FEATURES]

    numeric_branch = Pipeline(
        [
            ("cast", FunctionTransformer(to_float, feature_names_out="one-to-one")),
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]
    )

    categorical_branch = Pipeline(
        [
            ("cast", FunctionTransformer(to_text, feature_names_out="one-to-one")),
            (
                "onehot",
                OneHotEncoder(
                    handle_unknown="infrequent_if_exist",
                    min_frequency=settings.rare_level_threshold,
                    sparse_output=False,
                ),
            ),
        ]
    )

    return ColumnTransformer(
        [("num", numeric_branch, numeric), ("cat", categorical_branch, categorical)],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def feature_names(preprocessor: ColumnTransformer) -> list[str]:
    """The ordered feature names a fitted preprocessor produces.

    SHAP and `/model-info` both report these, so they must come from the fitted object
    rather than from a list someone maintains by hand.
    """
    return list(preprocessor.get_feature_names_out())


def split_columns(name: cfg.FeatureSet = "shipping") -> tuple[list[str], list[str]]:
    """The numeric and categorical column lists for a feature set, in config order."""
    columns = list(cfg.feature_set(name))
    numeric = [c for c in columns if c in cfg.NUMERIC_FEATURES]
    categorical = [c for c in columns if c not in cfg.NUMERIC_FEATURES]
    return numeric, categorical


#: Every column name the config knows, longest first. One-hot feature names are built as
#: ``<column>_<category>``, and both parts can contain underscores and spaces — "What
#: matters most to you in choosing a course_Better Career Prospects" is one real example —
#: so a transformed name is matched by longest prefix rather than split on a separator.
_ALL_SOURCE_COLUMNS: tuple[str, ...] = tuple(
    sorted(set(cfg.shipping_features()) | set(cfg.excluded_columns()), key=len, reverse=True)
)


def source_column(name: str) -> str:
    """The raw column a transformed feature name came from.

    SHAP attributions arrive per transformed column — one per one-hot level — and both the
    reasons in the API and the gate's leakage check need them grouped back by the column a
    human would recognise. Returns the name unchanged when nothing matches, so an
    unexpected name shows up in output rather than being silently reassigned.
    """
    if name in _ALL_SOURCE_COLUMNS:
        return name
    for column in _ALL_SOURCE_COLUMNS:  # longest first, so the match is unambiguous
        if name.startswith(f"{column}_"):
            return column
    return name


def group_by_source(names: list[str], values: np.ndarray) -> dict[str, float]:
    """Sum per-feature values back onto their source columns, keeping the sign.

    Signed, not absolute: a one-hot level that pushes the score down and another that pushes
    it up are different facts about the same column, and summing them is what "this column
    moved the score by X" means.
    """
    if len(names) != len(values):
        raise ValueError(f"{len(names)} names against {len(values)} values")
    grouped: dict[str, float] = {}
    for name, value in zip(names, values, strict=True):
        grouped[source_column(name)] = grouped.get(source_column(name), 0.0) + float(value)
    return grouped
