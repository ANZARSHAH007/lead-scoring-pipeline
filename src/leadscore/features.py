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
