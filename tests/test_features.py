"""Tests for the preprocessing pipeline (docs/10, task T-301).

The three that matter are the serving-time behaviours: an unseen category, a null, and
stability across save/load. Everything the API relies on is asserted here, because the API
itself never preprocesses.
"""

from __future__ import annotations

import io

import joblib
import numpy as np
import pandas as pd
import pytest

from leadscore import config as cfg
from leadscore import features as ft


def make_frame(rows: int = 60) -> pd.DataFrame:
    """A frame carrying exactly the shipping feature set."""
    rng = np.random.default_rng(0)
    cities = ["Mumbai", "Thane", "Other Metro"]
    sources = ["Google", "Olark Chat", "Direct Traffic"]
    data: dict[str, object] = {}
    for column in cfg.shipping_features():
        if column in cfg.NUMERIC_FEATURES:
            data[column] = rng.integers(0, 500, rows).astype(float)
        elif column == "City":
            data[column] = [cities[i % len(cities)] for i in range(rows)]
        elif column == "Lead Source":
            data[column] = [sources[i % len(sources)] for i in range(rows)]
        else:
            data[column] = ["Yes" if i % 2 else "No" for i in range(rows)]
    return pd.DataFrame(data)


@pytest.fixture
def fitted() -> ft.ColumnTransformer:
    pre = ft.build_preprocessor("shipping")
    pre.fit(make_frame())
    return pre


# ---------------------------------------------------------------------------
# serving-time robustness
# ---------------------------------------------------------------------------


def test_a_category_never_seen_in_training_does_not_raise(fitted) -> None:
    unseen = make_frame(rows=1)
    unseen.loc[0, "City"] = "Gilgit"
    out = fitted.transform(unseen)
    assert out.shape[0] == 1
    assert np.isfinite(out).all()


def test_a_null_becomes_an_explicit_missing_category(fitted) -> None:
    frame = make_frame(rows=1)
    frame.loc[0, "City"] = None
    assert np.isfinite(fitted.transform(frame)).all()


def test_a_null_in_every_nullable_column_still_transforms(fitted) -> None:
    frame = make_frame(rows=1)
    for column in cfg.shipping_features():
        frame.loc[0, column] = None
    out = fitted.transform(frame)
    assert out.shape[0] == 1
    assert np.isfinite(out).all(), "a fully-empty lead must still score, not crash"


def test_pandas_nullable_string_dtype_is_handled(fitted) -> None:
    """pandas 3 hands text columns a `string` dtype whose NA breaks naive encoders."""
    frame = make_frame(rows=2)
    frame["City"] = pd.array(["Mumbai", pd.NA], dtype="string")
    assert np.isfinite(fitted.transform(frame)).all()


# ---------------------------------------------------------------------------
# determinism and stability
# ---------------------------------------------------------------------------


def test_transforming_twice_gives_the_same_answer(fitted) -> None:
    frame = make_frame(rows=5)
    np.testing.assert_allclose(fitted.transform(frame), fitted.transform(frame))


def test_the_fitted_transformer_survives_a_round_trip_through_joblib(fitted) -> None:
    frame = make_frame(rows=5)
    before = fitted.transform(frame)
    buffer = io.BytesIO()
    joblib.dump(fitted, buffer)
    buffer.seek(0)
    restored = joblib.load(buffer)
    np.testing.assert_allclose(before, restored.transform(frame))
    assert ft.feature_names(restored) == ft.feature_names(fitted)


def test_feature_names_match_the_transformed_width(fitted) -> None:
    assert len(ft.feature_names(fitted)) == fitted.transform(make_frame(rows=3)).shape[1]


# ---------------------------------------------------------------------------
# the feature sets
# ---------------------------------------------------------------------------


def test_the_shipping_preprocessor_touches_no_banned_column() -> None:
    numeric, categorical = ft.split_columns("shipping")
    used = set(numeric) | set(categorical)
    assert not used & set(cfg.BANNED_LEAKAGE_COLUMNS)
    assert not used & set(cfg.ACTIVITY_FEATURES)
    assert not used & set(cfg.FLAG_FEATURES)


def test_every_shipping_column_lands_in_exactly_one_branch() -> None:
    numeric, categorical = ft.split_columns("shipping")
    assert not set(numeric) & set(categorical)
    assert set(numeric) | set(categorical) == set(cfg.shipping_features())


def test_the_audit_set_is_wider_than_the_shipping_set() -> None:
    shipping = ft.build_preprocessor("shipping").fit(make_frame())
    assert len(cfg.feature_set("audit_with_leakage")) > len(cfg.feature_set("shipping"))
    assert len(ft.feature_names(shipping)) > 0


# ---------------------------------------------------------------------------
# rare levels
# ---------------------------------------------------------------------------


def test_rare_levels_are_grouped_rather_than_given_their_own_column() -> None:
    frame = make_frame(rows=200)
    frame.loc[0, "City"] = "Vanishingly Rare"  # 0.5%, under the 1% threshold
    pre = ft.build_preprocessor("shipping")
    pre.fit(frame)
    assert not any("Vanishingly Rare" in name for name in ft.feature_names(pre))


def test_to_text_fills_nulls_with_the_missing_category() -> None:
    frame = pd.DataFrame({"a": pd.array(["x", pd.NA], dtype="string")})
    assert ft.to_text(frame)["a"].tolist() == ["x", ft.MISSING_CATEGORY]


def test_to_float_coerces_junk_to_null_rather_than_raising() -> None:
    frame = pd.DataFrame({"a": ["1", "oops"]})
    out = ft.to_float(frame)
    assert out["a"].iloc[0] == 1.0
    assert pd.isna(out["a"].iloc[1])
