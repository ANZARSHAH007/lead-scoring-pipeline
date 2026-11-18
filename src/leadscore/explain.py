"""Three reasons per lead, in plain English (task T-405).

A score with no reasons is a number a salesperson is asked to trust. Three reasons is the
smallest count that explains a decision without turning into a feature dump, and docs/02
FR-6 fixes it at exactly three.

How a reason is built, in order:

1. **SHAP** attributions for one lead, from `TreeExplainer` on the tree model. These are
   per *transformed* column - one value per one-hot level - so they are summed back onto
   the source column a human recognises (`features.group_by_source`). A column's effect is
   the signed sum of its levels' effects.
2. **Direction** comes from the sign: positive pushes the score up, negative pushes it down.
3. **A phrase** comes from `PHRASES`, which holds a sentence for each direction of every
   admissible column. No raw SHAP value reaches the response text (FR-6), because "0.42" is
   not a reason; it is a measurement of one.

Two honest limits, stated here because they belong in the code that produces the claim:

* attributions explain *this model's* output, not the world. "Spent little time on the site"
  does not mean making them browse longer would convert them.
* calibration is monotone, so the explainer runs on the uncalibrated tree model that ships
  beside the calibrated one. Same ranking, same signs, and `TreeExplainer` stays applicable.
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from leadscore import config as cfg
from leadscore import features as ft

LOGGER = logging.getLogger(__name__)

REASON_COUNT = 3

#: One sentence per direction for every admissible column. Both directions are filled in for
#: all 13, because a lead with nothing going for it still needs three reasons, and a
#: fallback phrase like "this feature was influential" is not a reason.
PHRASES: dict[str, dict[str, str]] = {
    "Total Time Spent on Website": {
        "up": "spent a long time on the website",
        "down": "spent little time on the website",
    },
    "TotalVisits": {
        "up": "came back to the site several times",
        "down": "has barely visited the site",
    },
    "Page Views Per Visit": {
        "up": "reads several pages per visit",
        "down": "leaves after a page or two",
    },
    "Lead Origin": {
        "up": "arrived through a channel that converts well",
        "down": "arrived through a channel that rarely converts",
    },
    "Lead Source": {
        "up": "came from a source with a strong track record",
        "down": "came from a source that rarely converts",
    },
    "Specialization": {
        "up": "picked a specialisation that usually converts",
        "down": "picked a specialisation that rarely converts, or none at all",
    },
    "What is your current occupation": {
        "up": "is working or self-employed, which converts well",
        "down": "is a student or did not say, which converts less often",
    },
    "What matters most to you in choosing a course": {
        "up": "is motivated by career prospects",
        "down": "gave no clear motivation for the course",
    },
    "City": {
        "up": "is in a city that converts above average",
        "down": "is in a city that converts below average, or did not say",
    },
    "Country": {
        "up": "is in a country that converts above average",
        "down": "is outside the usual market, or did not say",
    },
    "How did you hear about X Education": {
        "up": "heard about the course through a channel that converts",
        "down": "did not say how they heard about the course",
    },
    "Do Not Email": {
        "up": "can be contacted by email",
        "down": "has opted out of email",
    },
    "Do Not Call": {
        "up": "can be contacted by phone",
        "down": "has opted out of calls",
    },
}


@dataclass(frozen=True)
class Reason:
    """One readable reason for one lead."""

    feature: str
    direction: str
    phrase: str
    weight: float

    def as_dict(self) -> dict[str, object]:
        return {
            "feature": self.feature,
            "direction": self.direction,
            "phrase": self.phrase,
        }


def missing_phrases() -> list[str]:
    """Admissible columns with no phrase for both directions. Should always be empty."""
    missing: list[str] = []
    for column in cfg.shipping_features():
        entry = PHRASES.get(column)
        if not entry or "up" not in entry or "down" not in entry:
            missing.append(column)
    return missing


def phrase_for(column: str, direction: str) -> str:
    """The sentence for a column and a direction, or a stated fallback."""
    entry = PHRASES.get(column)
    if entry and direction in entry:
        return entry[direction]
    LOGGER.warning("no phrase for %s/%s; add one to explain.PHRASES", column, direction)
    return f"{column.lower()} moved the score {direction}"


def _tree_pipeline(model: Any) -> Pipeline:
    """The `prep` + tree pipeline, whether a bundle, a pipeline or a calibrated wrapper."""
    if isinstance(model, dict):  # a loaded bundle
        return _tree_pipeline(model.get("explainer_model") or model["model"])
    if isinstance(model, Pipeline):
        return model
    inner = getattr(model, "calibrated_classifiers_", None)
    if inner:
        return _tree_pipeline(inner[0].estimator)
    estimator = getattr(model, "estimator", None)
    if estimator is not None:
        return _tree_pipeline(estimator)
    raise TypeError(
        f"cannot find a tree pipeline inside {type(model).__name__}. Explanations need the "
        "uncalibrated LightGBM pipeline that `save_model_bundle` stores alongside the model."
    )


class Explainer:
    """SHAP attributions, grouped by source column and turned into sentences.

    Built once and reused: `TreeExplainer` setup is the expensive part, and the API loads
    the model at startup rather than per request.
    """

    def __init__(self, model: Any) -> None:
        import shap

        self.pipeline = _tree_pipeline(model)
        self.preprocessor = self.pipeline.named_steps["prep"]
        self.tree = self.pipeline.named_steps["model"]
        self.feature_names = ft.feature_names(self.preprocessor)
        self._explainer = shap.TreeExplainer(self.tree)

    def contributions(self, row: pd.DataFrame) -> dict[str, float]:
        """Signed contribution of every source column, for a single lead."""
        if len(row) != 1:
            raise ValueError(f"explain one lead at a time, got {len(row)} rows")

        transformed = self.preprocessor.transform(row)
        with warnings.catch_warnings():
            # SHAP announces that its LightGBM binary output is now a list of arrays. Both
            # shapes are handled three lines below, so the notice is noise here - and this
            # runs once per scored lead in the API, where noise costs something.
            warnings.filterwarnings(
                "ignore", message="LightGBM binary classifier with TreeExplainer"
            )
            values = self._explainer.shap_values(transformed)
        values = np.asarray(values)

        # LightGBM binary classification returns (rows, features) in current SHAP, and
        # (classes, rows, features) in older ones. Take the positive class either way.
        if values.ndim == 3:
            values = values[-1]
        flat = np.asarray(values).reshape(len(self.feature_names))
        return ft.group_by_source(self.feature_names, flat)

    def reasons(self, row: pd.DataFrame, count: int = REASON_COUNT) -> list[Reason]:
        """The `count` strongest reasons for one lead, strongest first."""
        grouped = self.contributions(row)
        ranked = sorted(grouped.items(), key=lambda item: abs(item[1]), reverse=True)
        reasons: list[Reason] = []
        for column, weight in ranked[:count]:
            direction = "up" if weight >= 0 else "down"
            reasons.append(
                Reason(
                    feature=column,
                    direction=direction,
                    phrase=phrase_for(column, direction),
                    weight=float(weight),
                )
            )
        return reasons


def explain_frame(model: Any, frame: pd.DataFrame, count: int = REASON_COUNT) -> list[list[Reason]]:
    """Reasons for every row of a frame, in order. One explainer, many leads."""
    explainer = Explainer(model)
    return [explainer.reasons(frame.iloc[[index]], count) for index in range(len(frame))]
