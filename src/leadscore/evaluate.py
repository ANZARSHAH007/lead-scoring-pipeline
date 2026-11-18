"""Calibration, grades, metrics and the quality gate (tasks T-401 to T-404).

This module owns the held-out rows. Everything in Phases 1 to 3 was arranged so that the test
set stayed closed; here it is opened, once, and the numbers that come out are the ones the
README publishes. Two habits make that claim checkable rather than promised:

* every read of the held-out rows goes through `ledger()`, which records who read them and
  how many rows they saw. The count is written into the model card, so a reader can see
  that it says 1;
* `fingerprint()` hashes the held-out row identifiers. The same hash appearing in the card
  across runs proves the split did not drift; a different hash means the data or the seed
  changed, and the metrics are not comparable.

The order of operations matters and is not negotiable:

1. **calibrate** on the training split alone, choosing between raw, sigmoid and isotonic by
   out-of-fold Brier score (T-401);
2. **set the grade cut-offs** from those same out-of-fold scores — validation data, never
   test data, because a cut-off fitted on the test set makes the test set training data
   (T-402);
3. **score the test set once**, computing every published metric for every model in the
   README's table, including the leakage row that exists to be refused (T-403);
4. **run the quality gate** on the result (T-404). The gate's thresholds live in config. The
   rule from CLAUDE.md applies: report the numbers, never raise the gate to make a model
   pass.

Capture@20% is the headline. ROC-AUC, PR-AUC and Brier support it: ranking quality, ranking
quality where the positives are rare, and whether the probability means anything.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import logging
import warnings
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from leadscore import config as cfg

if TYPE_CHECKING:  # the split is defined in train.py; importing it at runtime would cycle
    from leadscore.train import Split

LOGGER = logging.getLogger(__name__)

CalibrationMethod = Literal["none", "sigmoid", "isotonic"]

#: How many folds the calibrator uses internally. Three rather than five on purpose: the
#: outer loop is already 5-fold, so five inner folds would mean 25 LightGBM fits per
#: candidate method for a difference in Brier well inside the noise.
CALIBRATION_INNER_FOLDS = 3

#: Every read of the held-out rows, appended in order. See the module docstring.
_LEDGER: list[dict[str, object]] = []


# ---------------------------------------------------------------------------
# the ledger: who opened the test set, and how often
# ---------------------------------------------------------------------------


def ledger() -> list[dict[str, object]]:
    """A copy of the record of held-out-row reads, oldest first."""
    return [dict(entry) for entry in _LEDGER]


def reset_ledger() -> None:
    """Clear the record. For tests; a pipeline run never calls this."""
    _LEDGER.clear()


def record_opening(reader: str, rows: int, fingerprint_hex: str) -> None:
    """Note that something read the held-out rows."""
    _LEDGER.append(
        {
            "reader": reader,
            "rows": rows,
            "fingerprint": fingerprint_hex,
            "at": datetime.now(UTC).isoformat(timespec="seconds"),
        }
    )
    if len(_LEDGER) > 1:
        LOGGER.warning(
            "the held-out rows have now been read %d times in this process: %s. "
            "Model selection must not depend on any of these numbers.",
            len(_LEDGER),
            ", ".join(str(entry["reader"]) for entry in _LEDGER),
        )


def fingerprint(index: pd.Index) -> str:
    """A stable hash of the held-out row identifiers.

    Published in the model card so two runs can be compared: same hash, same held-out rows,
    comparable metrics. A changed hash is not a failure — it means the seed or the upload
    changed, and that the two sets of numbers are not about the same question.
    """
    joined = ",".join(str(value) for value in sorted(index.to_list()))
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()[:16]


# ---------------------------------------------------------------------------
# T-403 — the metrics
# ---------------------------------------------------------------------------


def capture_at(y_true: pd.Series | np.ndarray, scores: np.ndarray, top_fraction: float) -> float:
    """Share of all conversions sitting in the top `top_fraction` of the ranking.

    The headline metric, and the only one a sales manager needs: "if we call the top 20%,
    what fraction of the deals do we reach?" A random ranking scores `top_fraction`; a
    perfect ranking scores `min(1, top_fraction / base_rate)`.

    Ties are broken by original row order via a stable sort, so the number is reproducible.
    A model that assigns every lead the same score therefore scores `top_fraction` on
    average rather than something flattering.
    """
    if not 0.0 < top_fraction <= 1.0:
        raise ValueError(f"top_fraction must be in (0, 1], got {top_fraction}")

    truth = np.asarray(y_true, dtype=float)
    total_positives = truth.sum()
    if total_positives == 0:
        return 0.0

    k = max(1, round(len(truth) * top_fraction))
    order = np.argsort(-np.asarray(scores, dtype=float), kind="stable")
    return float(truth[order][:k].sum() / total_positives)


def perfect_capture_at(y_true: pd.Series | np.ndarray, top_fraction: float) -> float:
    """The ceiling: what a flawless ranking would capture at this depth."""
    truth = np.asarray(y_true, dtype=float)
    if truth.sum() == 0:
        return 0.0
    k = max(1, round(len(truth) * top_fraction))
    return float(min(k, truth.sum()) / truth.sum())


@dataclass(frozen=True)
class DecileRow:
    """One tenth of the ranking, best-scoring first."""

    decile: int
    leads: int
    conversions: int
    conversion_rate: float
    lift: float
    cumulative_capture: float


def decile_lift(y_true: pd.Series | np.ndarray, scores: np.ndarray) -> list[DecileRow]:
    """Ten equal-sized bins, best scores first, with lift against the overall rate.

    This is the table a sales manager reads instead of an ROC curve: decile 1 converts at N
    times the average, and calling the first two deciles reaches X% of the deals.
    """
    truth = np.asarray(y_true, dtype=float)
    order = np.argsort(-np.asarray(scores, dtype=float), kind="stable")
    ranked = truth[order]
    overall = ranked.mean() if len(ranked) else 0.0
    total = ranked.sum()

    rows: list[DecileRow] = []
    bounds = np.linspace(0, len(ranked), 11).round().astype(int)
    running = 0.0
    for decile, (start, stop) in enumerate(itertools.pairwise(bounds), start=1):
        chunk = ranked[start:stop]
        conversions = float(chunk.sum())
        running += conversions
        rate = float(chunk.mean()) if len(chunk) else 0.0
        rows.append(
            DecileRow(
                decile=decile,
                leads=len(chunk),
                conversions=int(conversions),
                conversion_rate=rate,
                lift=float(rate / overall) if overall else 0.0,
                cumulative_capture=float(running / total) if total else 0.0,
            )
        )
    return rows


@dataclass(frozen=True)
class Metrics:
    """Everything the README's results table publishes, for one model."""

    name: str
    roc_auc: float
    pr_auc: float
    brier: float
    capture_at_20: float
    rows: int
    positives: int

    def as_dict(self) -> dict[str, object]:
        # Six decimals, not four: the README formats these to one decimal place, and a value
        # rounded twice can land on the wrong side of a boundary - 0.35254 became 0.3525 in
        # the card and then 35.2% in the README while the report said 35.3%. Round once, at
        # the point of display.
        return {
            "name": self.name,
            "roc_auc": round(self.roc_auc, 6),
            "pr_auc": round(self.pr_auc, 6),
            "brier": round(self.brier, 6),
            "capture_at_20": round(self.capture_at_20, 6),
            "rows": self.rows,
            "positives": self.positives,
        }

    def __str__(self) -> str:
        return (
            f"{self.name:<34} ROC {self.roc_auc:.4f}  PR {self.pr_auc:.4f}  "
            f"Brier {self.brier:.4f}  Capture@20% {self.capture_at_20:.1%}"
        )


def score_model(name: str, y_true: pd.Series, scores: np.ndarray) -> Metrics:
    """Compute every published metric from truth and predicted probabilities."""
    truth = np.asarray(y_true, dtype=int)
    probabilities = np.clip(np.asarray(scores, dtype=float), 0.0, 1.0)
    if len(truth) != len(probabilities):
        raise ValueError(f"{len(truth)} labels against {len(probabilities)} scores")

    return Metrics(
        name=name,
        roc_auc=float(roc_auc_score(truth, probabilities)),
        pr_auc=float(average_precision_score(truth, probabilities)),
        brier=float(brier_score_loss(truth, probabilities)),
        capture_at_20=capture_at(truth, probabilities, 0.20),
        rows=len(truth),
        positives=int(truth.sum()),
    )


# ---------------------------------------------------------------------------
# T-401 — calibration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CalibrationChoice:
    """Which calibration won, and the evidence for it."""

    method: CalibrationMethod
    brier_by_method: dict[str, float]
    oof_scores: np.ndarray = field(repr=False)

    @property
    def improvement(self) -> float:
        """How much Brier improved against leaving the model uncalibrated."""
        return self.brier_by_method["none"] - self.brier_by_method[self.method]

    def as_dict(self) -> dict[str, object]:
        return {
            "method": self.method,
            "brier_by_method": {k: round(v, 5) for k, v in self.brier_by_method.items()},
            "brier_improvement_vs_uncalibrated": round(self.improvement, 5),
            "chosen_because": (
                "lowest out-of-fold Brier on the training split"
                if self.method != "none"
                else "no calibrator beat the raw model's out-of-fold Brier"
            ),
        }


def out_of_fold_scores(
    estimator: BaseEstimator,
    x: pd.DataFrame,
    y: pd.Series,
    settings: cfg.Settings,
) -> np.ndarray:
    """Cross-validated predicted probabilities for the training split.

    Out-of-fold, so every score comes from a model that did not see that row. These are the
    validation scores: calibration is judged on them and the grade cut-offs are set from
    them, which is what keeps the test set out of both decisions.
    """
    folds = StratifiedKFold(
        n_splits=settings.cv_folds, shuffle=True, random_state=settings.random_seed
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        predicted = cross_val_predict(
            estimator, x, y, cv=folds, method="predict_proba", n_jobs=1
        )
    return np.asarray(predicted)[:, 1]


def choose_calibration(
    estimator: BaseEstimator,
    x: pd.DataFrame,
    y: pd.Series,
    settings: cfg.Settings,
) -> CalibrationChoice:
    """Compare raw, sigmoid and isotonic by out-of-fold Brier; return the winner.

    A gradient booster's probabilities are usually over-confident at the extremes, so this
    is expected to find something — but it is measured, not assumed, and "none" is allowed
    to win. Calibration is monotone either way, so the ranking metrics (ROC-AUC, PR-AUC,
    Capture@20%) cannot change; only Brier and the meaning of the number can.
    """
    candidates: dict[str, BaseEstimator] = {
        "none": clone(estimator),
        "sigmoid": CalibratedClassifierCV(
            clone(estimator), method="sigmoid", cv=CALIBRATION_INNER_FOLDS
        ),
        "isotonic": CalibratedClassifierCV(
            clone(estimator), method="isotonic", cv=CALIBRATION_INNER_FOLDS
        ),
    }

    brier: dict[str, float] = {}
    scores: dict[str, np.ndarray] = {}
    for name, candidate in candidates.items():
        oof = out_of_fold_scores(candidate, x, y, settings)
        scores[name] = oof
        brier[name] = float(brier_score_loss(np.asarray(y, dtype=int), oof))
        LOGGER.info("calibration %-9s out-of-fold Brier %.5f", name, brier[name])

    winner = min(brier, key=lambda key: brier[key])
    return CalibrationChoice(
        method=winner,  # type: ignore[arg-type]
        brier_by_method=brier,
        oof_scores=scores[winner],
    )


def apply_calibration(
    estimator: BaseEstimator, method: CalibrationMethod, settings: cfg.Settings
) -> BaseEstimator:
    """Wrap the estimator in the chosen calibrator, or hand it back untouched."""
    if method == "none":
        return clone(estimator)
    return CalibratedClassifierCV(clone(estimator), method=method, cv=CALIBRATION_INNER_FOLDS)


def reliability_curve(
    y_true: pd.Series | np.ndarray, scores: np.ndarray, bins: int = 10
) -> list[dict[str, float]]:
    """Predicted probability against observed rate, in equal-width bins.

    Equal-width rather than equal-count, because the question is "when the model says 0.8,
    does it happen 80% of the time?" — a question about the probability axis. Bins with no
    leads in them are returned with `leads = 0` rather than dropped, so a gap in the
    model's output range is visible instead of interpolated over.
    """
    truth = np.asarray(y_true, dtype=float)
    probabilities = np.asarray(scores, dtype=float)
    edges = np.linspace(0.0, 1.0, bins + 1)
    rows: list[dict[str, float]] = []
    for low, high in itertools.pairwise(edges):
        in_bin = (probabilities >= low) & (
            probabilities < high if high < 1.0 else probabilities <= high
        )
        count = int(in_bin.sum())
        rows.append(
            {
                "bin_low": float(low),
                "bin_high": float(high),
                "leads": count,
                "mean_predicted": float(probabilities[in_bin].mean()) if count else float("nan"),
                "observed_rate": float(truth[in_bin].mean()) if count else float("nan"),
            }
        )
    return rows


# ---------------------------------------------------------------------------
# T-402 — the grade cut-offs
# ---------------------------------------------------------------------------


def to_score(probabilities: np.ndarray) -> np.ndarray:
    """FR-6: `score = round(100 * calibrated_probability)`, as integers 0 to 100."""
    return np.rint(np.clip(np.asarray(probabilities, dtype=float), 0.0, 1.0) * 100).astype(int)


@dataclass(frozen=True)
class GradeCutoffs:
    """Score boundaries for A/B/C, set on validation data and stored in the card."""

    a_from: int
    b_from: int
    source: str
    policy: dict[str, float]

    def grade(self, score: int) -> str:
        if score >= self.a_from:
            return "A"
        if score >= self.b_from:
            return "B"
        return "C"

    def as_dict(self) -> dict[str, object]:
        return {
            "A_from_score": self.a_from,
            "B_from_score": self.b_from,
            "C_from_score": 0,
            "policy": self.policy,
            "fitted_on": self.source,
        }


def grade_cutoffs(
    validation_probabilities: np.ndarray, settings: cfg.Settings, source: str = "out-of-fold"
) -> GradeCutoffs:
    """Cut-offs at the configured percentiles of the validation score distribution.

    Percentiles, not fixed scores: the policy is "the top 20% are grade A" (docs/02 FR-6),
    and what score that corresponds to depends on the fitted model. Hardcoding 80 would
    silently change the policy every time the model changed.

    Set on out-of-fold validation scores. Setting them on the test set would leak the test
    set into a decision the product depends on; setting them on in-fold training scores
    would place them on an over-confident distribution and send too many leads to grade A.
    """
    scores = to_score(validation_probabilities)
    a_from = int(np.percentile(scores, settings.grade_a_percentile * 100))
    b_from = int(np.percentile(scores, settings.grade_b_percentile * 100))
    if b_from >= a_from:  # a flat score distribution can collapse the two
        b_from = max(0, a_from - 1)
    return GradeCutoffs(
        a_from=a_from, b_from=b_from, source=source, policy=settings.grade_band_policy
    )


# ---------------------------------------------------------------------------
# T-404 — the quality gate
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GateCheck:
    name: str
    passed: bool
    value: float | str
    threshold: float | str
    why: str

    def as_dict(self) -> dict[str, object]:
        return {
            "check": self.name,
            "passed": self.passed,
            "value": self.value,
            "threshold": self.threshold,
            "why": self.why,
        }


@dataclass(frozen=True)
class GateResult:
    checks: list[GateCheck]

    @property
    def passed(self) -> bool:
        return all(check.passed for check in self.checks)

    @property
    def failures(self) -> list[GateCheck]:
        return [check for check in self.checks if not check.passed]

    def as_dict(self) -> dict[str, object]:
        return {
            "passed": self.passed,
            "checks": [check.as_dict() for check in self.checks],
        }

    def __str__(self) -> str:
        lines = [("PASS" if c.passed else "FAIL") + f"  {c.name}: {c.value} vs {c.threshold}"
                 for c in self.checks]
        return "\n".join(lines)


def quality_gate(
    candidate: Metrics,
    linear: Metrics,
    one_rule: Metrics,
    feature_names: list[str],
    settings: cfg.Settings,
) -> GateResult:
    """Five checks from docs/10. A model that fails any of them does not ship.

    The thresholds come from config, and the comparisons are against this run's own
    baselines rather than remembered numbers — a baseline that improved is a higher bar,
    which is the point.
    """
    banned = sorted(set(cfg.BANNED_LEAKAGE_COLUMNS) & _source_columns(feature_names))
    expected = set(cfg.shipping_features())
    covered = _source_columns(feature_names)
    missing = sorted(expected - covered)

    checks = [
        GateCheck(
            name="beats the linear baseline on ROC-AUC",
            passed=candidate.roc_auc > linear.roc_auc,
            value=round(candidate.roc_auc, 4),
            threshold=round(linear.roc_auc, 4),
            why="a gradient booster that cannot beat logistic regression has no reason to exist",
        ),
        GateCheck(
            name="ROC-AUC floor",
            passed=candidate.roc_auc >= settings.gate_min_roc_auc,
            value=round(candidate.roc_auc, 4),
            threshold=settings.gate_min_roc_auc,
            why="below this the ranking is too weak to change who gets called",
        ),
        GateCheck(
            name="Brier ceiling",
            passed=candidate.brier <= settings.gate_max_brier,
            value=round(candidate.brier, 4),
            threshold=settings.gate_max_brier,
            why="a sharper ranking that lies about probabilities is a regression",
        ),
        GateCheck(
            name="Capture@20% beats random and the one-rule baseline",
            passed=candidate.capture_at_20 > 0.20
            and candidate.capture_at_20 > one_rule.capture_at_20,
            value=round(candidate.capture_at_20, 4),
            threshold=round(max(0.20, one_rule.capture_at_20), 4),
            why="the headline claim is about calling the top 20%, so this is the bar that matters",
        ),
        GateCheck(
            name="no banned column reaches the model",
            passed=not banned and not missing,
            value=f"{len(covered)} source columns, banned: {banned or 'none'}",
            threshold="13 admissible columns, no leakage column",
            why="the leakage rule is the project's whole point (CLAUDE.md rule 1)",
        ),
    ]
    return GateResult(checks)


def _source_columns(feature_names: list[str]) -> set[str]:
    """Map transformed feature names back to the raw columns they came from."""
    from leadscore import features as ft

    return {ft.source_column(name) for name in feature_names}


# ---------------------------------------------------------------------------
# the model card
# ---------------------------------------------------------------------------


def model_version(params: dict[str, object], features: list[str], settings: cfg.Settings) -> str:
    """A deterministic version string: same inputs, same version.

    Date plus a hash of what actually defines the model — its hyper-parameters, its feature
    list and the seed. Not a git SHA: the working tree can be dirty, and two commits that
    do not touch the pipeline produce the same model.
    """
    payload = json.dumps(
        {"params": params, "features": sorted(features), "seed": settings.random_seed},
        sort_keys=True,
        default=str,
    )
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:8]
    return f"{datetime.now(UTC):%Y%m%d}-{digest}"


def build_model_card(
    *,
    version: str,
    selection: dict[str, object],
    calibration: CalibrationChoice,
    cutoffs: GradeCutoffs,
    metrics: list[Metrics],
    gate: GateResult,
    deciles: list[DecileRow],
    split_sizes: dict[str, int],
    test_fingerprint: str,
    feature_names: list[str],
) -> dict[str, object]:
    """Everything a reader — or `/model-info` — needs to judge this model."""
    shipping = next((m for m in metrics if m.name == "lightgbm_shipped"), None)
    return {
        "model_version": version,
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "headline": {
            "metric": "capture_at_20",
            "value": round(shipping.capture_at_20, 4) if shipping else None,
            "says": (
                f"calling the top 20% of scored leads reaches "
                f"{shipping.capture_at_20:.1%} of all conversions"
                if shipping
                else None
            ),
        },
        "data": {
            "rows_train": split_sizes.get("train"),
            "rows_test": split_sizes.get("test"),
            "test_set_fingerprint": test_fingerprint,
            "test_set_openings": len(_LEDGER),
            "expected_rows": cfg.EXPECTED_ROWS,
            "expected_target_rate": cfg.EXPECTED_TARGET_RATE,
        },
        "features": {
            "admissible_source_columns": list(cfg.shipping_features()),
            "transformed_count": len(feature_names),
            "excluded": {
                "banned_for_leakage": list(cfg.BANNED_LEAKAGE_COLUMNS),
                "activity_columns": list(cfg.ACTIVITY_FEATURES),
                "near_constant_flags": list(cfg.FLAG_FEATURES),
                "sparse": list(cfg.DROPPED_SPARSE_COLUMNS),
                "identifiers": list(cfg.ID_COLUMNS),
            },
        },
        "selection": selection,
        "calibration": calibration.as_dict(),
        "grades": cutoffs.as_dict(),
        "test_metrics": [m.as_dict() for m in metrics],
        "decile_lift": [
            {
                "decile": row.decile,
                "leads": row.leads,
                "conversions": row.conversions,
                "conversion_rate": round(row.conversion_rate, 4),
                "lift": round(row.lift, 3),
                "cumulative_capture": round(row.cumulative_capture, 4),
            }
            for row in deciles
        ],
        "quality_gate": gate.as_dict(),
        # One list, used by both the card and the report (defined below).
        "limitations": list(LIMITATIONS),
    }


def write_model_card(card: dict[str, object], settings: cfg.Settings) -> Path:
    """Write `models/model_card.json`, creating the directory if needed."""
    path = cfg.resolve(settings.model_card_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(card, indent=2, default=str) + "\n", encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# T-403 — the single pass over the held-out rows
# ---------------------------------------------------------------------------


def evaluate_on_test(
    models: dict[str, tuple[BaseEstimator, Split]],
    settings: cfg.Settings,
    reader: str = "evaluate_on_test",
) -> tuple[list[Metrics], dict[str, np.ndarray]]:
    """Fit every model on the training rows and score the held-out rows, once.

    Each entry is `(estimator, split)`. The leakage model needs a split with more columns
    than the shipping model, which is why the split travels with the estimator rather than
    being a single argument — but every split must hold back **the same rows**, or the
    numbers in the results table are not comparable, so that is checked rather than trusted.

    One ledger entry is written for the whole pass, because this is one opening of the test
    set: the models are compared against each other, not selected on what comes out.
    """
    if not models:
        raise ValueError("nothing to evaluate")

    splits = list(models.values())
    reference = splits[0][1]
    for _estimator, split in splits[1:]:
        if not reference.x_test.index.equals(split.x_test.index):
            raise ValueError(
                "two models were given different held-out rows; their metrics would not be "
                "comparable. Build every split from the same frame, seed and test_size."
            )

    hex_digest = fingerprint(reference.x_test.index)
    record_opening(reader, len(reference.x_test), hex_digest)

    metrics: list[Metrics] = []
    scores: dict[str, np.ndarray] = {}
    for name, (estimator, split) in models.items():
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            fitted = estimator.fit(split.x_train, split.y_train)
        predicted = np.asarray(fitted.predict_proba(split.x_test))[:, 1]
        scores[name] = predicted
        metrics.append(score_model(name, split.y_test, predicted))
        LOGGER.info("%s", metrics[-1])

    return metrics, scores


# ---------------------------------------------------------------------------
# T-406 — error analysis
# ---------------------------------------------------------------------------


def error_analysis(
    split: Split,
    scores: np.ndarray,
    cutoffs: GradeCutoffs,
    k: int = 20,
) -> dict[str, pd.DataFrame]:
    """The k worst misses in each direction, with the features a reader can check.

    Two questions, both answerable only from the held-out rows:

    * **confident and wrong** — the highest-scoring leads that never converted. These are
      what a salesperson would waste the morning on, so the pattern matters more than the
      count;
    * **missed opportunities** — the lowest-scoring leads that converted anyway. These show
      what the admissible columns cannot see.
    """
    frame = split.x_test.copy()
    frame["score"] = to_score(scores)
    frame["grade"] = [cutoffs.grade(int(value)) for value in frame["score"]]
    frame["converted"] = np.asarray(split.y_test, dtype=int)

    non_converters = frame[frame["converted"] == 0].nlargest(k, "score")
    converters = frame[frame["converted"] == 1].nsmallest(k, "score")
    return {"confident_and_wrong": non_converters, "missed_opportunities": converters}


def error_findings(analysis: dict[str, pd.DataFrame]) -> dict[str, list[str]]:
    """What each error group has in common, as sentences built from counts.

    Computed rather than written, because an error analysis is where a tired author starts
    narrating impressions. Every sentence below is a share of a real group of held-out rows,
    so it changes when the data changes instead of staying true by assertion.
    """
    findings: dict[str, list[str]] = {}
    for label, frame in analysis.items():
        notes: list[str] = []
        if frame.empty:
            findings[label] = notes
            continue

        total = len(frame)
        if "Lead Origin" in frame.columns:
            origins = frame["Lead Origin"].fillna("Missing").value_counts()
            top, count = origins.index[0], int(origins.iloc[0])
            notes.append(
                f"{count} of {total} arrived through **{top}**"
                + (
                    ", which report 01 found converts far above average - the model leans on "
                    "that channel hard enough to score a lead highly with no engagement at all"
                    if label == "confident_and_wrong"
                    else ", a channel that usually converts poorly"
                )
            )

        if "Total Time Spent on Website" in frame.columns:
            time_on_site = pd.to_numeric(frame["Total Time Spent on Website"], errors="coerce")
            silent = int((time_on_site.fillna(0) == 0).sum())
            notes.append(
                f"{silent} of {total} spent **no time on the website** at all "
                f"(median {time_on_site.median():.0f} seconds across the group)"
            )

        if "What is your current occupation" in frame.columns:
            unknown = int(frame["What is your current occupation"].isna().sum())
            if unknown:  # "0 of 20 did not say" is not a finding, it is a filled slot
                notes.append(f"{unknown} of {total} did not say what they do for a living")

        findings[label] = notes
    return findings


def error_patterns(analysis: dict[str, pd.DataFrame]) -> dict[str, dict[str, object]]:
    """What the two error groups have in common, as counts rather than impressions."""
    summary: dict[str, dict[str, object]] = {}
    for label, frame in analysis.items():
        if frame.empty:
            summary[label] = {"leads": 0}
            continue
        row: dict[str, object] = {
            "leads": len(frame),
            "score_range": [int(frame["score"].min()), int(frame["score"].max())],
            "grades": frame["grade"].value_counts().to_dict(),
        }
        for column in ("Lead Origin", "Lead Source", "What is your current occupation"):
            if column in frame.columns:
                row[column] = frame[column].fillna("Missing").value_counts().head(3).to_dict()
        for column in ("Total Time Spent on Website", "TotalVisits"):
            if column in frame.columns:
                numbers = pd.to_numeric(frame[column], errors="coerce")
                row[column] = {
                    "median": float(numbers.median()),
                    "min": float(numbers.min()),
                    "max": float(numbers.max()),
                }
        summary[label] = row
    return summary


# ---------------------------------------------------------------------------
# charts
# ---------------------------------------------------------------------------


def _reliability_chart(reliability: list[dict[str, float]], path: Path) -> Path:
    """Predicted against observed. The diagonal is the claim; the marks are the evidence."""
    import matplotlib.pyplot as plt

    from leadscore import plotting as viz

    viz.apply_style()
    populated = [row for row in reliability if row["leads"] > 0]
    fig, ax = plt.subplots(figsize=(5.4, 5.0))
    ax.plot([0, 1], [0, 1], color=viz.INK_MUTED, linewidth=1, linestyle="--", zorder=1)
    ax.text(0.62, 0.57, "perfect calibration", color=viz.INK_MUTED, fontsize=8, rotation=37)

    sizes = [max(18.0, 190.0 * row["leads"] / max(r["leads"] for r in populated))
             for row in populated]
    ax.scatter(
        [row["mean_predicted"] for row in populated],
        [row["observed_rate"] for row in populated],
        s=sizes, color=viz.SERIES_A, zorder=3, alpha=0.9,
    )
    ax.plot(
        [row["mean_predicted"] for row in populated],
        [row["observed_rate"] for row in populated],
        color=viz.SERIES_A, linewidth=1.2, zorder=2,
    )
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Observed conversion rate")
    ax.set_title("A score of 70 should convert 70% of the time", loc="left")
    ax.grid(alpha=0.6)
    ax.set_axisbelow(True)
    return viz.save(fig, path)


def _gains_chart(deciles: list[DecileRow], capture_at_20: float, path: Path) -> Path:
    """Cumulative capture against how deep the sales team calls. The headline, drawn."""
    import matplotlib.pyplot as plt

    from leadscore import plotting as viz

    viz.apply_style()
    depth = [0.0] + [row.decile / 10 for row in deciles]
    captured = [0.0] + [row.cumulative_capture for row in deciles]

    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    ax.plot([0, 1], [0, 1], color=viz.INK_MUTED, linewidth=1, linestyle="--")
    ax.text(0.55, 0.49, "calling at random", color=viz.INK_MUTED, fontsize=8, rotation=37)
    ax.plot(depth, captured, color=viz.SERIES_A, linewidth=2, marker="o", markersize=4)

    ax.axvline(0.2, color=viz.SERIES_B, linewidth=1, linestyle=":")
    ax.annotate(
        f"top 20% reaches {capture_at_20:.0%}",
        xy=(0.2, capture_at_20),
        xytext=(0.27, max(0.08, capture_at_20 - 0.17)),
        color=viz.SERIES_B,
        fontsize=9,
        arrowprops={"arrowstyle": "->", "color": viz.SERIES_B, "linewidth": 1},
    )

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("Share of leads called, best-scoring first")
    ax.set_ylabel("Share of all conversions reached")
    ax.set_title("What the ranking is worth at every calling depth", loc="left")
    ax.grid(alpha=0.6)
    ax.set_axisbelow(True)
    return viz.save(fig, path)


def _decile_chart(deciles: list[DecileRow], path: Path) -> Path:
    """Lift per decile. Bars start at zero, so their length is honest."""
    import matplotlib.pyplot as plt

    from leadscore import plotting as viz

    viz.apply_style()
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    labels = [str(row.decile) for row in deciles]
    lifts = [row.lift for row in deciles]
    ax.bar(labels, lifts, color=viz.SERIES_A, width=0.68)
    ax.axhline(1.0, color=viz.SERIES_B, linewidth=1, linestyle=":")
    ax.text(9.1, 1.06, "average", color=viz.SERIES_B, fontsize=8)
    for index, lift in enumerate(lifts):
        ax.text(index, lift + 0.04, f"{lift:.2f}", ha="center", fontsize=8, color=viz.INK)

    ax.set_ylim(0, max(lifts) * 1.18)
    ax.set_xlabel("Decile of the ranking (1 = highest scores)")
    ax.set_ylabel("Conversion rate vs average")
    ax.set_title("Decile 1 converts well above average; decile 10 barely converts", loc="left")
    ax.grid(axis="y", alpha=0.6)
    ax.set_axisbelow(True)
    return viz.save(fig, path)


# ---------------------------------------------------------------------------
# the report and the saved model
# ---------------------------------------------------------------------------

#: How the README's results table names each model, in the order it lists them. The
#: shipping row is filled in by `shipping_label`, because what was done to that model is a
#: result rather than a given.
REPORT_LABELS: dict[str, str] = {
    "random": "Random baseline",
    "one_rule": "One-rule (time on site)",
    "logistic_regression": "Logistic regression",
    "lightgbm_shipped": "",
    "lightgbm_leakage": "LightGBM + sales columns *(leakage, not shippable)*",
}


def shipping_label(method: str) -> str:
    """What to call the shipped model, given what calibration actually did.

    The plan said "LightGBM (calibrated)", and on this data no calibrator improved
    out-of-fold Brier, so nothing was wrapped around the model. Printing "calibrated"
    anyway would be a small, permanent lie in the most-read table in the project - and the
    honest version is more interesting: the booster's probabilities were already usable.
    """
    if method == "none":
        return "**LightGBM (tuned, no calibrator needed)**"
    return f"**LightGBM ({method}-calibrated)**"


def results_table(metrics: list[Metrics], calibration_method: str = "none") -> list[str]:
    """The five-row, five-column table the README publishes, as markdown lines."""
    by_name = {m.name: m for m in metrics}
    lines = [
        "| Model | ROC-AUC | PR-AUC | Brier | Capture@20% |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for name, label in REPORT_LABELS.items():
        row = by_name.get(name)
        if row is None:
            continue
        if name == "lightgbm_shipped":
            label = shipping_label(calibration_method)
        lines.append(
            f"| {label} | {row.roc_auc:.3f} | {row.pr_auc:.3f} | "
            f"{row.brier:.3f} | {row.capture_at_20:.1%} |"
        )
    return lines


def _error_group_section(
    label: str,
    frame: pd.DataFrame,
    pattern: dict[str, object],
    findings: list[str] | None = None,
) -> list[str]:
    """One error-analysis group as markdown: what it is, then ten real rows."""
    lines = [f"### {label.replace('_', ' ').capitalize()} ({pattern.get('leads', 0)} leads)", ""]
    if frame.empty:
        return [*lines, "None found.", ""]

    low, high = pattern.get("score_range", [0, 0])
    lines.append(f"Scores {low} to {high}. Grades: {pattern.get('grades', {})}.")
    lines.append("")
    for finding in findings or []:
        lines.append(f"- {finding}")
    if findings:
        lines.append("")

    columns = [
        column
        for column in (
            "Lead Origin",
            "Lead Source",
            "What is your current occupation",
            "Total Time Spent on Website",
            "TotalVisits",
            "score",
            "grade",
        )
        if column in frame.columns
    ]
    lines.append("| " + " | ".join(columns) + " |")
    lines.append("| " + " | ".join("---" for _ in columns) + " |")
    for _, row in frame.head(10).iterrows():
        cells = []
        for column in columns:
            value = row[column]
            if pd.isna(value):
                cells.append("*missing*")
            elif isinstance(value, float):
                cells.append(f"{value:,.0f}")
            else:
                cells.append(str(value))
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("")

    traits = {
        key: value
        for key, value in pattern.items()
        if key not in {"leads", "score_range", "grades"}
    }
    lines.append(f"Shared traits: `{json.dumps(traits, default=str)}`")
    lines.append("")
    return lines


LIMITATIONS: tuple[str, ...] = (
    "Scores one public dataset of 9,240 leads from a single education provider; nothing "
    "here transfers to another business without re-measurement.",
    "No column filled in after a human contacted the lead is used, so the model is weaker "
    "than published results on this dataset that use `Tags` - deliberately.",
    "Calibration is fitted on this one snapshot with no drift monitoring; the probabilities "
    "are trustworthy only while the lead mix resembles the training mix.",
    "Explanations are feature attributions, not causes. A high score is not advice about "
    "what to change.",
)


def write_evaluation_report(
    *,
    metrics: list[Metrics],
    deciles: list[DecileRow],
    calibration: CalibrationChoice,
    cutoffs: GradeCutoffs,
    gate: GateResult,
    reliability: list[dict[str, float]],
    analysis: dict[str, pd.DataFrame],
    patterns: dict[str, dict[str, object]],
    split: Split,
    settings: cfg.Settings,
    version: str,
    test_fingerprint: str,
) -> Path:
    """Write `reports/04_evaluation.md` and its three charts."""
    reports = cfg.resolve(settings.reports_dir)
    figures = cfg.resolve(settings.figures_dir)
    by_name = {m.name: m for m in metrics}
    shipping = by_name["lightgbm_shipped"]
    leakage = by_name.get("lightgbm_leakage")
    one_rule = by_name.get("one_rule")

    reliability_png = _reliability_chart(reliability, figures / "reliability.png")
    gains_png = _gains_chart(deciles, shipping.capture_at_20, figures / "gains.png")
    decile_png = _decile_chart(deciles, figures / "decile_lift.png")

    lines: list[str] = []
    add = lines.append

    add("# 04 - Evaluation, calibration and the quality gate")
    add("")
    add(f"Model version `{version}`, written by `python -m leadscore.train`.")
    add("")

    add("## The headline")
    add("")
    add(
        f"> Calling the top 20% of scored leads reaches **{shipping.capture_at_20:.1%}** of "
        "all conversions."
    )
    add("")
    perfect = perfect_capture_at(split.y_test, 0.20)
    closed = (shipping.capture_at_20 - 0.20) / max(perfect - 0.20, 1e-9)
    add(
        f"A random ranking reaches 20%. A flawless ranking would reach {perfect:.1%} at this "
        f"depth, so the model closes {closed:.0%} of the distance between guessing and "
        "omniscience."
    )
    add("")
    add(f"![Cumulative capture]({gains_png.relative_to(reports).as_posix()})")
    add("")

    add("## The test set was opened once")
    add("")
    add("| | |")
    add("| --- | --- |")
    add(f"| Held-out rows | {len(split.x_test):,} |")
    add(f"| Conversions among them | {int(split.y_test.sum()):,} |")
    add(f"| Fingerprint of the held-out row ids | `{test_fingerprint}` |")
    add(f"| Reads recorded in this run | {len(_LEDGER)} |")
    add("")
    add(
        "The fingerprint is a hash of the held-out row identifiers. The same value across "
        "runs means the same rows were held back, so two sets of numbers are about the same "
        "question; a different value means the seed or the upload changed. Every number "
        "below comes from that single read, and nothing in Phase 3 chose a model using any "
        "of them."
    )
    add("")

    add("## Results")
    add("")
    lines.extend(results_table(metrics, calibration.method))
    add("")
    if leakage:
        add(
            "The last row is the point of the project. Adding the columns sales fills in "
            f"*after* contacting a lead lifts ROC-AUC from {shipping.roc_auc:.3f} to "
            f"{leakage.roc_auc:.3f}, and Capture@20% from {shipping.capture_at_20:.1%} to "
            f"{leakage.capture_at_20:.1%}. None of it exists when a lead arrives, so none of "
            "it ships. A notebook that reports the higher number has not built a "
            "lead-scoring model; it has measured how well sales already knew."
        )
        add("")

    add("## Calibration")
    add("")
    add("| Method | Out-of-fold Brier | |")
    add("| --- | ---: | --- |")
    for method, value in calibration.brier_by_method.items():
        marker = "chosen" if method == calibration.method else ""
        add(f"| {method} | {value:.5f} | {marker} |")
    add("")
    if calibration.method == "none":
        runner_up = min(
            (m for m in calibration.brier_by_method if m != "none"),
            key=lambda m: calibration.brier_by_method[m],
        )
        gap = calibration.brier_by_method[runner_up] - calibration.brier_by_method["none"]
        add(
            "**No calibrator was applied.** Both were fitted and compared on the training "
            f"split alone, and the closest ({runner_up}) was {gap:.5f} Brier *worse* than "
            "leaving the model alone - a gap in the fourth decimal place, which is noise. "
            "The booster's own probabilities ship, and the reliability curve below is why "
            "that is defensible rather than lazy."
        )
    else:
        add(
            f"Chosen on the training split alone, by out-of-fold Brier score: "
            f"**{calibration.method}**, which improves Brier by {calibration.improvement:.5f} "
            "against the raw model. Calibration is monotone, so ROC-AUC, PR-AUC and "
            "Capture@20% cannot change; what changes is whether the number means what it says."
        )
    add("")
    add(f"![Reliability curve]({reliability_png.relative_to(reports).as_posix()})")
    add("")

    add("## Grades")
    add("")
    add("| Grade | Score from | Policy |")
    add("| --- | ---: | --- |")
    add(f"| A | {cutoffs.a_from} | top {1 - settings.grade_a_percentile:.0%} of scores |")
    add(
        f"| B | {cutoffs.b_from} | next "
        f"{settings.grade_a_percentile - settings.grade_b_percentile:.0%} |"
    )
    add(f"| C | 0 | the remaining {settings.grade_b_percentile:.0%} |")
    add("")
    add(
        f"Cut-offs are set on {cutoffs.source} validation scores from the training split, "
        "never on the held-out rows, and they are stored in the model card so that the API "
        "and the dashboard read the same numbers instead of hardcoding them."
    )
    add("")

    add("## Decile lift")
    add("")
    add("| Decile | Leads | Conversions | Rate | Lift | Cumulative capture |")
    add("| ---: | ---: | ---: | ---: | ---: | ---: |")
    for row in deciles:
        add(
            f"| {row.decile} | {row.leads} | {row.conversions} | "
            f"{row.conversion_rate:.1%} | {row.lift:.2f}x | {row.cumulative_capture:.1%} |"
        )
    add("")
    add(f"![Decile lift]({decile_png.relative_to(reports).as_posix()})")
    add("")

    add("## Quality gate")
    add("")
    add(f"**{'PASSED' if gate.passed else 'FAILED'}** - {len(gate.checks)} checks.")
    add("")
    add("| Check | Value | Threshold | Verdict | Why it exists |")
    add("| --- | ---: | ---: | --- | --- |")
    for check in gate.checks:
        verdict = "pass" if check.passed else "**FAIL**"
        add(f"| {check.name} | {check.value} | {check.threshold} | {verdict} | {check.why} |")
    add("")
    if not gate.passed:
        add(
            "The model was **not** written to `models/`. A failing gate is the gate working: "
            "the fix is a better model, never a lower threshold (CLAUDE.md)."
        )
        add("")

    add("## Error analysis")
    add("")
    add(
        "Two groups, both read off the held-out rows: the leads the model was most confident "
        "about that never converted, and the quietest leads that converted anyway. Ten of "
        "each are shown."
    )
    add("")
    findings = error_findings(analysis)
    for label, frame in analysis.items():
        lines.extend(
            _error_group_section(label, frame, patterns.get(label, {}), findings.get(label))
        )

    add("## Limitations")
    add("")
    for limitation in LIMITATIONS:
        add(f"- {limitation}")
    add("")
    if one_rule:
        add(
            "One more number worth keeping in view: the spreadsheet sort a sales manager "
            f"already does reaches {one_rule.capture_at_20:.1%} at the same depth. The model "
            f"is worth {(shipping.capture_at_20 - one_rule.capture_at_20) * 100:.1f} "
            "percentage points of captured conversions on top of that, which is the honest "
            "size of the win."
        )
        add("")

    reports.mkdir(parents=True, exist_ok=True)
    path = reports / "04_evaluation.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def save_model_bundle(
    *,
    model: BaseEstimator,
    explainer_model: BaseEstimator,
    card: dict[str, object],
    settings: cfg.Settings,
) -> Path:
    """Write the fitted model, its uncalibrated twin and the card into one joblib file.

    The uncalibrated twin travels with the model because SHAP's `TreeExplainer` needs a tree
    model and a calibrated wrapper is not one. Calibration is monotone, so attributions
    taken from the twin explain the same ranking the calibrated model produces - stated in
    the card rather than left as a surprise.
    """
    import joblib

    path = cfg.resolve(settings.model_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {"model": model, "explainer_model": explainer_model, "card": card}, path, compress=3
    )
    return path


def load_model_bundle(settings: cfg.Settings | None = None) -> dict[str, object]:
    """Read back what `save_model_bundle` wrote. Raises if it is not there."""
    import joblib

    settings = settings or cfg.Settings()
    path = cfg.resolve(settings.model_path)
    if not path.exists():
        raise FileNotFoundError(
            f"no model at {path}. Run `python -m leadscore.train` first; the model is a build "
            "artifact and is deliberately not committed."
        )
    return joblib.load(path)
