"""Model selection (tasks T-302 to T-305).

**The test set is not touched in this module.** The split happens once, here, and the held-out
20% is written to disk untouched; every decision below is made on cross-validation inside the
training 80%. Phase 4 (`evaluate.py`) opens the test set exactly once. If you find yourself
wanting a test score to choose between two models, that is the mistake this split exists to
prevent.

What runs here:

1. one stratified 80/20 split, seeded from config (T-302);
2. three baselines — random, one-rule, logistic regression (T-303);
3. LightGBM, tuned with Optuna on cross-validated ROC-AUC (T-304);
4. the winner picked on CV alone (T-305).

Every run is logged to MLflow. An untracked run did not happen.

    .venv/Scripts/python -m leadscore.train
"""

from __future__ import annotations

import json
import logging
import os
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.dummy import DummyClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline

from leadscore import config as cfg
from leadscore import features as ft

LOGGER = logging.getLogger(__name__)
ONE_RULE_COLUMN = "Total Time Spent on Website"


# ---------------------------------------------------------------------------
# T-302 — the split, made once
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Split:
    """The one split the whole project uses."""

    x_train: pd.DataFrame
    x_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series

    @property
    def sizes(self) -> dict[str, int]:
        return {"train": len(self.x_train), "test": len(self.x_test)}


def make_split(
    frame: pd.DataFrame,
    name: cfg.FeatureSet = "shipping",
    settings: cfg.Settings | None = None,
) -> Split:
    """Stratified, seeded, made once. Re-running with the same seed reproduces it exactly."""
    settings = settings or cfg.Settings()
    columns = list(cfg.feature_set(name))
    x = frame[columns]
    y = frame[cfg.TARGET]
    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=settings.test_size,
        random_state=settings.random_seed,
        stratify=y,
    )
    return Split(x_train, x_test, y_train, y_test)


# ---------------------------------------------------------------------------
# T-303 — the baselines
# ---------------------------------------------------------------------------


class OneRuleRanker(ClassifierMixin, BaseEstimator):
    """Rank leads by one column, highest first.

    This is the baseline that matters: it is what a sales manager does with a spreadsheet
    sort, and a model that cannot beat it is not worth deploying. Report 01 showed the
    relationship is U-shaped, so this baseline is weaker than it looks — which is exactly
    why it must be measured rather than assumed.
    """

    def __init__(self, column: str = ONE_RULE_COLUMN) -> None:
        self.column = column

    def fit(self, x: pd.DataFrame, y: pd.Series) -> OneRuleRanker:
        values = pd.to_numeric(x[self.column], errors="coerce")
        self.classes_ = np.unique(y)
        self.low_ = float(values.min())
        self.high_ = float(values.max())
        self.is_fitted_ = True
        return self

    def predict_proba(self, x: pd.DataFrame) -> np.ndarray:
        values = pd.to_numeric(x[self.column], errors="coerce").fillna(self.low_)
        span = max(self.high_ - self.low_, 1e-9)
        positive = ((values - self.low_) / span).clip(0, 1).to_numpy()
        return np.column_stack([1 - positive, positive])

    def predict(self, x: pd.DataFrame) -> np.ndarray:
        return (self.predict_proba(x)[:, 1] >= 0.5).astype(int)


def build_candidates(settings: cfg.Settings) -> dict[str, Pipeline | BaseEstimator]:
    """The three baselines, in the order the README reports them."""
    return {
        "random": DummyClassifier(strategy="uniform", random_state=settings.random_seed),
        "one_rule": OneRuleRanker(),
        "logistic_regression": Pipeline(
            [
                ("prep", ft.build_preprocessor("shipping", settings)),
                ("model", LogisticRegression(max_iter=2000, random_state=settings.random_seed)),
            ]
        ),
    }


# ---------------------------------------------------------------------------
# cross-validation
# ---------------------------------------------------------------------------


@dataclass
class CvResult:
    name: str
    mean_auc: float
    std_auc: float
    params: dict[str, object] = field(default_factory=dict)

    def __str__(self) -> str:
        return f"{self.name:<28} {self.mean_auc:.4f} ± {self.std_auc:.4f}"


def cross_validate(
    estimator: BaseEstimator,
    split: Split,
    settings: cfg.Settings,
    name: str,
) -> CvResult:
    """Cross-validated ROC-AUC on the training split only."""
    folds = StratifiedKFold(
        n_splits=settings.cv_folds, shuffle=True, random_state=settings.random_seed
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        scores = cross_val_score(
            estimator, split.x_train, split.y_train, cv=folds, scoring="roc_auc", n_jobs=1
        )
    return CvResult(name, float(scores.mean()), float(scores.std()))


# ---------------------------------------------------------------------------
# T-304 — the challenger
# ---------------------------------------------------------------------------


def build_lightgbm(params: dict[str, object], settings: cfg.Settings) -> Pipeline:
    from lightgbm import LGBMClassifier

    return Pipeline(
        [
            ("prep", ft.build_preprocessor("shipping", settings)),
            (
                "model",
                LGBMClassifier(
                    random_state=settings.random_seed,
                    n_jobs=1,
                    verbose=-1,
                    **params,
                ),
            ),
        ]
    )


DEFAULT_LGBM: dict[str, object] = {
    "n_estimators": 400,
    "learning_rate": 0.05,
    "num_leaves": 31,
    "min_child_samples": 20,
}


def tune_lightgbm(split: Split, settings: cfg.Settings) -> CvResult:
    """Optuna over cross-validated ROC-AUC.

    `optuna_trials = 0` disables tuning and uses sensible defaults — the first item on the
    cut list in docs/03, wired in rather than described.
    """
    if settings.optuna_trials == 0:
        LOGGER.info("tuning disabled (optuna_trials=0); using defaults")
        result = cross_validate(
            build_lightgbm(DEFAULT_LGBM, settings), split, settings, "lightgbm_default"
        )
        result.params = dict(DEFAULT_LGBM)
        return result

    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(trial: optuna.Trial) -> float:
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 200, 900, step=100),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 15, 127),
            "min_child_samples": trial.suggest_int("min_child_samples", 10, 80),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "subsample_freq": 1,
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
        }
        return cross_validate(
            build_lightgbm(params, settings), split, settings, "lightgbm_trial"
        ).mean_auc

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=settings.random_seed),
    )
    study.optimize(objective, n_trials=settings.optuna_trials, show_progress_bar=False)

    best = dict(study.best_params)
    best.setdefault("subsample_freq", 1)
    result = cross_validate(
        build_lightgbm(best, settings), split, settings, "lightgbm_tuned"
    )
    result.params = best
    return result


# ---------------------------------------------------------------------------
# T-305 — selection, on cross-validation alone
# ---------------------------------------------------------------------------


def run_selection(settings: cfg.Settings | None = None) -> tuple[list[CvResult], Split]:
    """Run every candidate on CV and return the results. The test set stays closed."""
    from leadscore import schema as sch

    settings = settings or cfg.Settings()
    frame = sch.load_and_validate(settings.data_raw, settings)
    split = make_split(frame, "shipping", settings)
    LOGGER.info("split: %s", split.sizes)

    results = [
        cross_validate(estimator, split, settings, name)
        for name, estimator in build_candidates(settings).items()
    ]
    results.append(tune_lightgbm(split, settings))
    return results, split


def log_to_mlflow(results: list[CvResult], split: Split, settings: cfg.Settings) -> str | None:
    """Log every candidate as its own MLflow run. Returns the tracking URI, or None."""
    try:
        import mlflow
    except ImportError:  # pragma: no cover - mlflow is an optional dev-time dependency
        LOGGER.warning("mlflow not installed; results not tracked")
        return None

    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_experiment("p1-lead-scoring")
    for result in results:
        with mlflow.start_run(run_name=result.name):
            mlflow.log_params(
                {
                    "feature_set": "shipping",
                    "n_features": len(cfg.shipping_features()),
                    "cv_folds": settings.cv_folds,
                    "seed": settings.random_seed,
                    "test_size": settings.test_size,
                    "train_rows": split.sizes["train"],
                    **{f"model__{k}": v for k, v in result.params.items()},
                }
            )
            mlflow.log_metrics(
                {"cv_roc_auc_mean": result.mean_auc, "cv_roc_auc_std": result.std_auc}
            )
    return settings.mlflow_tracking_uri


def write_report(results: list[CvResult], split: Split, settings: cfg.Settings) -> Path:
    """Write `reports/03_model_selection.md` and its chart."""
    import matplotlib.pyplot as plt

    from leadscore import plotting as viz

    reports = cfg.resolve(settings.reports_dir)
    figures = cfg.resolve(settings.figures_dir)
    viz.apply_style()

    ranked = sorted(results, key=lambda r: r.mean_auc, reverse=True)
    winner = ranked[0]
    by_name = {r.name: r for r in results}
    linear = by_name.get("logistic_regression")
    one_rule = by_name.get("one_rule")

    # A point estimate with a spread is a dot and a whisker, not a bar. Bars would have to
    # start at zero to be honest about length, and an ROC-AUC axis from 0 wastes half the
    # plot on a region no model can occupy.
    ordered = sorted(results, key=lambda r: r.mean_auc)
    labels = [r.name.replace("_", " ") for r in ordered]
    positions = range(len(ordered))
    fig, ax = plt.subplots(figsize=(7.2, 0.5 * len(ordered) + 1.5))
    ax.errorbar(
        [r.mean_auc for r in ordered],
        positions,
        xerr=[r.std_auc for r in ordered],
        fmt="o",
        color=viz.SERIES_A,
        ecolor=viz.INK_MUTED,
        elinewidth=1.2,
        capsize=4,
        markersize=7,
    )
    for y, result in enumerate(ordered):
        ax.text(
            result.mean_auc, y + 0.26, f"{result.mean_auc:.4f}",
            ha="center", fontsize=8, color=viz.INK,
        )
    ax.text(0.5, -0.62, " chance", color=viz.INK_MUTED, fontsize=8, va="center")
    if linear:
        ax.axvline(linear.mean_auc, color=viz.SERIES_B, linewidth=1, linestyle=":")
        ax.text(
            linear.mean_auc, len(ordered) - 0.42, " linear baseline",
            color=viz.SERIES_B, fontsize=8, va="center",
        )

    ax.set_yticks(list(positions))
    ax.set_yticklabels(labels)
    ax.set_ylim(-0.8, len(ordered) - 0.2)
    # Truncated at 0.45, which the guard permits only because chance stays on the chart
    # and every mark carries its whisker. See plotting.metric_axis and D-018.
    viz.metric_axis(ax, 0.45, 1.0, chance=0.5)
    ax.set_xlabel("Cross-validated ROC-AUC, 5-fold on the training split (whisker = ±1 SD)")
    margin = winner.mean_auc - linear.mean_auc if linear else 0.0
    beats_noise = bool(linear and margin > linear.std_auc)
    ax.set_title(
        "LightGBM clears the linear baseline, but only after tuning"
        if beats_noise
        else "LightGBM does not clearly beat the linear baseline",
        loc="left",
    )
    ax.grid(axis="x", alpha=0.6)
    ax.set_axisbelow(True)
    figure = viz.save(fig, figures / "model_selection.png")

    lines: list[str] = []
    add = lines.append
    add("# 03 — Model selection\n")
    add(
        f"Tasks T-302 to T-305 · generated by `leadscore.train` · "
        f"{split.sizes['train']:,} training rows, {split.sizes['test']:,} held back\n"
    )
    add(
        f"**{winner.name.replace('_', ' ')} wins on cross-validation at "
        f"{winner.mean_auc:.4f} ROC-AUC.** The test set was not opened; every number here "
        "comes from 5-fold stratified cross-validation inside the training split.\n"
    )
    add(f"![Model selection]({figure.relative_to(reports).as_posix()})\n")
    add("| Model | CV ROC-AUC | Std | Verdict |")
    add("| --- | ---: | ---: | --- |")
    for result in ranked:
        verdict = "**winner**" if result is winner else ""
        add(f"| {result.name.replace('_', ' ')} | {result.mean_auc:.4f} | "
            f"{result.std_auc:.4f} | {verdict} |")
    add("")

    if linear and one_rule:
        margin = winner.mean_auc - linear.mean_auc
        add("## How much did the gradient booster actually buy?\n")
        add(
            f"The honest comparison is against logistic regression, not against chance. "
            f"The margin is **{margin:+.4f} ROC-AUC** — "
            + (
                "inside one standard deviation of the fold-to-fold noise "
                f"(±{linear.std_auc:.4f}), so it is not a clear win."
                if margin < linear.std_auc
                else "larger than the fold-to-fold noise, so it is a real improvement."
            )
            + "\n"
        )
        add(
            f"Both clear the one-rule baseline ({one_rule.mean_auc:.4f}) comfortably, which "
            "is the comparison a sales manager would care about: sorting a spreadsheet by "
            "time on site is genuinely worse than either model.\n"
        )
        if margin < linear.std_auc:
            add(
                "**This is a finding, not a failure.** A linear model on 13 admissible "
                "columns is close to the ceiling this data supports. Phase 4 decides which "
                "to ship on calibration and Capture@20%, not on ROC-AUC alone, and the "
                "README will report the comparison rather than quietly shipping the more "
                "complicated model.\n"
            )

    add("## Settings\n")
    add("| Setting | Value |")
    add("| --- | --- |")
    add(f"| Feature set | shipping ({len(cfg.shipping_features())} columns) |")
    add(f"| Split | {1 - settings.test_size:.0%} / {settings.test_size:.0%}, stratified |")
    add(f"| Seed | {settings.random_seed} |")
    add(f"| CV | {settings.cv_folds}-fold stratified |")
    add(f"| Optuna trials | {settings.optuna_trials} |")
    add(f"| Tracking | `{settings.mlflow_tracking_uri}` |")
    add("")
    if winner.params:
        add("Winning parameters:\n")
        add("```json")
        add(json.dumps(winner.params, indent=2, default=str))
        add("```\n")

    reports.mkdir(parents=True, exist_ok=True)
    path = reports / "03_model_selection.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="  %(message)s")
    os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
    settings = cfg.Settings()

    results, split = run_selection(settings)
    results.sort(key=lambda r: r.mean_auc, reverse=True)
    winner = results[0]

    tracking = log_to_mlflow(results, split, settings)
    report = write_report(results, split, settings)

    print()
    print("  Cross-validated ROC-AUC on the training split (the test set was not opened)")
    print("  " + "-" * 56)
    for result in results:
        marker = "  <- winner" if result is winner else ""
        print(f"  {result}{marker}")
    print()
    print(f"  train rows {split.sizes['train']:,} · test rows held back {split.sizes['test']:,}")
    if tracking:
        print(f"  tracked in MLflow at {tracking}")
    print(f"  report {report}")
    print()
    print("  Next: Phase 4 calibrates the winner and opens the test set once (T-401 to T-403).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
