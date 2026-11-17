"""The leakage audit (tasks T-203 and T-204) — the centrepiece of this project.

Produces `reports/02_leakage_audit.md`, which records two things:

1. **A yes/no admissibility decision for every one of the 37 columns**, with the reason. This
   is the record docs/02 FR-2 promises. A column is admissible only if its value is known at
   the moment the lead is scored.
2. **What the banned columns are worth**, measured. Three feature sets are cross-validated
   with the same logistic regression, so the gap between them is the price of honesty — the
   number the README's last row reports.

A per-column univariate score is also computed: one logistic regression per column, 5-fold.
It answers "how much does this column alone tell you", which is what makes the leakage case
obvious rather than asserted.

    .venv/Scripts/python scripts/leakage_audit.py
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from leadscore import config as cfg  # noqa: E402
from leadscore import plotting as viz  # noqa: E402
from leadscore import schema as sch  # noqa: E402

REPORTS = ROOT / "reports"
FIGURES = REPORTS / "figures"
NEAR_CONSTANT_SHARE = 0.95


# ---------------------------------------------------------------------------
# the decision record
# ---------------------------------------------------------------------------

REASONS: dict[str, tuple[str, str]] = {}


def classify(frame: pd.DataFrame) -> pd.DataFrame:
    """One row per column: admissible yes/no, and why."""
    rows = []

    def add(column: str, admissible: str, reason: str, group: str) -> None:
        rows.append(
            {"column": column, "group": group, "admissible": admissible, "reason": reason}
        )

    add(cfg.TARGET, "target", "this is the label", "Target")
    for column in cfg.ID_COLUMNS:
        add(
            column,
            "No",
            "an identifier carries no signal; `Prospect ID` is kept only to join results back",
            "Identifier",
        )
    for column in cfg.SOURCE_FEATURES:
        add(column, "Yes", "known the moment the lead arrives", "Source")
    for column in cfg.NUMERIC_FEATURES:
        add(column, "Yes", "pre-contact website behaviour, recorded before any rep acts", "Behaviour")
    for column in cfg.FORM_FEATURES:
        add(column, "Yes", "self-reported by the lead on the form at submission", "Form answer")
    for column in cfg.CONTACT_PREFERENCE_FEATURES:
        add(column, "Yes", "stated by the lead on the form", "Contact preference")
    for column in cfg.FLAG_FEATURES:
        share = frame[column].value_counts(normalize=True, dropna=True)
        top = share.iloc[0] if len(share) else 1.0
        add(
            column,
            "No",
            f"near-constant: one value covers {top:.1%} of rows, so it cannot discriminate",
            "Ad / opt-in flag",
        )
    for column in cfg.ACTIVITY_FEATURES:
        add(
            column,
            "Tested both ways",
            "may encode that a rep already made contact; measured below rather than assumed",
            "Activity",
        )
    for column in cfg.BANNED_LEAKAGE_COLUMNS:
        add(
            column,
            "No - leakage",
            "written by the sales team after working the lead, so it encodes the outcome",
            "Sales-filled",
        )
    for column in cfg.DROPPED_SPARSE_COLUMNS:
        add(column, "No", "74.2% empty once `Select` counts as missing; too sparse to learn from",
            "Sparse")

    out = pd.DataFrame(rows)
    missing = sorted(set(frame.columns) - set(out["column"]))
    if missing:
        raise SystemExit(f"columns with no decision: {missing}")
    return out


# ---------------------------------------------------------------------------
# modelling
# ---------------------------------------------------------------------------


def build_estimator(numeric: list[str], categorical: list[str]) -> Pipeline:
    return Pipeline(
        [
            (
                "prep",
                ColumnTransformer(
                    [
                        (
                            "num",
                            Pipeline(
                                [
                                    ("impute", SimpleImputer(strategy="median")),
                                    ("scale", StandardScaler()),
                                ]
                            ),
                            numeric,
                        ),
                        (
                            "cat",
                            Pipeline(
                                [
                                    (
                                        "impute",
                                        SimpleImputer(
                                            strategy="constant", fill_value="Missing"
                                        ),
                                    ),
                                    (
                                        "onehot",
                                        OneHotEncoder(
                                            handle_unknown="infrequent_if_exist",
                                            min_frequency=0.01,
                                            sparse_output=False,
                                        ),
                                    ),
                                ]
                            ),
                            categorical,
                        ),
                    ],
                    remainder="drop",
                ),
            ),
            ("model", LogisticRegression(max_iter=2000)),
        ]
    )


def score(frame: pd.DataFrame, columns: list[str], settings: cfg.Settings) -> float:
    numeric = [c for c in columns if c in cfg.NUMERIC_FEATURES]
    categorical = [c for c in columns if c not in cfg.NUMERIC_FEATURES]
    if not numeric and not categorical:
        return float("nan")

    x = frame[columns]
    y = frame[cfg.TARGET]
    folds = StratifiedKFold(
        n_splits=settings.cv_folds, shuffle=True, random_state=settings.random_seed
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        scores = cross_val_score(
            build_estimator(numeric, categorical), x, y, cv=folds, scoring="roc_auc"
        )
    return float(scores.mean())


def univariate_scores(frame: pd.DataFrame, settings: cfg.Settings) -> pd.DataFrame:
    """One model per column: how much does this column alone tell you?"""
    rows = []
    for column in frame.columns:
        if column in {cfg.TARGET, *cfg.ID_COLUMNS}:
            continue
        rows.append({"column": column, "auc": score(frame, [column], settings)})
    return pd.DataFrame(rows).sort_values("auc", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------------------
# chart
# ---------------------------------------------------------------------------


def chart_univariate(data: pd.DataFrame, decisions: pd.DataFrame, path: Path) -> Path:
    import matplotlib.pyplot as plt

    merged = data.merge(decisions, on="column", how="left").head(16).sort_values("auc")
    banned = merged["admissible"].str.startswith("No").fillna(False)
    colors = [viz.SERIES_B if b else viz.SERIES_A for b in banned]

    fig, ax = plt.subplots(figsize=(7.4, 0.34 * len(merged) + 1.3))
    labels = [c if len(c) <= 42 else c[:41] + "…" for c in merged["column"]]
    ax.barh(labels, merged["auc"], color=colors, height=0.62)
    ax.axvline(0.5, color=viz.INK_MUTED, linewidth=1, linestyle="--")
    ax.text(0.5, -0.9, " no signal", color=viz.INK_MUTED, fontsize=8, va="center")

    for y, auc in enumerate(merged["auc"]):
        ax.text(auc + 0.006, y, f"{auc:.3f}", va="center", fontsize=8, color=viz.INK_MUTED)

    ax.set_xlim(0.45, min(1.0, merged["auc"].max() + 0.07))
    ax.set_xlabel("Cross-validated ROC-AUC from that column alone")
    ax.set_title("The most predictive columns are the ones a rep fills in after calling", loc="left")
    ax.text(
        0, -0.09, "amber = inadmissible", transform=ax.transAxes, fontsize=7.5,
        color=viz.INK_MUTED,
    )
    ax.grid(axis="x", alpha=0.6)
    ax.set_axisbelow(True)
    return viz.save(fig, path)


# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------


def main() -> int:
    settings = cfg.Settings(_env_file=None)
    try:
        frame = sch.load_and_validate(settings.data_raw, settings)
    except sch.DataContractError as error:
        print(f"  {error}")
        return 1

    viz.apply_style()
    decisions = classify(frame)

    sets = {
        "Shipping (admissible only)": list(cfg.shipping_features()),
        "+ activity columns": list(cfg.with_activity_features()),
        "+ sales-filled (leaks)": list(cfg.audit_features()),
    }
    results = {name: score(frame, columns, settings) for name, columns in sets.items()}
    univariate = univariate_scores(frame, settings)
    figure = chart_univariate(univariate, decisions, FIGURES / "univariate_auc.png")

    honest = results["Shipping (admissible only)"]
    leaking = results["+ sales-filled (leaks)"]
    with_activity = results["+ activity columns"]

    lines: list[str] = []
    add = lines.append
    add("# 02 — Leakage audit\n")
    add(f"Tasks T-203 and T-204 · generated by `scripts/leakage_audit.py` · {len(frame):,} leads\n")
    add(
        f"**Using the columns a sales rep fills in after calling a lead lifts cross-validated "
        f"ROC-AUC from {honest:.3f} to {leaking:.3f}.** That gap is not a better model. It is "
        "the model being told the answer. Every number in this project's results table comes "
        f"from the {honest:.3f} column set.\n"
    )

    add("## What the banned columns are worth\n")
    add(
        "Same logistic regression, same 5-fold stratified split, same seed. Only the feature "
        "set changes.\n"
    )
    add("| Feature set | Columns | CV ROC-AUC | Verdict |")
    add("| --- | ---: | ---: | --- |")
    verdicts = {
        "Shipping (admissible only)": "**the set this project ships**",
        "+ activity columns": "not shippable — values a fresh lead cannot have",
        "+ sales-filled (leaks)": "not shippable — written after the rep calls",
    }
    for name, columns in sets.items():
        add(f"| {name} | {len(columns)} | {results[name]:.3f} | {verdicts[name]} |")
    add("")
    add(
        f"Adding the two activity columns moves ROC-AUC from {honest:.3f} to "
        f"{with_activity:.3f}. Adding the six sales-filled columns moves it to {leaking:.3f}. "
        f"A public notebook reporting a number near {leaking:.2f} on "
        "this dataset is almost certainly scoring leads with information that does not exist "
        "when a lead arrives.\n"
    )

    add("## Which columns carry the signal\n")
    add(f"![Univariate AUC]({figure.relative_to(REPORTS).as_posix()})\n")
    add(
        "One logistic regression per column, cross-validated, nothing else in the model. "
        "0.500 means the column tells you nothing.\n"
    )
    add("| Column | AUC alone | Admissible |")
    add("| --- | ---: | --- |")
    merged = univariate.merge(decisions, on="column", how="left")
    for _, row in merged.head(20).iterrows():
        add(f"| `{row['column']}` | {row['auc']:.3f} | {row['admissible']} |")
    add("")

    add("## The activity columns: contaminated, not clean\n")
    add(
        "docs/02 said to test `Last Activity` and `Last Notable Activity` both ways rather "
        "than assume. Here is why they cannot simply be kept: their values mix things the "
        "**lead** did with things the **company** did.\n"
    )
    for column in cfg.ACTIVITY_FEATURES:
        grouped = (
            frame.groupby(column, dropna=False)[cfg.TARGET]
            .agg(leads="size", rate="mean")
            .sort_values("leads", ascending=False)
            .head(8)
        )
        add(f"**`{column}`**\n")
        add("| Value | Leads | Conversion | Who did it |")
        add("| --- | ---: | ---: | --- |")
        for value, row in grouped.iterrows():
            label = str(value)
            actor = (
                "**the company**"
                if any(k in label for k in ("SMS Sent", "Unreachable", "Bounced", "Modified"))
                else "the lead"
            )
            add(f"| {label} | {int(row['leads']):,} | {row['rate']:.1%} | {actor} |")
        add("")
    add(
        "`SMS Sent` is an outbound message. `Unreachable` is the outcome of a call attempt. "
        "Both convert far above average, and **neither can exist for a lead that has just "
        "arrived** — which is the definition of a feature that is not known at scoring time. "
        "Keeping these columns trains the model on a world that does not exist when it is "
        "asked to predict.\n"
    )
    add(
        f"That is the whole gap between the first two rows of the table above: "
        f"{with_activity - honest:+.3f} ROC-AUC bought with contaminated values. **Decided "
        "2026-11-17: both columns are excluded from the shipping model** and reported as a "
        "labelled comparison, exactly as the sales-filled row is. See D-016.\n"
    )

    add("## The decision record — every column\n")
    add(
        "A column is admissible only if its value is known at the moment the lead is scored. "
        "This table is the record docs/02 FR-2 requires; no column is left unclassified.\n"
    )
    add("| Column | Group | Admissible | Why |")
    add("| --- | --- | --- | --- |")
    order = {"target": 0, "Yes": 1, "Tested both ways": 2, "No": 3, "No - leakage": 4}
    for _, row in decisions.sort_values(
        by=["admissible", "column"], key=lambda s: s.map(order).fillna(9) if s.name == "admissible" else s
    ).iterrows():
        add(f"| `{row['column']}` | {row['group']} | {row['admissible']} | {row['reason']} |")
    add("")

    add("## Consequences for the build\n")
    add(
        f"- The shipping feature set is frozen at **{len(cfg.shipping_features())} columns** "
        "(T-205). The 12 ad and opt-in flags are out on the evidence in report 01, and the 2 "
        "activity columns on the evidence above — neither on suspicion.\n"
        f"- The honest baseline to beat is **{honest:.3f}** ROC-AUC from logistic regression "
        "(T-303), not the leaking figure.\n"
        "- The README results table carries the leaking row, labelled unshippable, so the "
        "comparison is visible rather than hidden.\n"
        "- `Prospect ID` stays in the data only to join predictions back; it is never a feature.\n"
    )

    REPORTS.mkdir(parents=True, exist_ok=True)
    path = REPORTS / "02_leakage_audit.md"
    path.write_text("\n".join(lines), encoding="utf-8")

    print(f"  report  {path.relative_to(ROOT)}")
    print(f"  figure  {figure.relative_to(ROOT)}")
    print()
    for name in sets:
        print(f"  {name:<36} {results[name]:.4f}")
    print(f"\n  leakage gap: {leaking - honest:+.4f} ROC-AUC")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
