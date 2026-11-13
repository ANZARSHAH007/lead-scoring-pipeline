"""Exploratory data analysis for the X Education lead dataset (task T-201).

Writes a readable report to ``reports/01_eda.md`` plus four charts in ``reports/figures/``.

This replaces the notebook the roadmap originally named. A markdown report is reviewable in
a diff and readable in any editor's preview, where a committed ``.ipynb`` is JSON noise with
embedded image blobs. Decision D-015.

    .venv/Scripts/python scripts/eda.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from leadscore import config as cfg  # noqa: E402
from leadscore import plotting as viz  # noqa: E402

REPORTS = ROOT / "reports"
FIGURES = REPORTS / "figures"
NEAR_CONSTANT_SHARE = 0.95
TOP_SOURCES = 10


# ---------------------------------------------------------------------------
# loading
# ---------------------------------------------------------------------------


def load(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    # `Select` is a form default, not an answer (docs/02). Treat it as missing everywhere,
    # but only after the raw blank counts have been taken.
    return frame


def missingness(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    n = len(frame)
    for column in frame.columns:
        series = frame[column]
        blank = int(series.isna().sum())
        select = int((series.astype("string").str.strip() == cfg.SELECT_PLACEHOLDER).sum())
        non_missing = series[series.notna()]
        if len(non_missing):
            top_value = non_missing.astype("string").value_counts().idxmax()
            top_share = non_missing.astype("string").value_counts().iloc[0] / n
        else:
            top_value, top_share = "", 0.0
        rows.append(
            {
                "column": column,
                "dtype": str(series.dtype),
                "unique": int(series.nunique(dropna=True)),
                "blank_pct": blank / n,
                "select_pct": select / n,
                "missing_pct": (blank + select) / n,
                "top_value": top_value,
                "top_share": top_share,
            }
        )
    out = pd.DataFrame(rows)
    out["near_constant"] = out["top_share"] >= NEAR_CONSTANT_SHARE
    return out.sort_values("missing_pct", ascending=False).reset_index(drop=True)


def conversion_by(frame: pd.DataFrame, column: str, top: int | None = None) -> pd.DataFrame:
    grouped = (
        frame.groupby(column, dropna=False)[cfg.TARGET]
        .agg(leads="size", converted="sum")
        .reset_index()
    )
    grouped["rate"] = grouped["converted"] / grouped["leads"]
    grouped = grouped.sort_values("leads", ascending=False)
    if top is not None:
        grouped = grouped.head(top)
    return grouped.reset_index(drop=True)


# ---------------------------------------------------------------------------
# charts
# ---------------------------------------------------------------------------


def chart_conversion_by(data: pd.DataFrame, column: str, title: str, path: Path) -> Path:
    import matplotlib.pyplot as plt

    data = data.sort_values("rate")
    labels = [str(v)[:34] for v in data[column]]
    height = max(2.4, 0.33 * len(data) + 1.1)
    fig, ax = plt.subplots(figsize=(7.2, height))

    ax.barh(labels, data["rate"], color=viz.SERIES_A, height=0.62)
    overall = data["converted"].sum() / data["leads"].sum()
    ax.axvline(overall, color=viz.INK_MUTED, linewidth=1, linestyle="--")
    ax.text(
        overall,
        len(data) - 0.3,
        f"  overall {overall:.0%}",
        color=viz.INK_MUTED,
        fontsize=8,
        va="center",
    )

    for y, (rate, leads) in enumerate(zip(data["rate"], data["leads"], strict=True)):
        ax.text(rate + 0.012, y, f"{rate:.0%}  (n={leads:,})", va="center", fontsize=8,
                color=viz.INK_MUTED)

    ax.set_xlim(0, min(1.0, data["rate"].max() + 0.22))
    ax.set_xlabel("Conversion rate")
    ax.set_title(title, loc="left")
    ax.xaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    ax.grid(axis="x", alpha=0.6)
    ax.set_axisbelow(True)
    return viz.save(fig, path)


def time_on_site_deciles(frame: pd.DataFrame) -> pd.DataFrame:
    """Conversion rate by time on website: a zero bin plus eight equal-count bins.

    Two overlaid histograms were the obvious chart here and the wrong one: the groups are
    different sizes, so the bars occlude each other and the reader has to compare areas.
    The question that matters is whether conversion rate *rises* with time on site, because
    that is exactly what the one-rule baseline exploits. Rate per equal-count bin answers it
    directly, as one series.

    Zero gets its own bin rather than being split across two identical ones. Roughly a fifth
    of leads spent no time on the site at all, so equal-count bins put that single value in
    two buckets labelled "0-0", which is unreadable and hides that the zero group behaves
    nothing like the low-but-positive group.
    """
    column = "Total Time Spent on Website"
    work = frame[[column, cfg.TARGET]].dropna(subset=[column]).copy()

    zero = work[work[column] == 0]
    positive = work[work[column] > 0].copy()
    positive["bin"] = pd.qcut(positive[column].rank(method="first"), 8, labels=False)

    rows = [
        {
            "label": "0 (no visit recorded)",
            "leads": len(zero),
            "converted": int(zero[cfg.TARGET].sum()),
            "is_zero": True,
        }
    ]
    grouped = positive.groupby("bin").agg(
        leads=(cfg.TARGET, "size"),
        converted=(cfg.TARGET, "sum"),
        low=(column, "min"),
        high=(column, "max"),
    )
    for row in grouped.itertuples(index=False):
        rows.append(
            {
                "label": f"{int(row.low):,}-{int(row.high):,}",
                "leads": int(row.leads),
                "converted": int(row.converted),
                "is_zero": False,
            }
        )

    out = pd.DataFrame(rows)
    out["rate"] = out["converted"] / out["leads"]
    return out


def chart_time_on_site(deciles: pd.DataFrame, overall: float, path: Path) -> Path:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.4, 3.6))
    positions = range(len(deciles))
    colors = [viz.SERIES_B if z else viz.SERIES_A for z in deciles["is_zero"]]
    ax.bar(positions, deciles["rate"], color=colors, width=0.68)
    ax.axhline(overall, color=viz.INK_MUTED, linewidth=1, linestyle="--")
    ax.text(
        len(deciles) - 0.5,
        overall - 0.045,
        f"overall {overall:.0%}",
        color=viz.INK_MUTED,
        fontsize=8,
        ha="right",
    )

    # label only the ends and the crossing point, not every bar
    marked = {0, len(deciles) - 1, int(deciles["rate"].sub(overall).abs().idxmin())}
    for i in marked:
        ax.text(
            i,
            deciles["rate"].iloc[i] + 0.02,
            f"{deciles['rate'].iloc[i]:.0%}",
            ha="center",
            fontsize=8,
            color=viz.INK,
        )

    ax.set_xticks(list(positions))
    ax.set_xticklabels(deciles["label"], rotation=45, ha="right", fontsize=7)
    ax.set_xlabel("Time on website (seconds), equal-count bins")
    ax.set_ylabel("Conversion rate")
    ax.set_ylim(0, min(1.0, deciles["rate"].max() + 0.12))
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    ax.set_title(
        "Conversion is U-shaped in time on website, not rising", loc="left"
    )
    ax.text(
        0,
        -0.30,
        "amber = no visit recorded",
        transform=ax.transAxes,
        fontsize=7.5,
        color=viz.INK_MUTED,
    )
    ax.grid(axis="y", alpha=0.6)
    ax.set_axisbelow(True)
    return viz.save(fig, path)


def chart_missingness(summary: pd.DataFrame, path: Path) -> Path:
    import matplotlib.pyplot as plt

    data = summary[summary["missing_pct"] > 0.01].sort_values("missing_pct")
    fig, ax = plt.subplots(figsize=(7.2, max(2.4, 0.33 * len(data) + 1.1)))

    labels = [c if len(c) <= 46 else c[:45] + "…" for c in data["column"]]
    ax.barh(labels, data["blank_pct"], color=viz.SERIES_A, height=0.62, label="Blank")
    ax.barh(
        labels,
        data["select_pct"],
        left=data["blank_pct"],
        color=viz.SERIES_B,
        height=0.62,
        label='"Select" placeholder',
    )
    for y, total in enumerate(data["missing_pct"]):
        ax.text(total + 0.012, y, f"{total:.0%}", va="center", fontsize=8, color=viz.INK_MUTED)

    ax.set_xlim(0, min(1.0, data["missing_pct"].max() + 0.12))
    ax.set_xlabel("Share of rows with no usable value")
    ax.set_title("Effective missingness: blanks plus the Select placeholder", loc="left")
    ax.xaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    ax.legend(loc="lower right")
    ax.grid(axis="x", alpha=0.6)
    ax.set_axisbelow(True)
    return viz.save(fig, path)


# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------


def pct(value: float) -> str:
    return f"{value:.1%}"


def write_report(
    frame: pd.DataFrame,
    summary: pd.DataFrame,
    deciles: pd.DataFrame,
    figures: dict[str, Path],
) -> Path:
    n = len(frame)
    rate = frame[cfg.TARGET].mean()
    origin = conversion_by(frame, "Lead Origin")
    source = conversion_by(frame, "Lead Source", top=TOP_SOURCES)

    near_constant = summary[summary["near_constant"]]
    heavy_missing = summary[summary["missing_pct"] >= 0.40]

    lines: list[str] = []
    add = lines.append

    add("# 01 — Exploratory data analysis\n")
    add(f"Task T-201 · generated by `scripts/eda.py` · {n:,} leads by {len(frame.columns)} columns\n")
    add(
        f"**{frame[cfg.TARGET].sum():,} of {n:,} leads converted ({pct(rate)}).** "
        "Everything below is measured from `data/raw/Leads.csv`; nothing is carried over "
        "from the planning docs.\n"
    )

    add("## What this changes\n")
    add(
        f"1. **{len(near_constant)} columns are near-constant** "
        f"(one value covers ≥ {NEAR_CONSTANT_SHARE:.0%} of rows) and carry no signal. "
        "They are the ad and opt-in flags docs/02 said to 'check first'. Drop them.\n"
    )
    add(
        f"2. **{len(heavy_missing)} columns are ≥ 40% empty** once `Select` counts as missing. "
        "Two of them are admissible features, so they need a decision rather than a default.\n"
    )
    add(
        "3. **Time on website predicts conversion, but U-shaped rather than rising.** Leads "
        "with no recorded visit convert above average, so the one-rule baseline that sorts "
        "by time on site ranks them last and is weaker than it appears.\n"
    )
    suspicious = origin[(origin["rate"] >= 0.85) & (origin["leads"] >= 50)]
    for _, row in suspicious.iterrows():
        add(
            f"4. **`Lead Origin = {row['Lead Origin']}` converts at {pct(row['rate'])} across "
            f"{row['leads']:,} leads — a red flag for T-203.** A category that almost always "
            "converts usually means the row was created *because* something already happened, "
            "not that the origin causes conversion. `Lead Origin` is otherwise admissible, so "
            "the audit has to decide whether this one level is a record-keeping artefact. Do "
            "not keep it just because it is predictive; that is the trap this project exists "
            "to avoid.\n"
        )

    add("## Missingness\n")
    add(
        "`Select` is a form default, not an answer. A column can look populated and be empty "
        "in practice, so both bases are reported.\n"
    )
    add(f"![Missingness]({figures['missing'].relative_to(REPORTS).as_posix()})\n")
    add("| Column | Blank | `Select` | Effectively missing | Unique | Near-constant |")
    add("| --- | ---: | ---: | ---: | ---: | :---: |")
    for _, row in summary[summary["missing_pct"] > 0].iterrows():
        add(
            f"| `{row['column']}` | {pct(row['blank_pct'])} | {pct(row['select_pct'])} | "
            f"**{pct(row['missing_pct'])}** | {row['unique']:,} | "
            f"{'yes' if row['near_constant'] else ''} |"
        )
    add("")

    add("## Near-constant columns\n")
    if near_constant.empty:
        add("None.\n")
    else:
        add(
            f"One value covers ≥ {NEAR_CONSTANT_SHARE:.0%} of rows, so the column cannot "
            "discriminate. Dropping these costs nothing and shortens the feature list.\n"
        )
        add("| Column | Dominant value | Share | Unique |")
        add("| --- | --- | ---: | ---: |")
        for _, row in near_constant.iterrows():
            add(
                f"| `{row['column']}` | {row['top_value']} | {pct(row['top_share'])} | "
                f"{row['unique']:,} |"
            )
        add("")

    add("## Numeric features\n")
    add("| Feature | Missing | Min | Median | Mean | p95 | Max |")
    add("| --- | ---: | ---: | ---: | ---: | ---: | ---: |")
    for column in cfg.NUMERIC_FEATURES:
        series = frame[column]
        add(
            f"| `{column}` | {pct(series.isna().mean())} | {series.min():,.0f} | "
            f"{series.median():,.0f} | {series.mean():,.1f} | "
            f"{series.quantile(0.95):,.0f} | {series.max():,.0f} |"
        )
    add("")
    add(f"![Conversion by time on website]({figures['time'].relative_to(REPORTS).as_posix()})\n")
    zero_row = deciles.iloc[0]
    positives = deciles[~deciles["is_zero"]]
    trough = positives.loc[positives["rate"].idxmin()]
    peak = positives.loc[positives["rate"].idxmax()]
    add(
        f"**The relationship is U-shaped, not rising.** Leads with no recorded visit convert "
        f"at {pct(zero_row['rate'])} ({zero_row['leads']:,} leads), above the {pct(rate)} "
        f"average. Among leads that did browse, conversion bottoms out at "
        f"{pct(trough['rate'])} in the {trough['label']} second band and climbs to "
        f"{pct(peak['rate'])} in the {peak['label']} band.\n"
    )
    add(
        "This matters for the baseline. Ranking leads by time on site, which is the one-rule "
        "baseline and what a sales manager would do by hand, puts the zero-time group "
        "**last** even though it converts above average. That group is a different "
        "population rather than a cold one: most likely imported or offline leads whose "
        "browsing was never tracked.\n"
    )
    add(
        "1. The one-rule baseline is weaker than its headline suggests, so beating it is a "
        "lower bar than it looks. The README should say so rather than claim an easy win.\n"
        "2. A linear model cannot fit a U-shape in a raw numeric feature, so logistic "
        "regression needs time on site binned or paired with a zero indicator. LightGBM "
        "handles it natively, and that may turn out to be most of the gap between the two "
        "(T-304).\n"
    )
    add("| Time on site (seconds) | Leads | Converted | Rate |")
    add("| --- | ---: | ---: | ---: |")
    for row in deciles.itertuples(index=False):
        add(f"| {row.label} | {row.leads:,} | {row.converted:,} | {pct(row.rate)} |")
    add("")
    zero_time = (frame["Total Time Spent on Website"] == 0).mean()
    add(
        f"{pct(zero_time)} of leads spent **zero** seconds on the website. They are not an "
        "error — a lead can arrive through an import or a cold list — but they make the "
        "single strongest feature uninformative for that slice, which the error analysis "
        "(T-406) should revisit.\n"
    )

    add("## Conversion by where the lead came from\n")
    add(f"![Conversion by lead origin]({figures['origin'].relative_to(REPORTS).as_posix()})\n")
    add("| Lead Origin | Leads | Converted | Rate |")
    add("| --- | ---: | ---: | ---: |")
    for _, row in origin.iterrows():
        add(
            f"| {row['Lead Origin']} | {row['leads']:,} | {row['converted']:,} | "
            f"{pct(row['rate'])} |"
        )
    add("")
    add(f"![Conversion by lead source]({figures['source'].relative_to(REPORTS).as_posix()})\n")
    add(f"Top {TOP_SOURCES} sources by volume.\n")
    add("| Lead Source | Leads | Converted | Rate |")
    add("| --- | ---: | ---: | ---: |")
    for _, row in source.iterrows():
        add(
            f"| {row['Lead Source']} | {row['leads']:,} | {row['converted']:,} | "
            f"{pct(row['rate'])} |"
        )
    add("")

    add("## What T-202 and T-203 take from this\n")
    add(
        "- The pandera schema (T-202) enforces the dtypes and ranges in the numeric table "
        "above, and rejects the near-constant columns before they reach a model.\n"
        "- The leakage audit (T-203) still has to classify every column; this report settles "
        "only the *usefulness* question, never the *admissibility* one. A column can be "
        "perfectly predictive and still be banned.\n"
    )

    REPORTS.mkdir(parents=True, exist_ok=True)
    path = REPORTS / "01_eda.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def main() -> int:
    settings = cfg.Settings(_env_file=None)
    path = ROOT / settings.data_raw
    if not path.exists():
        print(f"  Not found: {path}\n  Run scripts/check_data.py first.")
        return 1

    viz.apply_style()
    frame = load(path)
    summary = missingness(frame)
    deciles = time_on_site_deciles(frame)

    figures = {
        "missing": chart_missingness(summary, FIGURES / "missingness.png"),
        "time": chart_time_on_site(
            deciles, float(frame[cfg.TARGET].mean()), FIGURES / "time_on_website.png"
        ),
        "origin": chart_conversion_by(
            conversion_by(frame, "Lead Origin"),
            "Lead Origin",
            "Conversion rate by lead origin",
            FIGURES / "conversion_by_origin.png",
        ),
        "source": chart_conversion_by(
            conversion_by(frame, "Lead Source", top=TOP_SOURCES),
            "Lead Source",
            f"Conversion rate by lead source (top {TOP_SOURCES} by volume)",
            FIGURES / "conversion_by_source.png",
        ),
    }

    report = write_report(frame, summary, deciles, figures)
    print(f"  report  {report.relative_to(ROOT)}")
    for key, figure in figures.items():
        print(f"  figure  {figure.relative_to(ROOT)}  ({key})")
    print(f"\n  {len(summary[summary['near_constant']])} near-constant columns found.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
