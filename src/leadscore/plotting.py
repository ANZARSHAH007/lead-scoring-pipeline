"""Chart styling, shared by the EDA, the leakage audit and the evaluation reports.

Colors come from the project design system (`design-system/lead-scoring-pipeline/MASTER.md`)
with one validated deviation: chart marks use ``SERIES_A = #2563EB`` rather than the UI
primary ``#1E40AF``, because the darker blue falls outside the readable lightness band for
data marks. The pair (#2563EB, #D97706) passes all six palette checks — lightness band,
chroma floor, CVD separation (ΔE 32.3 protan), normal-vision separation (ΔE 38.3) and
contrast against the surface. See docs/08 and decision D-014.

Rules enforced here rather than remembered per chart: two series maximum before a legend is
mandatory, categorical hues assigned in fixed order and never cycled, text in ink tokens
rather than series color, recessive axes, no gridlines on the category axis.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # no display in CI or a terminal session
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure

# --- validated categorical pair, in fixed order ---
SERIES_A = "#2563EB"  # first series / single-series bars
SERIES_B = "#D97706"  # second series only

# --- ink and surface, from MASTER.md ---
INK = "#1E293B"
INK_MUTED = "#475569"
SURFACE = "#FFFFFF"
GRID = "#DBEAFE"

FIGURE_DPI = 140


def apply_style() -> None:
    """Set the rcParams every chart in this project shares."""
    plt.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "font.family": "sans-serif",
            "font.sans-serif": ["Fira Sans", "Segoe UI", "DejaVu Sans"],
            "font.size": 9,
            "axes.titlesize": 11,
            "axes.titleweight": "600",
            "axes.titlecolor": INK,
            "axes.labelsize": 9,
            "axes.labelcolor": INK_MUTED,
            "axes.edgecolor": GRID,
            "axes.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.color": INK_MUTED,
            "ytick.color": INK_MUTED,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.frameon": False,
            "legend.fontsize": 8,
            "grid.color": GRID,
            "grid.linewidth": 0.6,
        }
    )


def metric_axis(
    ax: Axes,
    lo: float,
    hi: float = 1.0,
    chance: float | None = None,
    axis: str = "x",
) -> None:
    """Set a metric axis, refusing limits that would hide the baseline.

    A truncated axis is the easiest way to publish a misleading chart, and this project made
    that mistake once: a bar chart of ROC-AUC starting at 0.45 made a 0.015 difference look
    decisive (D-018). The rule that replaced it is enforceable, so it is enforced here rather
    than written in a doc and forgotten:

    * a metric axis may start above zero **only** if the chance line stays inside the limits,
      so a reader can see how far above chance the marks actually sit;
    * the mark must carry its own spread — a dot and a whisker, not a bar, whose length
      invites comparison against an origin that is not on the chart.

    Raises:
        ValueError: if `chance` falls outside `[lo, hi]`, or the limits are reversed.
    """
    if not lo < hi:
        raise ValueError(f"axis limits must increase: got lo={lo}, hi={hi}")
    if chance is not None and not lo <= chance <= hi:
        raise ValueError(
            f"a truncated metric axis must keep the chance line visible: chance={chance} "
            f"falls outside [{lo}, {hi}]. Widen the limits or plot the full range."
        )
    if axis == "x":
        ax.set_xlim(lo, hi)
    else:
        ax.set_ylim(lo, hi)
    if chance is not None:
        line = ax.axvline if axis == "x" else ax.axhline
        line(chance, color=INK_MUTED, linewidth=1, linestyle="--")


def save(fig: Figure, path: Path) -> Path:
    """Write a figure and close it. Returns the path, for the report to link."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    return path
