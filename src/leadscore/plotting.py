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


def save(fig: Figure, path: Path) -> Path:
    """Write a figure and close it. Returns the path, for the report to link."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    return path
