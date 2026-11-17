"""Tests for the chart rules (task T-110).

Report 03's first draft was a bar chart of ROC-AUC with the x-axis starting at 0.45, which
made a 0.015 difference between two models look decisive. The chart was replaced; these tests
make sure the *rule* that replaced it cannot be forgotten, because a rule written only in a
doc is a rule that gets broken the next time someone is in a hurry.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import pytest

from leadscore import plotting as viz


@pytest.fixture
def ax():
    figure, axes = plt.subplots()
    yield axes
    plt.close(figure)


def test_a_truncated_axis_is_allowed_when_chance_stays_visible(ax) -> None:
    viz.metric_axis(ax, 0.45, 1.0, chance=0.5)
    assert ax.get_xlim() == (0.45, 1.0)


def test_a_truncated_axis_that_hides_chance_is_refused(ax) -> None:
    """The actual mistake: start the axis above the baseline and the gap looks enormous."""
    with pytest.raises(ValueError, match="chance line visible"):
        viz.metric_axis(ax, 0.55, 1.0, chance=0.5)


def test_reversed_limits_are_refused(ax) -> None:
    with pytest.raises(ValueError, match="must increase"):
        viz.metric_axis(ax, 1.0, 0.45)


def test_the_chance_reference_is_drawn_once(ax) -> None:
    viz.metric_axis(ax, 0.4, 1.0, chance=0.5)
    positions = [line.get_xdata()[0] for line in ax.get_lines()]
    assert positions.count(0.5) == 1


def test_no_reference_line_without_a_chance_value(ax) -> None:
    viz.metric_axis(ax, 0.0, 1.0)
    assert ax.get_lines() == []


def test_the_vertical_form_sets_the_y_axis(ax) -> None:
    viz.metric_axis(ax, 0.4, 1.0, chance=0.5, axis="y")
    assert ax.get_ylim() == (0.4, 1.0)


def test_the_series_colours_are_the_validated_pair() -> None:
    """Changing these silently would break the palette validation recorded in D-014."""
    assert viz.SERIES_A == "#2563EB"
    assert viz.SERIES_B == "#D97706"


def test_save_creates_missing_directories(tmp_path) -> None:
    figure, axes = plt.subplots()
    axes.plot([0, 1], [0, 1])
    path = viz.save(figure, tmp_path / "deep" / "nested" / "chart.png")
    assert path.exists() and path.stat().st_size > 0
