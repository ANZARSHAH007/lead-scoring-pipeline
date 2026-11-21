"""The README-drift comparator (task T-407, D-033).

CI retrains from scratch on every push to `main`, which means it produces a card the
committed README cannot possibly equal: `model_version` carries the date of the run. Run #3
failed on precisely that, reported only "README does not match", and said nothing at all
about whether the model was still any good.

`compare()` is the replacement, and these tests pin the three distinctions it draws:
a new model version is expected, a moved number is not, and a changed test-set fingerprint
is the one thing that must never be tolerated - every published figure is measured on those
rows, so if they moved, nothing in the README means what it says.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

_spec = importlib.util.spec_from_file_location(
    "publish_results", ROOT / "scripts" / "publish_results.py"
)
assert _spec and _spec.loader
publish_results = importlib.util.module_from_spec(_spec)
sys.modules["publish_results"] = publish_results
_spec.loader.exec_module(publish_results)

compare = publish_results.compare
numeric = publish_results.numeric
skeleton = publish_results.skeleton


PUBLISHED = [
    "<!-- headline:start -->",
    "> If sales calls only the top 20% of scored leads, they reach **44.1%** of all conversions.",
    "",
    "Measured once on 1,848 held-out leads the model never saw (fingerprint `e30abe9c3fe596c1`), "
    "by model version `20261118-6ae06cc8`. Quality gate: **passed**.",
    "<!-- headline:end -->",
]


def retrained(old: str, new: str) -> list[str]:
    """The published block with one substring swapped, as a fresh card would render it."""
    return [line.replace(old, new) for line in PUBLISHED]


# --- numbers --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "expected"),
    [("44.1%", 0.441), ("1,848", 1848.0), ("0.865", 0.865), ("20", 20.0)],
)
def test_numeric_reads_percentages_and_thousands(text: str, expected: float) -> None:
    assert numeric(text) == pytest.approx(expected)


def test_skeleton_blanks_numbers_and_tokens() -> None:
    a = skeleton("reach **44.1%** by version `20261118-6ae06cc8`")
    b = skeleton("reach **43.0%** by version `20261120-ffffffff`")
    assert a == b, "two renderings of the same sentence must have the same skeleton"


# --- what a retrain is allowed to change ------------------------------------------------


def test_identical_blocks_have_no_problems() -> None:
    assert compare(PUBLISHED, list(PUBLISHED)) == []


def test_a_new_model_version_is_expected_not_drift() -> None:
    """The failure that made run #3 red. A retrain on another day changes the date."""
    fresh = retrained("`20261118-6ae06cc8`", "`20261120-6ae06cc8`")
    assert compare(PUBLISHED, fresh) == []


def test_a_tiny_metric_move_is_tolerated() -> None:
    """A different OS and thread count move the last decimals. That is not a regression."""
    fresh = retrained("44.1%", "44.0%")
    assert compare(PUBLISHED, fresh) == []


# --- what it must catch -----------------------------------------------------------------


def test_a_real_metric_move_is_reported() -> None:
    fresh = retrained("44.1%", "38.0%")
    problems = compare(PUBLISHED, fresh)
    assert len(problems) == 1
    assert "44.1%" in problems[0] and "38.0%" in problems[0]


def test_a_changed_fingerprint_is_never_tolerated() -> None:
    """The strictest rule here: the held-out rows are what every number is measured on."""
    fresh = retrained("`e30abe9c3fe596c1`", "`0000000000000000`")
    problems = compare(PUBLISHED, fresh)
    assert len(problems) == 1
    assert "held-out rows" in problems[0]


def test_a_failed_quality_gate_is_a_wording_change_and_is_caught() -> None:
    fresh = retrained("**passed**", "**failed**")
    problems = compare(PUBLISHED, fresh)
    assert problems and "wording changed" in problems[0]


def test_a_changed_row_count_is_caught_even_though_it_is_a_number() -> None:
    """1,848 held-out rows is the split, not a metric: one row off is a different test set."""
    fresh = retrained("1,848", "1,840")
    assert compare(PUBLISHED, fresh), "a row-count change exceeds the metric tolerance"


def test_an_added_row_is_a_shape_change() -> None:
    assert compare(PUBLISHED, [*PUBLISHED, "| extra | 1 | 2 | 3 | 4% |"])


def test_tolerance_is_configurable_and_not_silently_wide() -> None:
    assert publish_results.DRIFT_TOLERANCE == 0.02
    fresh = retrained("44.1%", "42.5%")
    assert compare(PUBLISHED, fresh) == [], "1.6 points is inside the documented tolerance"
    assert compare(PUBLISHED, fresh, tolerance=0.005), "and outside a stricter one"
