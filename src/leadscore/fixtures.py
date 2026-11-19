"""A synthetic stand-in for `Leads.csv`, for tests, CI and the preflight check.

The real dataset is third-party data and is never committed (CLAUDE.md rule 2), so anything
that must run without it — the CI pipeline, the smoke test, `scripts/doctor.py` — needs a file
with the same *shape* and none of the data. That is what this module builds.

Three properties matter, and each one is deliberate:

1. **Contract-identical.** Exactly the 37 columns `schema.known_columns()` names, with the
   same dtypes and the same uniqueness constraints, so a fixture run exercises the real data
   contract rather than a relaxed one. If a column is added to config and not here, the
   fixture stops validating — which is the point.
2. **Learnable.** Conversion depends on time on site, so a model can beat chance and the
   selection code runs its real path. The signal is synthetic and means nothing; no number
   produced from this file may ever be reported as a result.
3. **Realistic in its defects.** `Select` placeholders, missing numerics, near-constant
   flag columns and a leaky `Tags` column are all present, because those are the shapes the
   pipeline has code to handle, and code that is never exercised is not known to work.

The row count differs from the real file on purpose, so `check_shape` emits its drift warning
and the fixture can never be mistaken for the genuine upload.

    from leadscore.fixtures import synthetic_leads, write_fixture
    frame = synthetic_leads(rows=300)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from leadscore import config as cfg

DEFAULT_ROWS = 300
DEFAULT_SEED = 7

#: The column that carries the synthetic signal, matching `train.ONE_RULE_COLUMN`.
SIGNAL_COLUMN = "Total Time Spent on Website"

#: Origin and source are different questions in the real upload, so they get different
#: levels here too - see the note on _FORM_LEVELS below.
_ORIGIN_LEVELS = ("Landing Page Submission", "API", "Lead Add Form", "Lead Import")
_LEAD_SOURCE_LEVELS = (
    "Google", "Direct Traffic", "Olark Chat", "Organic Search", "Reference",
    "Welingak Website",
)
_ACTIVITY_LEVELS = ("Email Opened", "SMS Sent", "Olark Chat Conversation", "Page Visited")
_YES_NO = ("No", "Yes")

#: Per-column levels. One shared list would be simpler and would put specialisations in the
#: city column, which matters because this fixture is also the sample CSV a reviewer
#: downloads from the dashboard - a sample full of nonsense teaches the wrong thing about the
#: data. `Select` stays in the columns where the real form offers it.
_FORM_LEVELS: dict[str, tuple[str, ...]] = {
    "Specialization": (
        "Select", "Finance Management", "Marketing Management", "Operations Management",
        "Business Administration",
    ),
    "What is your current occupation": (
        "Select", "Unemployed", "Working Professional", "Student", "Businessman",
    ),
    "What matters most to you in choosing a course": (
        "Select", "Better Career Prospects", "Flexibility & Convenience", "Other",
    ),
    "City": (
        "Select", "Mumbai", "Thane & Outskirts", "Other Metro Cities", "Pune",
    ),
    "Country": (
        "Select", "India", "United States", "United Arab Emirates", "Singapore",
    ),
    "How did you hear about X Education": (
        "Select", "Online Search", "Word Of Mouth", "Advertisements", "Email",
    ),
}
_FORM_FALLBACK = ("Select", "Other")


def _intercept_for_target_rate(signal: np.ndarray, scale: float, rate: float) -> float:
    """Find the offset that makes the mean conversion probability equal `rate`.

    Bisection rather than algebra: the mean of a logistic has no closed form, and 60
    iterations on a 300-row array costs nothing. Deterministic, so the fixture is stable.
    """
    low, high = float(signal.min()) - 10_000.0, float(signal.max()) + 10_000.0
    for _ in range(60):
        middle = (low + high) / 2
        mean = float(np.mean(1 / (1 + np.exp(-(signal - middle) / scale))))
        if mean > rate:  # too many conversions, push the threshold up
            low = middle
        else:
            high = middle
    return (low + high) / 2


def synthetic_leads(
    rows: int = DEFAULT_ROWS,
    seed: int = DEFAULT_SEED,
    target_rate: float = cfg.EXPECTED_TARGET_RATE,
) -> pd.DataFrame:
    """A frame that satisfies the data contract and carries a learnable synthetic signal."""
    rng = np.random.default_rng(seed)

    time_on_site = rng.integers(0, 2_000, rows).astype(float)
    offset = _intercept_for_target_rate(time_on_site, scale=300.0, rate=target_rate)
    probability = 1 / (1 + np.exp(-(time_on_site - offset) / 300.0))
    target = rng.binomial(1, probability)

    data: dict[str, object] = {
        cfg.TARGET: target,
        "Prospect ID": [f"synthetic-{index:06d}" for index in range(rows)],
        "Lead Number": np.arange(600_000, 600_000 + rows),
        SIGNAL_COLUMN: time_on_site,
    }

    # Numerics, with a little missingness so the imputer is exercised rather than assumed.
    visits = rng.integers(0, 20, rows).astype(float)
    pages = rng.uniform(0, 10, rows).round(2)
    for column, values in (("TotalVisits", visits), ("Page Views Per Visit", pages)):
        values = values.copy()
        values[rng.random(rows) < 0.03] = np.nan
        data[column] = values

    data["Lead Origin"] = rng.choice(_ORIGIN_LEVELS, rows, p=(0.53, 0.31, 0.08, 0.08))
    data["Lead Source"] = rng.choice(_LEAD_SOURCE_LEVELS, rows)

    # `Select` is a form default, not an answer: the fixture must contain it so that
    # `normalise_select` runs on the fixture path too (docs/02 FR-1).
    for column in cfg.FORM_FEATURES:
        levels = _FORM_LEVELS.get(column, _FORM_FALLBACK)
        # `Select` is deliberately the most common, as it is in the real upload.
        weights = [0.4] + [0.6 / (len(levels) - 1)] * (len(levels) - 1)
        data[column] = rng.choice(levels, rows, p=weights)

    for column in cfg.CONTACT_PREFERENCE_FEATURES:
        data[column] = rng.choice(_YES_NO, rows, p=(0.92, 0.08))

    # Near-constant, exactly like the real ad and opt-in flags: report 01 found most of them
    # are 99%+ a single value, and the preprocessor has to survive that.
    for column in cfg.FLAG_FEATURES:
        data[column] = rng.choice(_YES_NO, rows, p=(0.995, 0.005))

    for column in cfg.ACTIVITY_FEATURES:
        data[column] = rng.choice(_ACTIVITY_LEVELS, rows)

    # Leaky on purpose. `Tags` is filled in after a human has spoken to the lead, so it
    # predicts the outcome almost perfectly — which is what the leakage audit demonstrates.
    noise = rng.random(rows) < 0.1
    leaked = np.where(target.astype(bool) ^ noise, "Will revert after reading the email", "Ringing")
    for column in cfg.BANNED_LEAKAGE_COLUMNS:
        data[column] = leaked if column == "Tags" else rng.choice(("01.High", "02.Medium"), rows)

    for column in cfg.DROPPED_SPARSE_COLUMNS:
        data[column] = rng.choice(("Select", "Potential Lead"), rows, p=(0.75, 0.25))

    frame = pd.DataFrame(data)

    # The contract, asserted here rather than discovered downstream. A column added to config
    # without being added to this fixture should fail loudly at the source.
    from leadscore import schema as sch

    absent = sch.missing_columns(frame)
    extra = sch.unclassified_columns(frame)
    if absent or extra:
        raise RuntimeError(
            "the synthetic fixture no longer matches the data contract: "
            f"missing={absent}, unclassified={extra}. Add the column to fixtures.py."
        )
    return frame


def write_fixture(path: Path, rows: int = DEFAULT_ROWS, seed: int = DEFAULT_SEED) -> Path:
    """Write the synthetic frame as a CSV that `load_and_validate` can read."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    synthetic_leads(rows=rows, seed=seed).to_csv(path, index=False)
    return path
