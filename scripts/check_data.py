"""Verify that data/raw/Leads.csv is the dataset this pipeline was designed against.

Task T-107. Phase 2 does not start until this exits 0.

Uses only the standard library, so it runs before the ML stack is installed. It imports
``leadscore.config`` so that the column names it checks are the same ones the pipeline will
use: if an upload renames a column, this fails here rather than halfway through training.

    .venv/Scripts/python scripts/check_data.py        # Windows
    .venv/bin/python scripts/check_data.py            # macOS / Linux
"""

from __future__ import annotations

import csv
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from leadscore import config as cfg  # noqa: E402  (path set above)

OK = "PASS"
BAD = "FAIL"
WARN = "WARN"

#: Documented emptiness, counting `Select` as missing (docs/02). Used as a warning, not a gate:
#: a few points of drift is a different upload, not necessarily a wrong one.
DOCUMENTED_MISSING_RATES = {
    "Lead Quality": 0.516,
    "Tags": 0.363,
    "What is your current occupation": 0.291,
    "Country": 0.266,
    "Specialization": 0.156,
    "Lead Profile": 0.742,
    "How did you hear about X Education": 0.785,
}

results: list[tuple[str, str, str]] = []


def record(status: str, check: str, detail: str) -> None:
    results.append((status, check, detail))


def is_missing(value: str) -> bool:
    return value.strip() == "" or value.strip() == cfg.SELECT_PLACEHOLDER


def main() -> int:
    settings = cfg.Settings(_env_file=None)
    path = ROOT / settings.data_raw

    if not path.exists():
        print(f"\n  Not found: {path}\n")
        print("  The dataset is not in this repo on purpose. To get it:")
        print(f"    {ROOT / 'data' / 'raw' / 'README.md'}\n")
        print("  Short version: download the X Education lead-scoring dataset from Kaggle,")
        print("  then put Leads.csv at data/raw/Leads.csv and run this again.\n")
        return 1

    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        header = reader.fieldnames or []
        rows = list(reader)

    # --- shape ---
    record(
        OK if len(rows) == cfg.EXPECTED_ROWS else BAD,
        "row count",
        f"{len(rows):,} (expected {cfg.EXPECTED_ROWS:,})",
    )
    record(
        OK if len(header) == cfg.EXPECTED_COLUMNS else BAD,
        "column count",
        f"{len(header)} (expected {cfg.EXPECTED_COLUMNS})",
    )

    # --- the check that actually matters: do our column names exist in this file? ---
    present = set(header)
    required = set(cfg.shipping_features()) | set(cfg.excluded_columns()) | {cfg.TARGET}
    absent = sorted(required - present)
    record(
        OK if not absent else BAD,
        "columns named in config exist",
        "all present" if not absent else f"{len(absent)} missing: {', '.join(absent[:6])}",
    )
    unexpected = sorted(present - required)
    if unexpected:
        record(WARN, "columns not named in config", ", ".join(unexpected[:6]))

    # --- target ---
    if cfg.TARGET not in present:
        record(BAD, "target column", f"{cfg.TARGET!r} is not in the file")
        return report()

    values = Counter(row[cfg.TARGET].strip() for row in rows)
    stray = sorted(set(values) - {"0", "1"})
    record(
        OK if not stray else BAD,
        "target values are 0/1",
        "clean" if not stray else f"also found: {', '.join(stray[:5])}",
    )

    converted = values.get("1", 0)
    rate = converted / len(rows) if rows else 0.0
    drift = abs(rate - cfg.EXPECTED_TARGET_RATE)
    record(
        OK if drift <= settings.target_rate_tolerance else BAD,
        "conversion rate",
        f"{rate:.4f} ({converted:,} of {len(rows):,}); expected "
        f"{cfg.EXPECTED_TARGET_RATE:.4f}, drift {drift:.4f}",
    )

    # --- sparseness, reported for information ---
    for column, documented in DOCUMENTED_MISSING_RATES.items():
        if column not in present:
            continue
        empty = sum(1 for row in rows if is_missing(row.get(column) or ""))
        actual = empty / len(rows)
        close = abs(actual - documented) <= 0.05
        record(
            OK if close else WARN,
            f"missing rate: {column}",
            f"{actual:.1%} (docs say {documented:.1%})",
        )

    return report()


def report() -> int:
    width = max(len(check) for _, check, _ in results)
    print()
    for status, check, detail in results:
        print(f"  [{status}] {check.ljust(width)}  {detail}")

    failures = sum(1 for status, _, _ in results if status == BAD)
    warnings = sum(1 for status, _, _ in results if status == WARN)
    print()
    if failures:
        print(f"  {failures} check(s) FAILED, {warnings} warning(s).")
        print("  Do not start Phase 2. Either the wrong file was downloaded, or docs/02 was")
        print("  written from a different upload. Resolve it and record the finding (docs/13, A-1).")
        return 1

    print(f"  All checks passed, {warnings} warning(s). T-107 is met; Phase 2 can start.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
