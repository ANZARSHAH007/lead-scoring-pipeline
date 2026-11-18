"""Fill the README's headline and results table from the model card (task T-407).

The release checklist in docs/12 has a line that says "README results table matches the
card". A human checking that by eye will be right most times and wrong once, and the once is
the version that gets read by someone deciding whether to trust the project. So the README is
generated from `models/model_card.json` instead of transcribed from it.

Three things are replaced, each between explicit markers so the surrounding prose stays
hand-written:

* the headline sentence - Capture@20% for the shipping model;
* the five-row results table;
* the "measured on" line - model version, held-out row count and fingerprint.

    .venv/Scripts/python scripts/publish_results.py            # rewrite the README
    .venv/Scripts/python scripts/publish_results.py --check    # fail if it is out of date

`--check` is what CI runs: it never edits, it only reports whether the published numbers
still match the card.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from leadscore import config as cfg  # noqa: E402
from leadscore import evaluate as ev  # noqa: E402

HEADLINE_START = "<!-- headline:start -->"
HEADLINE_END = "<!-- headline:end -->"
RESULTS_START = "<!-- results:start -->"
RESULTS_END = "<!-- results:end -->"


def load_card(settings: cfg.Settings) -> dict[str, object]:
    path = cfg.resolve(settings.model_card_path)
    if not path.exists():
        raise SystemExit(
            f"no model card at {path}. Run `.venv/Scripts/python -m leadscore.train` first — "
            "the README's numbers are only ever written from a real run."
        )
    return json.loads(path.read_text(encoding="utf-8"))


def headline_block(card: dict[str, object]) -> list[str]:
    headline = card["headline"]  # type: ignore[index]
    data = card["data"]  # type: ignore[index]
    gate = card["quality_gate"]  # type: ignore[index]
    value = headline["value"]
    return [
        HEADLINE_START,
        f"> If sales calls only the top 20% of scored leads, they reach **{value:.1%}** of all "
        "conversions.",
        "",
        f"Measured once on {data['rows_test']:,} held-out leads the model never saw "
        f"(fingerprint `{data['test_set_fingerprint']}`), by model version "
        f"`{card['model_version']}`. Quality gate: "
        f"**{'passed' if gate['passed'] else 'failed'}**.",
        HEADLINE_END,
    ]


def results_block(card: dict[str, object]) -> list[str]:
    metrics = [
        ev.Metrics(
            name=str(row["name"]),
            roc_auc=float(row["roc_auc"]),
            pr_auc=float(row["pr_auc"]),
            brier=float(row["brier"]),
            capture_at_20=float(row["capture_at_20"]),
            rows=int(row["rows"]),
            positives=int(row["positives"]),
        )
        for row in card["test_metrics"]  # type: ignore[union-attr]
    ]
    method = str(card["calibration"]["method"])  # type: ignore[index]
    return [RESULTS_START, *ev.results_table(metrics, method), RESULTS_END]


def replace_block(text: str, start: str, end: str, lines: list[str]) -> str:
    if start not in text or end not in text:
        raise SystemExit(
            f"markers {start} / {end} are missing from the README. Add them around the block "
            "this script owns, so the hand-written prose around it is never overwritten."
        )
    head = text[: text.index(start)]
    tail = text[text.index(end) + len(end) :]
    return head + "\n".join(lines) + tail


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--check", action="store_true", help="report drift and exit 1; never edit the README"
    )
    args = parser.parse_args()

    settings = cfg.Settings()
    card = load_card(settings)
    readme = ROOT / "README.md"
    current = readme.read_text(encoding="utf-8")

    updated = replace_block(current, HEADLINE_START, HEADLINE_END, headline_block(card))
    updated = replace_block(updated, RESULTS_START, RESULTS_END, results_block(card))

    if updated == current:
        print("  README already matches the model card.")
        return 0

    if args.check:
        print("  README does not match models/model_card.json.")
        print("  Run: .venv/Scripts/python scripts/publish_results.py")
        return 1

    readme.write_text(updated, encoding="utf-8")
    headline = card["headline"]  # type: ignore[index]
    print(f"  README updated from model card {card['model_version']}.")
    print(f"  headline: Capture@20% = {headline['value']:.1%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
