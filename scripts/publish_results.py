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
    .venv/Scripts/python scripts/publish_results.py --drift    # fail only if the NUMBERS moved

`--check` demands the README be byte-identical to what the current card generates. That is
the right check on this machine, where the card and the README come from the same run.

`--drift` is what CI runs, and the difference is not a loosening - it is a different
question. CI retrains from scratch, and `model_version` is date-stamped by design, so a
freshly trained card can never equal a committed README on any later day. Run #3 failed on
exactly that and told us nothing about the model (D-033). What is worth asserting after a
retrain is that the *numbers* did not move: same data, same seed, same metrics to within the
slack a different OS and CPU count deserve. The test-set fingerprint is held to a stricter
rule than the metrics - it must be identical, because a changed fingerprint means the
held-out rows moved, and every number in the project is measured on them.
"""

from __future__ import annotations

import argparse
import difflib
import json
import re
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from leadscore import config as cfg  # noqa: E402
from leadscore import evaluate as ev  # noqa: E402

#: How far a retrained metric may move from the published one before CI calls it drift.
#: Absolute, on each metric's own scale: 0.02 is two points of Capture@20%, or 0.02 of
#: ROC-AUC or Brier. A retrain on the same data with the same seed should move far less than
#: this; the slack is for a different OS and a different CPU count, not for a different
#: model. Raise it only with a reason written in docs/14 - a tolerance quietly widened until
#: the build passes is the same failure as a quality gate quietly lowered.
DRIFT_TOLERANCE = 0.02

#: A model version looks like 20261118-6ae06cc8: a date and a hash of the parameters. The
#: date changes on every retrain, so it is reported and never failed on.
VERSION_TOKEN = re.compile(r"^\d{8}-[0-9a-f]{8}$")

_NUMBER = re.compile(r"\d[\d,]*\.?\d*%?")
_TOKEN = re.compile(r"`[^`]*`")

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


def current_block(text: str, start: str, end: str) -> list[str]:
    """The block as the README currently has it, markers included."""
    if start not in text or end not in text:
        raise SystemExit(f"markers {start} / {end} are missing from the README.")
    return text[text.index(start) : text.index(end) + len(end)].splitlines()


def numeric(text: str) -> float:
    """'44.1%' -> 0.441, '1,848' -> 1848.0, '0.865' -> 0.865."""
    raw = text.replace(",", "")
    return float(raw[:-1]) / 100 if raw.endswith("%") else float(raw)


def masked(line: str) -> str:
    """The line with every backticked token blanked out, numbers left alone.

    Tokens are masked before numbers are read, because a fingerprint and a model version are
    full of digits that are not measurements: counting them makes `0000000000000000` look
    like a number that moved, which is how the first version of this got it wrong.
    """
    return _TOKEN.sub("`?`", line)


def skeleton(line: str) -> str:
    """The line with every number and every backticked token blanked out."""
    return _NUMBER.sub("#", masked(line))


def compare(
    published: list[str], fresh: list[str], tolerance: float = DRIFT_TOLERANCE
) -> list[str]:
    """What actually differs between a published block and a freshly generated one.

    Three kinds of difference, treated differently on purpose:

    * **wording or shape** - a row added, a label changed, a calibrator that used to be
      `none` and is now `sigmoid`. Always a problem: the README no longer describes the
      model.
    * **a number** - a problem only if it moved by more than `tolerance`. Floating-point
      arithmetic is not identical across operating systems and thread counts, and failing a
      build over the sixth decimal teaches people to ignore the build.
    * **a backticked token** - the model version is expected to change on every retrain and
      is ignored; anything else backticked, which in practice means the test-set
      fingerprint, must be identical.
    """
    problems: list[str] = []
    if len(published) != len(fresh):
        return [f"the block changed shape: {len(published)} lines published, {len(fresh)} now"]

    for old, new in zip(published, fresh, strict=True):
        if skeleton(old) != skeleton(new):
            problems.append(f"wording changed:\n      published: {old}\n      retrained: {new}")
            continue

        for was, now in zip(
            _NUMBER.findall(masked(old)), _NUMBER.findall(masked(new)), strict=True
        ):
            if abs(numeric(was) - numeric(now)) > tolerance:
                problems.append(
                    f"{was} -> {now} (moved more than {tolerance}) in: {skeleton(old).strip()}"
                )

        for was, now in zip(_TOKEN.findall(old), _TOKEN.findall(new), strict=True):
            if was == now:
                continue
            if VERSION_TOKEN.match(was.strip("`")) and VERSION_TOKEN.match(now.strip("`")):
                continue  # a new model version is what a retrain is for
            problems.append(f"{was} -> {now}: this identifies the held-out rows, so it must match")

    return problems


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
    parser.add_argument(
        "--drift",
        action="store_true",
        help="compare the numbers only, with a tolerance; what CI runs after a retrain",
    )
    args = parser.parse_args()

    settings = cfg.Settings()
    card = load_card(settings)
    readme = ROOT / "README.md"
    current = readme.read_text(encoding="utf-8")

    if args.drift:
        print("  published : README.md as committed")
        print(
            f"  retrained : {cfg.resolve(settings.model_card_path)}, "
            f"version {card['model_version']}"
        )
        print(f"  tolerance : {DRIFT_TOLERANCE} absolute, per metric")
        problems = compare(
            current_block(current, HEADLINE_START, HEADLINE_END), headline_block(card)
        ) + compare(current_block(current, RESULTS_START, RESULTS_END), results_block(card))
        if problems:
            print(f"\n  {len(problems)} difference(s) the retrain should not have produced:")
            for problem in problems:
                print(f"    - {problem}")
            print(
                "\n  Either the data or the pipeline changed. If the new numbers are the "
                "right ones,\n  run `publish_results.py` on a machine with the real dataset "
                "and commit the README."
            )
            return 1
        print("\n  The retrained model reproduces the published numbers.")
        return 0

    updated = replace_block(current, HEADLINE_START, HEADLINE_END, headline_block(card))
    updated = replace_block(updated, RESULTS_START, RESULTS_END, results_block(card))

    if updated == current:
        print("  README already matches the model card.")
        return 0

    if args.check:
        print("  README does not match models/model_card.json. What differs:")
        for line in difflib.unified_diff(
            current.splitlines(), updated.splitlines(), "README.md", "from the card", lineterm="", n=1
        ):
            if line.startswith(("+", "-")) and not line.startswith(("+++", "---")):
                print(f"    {line}")
        print("  Run: .venv/Scripts/python scripts/publish_results.py")
        return 1

    readme.write_text(updated, encoding="utf-8")
    headline = card["headline"]  # type: ignore[index]
    print(f"  README updated from model card {card['model_version']}.")
    print(f"  headline: Capture@20% = {headline['value']:.1%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
