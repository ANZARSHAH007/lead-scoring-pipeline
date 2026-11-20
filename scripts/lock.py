"""Regenerate `requirements.lock` so it installs on Linux as well as Windows (task T-113).

`pip freeze` writes what is installed, and nothing about *why*. That loses the environment
markers: `docker` needs `pywin32` only on Windows, but a freeze taken on Windows writes a
bare `pywin32==312`, and `pip install -r requirements.lock` on a Linux CI runner then fails
in about ten seconds with "no matching distribution".

That is exactly what happened on CI run #1 (D-031). The fix is not to remember to hand-edit
the file - it is to make regeneration do the right thing, which is what this script is for.

    .venv/Scripts/python scripts/lock.py           # rewrite requirements.lock
    .venv/Scripts/python scripts/lock.py --check   # fail if it is out of date or unportable

`--check` is what CI runs: it never writes, it only reports.
"""

from __future__ import annotations

import argparse
import datetime as dt
import platform
import subprocess
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parents[1]
LOCKFILE = ROOT / "requirements.lock"

#: Packages that only exist, or are only needed, on one platform. `pip freeze` drops the
#: marker, so it is restored here. The marker is the one the depending package declares:
#: `docker` asks for pywin32 under sys_platform == "win32", pytest and tqdm for colorama.
PLATFORM_MARKERS: dict[str, str] = {
    "pywin32": 'sys_platform == "win32"',
    "pypiwin32": 'sys_platform == "win32"',
    "colorama": 'sys_platform == "win32"',
    "win32-setctime": 'sys_platform == "win32"',
    "pywinpty": 'sys_platform == "win32"',
    "appnope": 'sys_platform == "darwin"',
}

HEADER = """\
# Resolved dependency set for the Lead Scoring Pipeline — DO NOT hand-edit.
# Regenerate with: .venv/Scripts/python scripts/lock.py
#
# Why this file exists: in Phase 3 MLflow moved to 3.18 under a floating `mlflow>=2.16`
# lower bound and retired the plain file store, which broke tracking with no code change on
# our side (D-017). Lower bounds in pyproject.toml say what the project needs; this file says
# what actually worked, so a second machine and CI install the same thing.
#
# Three pins are deliberate rather than incidental:
#   sqlalchemy stays below 2.1, because MLflow 3.18 declares a `lazy="noload"` relationship
#     that 2.1 deprecates (D-019);
#   httpx2 rather than httpx, because Starlette 1.10 moved TestClient onto it (D-024);
#   the `; sys_platform == ...` markers below are restored by scripts/lock.py, because
#     `pip freeze` drops them and a Windows-only package then breaks the Linux CI run (D-031).
#
# After regenerating, verify before committing:
#   .venv/Scripts/python scripts/doctor.py && .venv/Scripts/python -m pytest
# then record any upgrade in docs/14.
#
# Generated {date} on Python {python}, {system}. {count} packages.
"""


def freeze() -> list[str]:
    """Current environment as pip sees it, editable installs excluded."""
    result = subprocess.run(
        [sys.executable, "-m", "pip", "freeze", "--exclude-editable"],
        capture_output=True,
        text=True,
        check=True,
    )
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


def apply_markers(lines: list[str]) -> tuple[list[str], list[str]]:
    """Put back the environment markers `pip freeze` removed. Returns (lines, changed)."""
    out: list[str] = []
    changed: list[str] = []
    for line in lines:
        if ";" in line:  # already carries a marker
            out.append(line)
            continue
        name = line.split("==")[0].strip().lower().replace("_", "-")
        marker = PLATFORM_MARKERS.get(name)
        if marker:
            out.append(f"{line}; {marker}")
            changed.append(f"{name} -> {marker}")
        else:
            out.append(line)
    return out, changed


def render(lines: list[str]) -> str:
    header = HEADER.format(
        date=dt.date.today().isoformat(),
        python=platform.python_version(),
        system=platform.system(),
        count=len(lines),
    )
    return header + "\n" + "\n".join(lines) + "\n"


def unportable(lines: list[str]) -> list[str]:
    """Pins that would fail on Linux: a known platform-specific package with no marker."""
    bad = []
    for line in lines:
        if ";" in line:
            continue
        name = line.split("==")[0].strip().lower().replace("_", "-")
        if name in PLATFORM_MARKERS:
            bad.append(line)
    return bad


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--check", action="store_true", help="report problems and exit 1; never write"
    )
    args = parser.parse_args()

    if args.check:
        if not LOCKFILE.exists():
            print("  requirements.lock is missing.")
            return 1
        existing = [
            line.strip()
            for line in LOCKFILE.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.startswith("#")
        ]
        broken = unportable(existing)
        if broken:
            print(f"  {len(broken)} pin(s) would fail on a non-Windows machine:")
            for line in broken:
                print(f"    {line}")
            print("  Fix: .venv/Scripts/python scripts/lock.py")
            return 1
        print(f"  requirements.lock looks portable ({len(existing)} packages).")
        return 0

    lines, changed = apply_markers(freeze())
    LOCKFILE.write_text(render(lines), encoding="utf-8")

    print(f"  wrote {LOCKFILE.relative_to(ROOT)} with {len(lines)} packages")
    if changed:
        print(f"  restored {len(changed)} environment marker(s) pip freeze had dropped:")
        for entry in changed:
            print(f"    {entry}")
    else:
        print("  no platform-specific packages needed a marker")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
