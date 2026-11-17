"""Preflight check: is this machine able to run the pipeline? (task T-108)

Run this before a training run, after any dependency change, and first thing in a new
session. It answers in about two seconds what a training run would otherwise tell you three
minutes in, which is how two of Phase 3's bugs were actually found:

* `python -m leadscore.train` failed on import, because the package was never installed;
* MLflow 3.18 refused the `file:./mlruns` store that the config still asked for.

Both were environment facts, not logic errors, and both were invisible to `pytest` and
`ruff`. This script exists so that class of problem is reported by name, up front, with the
command that fixes it.

Deliberately dependency-light where it can be: the first three checks use nothing but the
standard library, so a broken install still produces a readable diagnosis rather than a
traceback. Later checks import the ML stack, and say so when it is missing.

    .venv/Scripts/python scripts/doctor.py
    .venv/Scripts/python scripts/doctor.py --strict   # warnings count as failures

Exit code 0 if the pipeline can run, 1 if it cannot.
"""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import logging
import os
import sys
import tempfile
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # Windows consoles are cp1252

OK = "ok  "
WARN = "warn"
BAD = "FAIL"

MIN_PYTHON = (3, 11)
ROOT = Path(__file__).resolve().parents[1]

#: Import name -> what breaks without it. Checked by import, not by `pip list`, because a
#: package that is listed but not importable is exactly the failure mode being hunted.
REQUIRED = {
    "pandas": "loading the dataset",
    "pandera": "the data contract",
    "sklearn": "the preprocessor and the baselines",
    "lightgbm": "the challenger model",
    "optuna": "tuning (set LEADSCORE_OPTUNA_TRIALS=0 to skip)",
    "mlflow": "experiment tracking",
    "matplotlib": "report figures",
    "joblib": "saving the model",
}

OPTIONAL = {
    "fastapi": "the API (Phase 5)",
    "uvicorn": "serving the API (Phase 5)",
    "streamlit": "the dashboard (Phase 6)",
    "shap": "score explanations (Phase 4)",
}


class Report:
    """Collects results so every check runs, instead of stopping at the first failure."""

    def __init__(self) -> None:
        self.failures: list[str] = []
        self.warnings: list[str] = []

    def line(self, status: str, label: str, detail: str = "", fix: str = "") -> None:
        print(f"  [{status}] {label}" + (f" — {detail}" if detail else ""))
        if fix:
            print(f"         fix: {fix}")
        if status == BAD:
            self.failures.append(label)
        elif status == WARN:
            self.warnings.append(label)


def check_python(report: Report) -> None:
    version = sys.version_info
    actual = f"{version.major}.{version.minor}.{version.micro}"
    if version[:2] < MIN_PYTHON:
        report.line(
            BAD,
            "Python version",
            f"{actual}, need >= {MIN_PYTHON[0]}.{MIN_PYTHON[1]}",
            "install a newer Python and recreate .venv",
        )
    else:
        report.line(OK, "Python version", actual)

    in_venv = sys.prefix != sys.base_prefix
    if in_venv:
        report.line(OK, "virtualenv", Path(sys.prefix).name)
    else:
        report.line(
            WARN,
            "virtualenv",
            "running against the system Python",
            ".venv/Scripts/python scripts/doctor.py",
        )


def check_package_installed(report: Report) -> bool:
    """The bug that cost a run in Phase 3: the package was importable from the repo root only."""
    spec = importlib.util.find_spec("leadscore")
    if spec is None or not spec.origin:
        report.line(
            BAD,
            "leadscore importable",
            "not installed",
            '.venv/Scripts/python -m pip install -e . --no-deps',
        )
        return False

    origin = Path(spec.origin).resolve()
    expected = (ROOT / "src" / "leadscore" / "__init__.py").resolve()
    if origin != expected:
        report.line(
            WARN,
            "leadscore importable",
            f"resolves to {origin}, not this working tree",
            ".venv/Scripts/python -m pip install -e . --no-deps",
        )
        return True

    report.line(OK, "leadscore importable", "editable, from src/")
    return True


def check_dependencies(report: Report) -> None:
    missing = []
    for name, why in REQUIRED.items():
        if importlib.util.find_spec(name) is None:
            missing.append(f"{name} ({why})")
    if missing:
        report.line(
            BAD,
            "required packages",
            f"{len(missing)} missing: " + ", ".join(missing),
            '.venv/Scripts/python -m pip install -e ".[dev]"',
        )
    else:
        report.line(OK, "required packages", f"all {len(REQUIRED)} importable")

    absent = [f"{n} ({w})" for n, w in OPTIONAL.items() if importlib.util.find_spec(n) is None]
    if absent:
        report.line(WARN, "later-phase packages", ", ".join(absent), "install when that phase starts")
    else:
        report.line(OK, "later-phase packages", f"all {len(OPTIONAL)} importable")


def check_settings(report: Report):
    try:
        from leadscore import config as cfg

        settings = cfg.Settings()
    except Exception as error:
        report.line(BAD, "settings load", f"{type(error).__name__}: {error}", "check .env / .env.local")
        return None
    report.line(OK, "settings load", f"seed {settings.random_seed}, {settings.cv_folds}-fold CV")
    return settings


def check_lockfile(report: Report) -> None:
    """A floating dependency is how MLflow changed under this project without a code change."""
    lock = ROOT / "requirements.lock"
    if not lock.exists():
        report.line(
            WARN,
            "lockfile",
            "requirements.lock missing — installs are not reproducible",
            ".venv/Scripts/python -m pip freeze --exclude-editable > requirements.lock",
        )
        return

    pinned = [
        line
        for line in lock.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith("#")
    ]
    report.line(OK, "lockfile", f"{len(pinned)} pinned packages")


def check_mlflow_store(report: Report, settings) -> None:
    """MLflow 3 removed the plain file store. Prove the configured URI actually accepts a run."""
    if settings is None or importlib.util.find_spec("mlflow") is None:
        report.line(WARN, "MLflow store", "skipped — mlflow not importable")
        return

    uri = settings.mlflow_tracking_uri
    if uri.startswith("file:") or uri.startswith("./") or uri == "mlruns":
        report.line(
            BAD,
            "MLflow store",
            f"{uri} is a plain file store, retired in MLflow 3",
            "set MLFLOW_TRACKING_URI=sqlite:///mlflow.db",
        )
        return

    # Both before the import: the agent hint and the migration chatter are emitted on load.
    os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
    logging.getLogger("mlflow").setLevel(logging.WARNING)
    mlflow = importlib.import_module("mlflow")
    for noisy in ("mlflow.store.db.utils", "mlflow.tracking.fluent", "alembic"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    # ignore_cleanup_errors: MLflow keeps its SQLAlchemy pool open, and Windows will not
    # delete a file that is still held. The probe has already proved what it needed to.
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        probe = f"sqlite:///{Path(tmp).as_posix()}/doctor.db"
        try:
            mlflow.set_tracking_uri(probe)
            mlflow.set_experiment("doctor-probe")
            with mlflow.start_run(run_name="probe"):
                mlflow.log_metric("probe", 1.0)
        except Exception as error:
            report.line(
                BAD,
                "MLflow store",
                f"a run could not be written: {type(error).__name__}: {error}",
                "check the MLflow version against requirements.lock",
            )
            return
        finally:
            mlflow.set_tracking_uri(uri)  # never leave the process pointed at the probe
    report.line(OK, "MLflow store", f"{uri}, a probe run wrote cleanly")


def check_dataset(report: Report, settings) -> None:
    if settings is None:
        return
    path = settings.data_raw if settings.data_raw.is_absolute() else ROOT / settings.data_raw
    if not path.exists():
        report.line(
            WARN,
            "dataset",
            f"{settings.data_raw} not found — training cannot run, tests still can",
            "download Leads.csv per data/raw/README.md",
        )
        return
    size_mb = path.stat().st_size / 1_048_576
    report.line(OK, "dataset", f"{settings.data_raw} present, {size_mb:.1f} MB")


def check_fixture(report: Report) -> None:
    """If the synthetic fixture stops matching the contract, CI loses its only dataset."""
    if importlib.util.find_spec("pandas") is None:
        report.line(WARN, "synthetic fixture", "skipped — pandas not importable")
        return
    try:
        import warnings

        from leadscore import fixtures, schema

        frame = fixtures.synthetic_leads(rows=120)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", schema.DataDriftWarning)
            schema.validate(frame)
    except Exception as error:
        report.line(
            BAD,
            "synthetic fixture",
            f"{type(error).__name__}: {error}",
            "update src/leadscore/fixtures.py to match config",
        )
        return
    report.line(OK, "synthetic fixture", f"{frame.shape[0]} rows x {frame.shape[1]} cols, validates")


def check_write_targets(report: Report, settings) -> None:
    if settings is None:
        return
    for label, directory in (
        ("reports", settings.reports_dir),
        ("models", settings.model_path.parent),
    ):
        target = directory if directory.is_absolute() else ROOT / directory
        try:
            target.mkdir(parents=True, exist_ok=True)
            probe = target / ".doctor-write-probe"
            probe.write_text("probe", encoding="utf-8")
            probe.unlink()
        except OSError as error:
            report.line(BAD, f"{label}/ writable", str(error), "check permissions")
            return
    report.line(OK, "output directories", "reports/ and models/ are writable")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--strict", action="store_true", help="treat warnings as failures")
    args = parser.parse_args()

    print("\n  Lead Scoring Pipeline — environment check")
    print("  " + "-" * 56)

    report = Report()
    check_python(report)
    installed = check_package_installed(report)
    check_dependencies(report)
    check_lockfile(report)

    settings = check_settings(report) if installed else None
    if installed:
        check_fixture(report)
        check_mlflow_store(report, settings)
        check_dataset(report, settings)
        check_write_targets(report, settings)

    print("  " + "-" * 56)
    if report.failures:
        print(f"  {len(report.failures)} blocking problem(s): " + ", ".join(report.failures))
        print("  The pipeline will not run until these are fixed.\n")
        return 1
    if report.warnings and args.strict:
        print(f"  {len(report.warnings)} warning(s), --strict: " + ", ".join(report.warnings) + "\n")
        return 1
    if report.warnings:
        print(f"  Ready, with {len(report.warnings)} warning(s): " + ", ".join(report.warnings) + "\n")
        return 0
    print("  Everything checks out.\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
