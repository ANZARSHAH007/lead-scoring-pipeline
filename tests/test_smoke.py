"""The end-to-end smoke test (task T-109).

Phase 3 shipped two bugs that neither `ruff` nor the 81 unit tests could have caught, because
both only exist when the whole chain actually runs: the package was not importable as
`python -m leadscore.train`, and MLflow refused the configured tracking store. A unit test
that mocks MLflow would have passed on a broken install.

So this file runs the real chain — validate, split, cross-validate, tune, track, report — on
the synthetic fixture, in a temporary directory, in a few seconds. It asserts almost nothing
about the *numbers*, because the fixture's numbers are meaningless; it asserts that every
stage executes and produces the artifact the next stage expects.

If this test passes and a real training run then fails, something about the environment is
different from CI, and `scripts/doctor.py` is the next thing to run.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import pytest

from leadscore import config as cfg
from leadscore import fixtures, schema, train


@pytest.fixture
def fixture_csv(tmp_path: Path) -> Path:
    return fixtures.write_fixture(tmp_path / "fixture" / "Leads.csv", rows=300)


@pytest.fixture
def settings(tmp_path: Path, fixture_csv: Path) -> cfg.Settings:
    """A full configuration pointed entirely at the temporary directory.

    Nothing here may write to the repository: a fixture run that overwrote
    `reports/03_model_selection.md` would replace real results with synthetic ones, which is
    exactly the kind of quiet corruption the project's rules exist to prevent.
    """
    return cfg.Settings(
        _env_file=None,
        data_raw=fixture_csv,
        reports_dir=tmp_path / "reports",
        model_path=tmp_path / "models" / "model.joblib",
        cv_folds=3,
        optuna_trials=0,  # tuning is covered in test_train; this is about the chain
        mlflow_tracking_uri=f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}",
    )


def test_the_whole_chain_runs_on_the_fixture(settings: cfg.Settings) -> None:
    """validate -> split -> cross-validate -> track -> report, with no real data."""
    with warnings.catch_warnings():
        # The fixture has 300 rows, not 9,240, so the shape check warns by design.
        warnings.simplefilter("ignore", schema.DataDriftWarning)
        results, split = train.run_selection(settings)

    names = {result.name for result in results}
    assert {"random", "one_rule", "logistic_regression"} <= names
    assert any(name.startswith("lightgbm") for name in names)

    assert split.sizes["train"] + split.sizes["test"] == 300
    assert split.sizes["test"] == 60

    # Chance is 0.5 by construction; a fixture with signal in it should clear that.
    best = max(results, key=lambda result: result.mean_auc)
    assert best.mean_auc > 0.6, f"the fixture carries no learnable signal: {best}"


def test_the_report_and_its_figure_are_written(settings: cfg.Settings) -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", schema.DataDriftWarning)
        results, split = train.run_selection(settings)

    report = train.write_report(results, split, settings)

    assert report.exists()
    assert report.parent == settings.reports_dir
    text = report.read_text(encoding="utf-8")
    assert "ROC-AUC" in text
    assert (settings.figures_dir / "model_selection.png").exists()


def test_nothing_in_the_repository_is_touched_by_a_fixture_run(
    settings: cfg.Settings, tmp_path: Path
) -> None:
    """The guard on the guard: every output path must sit under the temporary directory."""
    for path in (settings.reports_dir, settings.figures_dir, settings.model_path, settings.data_raw):
        assert tmp_path in Path(path).resolve().parents or Path(path).resolve() == tmp_path

    # And the default stays the real one, so this test cannot pass by weakening the default.
    assert cfg.Settings(_env_file=None).reports_dir == Path("reports")


def test_mlflow_accepts_a_real_run(settings: cfg.Settings) -> None:
    """The Phase 3 bug, as a test: MLflow 3 rejects a file store, and only a run finds out."""
    mlflow = pytest.importorskip("mlflow")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", schema.DataDriftWarning)
        results, split = train.run_selection(settings)

    tracking = train.log_to_mlflow(results, split, settings)
    assert tracking == settings.mlflow_tracking_uri

    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    runs = mlflow.search_runs(experiment_names=["p1-lead-scoring"])
    assert len(runs) == len(results)
    assert "metrics.cv_roc_auc_mean" in runs.columns


def test_a_file_store_uri_is_not_silently_accepted() -> None:
    """`file:./mlruns` worked in MLflow 2 and raises in 3. Keep the default off that path."""
    assert cfg.Settings(_env_file=None).mlflow_tracking_uri.startswith("sqlite:///")
