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

import json
import re
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
        # Every path, not just the obvious ones. An earlier version of this fixture left
        # `model_card_path` at its default, and because a relative default resolves against
        # the project root, a test run wrote fixture numbers over the real model card.
        model_card_path=tmp_path / "models" / "model_card.json",
        db_path=tmp_path / "predictions.db",
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
    """The guard on the guard: every path this run could write to must sit under tmp_path.

    Enumerated from the model's own fields rather than listed by hand, because a hand-written
    list is how this went wrong once already: `model_card_path` was missing from it, so a
    fixture run wrote synthetic numbers over the real `models/model_card.json`. A path field
    added to `Settings` in future is covered by this test the day it is added.
    """
    paths = {
        name: getattr(settings, name)
        for name, field in type(settings).model_fields.items()
        if field.annotation is Path
    }
    paths["figures_dir"] = settings.figures_dir  # a property, so not in model_fields
    assert "model_card_path" in paths, "the enumeration stopped finding path fields"

    for name, value in paths.items():
        resolved = cfg.resolve(Path(value))
        assert resolved == tmp_path or tmp_path in resolved.parents, (
            f"{name} points at {resolved}, outside the temporary directory"
        )

    # And the defaults stay the real ones, so this test cannot pass by weakening them.
    defaults = cfg.Settings(_env_file=None)
    assert defaults.reports_dir == Path("reports")
    assert defaults.model_card_path == Path("models/model_card.json")


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


# ---------------------------------------------------------------------------
# Phase 4: the whole publishing chain, on synthetic data (T-401 to T-407)
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def clean_ledger():
    from leadscore import evaluate as ev

    ev.reset_ledger()
    yield
    ev.reset_ledger()


def _shipped(settings: cfg.Settings):
    """Run selection then the whole Phase 4 chain, as `python -m leadscore.train` does."""
    from leadscore import schema as sch

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", schema.DataDriftWarning)
        frame = sch.load_and_validate(settings.data_raw, settings)
        results, split = train.run_selection(settings, frame=frame)
        results.sort(key=lambda result: result.mean_auc, reverse=True)
        return train.ship(frame, split, results[0], settings), split


def test_the_publishing_chain_runs_end_to_end(settings: cfg.Settings) -> None:
    """Calibrate, grade, open the test set once, gate, write the card and the report."""
    outcome, _ = _shipped(settings)

    assert outcome["card_path"].exists()
    assert outcome["report_path"].exists()
    assert (settings.figures_dir / "gains.png").exists()
    assert (settings.figures_dir / "reliability.png").exists()
    assert (settings.figures_dir / "decile_lift.png").exists()

    report = outcome["report_path"].read_text(encoding="utf-8")
    assert "Capture@20%" in report
    assert "The test set was opened once" in report


def test_the_held_out_rows_are_read_exactly_once(settings: cfg.Settings) -> None:
    """The Phase 4 discipline, as a test rather than a promise."""
    from leadscore import evaluate as ev

    _shipped(settings)
    assert len(ev.ledger()) == 1
    assert ev.ledger()[0]["reader"] == "evaluate_on_test"


def test_the_card_carries_the_numbers_the_readme_publishes(settings: cfg.Settings) -> None:
    outcome, split = _shipped(settings)
    card = json.loads(outcome["card_path"].read_text(encoding="utf-8"))

    assert card["model_version"] == outcome["version"]
    assert card["data"]["rows_test"] == split.sizes["test"]
    assert card["data"]["test_set_openings"] == 1
    assert card["data"]["test_set_fingerprint"]
    assert len(card["decile_lift"]) == 10
    assert len(card["quality_gate"]["checks"]) == 5
    assert card["grades"]["A_from_score"] > card["grades"]["B_from_score"]
    assert card["calibration"]["method"] in {"none", "sigmoid", "isotonic"}
    assert card["features"]["admissible_source_columns"] == list(cfg.shipping_features())

    names = {row["name"] for row in card["test_metrics"]}
    assert {"random", "one_rule", "logistic_regression", "lightgbm_shipped",
            "lightgbm_leakage"} == names


def test_the_leakage_row_is_measured_and_not_shipped(settings: cfg.Settings) -> None:
    outcome, _ = _shipped(settings)
    card = json.loads(outcome["card_path"].read_text(encoding="utf-8"))
    rows = {row["name"]: row for row in card["test_metrics"]}
    assert rows["lightgbm_leakage"]["roc_auc"] > rows["lightgbm_shipped"]["roc_auc"]
    assert "Tags" in card["features"]["excluded"]["banned_for_leakage"]


def test_a_failing_gate_does_not_write_a_model(settings: cfg.Settings) -> None:
    """On the fixture the tree model loses to the one-rule baseline, so the gate fails.

    That is the fixture's doing, not the model's — and it makes this the cheapest possible
    test of the rule that matters: a model that fails the gate must not reach `models/`.
    """
    outcome, _ = _shipped(settings)
    if outcome["gate"].passed:
        pytest.skip("the fixture's model passed the gate; the refusal path is untested here")
    assert outcome["model_path"] is None
    assert not settings.model_path.exists()
    assert "FAILED" in outcome["report_path"].read_text(encoding="utf-8")


def test_a_passing_gate_writes_a_loadable_bundle(settings: cfg.Settings, monkeypatch) -> None:
    """With the gate satisfied, the artifact must be written and must load back.

    The gate is stubbed to pass rather than the thresholds lowered: the question here is
    whether the writing and loading work, and a test that moved a threshold to get there
    would be testing something else.
    """
    from leadscore import evaluate as ev
    from leadscore import explain as ex

    monkeypatch.setattr(
        ev,
        "quality_gate",
        lambda **_: ev.GateResult(
            [ev.GateCheck("stubbed for this test", True, 1.0, 1.0, "see the docstring")]
        ),
    )
    outcome, split = _shipped(settings)

    assert outcome["model_path"] is not None
    bundle = ev.load_model_bundle(settings)
    assert set(bundle) == {"model", "explainer_model", "card"}
    assert bundle["card"]["model_version"] == outcome["version"]

    # The round-tripped model scores, and its explanations still read as sentences.
    scores = bundle["model"].predict_proba(split.x_test.head(3))[:, 1]
    assert ((scores >= 0) & (scores <= 1)).all()
    reasons = ex.Explainer(bundle).reasons(split.x_test.iloc[[0]])
    assert len(reasons) == 3
    assert all(reason.phrase for reason in reasons)


def test_a_missing_model_says_what_to_run(settings: cfg.Settings) -> None:
    from leadscore import evaluate as ev

    with pytest.raises(FileNotFoundError, match=re.escape("leadscore.train")):
        ev.load_model_bundle(settings)
