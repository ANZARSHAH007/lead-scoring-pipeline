# Lead Scoring Pipeline — one entry point per action.
#
# `make` is not installed on the owner's Windows machine, so every target below is a single
# line that can be pasted into a shell by hand; the equivalents are listed in the README.
# CI runs on Linux, where make exists.

PY ?= python
VENV ?= .venv
ifeq ($(OS),Windows_NT)
	BIN := $(VENV)/Scripts
else
	BIN := $(VENV)/bin
endif

.PHONY: help venv install install-min lock doctor lint format test test-cov smoke train publish serve dashboard docker clean

help:
	@echo "venv         create $(VENV) with $(PY)"
	@echo "install      install the project and dev tools (full ML stack, large)"
	@echo "install-min  install only what the current phase needs (config + tests)"
	@echo "lock         rewrite requirements.lock from the current environment"
	@echo "doctor       check this machine can run the pipeline, in ~2 seconds"
	@echo "lint         ruff check"
	@echo "format       ruff format"
	@echo "test         pytest"
	@echo "smoke        run the whole chain on the synthetic fixture, no real data"
	@echo "test-cov     pytest with coverage"
	@echo "train        train, evaluate once, write models/ + reports/  [Phase 3+]"
	@echo "publish      fill the README numbers from the model card     [Phase 4+]"
	@echo "serve        run the API on :8000                     [Phase 5+]"
	@echo "dashboard    run the Streamlit dashboard on :8501     [Phase 6+]"
	@echo "docker       build and run both services              [Phase 6+]"

venv:
	$(PY) -m venv $(VENV)
	$(BIN)/python -m pip install --upgrade pip

# Full install. Pulls lightgbm, mlflow, shap, streamlit and friends — hundreds of MB.
install:
	$(BIN)/python -m pip install -e ".[dev]"

# Phase 1 subset: enough to import the config and run the suite, nothing more.
install-min:
	$(BIN)/python -m pip install "pydantic>=2.9" "pydantic-settings>=2.5" "pytest>=8.3" "ruff>=0.7"

# Pinned resolution of what actually works. Regenerate deliberately: a dependency that
# moves on its own is how MLflow 3.18 broke tracking with no change on our side (D-017).
lock:
	$(BIN)/python -m pip freeze --exclude-editable > requirements.lock

# Run this before training and after any dependency change. Two seconds, and it answers
# what a three-minute training run would otherwise tell you at the end.
doctor:
	$(BIN)/python scripts/doctor.py

lint:
	$(BIN)/ruff check src tests scripts

format:
	$(BIN)/ruff format src tests scripts

test:
	$(BIN)/python -m pytest

test-cov:
	$(BIN)/python -m pytest --cov=leadscore --cov-report=term-missing

# The whole pipeline on 300 synthetic rows: catches the failures that only appear when the
# chain actually runs, in seconds rather than minutes. This is what CI runs on every push.
smoke:
	$(BIN)/python -m pytest tests/test_smoke.py -v

train:
	$(BIN)/python -m leadscore.train

# Generated, never transcribed: the README's headline and results table come from
# models/model_card.json. `--check` is the CI form, which reports drift and edits nothing.
publish:
	$(BIN)/python scripts/publish_results.py

serve:
	$(BIN)/python -m uvicorn leadscore.api.main:app --host 0.0.0.0 --port 8000 --reload

dashboard:
	$(BIN)/python -m streamlit run dashboard/app.py

docker:
	docker compose up --build

clean:
	rm -rf .pytest_cache .ruff_cache **/__pycache__ .coverage
