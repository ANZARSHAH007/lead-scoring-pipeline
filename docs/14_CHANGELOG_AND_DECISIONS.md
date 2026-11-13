# 14 — Changelog and Decisions

Two sections. **Decisions** are permanent: each records what was chosen, what was rejected, and why, so a future session does not re-litigate it. **Changelog** records what actually changed, dated.

---

## Decisions

### D-001 — The deliverable is a pipeline, not an application
**Date:** 2026-11-13 · **Status:** accepted

**Context.** The initial framing was "an app". The owner corrected it: the target is an AI pipeline.

**Decision.** Weight the project and its docs toward data contract, training, evaluation, calibration, explanation, serving and the retraining loop. The Streamlit dashboard is a demo surface with no business logic.

**Rejected.** A UI-first build. It would have produced a nice-looking page over an unmeasured model, which is the exact failure mode the roadmap's gap table is designed to avoid.

**Consequence.** Doc 08 is short and honest about Streamlit's limits; docs 02, 03, 05, 06 and 10 carry the weight. The dashboard never imports the model — it speaks HTTP, like a CRM would.

---

### D-002 — Training and serving are split by one file
**Date:** 2026-11-13 · **Status:** accepted

**Decision.** Training writes `models/model.joblib` + `model_card.json`; serving only loads them. The two lanes share no code path beyond the fitted pipeline object.

**Why.** Retraining never touches the API. Rollback is swapping one artifact. The model is immutable at runtime, so instances cannot disagree.

**Rejected.** Training inside the API process (would couple deploy to retrain, and make rollback a code change); a model registry service (nothing to register — two files and MLflow are enough at this size).

---

### D-003 — All preprocessing lives inside the fitted pipeline
**Date:** 2026-11-13 · **Status:** accepted

**Decision.** Cleaning and encoding sit in a scikit-learn `ColumnTransformer` that is saved *with* the model. The API never transforms a raw column by hand.

**Why.** Training/serving skew is the most common silent failure in deployed ML, and it is structurally impossible if there is only one transformer and it travels with the model.

**Consequence.** An unseen category is handled by the transformer (`→ Other`), not by API code. Feature names round-trip through save/load, so SHAP and `/model-info` agree.

---

### D-004 — Leakage control is a product requirement, not a modelling detail
**Date:** 2026-11-13 · **Status:** accepted

**Decision.** A column is admissible only if known at scoring time. `Tags`, `Lead Quality` and the four `Asymmetrique` columns are banned from the shipping model and appear only in a labelled, unshippable "leakage row" of the results table. The audit records a yes/no decision for all 37 columns.

**Why.** Those columns are filled in by sales *after* contacting the lead, so a model using them scores leads it could never have scored in reality. Many public notebooks on this dataset report high accuracy this way. Publishing both numbers is the single most credible thing this repo can do.

**Consequence.** The headline number is lower than published notebooks' — and true. The README says why, so the comparison does not read as underperformance.

---

### D-005 — Capture@20% is the headline metric
**Date:** 2026-11-13 · **Status:** accepted

**Decision.** ROC-AUC, PR-AUC and Brier are reported; **Capture@20%** leads.

**Why.** A sales manager does not act on AUC. "Call the top 20% and you reach X% of conversions" is a decision. The one-rule baseline (sort by time on site) is the honest bar, because it is what the team would otherwise do by hand.

---

### D-006 — Calibration is required, not optional
**Date:** 2026-11-13 · **Status:** accepted

**Decision.** Check the reliability curve and the Brier score; wrap in `CalibratedClassifierCV` if the model is mis-confident. Grade cut-offs come from validation data and are stored in the model card.

**Why.** The product is a 0–100 *score*. If 70 does not mean roughly a 70% chance, the number is decoration and the grade bands are arbitrary.

**Consequence.** The dashboard reads bands from `/model-info`; a band is never hardcoded in UI code.

---

### D-007 — The quality gate runs as a test in CI
**Date:** 2026-11-13 · **Status:** accepted

**Decision.** A retrained model ships only if test ROC-AUC beats logistic regression and Brier does not worsen. The gate is a `pytest` test, so CI fails on a model regression exactly as it fails on a broken import. A shuffled-label model must be rejected, which tests the gate itself.

**Rejected.** Eyeballing metrics in a notebook — the thing that makes this not a notebook project is that the build can fail.

**Rule.** Report the numbers; never raise the gate to make a model pass.

---

### D-008 — Two tables in the prediction log, with `input_json` stored whole
**Date:** 2026-11-13 · **Status:** accepted

**Decision.** `predictions` and `feedback` as separate tables; the request payload kept as a JSON blob; `model_version` on every row.

**Why.** A score is a fact at a time; an outcome arrives later and may never arrive — so feedback must not rewrite history. The admissible feature list will change between model versions, and a JSON blob survives that without a schema migration. Accuracy is meaningless unless it is per model version.

**Rejected.** One wide table with a column per feature (migration on every model change); a full ORM (two tables; raw parameterised SQL is clearer and has no dependency).

---

### D-009 — SQLite now, isolated behind one module
**Date:** 2026-11-13 · **Status:** accepted

**Decision.** SQLite, with every statement confined to `api/store.py`.

**Why.** Zero setup, one file, ample for demo volume. The isolation is the point: the capstone's swap to Postgres is a one-file change, and the known single-writer limit is written down rather than discovered.

---

### D-010 — Methodology from the Vibe Coding guide, stack from the project
**Date:** 2026-11-13 · **Status:** accepted

**Context.** The reference guide (read 2026-11-13) prescribes planning before prompting, docs as the AI's context, small vertical slices, and a loop of read → plan → implement → test → review → commit → update docs. It also recommends a stack: Next.js, TypeScript, Tailwind, Supabase, Vercel.

**Decision.** Adopt the **methodology** in full — it is why docs 01–14 exist before any code, and it is encoded in `CLAUDE.md`'s session loop. **Do not** adopt the recommended stack: this is a Python tabular-ML pipeline, and Next.js/Supabase would add a frontend and a hosted database to a service whose job is to return a number.

**Mapping of the guide's doc set onto this one:** PRD → 01+02, ARCHITECTURE → 05, DESIGN → 08 + `design-system/`, RULES → `CLAUDE.md`, TASKS → 11, DECISIONS → this file, TEST_PLAN → 10, SECURITY → 09, README and `.env.example` → as named.

---

### D-011 — Design system generated, not invented
**Date:** 2026-11-13 · **Status:** accepted

**Decision.** Tokens come from the UI/UX Pro Max skill for category *Analytics Dashboard* at variance 3 / motion 2 / density 8, persisted to `design-system/lead-scoring-pipeline/MASTER.md`: `#1E40AF` primary, `#D97706` accent, Fira Code headings, Fira Sans body, dense spacing scale.

**Why.** The dials match the product honestly — an internal data tool read by someone scanning 200 rows wants density and restraint, not motion and ornament. A monospaced heading font suits a tool whose content is numbers.

**Deviation recorded.** Grade C uses muted grey, **not** the destructive red. A C lead is deprioritised, not an error, and red is reserved for actual errors. Grades also carry their letter, so colour is never the only encoding.

**Limit accepted.** Streamlit honours theme tokens and nothing more; see OQ-5.

---

### D-012 — Repo is `lead-scoring-pipeline`, public from the first commit
**Date:** 2026-11-13 · **Status:** accepted

**Decision.** Name states the deliverable (pipeline, not just an API) and survives the real-estate re-theme. Public from commit one.

**Why.** A private portfolio repo proves nothing: the CI badge, the demo link and the commit history only count if a stranger can open them. The `.gitignore` already excludes the dataset, models, the prediction DB and all secrets, so there is nothing sensitive to expose.

**Consequence.** The README states "in progress" and the results table stays blank until a real run fills it. Nothing unmeasured is ever published as a result.

---

### D-013 — Python 3.11, pip and venv; no machine-wide install
**Date:** 2026-11-13 · **Status:** accepted

**Context.** The machine has Python 3.11.9 and **no** `uv`, `docker` or `gh`. The design doc assumed 3.12 + uv + Docker. Installing any of them is a machine-wide change, which the owner's initialization prompt says needs approval.

**Proposal.** Target 3.11 (no install, every dependency supports it); approve `uv` (OQ-2); defer Docker to Phase 6 (OQ-3); create the GitHub repo in the browser instead of installing `gh` (OQ-11).

**Decision.** Python 3.11.9, `pip` + `venv` in a project-local `.venv`, Docker deferred to Phase 6, GitHub repo created in the browser. Nothing machine-wide was installed, because the owner's go-ahead for Phase 1 was not an explicit yes to installing `uv`, and a venv is project-local and reversible where a global tool is not.

**Consequence.** `pyproject.toml` stays uv-compatible — `uv sync` works later with no repo change. `make` is also absent, so every Makefile target has its raw command in the README. The full dependency install (`make install`) is therefore **untested**; only the Phase 1 subset has actually been installed and run.

---

## Changelog

### 2026-11-13 — Planning (Phase 0)

- Read the owner's P1 design doc, the master roadmap and the real-estate variant from the Claude doc `AI Engineer Portfolio Roadmap — Shah`. Project understanding is built from those three tabs, not invented.
- Read the Vibe Coding Development Guide; adopted its methodology, rejected its stack (D-010).
- Installed the UI/UX Pro Max skill bundle (7 skills) at user level after inspecting it in an isolated directory: MIT, v2.15.0, no hooks, no settings writes, no telemetry; the only outbound calls are opt-in image APIs needing the user's own keys.
- Generated the design system from the skill and persisted `design-system/lead-scoring-pipeline/MASTER.md` (D-011).
- Named the repo `lead-scoring-pipeline`, public (D-012). Renamed the local folder from `lead-scoring-api` to match.
- Wrote `README.md`, `CLAUDE.md`, `.env.example`, `.gitignore` and docs 01–14.
- Created the directory skeleton with placeholders. **No implementation code written** — the owner reads the docs first, then coding starts phase by phase.
- Recorded the toolchain gap (no `uv`, no Docker, no `gh`; Python 3.11 not 3.12) as D-013 and OQ-2/3/4/11. Phase 1 is `Blocked` on it in doc 11.

**Corrections made during the session, recorded so they are not repeated:** the project was initially framed as an app (corrected to a pipeline, D-001) and initially scaffolded as `lead-scoring-api` (renamed, D-012).

### 2026-11-13 (later the same day) — readability fix and capacity change

- **Fixed a real formatting bug:** 145 table rows across docs 08, 09, 10, 12, 13 and the three
  design-system page files were missing their leading `|`, so markdown renderers showed them as
  broken paragraphs instead of tables. The owner reported the docs as unreadable; this was the cause,
  not the rendering environment. All tables now parse.
- Published a rendered reader for the whole doc set, styled from the project's own generated tokens.
- **Capacity change:** mid-term exams finished. Planning assumption A-12 is resolved — budget 3–4 h a
  day until finals instead of 1.5–2 h. `CLAUDE.md` and doc 11 updated. The dates in doc 11 stand;
  the extra hours go into depth on work already planned, not new scope.

### 2026-11-13 (evening) — Phase 1: foundation

Verified by running, not by inspection: `ruff check` clean, **32 tests passing**.

- `pyproject.toml`: full dependency set declared, ruff and pytest configured, `pythonpath = ["src"]`
  so the suite runs without installing the package.
- `src/leadscore/config.py`: paths, seed, split, CV, tuning budget, gate floors, grade policy and
  API limits as a frozen `pydantic-settings` object; the column groups as module constants.
- `tests/test_config.py`: 32 tests. The one that matters asserts no banned column can reach the
  shipping feature set. One test also proves the column groups account for all 37 dataset columns,
  which keeps config and docs/02 from drifting apart.
- `Makefile` with every target, plus raw commands in the README because `make` is not installed here.
- `.gitattributes` normalising line endings; `git init`, first commit `872d629`, 34 files,
  verified free of data, models, databases and secrets.
- Toolchain settled (D-013): Python 3.11.9, pip + venv, Docker deferred. OQ-2 and OQ-4 resolved.
- Finals confirmed for **2027-01-01**, so the free window effectively ends around 2026-12-20.

**Not done, and why:** T-106 is only half complete — the GitHub repo does not exist yet, so there
is nothing to push to (owner action, OQ-11). T-107 needs the Kaggle download, also owner action.
`make install` (the full ML stack) has never been run; only the Phase 1 subset is installed.

### Next entry

Phase 1 (T-101 → T-107), once OQ-2, OQ-3 and OQ-4 are answered.
