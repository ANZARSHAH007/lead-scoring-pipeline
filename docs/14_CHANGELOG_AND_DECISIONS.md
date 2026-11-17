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

### D-014 — Chart marks use #2563EB, not the UI primary
**Date:** 2026-11-13 · **Status:** accepted

**Context.** The design system's primary is `#1E40AF`. Run through the palette validator, it
**fails the lightness band** for data marks (L 0.424, outside 0.43–0.77): it is a good colour
for text and chrome and too dark for a filled mark.

**Decision.** Chart marks use `#2563EB` (first series) and `#D97706` (second series, the
design system's accent unchanged). UI chrome keeps `#1E40AF`. The pair passes all six checks:
lightness band, chroma floor, CVD separation (ΔE 32.3 protan, 29.3 tritan), normal-vision
separation (ΔE 38.3) and contrast against the surface.

**Why it is written down.** The validator was run rather than the colours eyeballed, and the
next person to add a chart needs to know the chart palette deliberately differs from the UI
palette instead of "fixing" it back. Tokens live in `src/leadscore/plotting.py`.

---

### D-015 — Analysis ships as markdown reports, not notebooks
**Date:** 2026-11-13 · **Status:** accepted

**Context.** The roadmap named `notebooks/01_eda.ipynb` and `notebooks/02_leakage_audit.ipynb`.

**Decision.** The analysis is a script in `scripts/` that writes a markdown report and PNG
figures into `reports/`, both committed.

**Why.** A committed notebook is JSON with embedded image blobs: it cannot be reviewed in a
diff, and the owner reviews every change before it enters history. A markdown report reads in
any editor's preview, and re-running the script reproduces it exactly. Notebooks stay useful
for throwaway exploration; they are just not the artefact.

**Rejected.** Committing executed notebooks (unreviewable), and stripping outputs before
commit (then the analysis is not in the repo at all).

---

### D-016 — The activity columns are excluded
**Date:** 2026-11-17 · **Status:** accepted

**Context.** `Last Activity` and `Last Notable Activity` were the one group docs/02 refused to
decide in advance, pending measurement. Measured, their values split into two kinds:

| Value | Conversion | Who did it |
| --- | ---: | --- |
| `Email Opened` | 36.5% | the lead |
| `Olark Chat Conversation` | 8.6% | the lead |
| `SMS Sent` | 62.9% / 69.4% | **the company** |
| `Unreachable` | 68.8% | **the company** |

**Decision.** Both columns are excluded from the shipping model. The shipping feature set is
frozen at **13 columns**. The variant that includes them is reported as a labelled comparison,
exactly as the sales-filled row is.

**Why.** `SMS Sent` is an outbound message and `Unreachable` is the outcome of a call attempt.
Neither can exist for a lead that has just arrived, which is the definition of a feature that
is not known at scoring time. Keeping them would have been the same mistake as keeping `Tags`,
made quieter by a smaller number.

**Cost, stated plainly.** 0.897 → 0.866 cross-validated ROC-AUC. The published headline is
lower because it is true.

**Rejected.** Keeping both (contradicts the project's own rule, and a reviewer who opens the
value list finds `SMS Sent` immediately). Filtering to lead-initiated values only (at scoring
time a fresh lead has no activity at all, so the feature would be almost always missing in
production — training and serving distributions would not match, for more code and no gain).

---

### D-017 — MLflow tracks to local SQLite, and the package is installed editable
**Date:** 2026-11-17 · **Status:** accepted

**Context.** Two things broke on contact with the real toolchain rather than in planning.

1. **MLflow 3.18 retired the plain file store.** `file:./mlruns` now raises outright:
   *"The filesystem tracking backend is in maintenance mode… migrate to a database backend."*
   docs/05 had specified the file store.
2. **`python -m leadscore.train` could not import the package.** The test suite reaches `src/`
   through pytest's `pythonpath`, which does nothing for a module run, so the documented
   command failed on a clean checkout.

**Decision.** Track to `sqlite:///mlflow.db` — still local, still no server and no account,
and the supported path going forward. Install the package editable (`pip install -e .`), which
is what the README already told a reader to do.

**Why not `MLFLOW_ALLOW_FILE_STORE=true`.** It silences the error and keeps a backend its own
maintainers have deprecated. Setting an opt-out flag to postpone a migration is the kind of
shortcut this project exists to avoid, and the SQLite switch cost one config line.

**Consequence.** `mlflow.db` is git-ignored alongside `mlruns/` and `mlartifacts/`.
`.env.example`, docs/05 and docs/12 updated.

### D-018 — A truncated metric axis must keep the baseline visible, enforced in code

**Date** 2026-11-17 · **Status** accepted

Report 03's first chart was a bar chart of cross-validated ROC-AUC with the x-axis starting at
0.45. The gap between LightGBM (0.8825) and logistic regression (0.8679) filled a third of the
plot. The difference is 0.0146. The chart was not wrong about the numbers and was still
misleading about the finding, which is worse than being wrong, because nothing in it is
checkable by a reader.

The rule now: a metric axis may start above zero only if the chance line stays on the chart,
and marks that invite length comparison (bars) are replaced by marks that carry their own
spread (a dot and a whisker). `plotting.metric_axis` raises `ValueError` when the chance value
falls outside the limits, so the rule is enforced where charts are drawn rather than
remembered from a doc. Eight tests in `test_plotting.py` hold it.

**Rejected:** writing the rule into docs/08 and relying on care. The mistake was made *after*
docs/08 was written, by the author of docs/08.

### D-019 — Dependencies are pinned in `requirements.lock`; one upstream warning is allowed

**Date** 2026-11-17 · **Status** accepted

Two of Phase 3's three bugs were environment facts, not logic: MLflow had moved to 3.18 under
a floating `mlflow>=2.16` bound and retired the plain file store, and the package was never
installed, so `python -m leadscore.train` could not import it. Neither was visible to `ruff`
or to 81 unit tests.

So: `requirements.lock` pins the 110-package resolution that actually works, the pyproject
floor is raised to `mlflow>=3.1` so a 2.x file store cannot return, and `scripts/doctor.py`
proves the configured store accepts a run by writing a probe run into a temporary database
rather than trusting the URI string.

One consequence is recorded rather than hidden: MLflow 3.18 reads runs through SQLAlchemy's
`noload` strategy, which SQLAlchemy 2.1 deprecates, and this project's pytest config turns
`DeprecationWarning` into an error. The global filter stays strict; a single message-matched
`ignore` is added with a comment naming this decision. It is an upstream deprecation in a
pinned dependency, so it cannot change under us without a deliberate upgrade, and the next
MLflow upgrade should try deleting the line.

**Rejected:** relaxing `error::DeprecationWarning` to a warning. That filter is how a
deprecation in our own code gets noticed; trading it away to silence someone else's would cost
more than it saves.

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

### 2026-11-13 (late) — T-107: the dataset is verified

The real `Leads.csv` is on disk and `scripts/check_data.py` exits 0. **Phase 2 is unblocked.**

- 9,240 rows x 37 columns, `Converted` rate **0.3854** against an expected 0.3854 — drift 0.0000.
- Every column named in `config.py` exists in the file, so the column groups and the upload agree.
- Source: the Kaggle archive also ships `Leads Data Dictionary.xlsx`, kept beside the CSV in
  `data/raw/` (git-ignored with it).

**Two corrections came out of the check, both in the docs rather than the data:**

1. The checker counted `Select` as missing for every column, while docs/02 quoted blank-only rates
   for most and blank-plus-`Select` for two. That mismatch raised a false warning on
   `Specialization`. The checker now tracks both bases and compares each against the one docs/02
   states; docs/02 carries the full table.
2. **`City` is 39.7% effectively missing** (15.4% blank plus 24.3% `Select`) and was not in the
   original figures at all — emptier than `Specialization`. It is an admissible feature, so T-201
   decides whether it earns its place.

### 2026-11-13 — T-201: EDA, and three findings that change the plan

`scripts/eda.py` writes `reports/01_eda.md` and four figures. Run on the real file.

**1. All 12 ad and opt-in flags are near-constant.** One value covers at least 95% of rows in
every one of them, so none can discriminate. docs/02 said "check first"; the answer is drop all
12. The admissible feature list falls from 27 columns to 15.

**2. Time on website is U-shaped, not rising.** Leads with no recorded visit convert at 42.0%,
*above* the 38.5% average; among leads that did browse, conversion bottoms out near 13% in the
1–84 second band and climbs to ~71% above 784 seconds. Two consequences: the one-rule baseline
(sort by time on site) ranks the zero-time group last even though it converts above average, so
it is a weaker bar than it looked; and logistic regression cannot fit a U-shape in a raw numeric
feature, so it needs the column binned or a zero indicator, while LightGBM handles it natively.
That may turn out to be most of the gap between them (T-304).

**3. `Lead Origin = Lead Add Form` converts at 92.5% across 718 leads.** A level that almost
always converts usually means the row was created *because* something already happened. T-203
has to decide whether it is a record-keeping artefact. Flagged, not dropped.

Also: the chart palette was validated rather than eyeballed, which rejected the UI primary for
data marks (D-014), and the analysis ships as markdown instead of notebooks (D-015).

**Not done:** T-202 (pandera schema) and T-203 (the leakage audit itself) are next.

### 2026-11-17 — Phase 2 complete: the data contract and the leakage audit

**T-202, the contract.** `src/leadscore/schema.py` enforces four things before any model sees
a row: every column config names is present; no column is unclassified; types and ranges hold;
shape and target rate are sane (a warning, not a failure). 20 tests, and the real file
validates clean. Two bugs were found and fixed while writing it — pandas 3 gives text columns
the `str` dtype rather than `object`, so the `Select` normalisation silently skipped every
column until the check was rewritten.

**T-203 and T-204, the audit.** `reports/02_leakage_audit.md` carries a yes/no decision with a
reason for all 37 columns, and measures what the banned columns are worth:

| Feature set | Columns | CV ROC-AUC |
| --- | ---: | ---: |
| Shipping (admissible only) | 13 | **0.866** |
| + activity columns | 15 | 0.897 |
| + sales-filled (leaks) | 21 | 0.982 |

**The leakage gap is +0.117 ROC-AUC.** A per-column univariate model makes the case visually:
`Tags` alone scores **0.921** — nearly the whole leaking model from one field a rep types after
calling. `Lead Quality` alone scores 0.800.

**T-205, frozen.** The shipping set is 13 columns, enforced by tests rather than convention.
Out: 12 near-constant flags (reports/01), 2 activity columns (D-016), 6 sales-filled, 1 sparse,
2 identifiers.

**What this means for the rest of the project:** the honest bar for LightGBM to beat is 0.866,
not 0.98. Any public notebook on this dataset reporting ~0.98 is scoring leads with information
that does not exist when a lead arrives.

### 2026-11-17 — Phase 3: the split, the baselines and the challenger

Verified by running: ruff clean, **81 tests**, selection reproduces on a second run with the
same seed.

| Model | CV ROC-AUC | Std |
| --- | ---: | ---: |
| **LightGBM, Optuna-tuned** | **0.8825** | 0.0045 |
| LightGBM, defaults | 0.8718 | 0.0072 |
| Logistic regression | 0.8679 | 0.0062 |
| One-rule (time on site) | 0.6517 | 0.0116 |
| Random | 0.5000 | 0.0000 |

5-fold stratified CV inside the training split. **The test set was not opened** — 1,848 rows
are held back for Phase 4, and a test spies on what `cross_val_score` actually receives to
prove no held-out index reaches it.

**The margin is the story.** Untuned, LightGBM beat logistic regression by +0.004, which is
inside the fold-to-fold noise and is not a win. Forty Optuna trials moved it to **+0.0146**,
which clears logistic regression's own standard deviation. So the gradient booster earns its
place, but narrowly: a linear model on 13 admissible columns is close to the ceiling this data
supports. Phase 4 decides what ships on calibration and Capture@20% as well, and the README
reports the comparison rather than quietly shipping the more complicated model.

The one-rule baseline lands at 0.6517 — far below both. Sorting a spreadsheet by time on site
really is worse than modelling, which report 01 had cast into doubt when it showed the
relationship was U-shaped.

Two environment problems were found by running rather than planning, both in D-017: MLflow
3.18 refuses the plain file store, and the package needed an editable install before
`python -m leadscore.train` would work at all.

### 2026-11-17 — Guardrails: the three Phase 3 bugs, fixed as classes

Phase 3's bugs had one shape in common: each appeared only when the pipeline actually ran, and
none was visible to `ruff` or to the 81 unit tests. Fixing the three instances was an hour's
work. Fixing the class took the rest of the session, and is what follows.

| Bug | Instance fix | Class fix |
| --- | --- | --- |
| MLflow 3.18 refuses `file:./mlruns` | tracking URI → `sqlite:///mlflow.db` (D-017) | `requirements.lock` pins 110 packages; pyproject floor raised to `mlflow>=3.1`; `doctor.py` writes a probe run to prove the store accepts one (D-019) |
| `python -m leadscore.train` could not import the package | `pip install -e . --no-deps` (D-017) | `doctor.py` checks the import *and* that it resolves to this working tree, and prints the exact install command |
| a bar chart truncated at 0.45 exaggerated a 0.0146 gap | replaced with a dot-and-whisker | `plotting.metric_axis` raises when a truncated axis would hide the chance line (D-018) |

Two more guards came out of the same reasoning:

- **An end-to-end smoke test on synthetic data** (`fixtures.py`, `test_smoke.py`). 300 rows
  built to satisfy the real data contract — all 37 columns, `Select` placeholders, missing
  numerics, near-constant flags, a deliberately leaky `Tags` — run through validate, split,
  cross-validate, track and report in about ten seconds. This is the test that would have
  caught both environment bugs, and it is what CI will run on every push without needing the
  dataset.
- **`reports_dir` is configurable.** `write_report` had the reports path hardcoded from
  `__file__`, so a fixture run would have overwritten `reports/03_model_selection.md` with
  synthetic numbers. Found while writing the smoke test, which is the sort of thing writing a
  smoke test finds. A test now asserts the default is still the real `reports/`, so the guard
  cannot be satisfied by weakening it.

**Verified:** `ruff check src tests scripts` clean; **103 tests pass** (81 + 22); `doctor.py`
exits 0 with two warnings (Streamlit not installed, a Phase 6 dependency — the lockfile
warning is now resolved); the real training run reproduces 0.8825 ± 0.0045 with the chart and
report paths changed.

Nothing about the model changed. No new dependency was added.

### Next entry

Phase 1 (T-101 → T-107), once OQ-2, OQ-3 and OQ-4 are answered.
