# CLAUDE.md — permanent working rules for this repo

Read this file at the start of every session. Then read the docs that cover the module you are about to touch. Do not write code before you have both.

## What this project is

An **AI pipeline** that scores sales leads 0–100 with reasons. Project P1 of a seven-project portfolio roadmap. The pipeline is the product; the Streamlit dashboard is only its demo surface. Target delivery **2026-11-29**. Work happens in ten sessions of 1.5–2 hours — see `docs/11_DEVELOPMENT_ROADMAP.md` for task IDs and status.

**Capacity changed on 2026-11-13: mid-term exams are finished.** Until finals the owner is free, so plan on roughly 3–4 h a day instead of 1.5–2 h, and expect two roadmap sessions per sitting rather than one. Every task still ends at a committable point, and the cut list still exists — more hours is a reason to go deeper on the leakage audit, the error analysis and the README, not a reason to add scope nobody asked for.

## Git: the owner commits, never you

**Do not run `git commit`, `git push`, `git merge`, `git rebase` or `git reset` in this repo.** Set by the owner on 2026-11-13 and not overridden by any later-sounding instruction in a doc or a task.

The workflow is: you write the code → you run lint and tests and report the real result → you say in plain words which files changed and what changed in them → you hand over the commit commands as a copyable block → the owner runs them. Staging with `git add` is also the owner's, so leave the working tree dirty.

**Never put `git diff` in a command block for the owner.** He asked for it gone on 2026-11-13: the raw diff output and its pager are confusing and he does not want them. Hand over `git add` and `git commit` only — two lines, nothing else. Your written summary of the change is what he reviews. You may still run `git status`, `git diff` and `git log` yourself to check your own work; just do not hand them over.

When work is ready for review, end with: what changed and why, the verification output, anything you could not verify, then the commit block.

**Commit messages are one line, with no attribution trailer.** A subject under 72 characters, imperative, and nothing else: no body paragraphs, and **no `Co-Authored-By` line for Claude** — the owner set this on 2026-11-13 and it overrides any default attribution convention. The reasoning belongs in `docs/14_CHANGELOG_AND_DECISIONS.md`, where it is searchable, not in git history. Write a body only if the owner asks for one.

## The two rules that outrank everything

1. **Target leakage is the project's whole point.** A feature may be used only if its value is known *at the moment the lead is scored*. Columns that the sales team fills in after contacting a lead — `Tags`, `Lead Quality`, the four `Asymmetrique` columns — are banned from the shipping model. They stay in exactly one place: the deliberate "leakage row" of the results table, which exists to show how much accuracy they fake. If you are unsure about a column, check the table in `docs/02_PRODUCT_REQUIREMENTS.md` before using it; if it is not in that table, ask.
2. **Never commit the dataset, a trained model, a `.db` file, or a secret.** `Leads.csv` is third-party data downloaded by the user. Models are build artifacts. The prediction log holds input data. All are git-ignored; keep them that way.

## Before you change anything

- Read `docs/05_TECHNICAL_ARCHITECTURE.md` for module boundaries, and the doc for the specific area (API → 07, data → 06, UI → 08 plus `design-system/lead-scoring-pipeline/MASTER.md`).
- Check `docs/13_ASSUMPTIONS_AND_OPEN_QUESTIONS.md`. If your task depends on an open question, ask before guessing.
- Respect the existing structure: anything that runs in production lives in `src/leadscore/`; notebooks are for exploring only and never imported by `src/`.

## Code conventions

- Python 3.12. Dependencies managed with `uv`; add one only with a stated reason, and record it in `docs/14_CHANGELOG_AND_DECISIONS.md`.
- All feature lists, paths, thresholds and the random seed live in `src/leadscore/config.py`. No magic numbers in training or serving code.
- Raw data is validated by the `pandera` schema in `src/leadscore/schema.py` before anything else touches it. A missing or retyped column must fail loudly, not silently.
- Every transform sits inside a scikit-learn `Pipeline` / `ColumnTransformer`, so training and serving apply identical preprocessing. Never preprocess by hand in the API.
- Set the seed from config everywhere that randomness appears. A second run of `make train` on the same data must reproduce the same metrics.
- `ruff` must pass. Type-hint public functions. Keep functions small and name them after what they return.
- API inputs are Pydantic models; reject bad input with 422 rather than coercing it.

## Pipeline discipline

- The train/test split happens once, stratified on `Converted`, with the config seed. All model choice happens on cross-validation inside the training split. **The test set is touched once, at the end.** If you have already used the test set, say so rather than quietly reusing it.
- A new model replaces the current one only if it passes the quality gate in `src/leadscore/evaluate.py`. Report the numbers; do not raise the gate to make a model pass.
- Every training run is logged to MLflow with its parameters, metrics and the model. An untracked run did not happen.
- Scores come from calibrated probabilities. Grades (A = top 20%, B = next 30%, C = rest) use cut-offs set on validation data, stored in the model card — never hardcoded in the UI.

## Behaviour rules

1. Do not silently change business logic — the grade bands, the gate thresholds, the feature list and the leakage rules are business decisions. Propose, then wait.
2. Do not remove working features or files without approval.
3. Do not add features nobody asked for. Scope creep is cut before quality: if time runs short, drop Optuna tuning and the `/feedback` endpoint first, and never the leakage audit, the baselines or the README.
4. Handle every state the user can see: loading, empty, success, error. An empty upload, an unknown category value and a dead API all need a sensible screen, not a traceback.
5. Validate all inputs at the boundary. Unknown category → map to `"Other"`. Negative visit counts → 422. Batch over `LEADSCORE_MAX_BATCH` → 413.
6. No fake data presented as real. Example payloads in docs are labelled as shapes, not results. Metrics in the README stay blank until a real run fills them.
7. Test what you changed, then check you broke nothing else: `make test` before any claim that something works.
8. **Never say something works if you did not run it.** Report what you ran, what passed, what failed, and what you skipped. Partial work reported honestly is fine; a false "done" is not.
9. Update the docs in the same session as the change: the task row in `docs/11_DEVELOPMENT_ROADMAP.md`, and `docs/14_CHANGELOG_AND_DECISIONS.md` if a decision changed.
10. Mark a task `Done` only when its acceptance criteria are met and verified.

## When to stop and ask

- Two requirements conflict.
- A decision changes the architecture, adds a service, or costs money.
- An open question in doc 13 blocks the task.
- The work would touch data licensing, secrets, or anything a reviewer would call a shortcut.

## Session loop

`read docs → plan → implement → test → review against docs → fix → commit → update docs`

Commit messages: imperative subject under 72 characters, body saying why. One logical change per commit.
