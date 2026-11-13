# 01 — Project Overview

**Project:** Lead Scoring Pipeline (P1)
**As of:** 2026-11-13 · **Target delivery:** 2026-11-29
**Type:** AI/ML pipeline with a thin demo surface. Not an end-user application.

## The problem

A sales team receives more inbound leads than it can call. X Education, the online-course company whose public dataset this project uses, converts about 38.5% of its leads but has no way to tell on arrival which of them is worth a call first. Reps work the list in the order it arrives, so a lead that would have converted goes cold while a rep is on the phone with one that never will.

The cost is not "lost leads" in the abstract — it is wasted call minutes. Any fix has to rank leads using only what is known *before* anyone picks up the phone.

## Purpose of the pipeline

Give every new lead a **0–100 score, a grade, and three plain-English reasons**, within one HTTP request, using only information available at the moment the lead arrives.

One sentence has to be true and measurable when this ships:

> If sales calls only the top 20% of scored leads, they reach **X%** of all conversions.

That number — **Capture@20%** — is the deliverable. Model accuracy matters only insofar as it moves it.

## Business objectives

| Objective | Measure | Why it matters |
| --- | --- | --- |
| Rank leads better than a human rule of thumb | Capture@20% beats the one-rule baseline (sort by time on site) | if a sorted spreadsheet column does as well, the model is not worth running |
| Make scores trustworthy, not merely ordered | Brier score; reliability curve near the diagonal | a score of 70 must mean roughly a 70% chance, or reps stop believing it |
| Make every score explainable | 3 SHAP-derived reasons per lead, in plain words | a rep ignores a number they cannot argue with |
| Prove honest measurement | published results table including a leakage row | most public notebooks on this dataset quietly use post-contact columns |
| Be deployable, not a notebook | `docker compose up`, live demo URL, CI quality gate | P1 exists to prove the owner can ship and measure |

## Secondary objective: portfolio proof

This repo is also a work sample. A reviewer spends about a minute on it, so the README leads with the problem, the number and a demo link. The pipeline is built so later projects reuse it: the capstone (P7) scores every inbound lead through this same API.

## Target users

P1 has no login and no multi-tenancy. "Users" here means the four parties whose needs shape the design — not account types in a system.

| Party | What they need from it | Where they meet it |
| --- | --- | --- |
| **Sales rep** | a ranked call list and a reason per lead | dashboard page 1 — upload a CSV, get it ranked |
| **Sales manager** | confidence the ranking is real, and how many leads to work | dashboard page 3, the model card: metrics, calibration plot, grade bands, stated limits |
| **A CRM or another service** | score one lead, or a batch, as JSON | `POST /score`, `POST /score/batch` |
| **The maintainer (owner)** | retrain, compare runs, roll back | `make train`, MLflow, versioned `model.joblib` + `model_card.json` |

Roles, and what P1 deliberately does *not* implement, are in `09_SECURITY_AND_PERMISSIONS.md`.

## What it does not do

Stated so nobody expects it later:

- No decision to contact or discard a lead. It ranks; a human calls.
- No writing back to any CRM. P1 returns a score; the capstone does the writing.
- No authentication, accounts, or per-user state.
- No automatic retraining. `/feedback` collects outcomes so retraining becomes possible; triggering it stays manual.
- No LLM. This is classic tabular ML on purpose — the roadmap's gap table calls for one solid classic-ML project.

## Where it fits in the roadmap

P1 is the first of seven projects that together build an AI layer over a sales CRM. Its pipeline is reused twice: directly by the capstone (P7) lead-intake flow, and structurally by the real-estate variant if that direction is chosen on 2026-11-25 — in which case only `config.py`, `schema.py` and the API's input fields change, and the target becomes a booked viewing or a signed lease. See `13_ASSUMPTIONS_AND_OPEN_QUESTIONS.md`, OQ-1.
