---
title: Lead Scoring Pipeline
emoji: 📈
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 8501
pinned: false
license: mit
---

# Lead Scoring Pipeline

Scores a sales lead 0–100 on arrival, with three reasons why.

**The constraint that defines this project:** a column may be used only if its value is known
at the moment the lead is scored. Six columns in the source dataset are filled in by the sales
team *after* they have contacted the lead — `Tags`, `Lead Quality` and four `Asymmetrique`
scores. Including them lifts ROC-AUC from 0.865 to 0.984 and makes the model look excellent.
They are excluded, and the gap is published rather than hidden.

| Model | ROC-AUC | Capture@20% |
| --- | ---: | ---: |
| One-rule (sort by time on site) | 0.646 | 35.3% |
| Logistic regression | 0.850 | 42.6% |
| **LightGBM, shipped** | **0.865** | **44.1%** |
| With the sales columns *(not shippable)* | 0.984 | 51.5% |

Measured once on 1,848 held-out leads the model never saw.

**Three pages:** rank an uploaded CSV, try a single lead, and read the model card. Start on
**Rank leads** and download the sample CSV if you have nothing to hand.

**This demo sleeps when idle**, so the first request after a pause takes a few seconds while
the container wakes and loads the model.

Source, documentation and the full evaluation report:
[github.com/ANZARSHAH007/lead-scoring-pipeline](https://github.com/ANZARSHAH007/lead-scoring-pipeline)
