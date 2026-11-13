# data/raw

`Leads.csv` goes here, at exactly `data/raw/Leads.csv`.

It is **git-ignored on purpose**. The dataset is third-party, so this repo never redistributes it — it tells you how to fetch it. See `docs/06_DATABASE_DESIGN.md` and `docs/09_SECURITY_AND_PERMISSIONS.md`.

## What you are looking for

The X Education lead-scoring dataset, created by UpGrad / IIIT-B as a case study. The file the pipeline expects:

| Property | Value |
| --- | --- |
| Filename | `Leads.csv` |
| Rows | 9,240 |
| Columns | 37 |
| Target column | `Converted` (1 = became a customer) |
| Conversion rate | about 38.5% |
| Quirk | the string `Select` is a form default and means "missing" |

Several Kaggle listings host the same case study under different names. These two are the ones to try, in order:

1. **[ashydv/leads-dataset](https://www.kaggle.com/datasets/ashydv/leads-dataset)** — the listing behind the widely-cited notebook for this case study.
2. **[amritachatterjee09/lead-scoring-dataset](https://www.kaggle.com/datasets/amritachatterjee09/lead-scoring-dataset)** — same case study, alternative upload.

Do not assume either is right because of its name. **Run the checker below**; it is what decides.

## Option A — download in the browser

1. Sign in to Kaggle and open one of the links above.
2. Download the dataset archive and extract it.
3. Put `Leads.csv` in this folder, so the path is `data/raw/Leads.csv`. If the archive names it differently (`leads.csv`, `Lead Scoring.csv`), rename it to `Leads.csv`.

## Option B — Kaggle CLI

Only if you already use the Kaggle CLI. It needs an API token, which is a **secret**: it lives in `%USERPROFILE%\.kaggle\kaggle.json` or in `.env.local`, never in this repo. `kaggle.json` is already in `.gitignore` by name.

```bash
pip install kaggle
kaggle datasets download -d ashydv/leads-dataset -p data/raw --unzip
```

## Verify what you got — do this before Phase 2

```bash
.venv/Scripts/python scripts/check_data.py        # Windows
# .venv/bin/python scripts/check_data.py          # macOS / Linux
```

The checker reads the CSV with the standard library (no pandas needed) and reports:

- the row and column counts against the expected 9,240 × 37
- the conversion rate against the expected 38.54%
- that **every column named in `src/leadscore/config.py` actually exists in the file** — this is the one that catches a different upload with renamed columns
- `Converted` holding nothing but 0 and 1
- how empty the known-sparse columns are, counting `Select` as missing

It exits 0 when the file matches and 1 when it does not, and it names what is wrong. A mismatch is not automatically a problem with the file — it may mean `docs/02_PRODUCT_REQUIREMENTS.md` was written from a different upload. Either way, resolve it before any modelling, and record what you found (assumption A-1 in `docs/13`).

This check is task **T-107** in `docs/11_DEVELOPMENT_ROADMAP.md`, and it is Phase 2's entry condition.

## Licence

Check the licence on the Kaggle page you download from before reusing the data anywhere beyond this portfolio project. Whatever it says, the file stays out of this repo.
