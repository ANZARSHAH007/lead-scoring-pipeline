# 09 — Security and Permissions

P1 is a public demo of a scoring pipeline with no accounts, holding no personal data. That makes the threat model small — and worth stating explicitly, because an unstated threat model is how a demo quietly becomes a liability.

## What P1 deliberately does not implement

| Not implemented | Why | When it arrives |
| --- | --- | --- |
| Authentication / login | nothing to protect: no user data, no writes to any system of record, read-only inference on a public dataset | P5–P6, where an agent writes to a CRM |
| Roles and permissions | there is one surface and it has one capability: score a lead | P7 capstone (rep vs manager vs admin) |
| Multi-tenancy | one model, one dataset | P7 |
| Audit trail of users | there are no users; the prediction log records *what was scored*, not *who asked* | P6, with approvals |

Documented so that nobody reads the absence as an oversight, and so the capstone knows what it inherits.

## Roles, as designed (not as enforced)

| Party | Capability in P1 | Enforced by |
| --- | --- | --- |
| Anyone with the demo URL | score a lead, read the model card | nothing — intentional |
| A client service / CRM | the same, over HTTP | nothing in P1; an API key is available, see below |
| Maintainer | retrain, deploy, roll back | GitHub repo access + host account |
| CI | build, test, gate, publish an artifact | GitHub Actions' own token, least privilege |

An optional shared-secret header exists (`LEADSCORE_API_KEY`, empty by default). If the public demo attracts abuse, setting it closes `/score/batch` and `/feedback` without a code change. It is not a real auth system and the docs do not claim it is.

## Secrets

**Rules, non-negotiable:**

- No secret, key, token or credential in source, in a notebook, in a docstring, in a test fixture, or in the README.
- `.env.example` lists every setting by **name** with a safe default; real values live in `.env.local`, which is git-ignored.
- The only genuine secret P1 needs is the **Kaggle API token**, used by CI to download the dataset. It lives in GitHub Actions secrets (`KAGGLE_USERNAME`, `KAGGLE_KEY`) and nowhere else. Never printed, never echoed in a log.
- Host configuration (Hugging Face Spaces / Render) is set in that platform's own secret store.
- `kaggle.json` is in `.gitignore` by name.

**Verification (T-702, before delivery):** grep the full git history for the data filename, `KAGGLE`, `api_key`, `secret`, `token`, and private-key headers. Zero findings required. If anything is ever committed, the key is rotated — history rewriting alone is not a fix.

## Input trust

Every request is untrusted input, even on a demo with no attacker in mind.

| Input | Risk | Control |
| --- | --- | --- |
| JSON body | malformed or hostile payload | Pydantic validation with `extra="forbid"`; unknown field → 422 rather than silent acceptance |
| Numeric fields | negative or absurd values skewing a score | range rules (`≥ 0`); out of range → 422 |
| Category fields | an unseen value crashing the transformer | `handle_unknown` → `Other`; never an exception |
| `prospect_id` | unbounded string, log-injection | length-capped 1–64 chars, stored as a parameter, never interpolated |
| Batch size | memory exhaustion from one request | hard cap `LEADSCORE_MAX_BATCH` (1,000) → 413 |
| Uploaded CSV (dashboard) | huge file, wrong columns, formula injection on re-export | size cap, required-column check before upload, values written as text on CSV download |
| SQL | injection via logged values | parameterised statements only, in `store.py`; no string-built SQL anywhere |
| Pickle / joblib | `joblib.load` executes code from its file | **only ever load a model this repo's own CI built**; never a model from an upload or a URL. Stated here because it is the one genuinely dangerous load in the system |

## Data protection

- **The dataset is not redistributed.** Third-party licence; downloaded by the user, git-ignored, never in the image unless CI built it for a training run.
- **No PII in P1.** The X Education data has no names, emails or phone numbers; `Prospect ID` is opaque; `City`/`Country` are coarse. So `input_json` in the prediction log is safe to store whole.
- **The carry-over case is planned now, not later.** If this pipeline is pointed at real property inquiries (the 2026-11-25 decision), payloads may carry a phone number. Then: hash the identifier before logging, drop free-text fields from `input_json`, set a retention window (OQ-7), and state what is collected to the pilot owner in writing. These controls are written down before the data exists, because retrofitting them after a pilot starts is how leaks happen.
- **Nothing is sent to a third party.** No analytics, no telemetry, no LLM API, no error-reporting SaaS. The pipeline makes no outbound network calls at serving time.

## Service hardening

| Control | Setting | Reason |
| --- | --- | --- |
| CORS | explicit origin list; the dashboard's origin only. No `*` in production | a wildcard on a write endpoint is a free abuse channel |
| Rate limiting | per-IP cap on `/score/batch` and `/feedback` if the public demo is abused | a 1,000-lead batch is cheap to send and not cheap to serve |
| Error responses | JSON `detail` + `code`; stack traces server-side only | a traceback tells an attacker the stack and the paths |
| Container | non-root user, no build tools in the final image, pinned base image digest | smaller blast radius, reproducible builds |
| Dependencies | pinned with a lockfile; Dependabot or `pip-audit` in CI | a transitive CVE is the most likely real vulnerability here |
| Health endpoint | no model internals, no paths, no versions of libraries | liveness only |
| Model file | read-only in the running container | nothing at runtime may replace the model |
| Logging | no secrets, no full payloads at `INFO`, one line per request | logs leak more often than databases |

## Model-specific risks

Not conventional security, but the risks a reviewer will actually ask about:

| Risk | Mitigation |
| --- | --- |
| **Target leakage** — a model that looks excellent and fails in production | the entire doc-02 admissibility rule, the audit notebook, and the published leakage row |
| **Misuse of the score** — treating a C as "do not contact" | the model card's explicit limits; the API returns a ranking, never a decision |
| **Stale model** — scoring 2027 traffic with a 2026 model | `model_version` on every prediction; `/feedback` makes decay measurable |
| **Silent regression** — a retrain that is quietly worse | the CI quality gate; the gate's floors are never raised to make a model pass |
| **Unfounded confidence** — a 70 that does not mean 70% | calibration is a requirement, with the reliability curve published |
| **Fairness** | `City`, `Country` and occupation are inputs. P1 states this plainly on the model card and does not claim a fairness audit it has not run. If this were deciding anything about a person rather than ranking a call list, an audit would be mandatory — that condition is written on the card. |
