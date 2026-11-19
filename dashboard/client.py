"""Everything the dashboard does that is not drawing (task T-601).

Streamlit code is hard to test because it is a script that renders as it runs, so the parts
that can be wrong on their own live here instead: reading an uploaded CSV, deciding whether
it is usable, turning it into API payloads, and interpreting what the API says back. `app.py`
calls these and draws the result.

The dashboard talks to the API over HTTP rather than importing the model. That is the point
of the split: the demo surface exercises the same contract an outside caller would, so a
break in the API shows up in the demo instead of being bypassed by it.
"""

from __future__ import annotations

import io
import math
from dataclasses import dataclass, field
from typing import Any

import pandas as pd
import requests

from leadscore import config as cfg
from leadscore.api import schemas as sc

#: Columns an uploaded CSV must carry. The rest are optional, exactly as in `/score`.
REQUIRED_COLUMNS: tuple[str, ...] = ("prospect_id", "lead_origin", "total_time_spent_on_website")

#: Every column the dashboard will send if present, in the order the form shows them.
OPTIONAL_COLUMNS: tuple[str, ...] = tuple(
    field_name for field_name in sc.FIELD_TO_COLUMN if field_name not in REQUIRED_COLUMNS
)

#: Accepted spellings for the two boolean fields, so a CSV exported from a CRM still loads.
_TRUE = {"true", "yes", "y", "1", "t"}
_FALSE = {"false", "no", "n", "0", "f", ""}

TIMEOUT_SECONDS = 30.0
BATCH_TIMEOUT_SECONDS = 120.0


class ApiUnreachable(RuntimeError):
    """The scoring service did not answer. The uploaded file is never discarded for this."""


@dataclass
class CsvCheck:
    """What a client-side look at the upload found, before anything is sent."""

    ok: bool
    rows: int
    missing: list[str] = field(default_factory=list)
    unknown: list[str] = field(default_factory=list)
    message: str = ""
    truncated_to: int | None = None


def read_csv(data: bytes | io.BytesIO) -> pd.DataFrame:
    """Read an uploaded CSV, keeping everything as text until it is validated."""
    buffer = io.BytesIO(data) if isinstance(data, bytes) else data
    return pd.read_csv(buffer, dtype=str, keep_default_na=True)


def check_frame(frame: pd.DataFrame, max_batch: int) -> CsvCheck:
    """Decide whether this upload can be scored, and say why not in one sentence.

    Checked here rather than by the API because a missing column should be reported without
    sending anything: the rep can fix the file and try again, and nothing was logged.
    """
    if frame.empty:
        return CsvCheck(ok=False, rows=0, message="That file has no rows in it.")

    columns = set(frame.columns)
    missing = [column for column in REQUIRED_COLUMNS if column not in columns]
    if missing:
        return CsvCheck(
            ok=False,
            rows=len(frame),
            missing=missing,
            message=(
                f"The file is missing {len(missing)} required column(s): "
                f"{', '.join(missing)}. Nothing was sent for scoring."
            ),
        )

    # `prospect_id` is a required API field but not a model feature, so it is not in the
    # field map; `converted` is the label, which a re-uploaded training export carries.
    unknown = sorted(columns - set(sc.FIELD_TO_COLUMN) - {"prospect_id", "converted"})
    truncated = len(frame) > max_batch
    return CsvCheck(
        ok=True,
        rows=len(frame),
        unknown=unknown,
        truncated_to=max_batch if truncated else None,
        message=(
            f"{len(frame):,} rows is over the {max_batch:,} limit for one request. "
            f"The first {max_batch:,} will be scored."
            if truncated
            else ""
        ),
    )


def _to_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in _TRUE:
        return True
    if text in _FALSE or text in {"nan", "none"}:
        return False
    return False


def _clean(value: Any) -> str | None:
    """A CSV cell as the API wants it: a trimmed string, or nothing at all."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none"} or text == cfg.SELECT_PLACEHOLDER:
        return None
    return text


def _number(value: Any) -> float | None:
    cleaned = _clean(value)
    if cleaned is None:
        return None
    try:
        return float(cleaned)
    except ValueError:
        return None


def to_payloads(frame: pd.DataFrame, limit: int | None = None) -> list[dict[str, Any]]:
    """Turn a checked frame into `/score` bodies, dropping anything the API will not take."""
    rows = frame.head(limit) if limit else frame
    payloads: list[dict[str, Any]] = []

    for position, (_, row) in enumerate(rows.iterrows()):
        prospect_id = _clean(row.get("prospect_id")) or f"row-{position + 1}"
        time_on_site = _number(row.get("total_time_spent_on_website")) or 0.0
        payload: dict[str, Any] = {
            "prospect_id": prospect_id[:64],
            "lead_origin": _clean(row.get("lead_origin")) or "Other",
            "total_time_spent_on_website": max(0, int(time_on_site)),
        }

        for column in ("total_visits",):
            value = _number(row.get(column))
            if value is not None:
                payload[column] = max(0, int(value))
        value = _number(row.get("page_views_per_visit"))
        if value is not None:
            payload["page_views_per_visit"] = max(0.0, value)

        for column in ("do_not_email", "do_not_call"):
            if column in rows.columns:
                payload[column] = _to_bool(row.get(column))

        for column in (
            "lead_source",
            "specialization",
            "current_occupation",
            "what_matters_most",
            "city",
            "country",
            "how_did_you_hear",
        ):
            if column in rows.columns:
                cleaned = _clean(row.get(column))
                if cleaned is not None:
                    payload[column] = cleaned

        payloads.append(payload)
    return payloads


# ---------------------------------------------------------------------------
# talking to the API
# ---------------------------------------------------------------------------


def _get(base_url: str, path: str, timeout: float = TIMEOUT_SECONDS) -> dict[str, Any]:
    try:
        response = requests.get(f"{base_url.rstrip('/')}{path}", timeout=timeout)
    except requests.RequestException as error:
        raise ApiUnreachable(f"the scoring service is not responding ({error})") from error
    response.raise_for_status()
    return dict(response.json())


def _post(
    base_url: str, path: str, body: dict[str, Any], timeout: float = TIMEOUT_SECONDS
) -> tuple[int, dict[str, Any]]:
    try:
        response = requests.post(f"{base_url.rstrip('/')}{path}", json=body, timeout=timeout)
    except requests.RequestException as error:
        raise ApiUnreachable(f"the scoring service is not responding ({error})") from error
    try:
        payload = dict(response.json())
    except ValueError:
        payload = {"detail": response.text[:300], "code": "unexpected_response"}
    return response.status_code, payload


def health(base_url: str) -> dict[str, Any]:
    return _get(base_url, "/health", timeout=5.0)


def model_info(base_url: str) -> dict[str, Any]:
    return _get(base_url, "/model-info")


def score_one(base_url: str, payload: dict[str, Any]) -> tuple[int, dict[str, Any]]:
    """Score one lead. Returns the status code so the form can show a field error."""
    return _post(base_url, "/score", payload)


def score_batch(
    base_url: str, payloads: list[dict[str, Any]]
) -> tuple[int, dict[str, Any]]:
    return _post(
        base_url, "/score/batch", {"leads": payloads}, timeout=BATCH_TIMEOUT_SECONDS
    )


# ---------------------------------------------------------------------------
# turning responses into something a table can show
# ---------------------------------------------------------------------------


def results_to_frame(results: list[dict[str, Any]]) -> pd.DataFrame:
    """One row per lead: score, grade, the three reasons, then the identifier."""
    rows = []
    for result in results:
        reasons = result.get("reasons", [])
        rows.append(
            {
                "score": result.get("score"),
                "grade": result.get("grade"),
                "prospect_id": result.get("prospect_id"),
                "probability": result.get("probability"),
                **{
                    f"reason_{index + 1}": reasons[index]["text"] if index < len(reasons) else ""
                    for index in range(3)
                },
            }
        )
    frame = pd.DataFrame(rows)
    return frame.sort_values("score", ascending=False, kind="stable").reset_index(drop=True)


def describe_error(status_code: int, payload: dict[str, Any]) -> str:
    """One readable sentence for any API error, never a traceback (docs/04)."""
    code = str(payload.get("code", "")) or "error"
    detail = str(payload.get("detail", "")) or "the request could not be completed"
    field_name = payload.get("field")

    if status_code == 413:
        return f"Too many leads in one request: {detail}"
    if status_code == 422:
        where = f" (field: {field_name})" if field_name else ""
        return f"That request did not pass validation{where}: {detail}"
    if status_code == 404:
        return f"Not found: {detail}"
    if status_code == 409:
        return f"Already recorded: {detail}"
    if status_code >= 500:
        return (
            "The scoring service hit an internal error. Nothing was scored; the details are "
            "in the service log."
        )
    return f"{code}: {detail}"


def sample_csv(rows: int = 25, seed: int = 11) -> str:
    """A small CSV in the dashboard's own column names, for the empty state.

    Built from the synthetic fixture, so it carries the shapes the pipeline handles - blanks,
    `Select`, missing numbers - rather than twenty-five tidy rows that prove nothing.
    """
    from leadscore import fixtures

    frame = fixtures.synthetic_leads(rows=rows, seed=seed)
    renamed = pd.DataFrame(
        {
            api_field: frame[column]
            for api_field, column in sc.FIELD_TO_COLUMN.items()
            if column in frame.columns
        }
    )
    renamed.insert(0, "prospect_id", [f"sample-{index + 1:03d}" for index in range(len(frame))])
    for column in ("do_not_email", "do_not_call"):
        if column in renamed.columns:
            renamed[column] = renamed[column].map(lambda value: str(value).lower() == "yes")
    return renamed.to_csv(index=False)
