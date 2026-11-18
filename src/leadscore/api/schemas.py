"""Request and response models for the API (task T-502).

The boundary is typed so that bad input fails here, with a field name and a reason, before
any model code runs. Three rules shape everything below:

1. **Only admissible columns are accepted.** The request carries exactly the 13 source
   columns the shipping model was fitted on. `last_activity` appeared in the first draft of
   docs/07 and is gone: the activity columns were excluded as leakage (D-016), and accepting
   a field the model cannot use would invite a caller to believe it matters.
2. **Unknown fields are rejected** (`extra="forbid"`). A typo'd field name is a bug in the
   caller, and silently dropping it would produce a confidently wrong score.
3. **Field names are snake_case; the model's are not.** The CSV's columns are
   `"What is your current occupation"` and friends, so every field carries its source column
   in `json_schema_extra` and `to_frame_row()` does the translation in one place.

`"Select"` is accepted anywhere a string is, because it is what the form submits when the
user chose nothing; it is normalised to missing on the way in, exactly as in training.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Annotated, Literal

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, field_validator

from leadscore import config as cfg

#: snake_case API field -> the CSV column the model was fitted on. The single place the two
#: naming schemes meet; `test_api.py` asserts it covers every shipping feature.
FIELD_TO_COLUMN: dict[str, str] = {
    "lead_origin": "Lead Origin",
    "lead_source": "Lead Source",
    "do_not_email": "Do Not Email",
    "do_not_call": "Do Not Call",
    "total_visits": "TotalVisits",
    "total_time_spent_on_website": "Total Time Spent on Website",
    "page_views_per_visit": "Page Views Per Visit",
    "specialization": "Specialization",
    "current_occupation": "What is your current occupation",
    "what_matters_most": "What matters most to you in choosing a course",
    "city": "City",
    "country": "Country",
    "how_did_you_hear": "How did you hear about X Education",
}

NonNegativeInt = Annotated[int, Field(ge=0)]
NonNegativeFloat = Annotated[float, Field(ge=0.0)]


def _yes_no(value: bool) -> str:
    """The CSV encodes the contact flags as 'Yes'/'No'; the API takes real booleans."""
    return "Yes" if value else "No"


class Lead(BaseModel):
    """One lead, as it is known at the moment it arrives."""

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "example": {
                "prospect_id": "7927b2df-8bba-4d29-b9a2-b6e0beafe620",
                "lead_origin": "Landing Page Submission",
                "lead_source": "Google",
                "do_not_email": False,
                "do_not_call": False,
                "total_visits": 5,
                "total_time_spent_on_website": 1532,
                "page_views_per_visit": 2.5,
                "specialization": "Finance Management",
                "current_occupation": "Working Professional",
                "what_matters_most": "Better Career Prospects",
                "city": "Mumbai",
                "country": "India",
                "how_did_you_hear": "Online Search",
            }
        },
    )

    prospect_id: str = Field(
        min_length=1,
        max_length=64,
        description="Your identifier for this lead. Echoed back and written to the log.",
    )
    lead_origin: str = Field(
        min_length=1, description="How the lead arrived. An unknown value scores as 'Other'."
    )
    total_time_spent_on_website: NonNegativeInt = Field(
        description="Seconds on the site before this scoring call. The strongest single signal."
    )

    lead_source: str | None = None
    do_not_email: bool = False
    do_not_call: bool = False
    total_visits: NonNegativeInt | None = None
    page_views_per_visit: NonNegativeFloat | None = None
    specialization: str | None = None
    current_occupation: str | None = None
    what_matters_most: str | None = None
    city: str | None = None
    country: str | None = None
    how_did_you_hear: str | None = None

    @field_validator(
        "lead_source",
        "specialization",
        "current_occupation",
        "what_matters_most",
        "city",
        "country",
        "how_did_you_hear",
        "lead_origin",
        mode="after",
    )
    @classmethod
    def _blank_and_select_are_missing(cls, value: str | None) -> str | None:
        """'Select' is the form's default, not an answer (docs/02 FR-1). So is whitespace."""
        if value is None:
            return None
        stripped = value.strip()
        if not stripped or stripped == cfg.SELECT_PLACEHOLDER:
            return None
        return stripped

    @field_validator("lead_origin", mode="after")
    @classmethod
    def _origin_is_required(cls, value: str | None) -> str:
        if value is None:
            raise ValueError("lead_origin must not be blank or 'Select'")
        return value

    def to_frame_row(self) -> dict[str, object]:
        """This lead as one row keyed by the model's own column names."""
        row: dict[str, object] = {}
        for field, column in FIELD_TO_COLUMN.items():
            value = getattr(self, field)
            if column in cfg.CONTACT_PREFERENCE_FEATURES:
                row[column] = _yes_no(bool(value))
            else:
                row[column] = value
        return row


def leads_to_frame(leads: list[Lead]) -> pd.DataFrame:
    """A frame with exactly the columns the fitted pipeline expects, in its order."""
    frame = pd.DataFrame([lead.to_frame_row() for lead in leads])
    return frame[list(cfg.shipping_features())]


class Reason(BaseModel):
    """One of the three reasons behind a score. No raw SHAP value (docs/02 FR-6)."""

    feature: str = Field(description="The API field name the reason is about.")
    direction: Literal["up", "down"]
    text: str = Field(description="A phrase a salesperson can read aloud.")


class ScoreResult(BaseModel):
    prospect_id: str
    score: int = Field(ge=0, le=100, description="round(100 * probability)")
    probability: float = Field(ge=0.0, le=1.0)
    grade: Literal["A", "B", "C"]
    reasons: list[Reason]
    model_version: str
    scored_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class BatchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    leads: list[Lead] = Field(min_length=1, description="Up to LEADSCORE_MAX_BATCH leads.")


class BatchResponse(BaseModel):
    results: list[ScoreResult]
    count: int
    model_version: str
    request_id: str = Field(description="Groups this batch in the prediction log.")


class FeedbackRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    prospect_id: str = Field(min_length=1, max_length=64)
    converted: bool
    note: str | None = Field(default=None, max_length=500)


class FeedbackResponse(BaseModel):
    recorded: bool
    prediction_id: int


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    model_version: str | None
    model_loaded: bool
    uptime_seconds: float


class GradeBand(BaseModel):
    min_score: int


class ModelInfoResponse(BaseModel):
    """Read from the model card, never recomputed."""

    model_config = ConfigDict(protected_namespaces=())

    model_version: str
    trained_at: str
    algorithm: str
    features: list[str]
    excluded_as_leakage: list[str]
    metrics: dict[str, float]
    baselines: dict[str, dict[str, float]]
    grade_bands: dict[str, GradeBand]
    limits: list[str]


class ErrorResponse(BaseModel):
    """The stable error shape from docs/07. Never an HTML traceback."""

    detail: str
    code: str
    field: str | None = None
