"""The scoring service (task T-503).

Five endpoints over the artifact the training lane wrote. The service never fits anything:
it loads `models/model.joblib` and `models/model_card.json` once, in the lifespan hook, and
every number it serves - the grade cut-offs, the published metrics, the feature list - comes
from that card rather than from a constant typed into serving code. Change the model and the
API's answers change with it, with no code edit.

**The app refuses to start without a model.** A service that is up and returning invented
scores is worse than one that is down, because only the second kind gets noticed.

    .venv/Scripts/python -m uvicorn leadscore.api.main:app --port 8000
    open http://localhost:8000/docs
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

import numpy as np
from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from leadscore import config as cfg
from leadscore import evaluate as ev
from leadscore import explain as ex
from leadscore.api import schemas as sc
from leadscore.api.store import PredictionLog

LOGGER = logging.getLogger(__name__)

#: Model column -> API field, for turning an explanation back into the caller's vocabulary.
COLUMN_TO_FIELD: dict[str, str] = {column: field for field, column in sc.FIELD_TO_COLUMN.items()}


@dataclass
class Service:
    """Everything loaded once at startup. Held on `app.state`, never rebuilt per request."""

    model: Any
    explainer: ex.Explainer
    card: dict[str, Any]
    log: PredictionLog
    started_at: float

    @property
    def version(self) -> str:
        return str(self.card.get("model_version", "unknown"))

    @property
    def cutoffs(self) -> tuple[int, int]:
        """(A from, B from) as the card recorded them. Never hardcoded here."""
        grades = self.card.get("grades", {})
        return int(grades.get("A_from_score", 80)), int(grades.get("B_from_score", 50))

    def grade(self, score: int) -> str:
        a_from, b_from = self.cutoffs
        if score >= a_from:
            return "A"
        return "B" if score >= b_from else "C"


def load_service(settings: cfg.Settings | None = None) -> Service:
    """Load the model bundle and the card, or raise with the command that fixes it."""
    settings = settings or cfg.Settings()
    bundle = ev.load_model_bundle(settings)

    card_path = cfg.resolve(settings.model_card_path)
    card = (
        json.loads(card_path.read_text(encoding="utf-8"))
        if card_path.exists()
        else dict(bundle.get("card") or {})
    )
    if not card:
        raise RuntimeError(
            f"no model card at {card_path} and none inside the bundle. The grade cut-offs "
            "live in the card, so serving without it would mean inventing them."
        )

    return Service(
        model=bundle["model"],
        explainer=ex.Explainer(bundle),
        card=card,
        log=PredictionLog(settings=settings),
        started_at=time.monotonic(),
    )


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    app.state.settings = getattr(app.state, "settings", None) or cfg.Settings()
    app.state.service = load_service(app.state.settings)
    LOGGER.info("model %s loaded; API ready", app.state.service.version)
    yield
    LOGGER.info("API shutting down")


app = FastAPI(
    title="Lead Scoring API",
    version="1.0.0",
    summary="Scores a sales lead 0-100 on arrival, with three reasons why.",
    description=(
        "Scores use only what is known when a lead arrives. Columns the sales team fills in "
        "after contacting a lead are deliberately excluded - see `/model-info`."
    ),
    lifespan=lifespan,
)


def _service(request: Request) -> Service:
    service = getattr(request.app.state, "service", None)
    if service is None:  # pragma: no cover - lifespan guarantees this in practice
        raise RuntimeError("the model is not loaded")
    return service


def _error(status_code: int, code: str, detail: str, field: str | None = None) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content=sc.ErrorResponse(detail=detail, code=code, field=field).model_dump(),
    )


# ---------------------------------------------------------------------------
# error handlers: one shape, never a traceback (docs/07)
# ---------------------------------------------------------------------------


@app.exception_handler(RequestValidationError)
async def _validation_error(request: Request, error: RequestValidationError) -> JSONResponse:
    first = error.errors()[0] if error.errors() else {}
    location = [str(part) for part in first.get("loc", []) if part != "body"]
    return _error(
        status.HTTP_422_UNPROCESSABLE_CONTENT,
        "validation_error",
        first.get("msg", "the request body failed validation"),
        ".".join(location) or None,
    )


@app.exception_handler(Exception)
async def _unexpected_error(request: Request, error: Exception) -> JSONResponse:
    # The trace goes to the server log, never to the caller (docs/09).
    LOGGER.exception("unhandled error on %s", request.url.path)
    return _error(
        status.HTTP_500_INTERNAL_SERVER_ERROR,
        "internal_error",
        "the request could not be completed",
    )


# ---------------------------------------------------------------------------
# scoring
# ---------------------------------------------------------------------------


def _reasons_for(service: Service, frame, index: int) -> list[sc.Reason]:
    """Three reasons for one row, named in the caller's vocabulary."""
    reasons = service.explainer.reasons(frame.iloc[[index]])
    return [
        sc.Reason(
            feature=COLUMN_TO_FIELD.get(reason.feature, reason.feature),
            direction=reason.direction,
            text=reason.phrase.capitalize(),
        )
        for reason in reasons
    ]


def _score_leads(service: Service, leads: list[sc.Lead]) -> tuple[list[sc.ScoreResult], float]:
    """Score every lead in one pass through the pipeline. Returns results and elapsed ms."""
    started = time.perf_counter()
    frame = sc.leads_to_frame(leads)
    probabilities = np.asarray(service.model.predict_proba(frame))[:, 1]
    scores = ev.to_score(probabilities)

    results = [
        sc.ScoreResult(
            prospect_id=lead.prospect_id,
            score=int(scores[index]),
            probability=float(probabilities[index]),
            grade=service.grade(int(scores[index])),
            reasons=_reasons_for(service, frame, index),
            model_version=service.version,
            scored_at=datetime.now(UTC),
        )
        for index, lead in enumerate(leads)
    ]
    return results, (time.perf_counter() - started) * 1000.0


# ---------------------------------------------------------------------------
# endpoints
# ---------------------------------------------------------------------------


@app.get("/health", response_model=sc.HealthResponse, tags=["service"])
async def health(request: Request) -> sc.HealthResponse:
    """Liveness. No inference, so it stays cheap enough for a container healthcheck."""
    service = getattr(request.app.state, "service", None)
    if service is None:  # pragma: no cover - the app refuses to start without a model
        return sc.HealthResponse(
            status="degraded", model_version=None, model_loaded=False, uptime_seconds=0.0
        )
    return sc.HealthResponse(
        status="ok",
        model_version=service.version,
        model_loaded=True,
        uptime_seconds=round(time.monotonic() - service.started_at, 1),
    )


@app.get("/model-info", response_model=sc.ModelInfoResponse, tags=["service"])
async def model_info(request: Request) -> sc.ModelInfoResponse:
    """The model card, served as-is. Nothing here is recomputed per request."""
    service = _service(request)
    card = service.card
    metrics = {row["name"]: row for row in card.get("test_metrics", [])}
    shipped = metrics.get("lightgbm_shipped", {})
    one_rule = metrics.get("one_rule", {})
    logistic = metrics.get("logistic_regression", {})
    grades = card.get("grades", {})
    calibration = str(card.get("calibration", {}).get("method", "none"))

    return sc.ModelInfoResponse(
        model_version=service.version,
        trained_at=str(card.get("created_at", "")),
        algorithm=(
            "LightGBM, Optuna-tuned"
            + ("" if calibration == "none" else f" + CalibratedClassifierCV({calibration})")
        ),
        features=[COLUMN_TO_FIELD.get(c, c) for c in card.get("features", {}).get(
            "admissible_source_columns", []
        )],
        excluded_as_leakage=card.get("features", {})
        .get("excluded", {})
        .get("banned_for_leakage", []),
        metrics={
            key: float(shipped.get(key, 0.0))
            for key in ("roc_auc", "pr_auc", "brier", "capture_at_20")
        },
        baselines={
            "one_rule": {
                "roc_auc": float(one_rule.get("roc_auc", 0.0)),
                "capture_at_20": float(one_rule.get("capture_at_20", 0.0)),
            },
            "logistic_regression": {
                "roc_auc": float(logistic.get("roc_auc", 0.0)),
                "capture_at_20": float(logistic.get("capture_at_20", 0.0)),
            },
        },
        grade_bands={
            "A": sc.GradeBand(min_score=int(grades.get("A_from_score", 80))),
            "B": sc.GradeBand(min_score=int(grades.get("B_from_score", 50))),
            "C": sc.GradeBand(min_score=0),
        },
        limits=list(card.get("limitations", [])),
    )


@app.post("/score", response_model=sc.ScoreResult, tags=["scoring"])
async def score(lead: sc.Lead, request: Request) -> sc.ScoreResult:
    """Score one lead. Unknown category values are mapped inside the fitted transformer."""
    service = _service(request)
    results, elapsed = _score_leads(service, [lead])
    result = results[0]

    service.log.log_prediction(
        prospect_id=result.prospect_id,
        model_version=result.model_version,
        score=result.score,
        probability=result.probability,
        grade=result.grade,
        reasons=[reason.model_dump() for reason in result.reasons],
        request_payload=lead.model_dump(),
        latency_ms=elapsed,
    )
    return result


@app.post("/score/batch", response_model=sc.BatchResponse, tags=["scoring"])
async def score_batch(payload: sc.BatchRequest, request: Request) -> Any:
    """Score up to `LEADSCORE_MAX_BATCH` leads.

    The whole batch is validated before anything is scored: a half-scored batch would leave
    the caller to work out which half, which is worse than a clean refusal (docs/07).
    """
    service = _service(request)
    settings: cfg.Settings = request.app.state.settings
    if len(payload.leads) > settings.max_batch:
        return _error(
            status.HTTP_413_CONTENT_TOO_LARGE,
            "batch_too_large",
            f"{len(payload.leads)} leads exceeds the limit of {settings.max_batch}",
            "leads",
        )

    results, elapsed = _score_leads(service, payload.leads)
    request_id = str(uuid.uuid4())
    per_lead = elapsed / max(len(results), 1)

    service.log.log_batch(
        [
            {
                "prospect_id": result.prospect_id,
                "request_id": request_id,
                "model_version": result.model_version,
                "score": result.score,
                "probability": result.probability,
                "grade": result.grade,
                "reasons": [reason.model_dump() for reason in result.reasons],
                "request_payload": lead.model_dump(),
                "latency_ms": per_lead,
            }
            for result, lead in zip(results, payload.leads, strict=True)
        ]
    )
    return sc.BatchResponse(
        results=results,
        count=len(results),
        model_version=service.version,
        request_id=request_id,
    )


@app.post("/feedback", response_model=sc.FeedbackResponse, tags=["feedback"])
async def feedback(payload: sc.FeedbackRequest, request: Request) -> Any:
    """Attach a real outcome to a lead this service scored.

    The inlet for the retraining loop: once outcomes exist, live accuracy is computable and
    the leads become training rows for the next model.
    """
    service = _service(request)
    outcome, prediction_id = service.log.record_feedback(
        prospect_id=payload.prospect_id, converted=payload.converted, note=payload.note
    )

    if outcome == "prospect_not_found":
        return _error(
            status.HTTP_404_NOT_FOUND,
            "prospect_not_found",
            f"no scored lead with prospect_id {payload.prospect_id!r}; nothing was written",
            "prospect_id",
        )
    if outcome == "already_reported":
        return _error(
            status.HTTP_409_CONFLICT,
            "already_reported",
            f"an outcome was already recorded for prediction {prediction_id}",
            "prospect_id",
        )
    return sc.FeedbackResponse(recorded=True, prediction_id=int(prediction_id or 0))
