"""The prediction log (task T-504).

Every score the API hands out is written here: what was asked, what came back, which model
version answered and how long it took. That turns three otherwise unanswerable questions into
queries — what is the live latency, what does the score distribution look like in production,
and once `/feedback` supplies outcomes, is the model still as accurate as the card claims.

Two design rules, both from docs/06 and docs/07:

* **A failed log write never fails a response.** Scoring is the service; logging is
  bookkeeping. `log_prediction` catches its own errors, reports them and returns `None`, so a
  full disk degrades observability rather than the product.
* **The input is stored as JSON, verbatim.** Not re-serialised from the parsed model and not
  normalised: when a score looks wrong later, the question is what the caller actually sent.

SQLite because the log is append-mostly, single-writer, and belongs in one file that is easy
to inspect with any tool. If this ever needs concurrent writers it needs Postgres, and that
is a different project.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from leadscore import config as cfg

LOGGER = logging.getLogger(__name__)

SCHEMA = """
CREATE TABLE IF NOT EXISTS predictions (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    prospect_id     TEXT    NOT NULL,
    request_id      TEXT,
    scored_at       TEXT    NOT NULL,
    model_version   TEXT    NOT NULL,
    score           INTEGER NOT NULL,
    probability     REAL    NOT NULL,
    grade           TEXT    NOT NULL,
    reasons_json    TEXT    NOT NULL,
    input_json      TEXT    NOT NULL,
    latency_ms      REAL    NOT NULL
);

-- Feedback is a separate table rather than a nullable column on `predictions`: an outcome
-- arrives later, from a different caller, and may never arrive at all.
CREATE TABLE IF NOT EXISTS feedback (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    prediction_id   INTEGER NOT NULL REFERENCES predictions(id),
    prospect_id     TEXT    NOT NULL,
    converted       INTEGER NOT NULL,
    note            TEXT,
    recorded_at     TEXT    NOT NULL,
    UNIQUE (prediction_id)
);

CREATE INDEX IF NOT EXISTS idx_predictions_prospect ON predictions(prospect_id);
CREATE INDEX IF NOT EXISTS idx_predictions_scored_at ON predictions(scored_at);
CREATE INDEX IF NOT EXISTS idx_predictions_request ON predictions(request_id);
"""


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="milliseconds")


class PredictionLog:
    """A SQLite prediction log. Cheap to construct; the file is created on first use."""

    def __init__(self, path: Path | str | None = None, settings: cfg.Settings | None = None):
        settings = settings or cfg.Settings()
        self.path = Path(path) if path is not None else cfg.resolve(settings.db_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._initialise()

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.path, timeout=5.0)
        connection.row_factory = sqlite3.Row
        try:
            yield connection
            connection.commit()
        finally:
            connection.close()

    def _initialise(self) -> None:
        with self._connect() as connection:
            # WAL so a reader (the dashboard) never blocks the writer (the API).
            connection.execute("PRAGMA journal_mode=WAL")
            connection.executescript(SCHEMA)

    # -- writes ------------------------------------------------------------

    def log_prediction(
        self,
        *,
        prospect_id: str,
        model_version: str,
        score: int,
        probability: float,
        grade: str,
        reasons: list[dict[str, Any]],
        request_payload: dict[str, Any],
        latency_ms: float,
        request_id: str | None = None,
    ) -> int | None:
        """Write one scored lead. Returns its row id, or `None` if the write failed.

        Never raises. A caller that cannot log a prediction has still produced a correct
        score, and the response belongs to the user rather than to the bookkeeping.
        """
        try:
            with self._connect() as connection:
                cursor = connection.execute(
                    """
                    INSERT INTO predictions (
                        prospect_id, request_id, scored_at, model_version, score,
                        probability, grade, reasons_json, input_json, latency_ms
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        prospect_id,
                        request_id,
                        _now(),
                        model_version,
                        int(score),
                        float(probability),
                        grade,
                        json.dumps(reasons, default=str),
                        json.dumps(request_payload, default=str),
                        float(latency_ms),
                    ),
                )
                return int(cursor.lastrowid or 0)
        except sqlite3.Error as error:
            LOGGER.error("prediction log write failed for %s: %s", prospect_id, error)
            return None

    def log_batch(self, rows: list[dict[str, Any]]) -> list[int]:
        """Write a whole batch in one transaction. Returns the ids it managed to write."""
        if not rows:
            return []
        try:
            with self._connect() as connection:
                written: list[int] = []
                for row in rows:
                    cursor = connection.execute(
                        """
                        INSERT INTO predictions (
                            prospect_id, request_id, scored_at, model_version, score,
                            probability, grade, reasons_json, input_json, latency_ms
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            row["prospect_id"],
                            row.get("request_id"),
                            _now(),
                            row["model_version"],
                            int(row["score"]),
                            float(row["probability"]),
                            row["grade"],
                            json.dumps(row["reasons"], default=str),
                            json.dumps(row["request_payload"], default=str),
                            float(row["latency_ms"]),
                        ),
                    )
                    written.append(int(cursor.lastrowid or 0))
                return written
        except sqlite3.Error as error:
            LOGGER.error("prediction log batch write failed (%d rows): %s", len(rows), error)
            return []

    def record_feedback(
        self, *, prospect_id: str, converted: bool, note: str | None = None
    ) -> tuple[str, int | None]:
        """Attach a real outcome to the most recent prediction for a lead.

        Returns `(status, prediction_id)` where status is one of `recorded`,
        `prospect_not_found` or `already_reported`. Unlike `log_prediction`, this one is
        allowed to report failure: the caller asked for something that did not happen.
        """
        with self._connect() as connection:
            row = connection.execute(
                "SELECT id FROM predictions WHERE prospect_id = ? ORDER BY id DESC LIMIT 1",
                (prospect_id,),
            ).fetchone()
            if row is None:
                return "prospect_not_found", None

            prediction_id = int(row["id"])
            existing = connection.execute(
                "SELECT id FROM feedback WHERE prediction_id = ?", (prediction_id,)
            ).fetchone()
            if existing is not None:
                return "already_reported", prediction_id

            connection.execute(
                """
                INSERT INTO feedback (prediction_id, prospect_id, converted, note, recorded_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (prediction_id, prospect_id, int(converted), note, _now()),
            )
            return "recorded", prediction_id

    # -- reads -------------------------------------------------------------

    def count(self) -> int:
        with self._connect() as connection:
            return int(connection.execute("SELECT COUNT(*) FROM predictions").fetchone()[0])

    def recent(self, limit: int = 50) -> list[dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM predictions ORDER BY id DESC LIMIT ?", (limit,)
            ).fetchall()
        return [dict(row) for row in rows]

    def by_request(self, request_id: str) -> list[dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM predictions WHERE request_id = ? ORDER BY id", (request_id,)
            ).fetchall()
        return [dict(row) for row in rows]

    def latency_percentile(self, percentile: float = 0.95) -> float | None:
        """Measured p95 from real traffic, which is how NFR-1 is checked rather than claimed."""
        with self._connect() as connection:
            values = [
                float(row[0])
                for row in connection.execute(
                    "SELECT latency_ms FROM predictions ORDER BY latency_ms"
                ).fetchall()
            ]
        if not values:
            return None
        index = min(len(values) - 1, round(percentile * (len(values) - 1)))
        return values[index]

    def live_accuracy(self) -> dict[str, float | int] | None:
        """Accuracy on the leads whose real outcome came back through `/feedback`."""
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT p.score, p.grade, f.converted
                FROM feedback f JOIN predictions p ON p.id = f.prediction_id
                """
            ).fetchall()
        if not rows:
            return None
        converted = [int(row["converted"]) for row in rows]
        grade_a = [int(row["converted"]) for row in rows if row["grade"] == "A"]
        return {
            "leads_with_outcomes": len(rows),
            "conversion_rate": sum(converted) / len(converted),
            "grade_a_leads": len(grade_a),
            "grade_a_conversion_rate": (sum(grade_a) / len(grade_a)) if grade_a else 0.0,
        }
