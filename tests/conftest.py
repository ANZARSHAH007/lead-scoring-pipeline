"""Test-suite setup that has to happen before anything imports MLflow.

MLflow prints an assistant hint and a stack of database-migration notices on import, which
bury the one line of pytest output that matters. Both are controlled by environment variables
read at import time, so they are set here rather than in a fixture.
"""

from __future__ import annotations

import logging
import os

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

for noisy in ("mlflow", "mlflow.store.db.utils", "mlflow.tracking.fluent", "alembic"):
    logging.getLogger(noisy).setLevel(logging.WARNING)
