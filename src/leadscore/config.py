"""Single source of truth for paths, thresholds, the random seed and the feature lists.

Nothing else in the pipeline may hardcode a threshold, a path or a column name. Two kinds of
setting live here, and they are deliberately different kinds of thing:

* **Column groups** are module constants. They encode the leakage decision recorded in
  docs/02_PRODUCT_REQUIREMENTS.md, which is a business rule, not a deployment knob. Nobody
  gets to re-admit a leaking column by setting an environment variable.
* **Everything else** is a :class:`Settings` field, read from the environment with the
  ``LEADSCORE_`` prefix and a documented default (see ``.env.example``).

See docs/03_FEATURE_BREAKDOWN.md (M1) and CLAUDE.md.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# ---------------------------------------------------------------------------
# The raw dataset's shape. Checked by schema.py (Phase 2) so drift is loud.
# ---------------------------------------------------------------------------

#: The repository root, derived from this file's location rather than the working
#: directory. `src/leadscore/config.py` -> up three -> the repo.
PROJECT_ROOT = Path(__file__).resolve().parents[2]


def resolve(path: Path) -> Path:
    """Turn a configured path into an absolute one, against the project root.

    Every path in `Settings` has a *relative* default, because a default that embedded one
    machine's directory layout would be useless to anyone else. Relative to what, though, is
    a decision: relative to the current directory means `python scripts/eda.py` from a
    subfolder quietly writes its report somewhere new, which is how a reports path ends up
    hardcoded from `__file__` to avoid the problem. Relative to the project root means the
    pipeline behaves the same from anywhere, and an absolute override — a temporary directory
    in a test, a mounted volume in a container — is still passed through untouched.
    """
    return path if path.is_absolute() else PROJECT_ROOT / path


TARGET = "Converted"
EXPECTED_ROWS = 9_240
EXPECTED_COLUMNS = 37
EXPECTED_TARGET_RATE = 0.3854

#: A form default, not an answer. Treated as missing everywhere it appears.
SELECT_PLACEHOLDER = "Select"

# ---------------------------------------------------------------------------
# Column groups. Admissibility comes from one rule: a column may be used only if its
# value is known at the moment the lead is scored. docs/02 holds the per-column reasoning.
# ---------------------------------------------------------------------------

#: Identifiers. `Prospect ID` is kept to join results back, never as a feature.
ID_COLUMNS: tuple[str, ...] = (
    "Prospect ID",
    "Lead Number",
)

#: Known when the lead arrives.
SOURCE_FEATURES: tuple[str, ...] = (
    "Lead Origin",
    "Lead Source",
)

#: Pre-contact behaviour. Expected to be the strongest admissible signals.
NUMERIC_FEATURES: tuple[str, ...] = (
    "TotalVisits",
    "Total Time Spent on Website",
    "Page Views Per Visit",
)

#: Self-reported at submission. `Select` is treated as missing; rare levels are grouped.
FORM_FEATURES: tuple[str, ...] = (
    "Specialization",
    "What is your current occupation",
    "What matters most to you in choosing a course",
    "City",
    "Country",
    "How did you hear about X Education",
)

#: Stated by the lead on the form.
CONTACT_PREFERENCE_FEATURES: tuple[str, ...] = (
    "Do Not Email",
    "Do Not Call",
)

#: Ad and opt-in flags. Near-constant ones are dropped in Phase 2 once measured (T-203).
FLAG_FEATURES: tuple[str, ...] = (
    "Search",
    "Magazine",
    "Newspaper Article",
    "X Education Forums",
    "Newspaper",
    "Digital Advertisement",
    "Through Recommendations",
    "Receive More Updates About Our Courses",
    "Update me on Supply Chain Content",
    "Get updates on DM Content",
    "I agree to pay the amount through cheque",
    "A free copy of Mastering The Interview",
)

#: Suspect: some values may record that a rep already made contact. Trained both ways,
#: both reported (docs/02 FR-2, assumption A-2). Never silently included.
ACTIVITY_FEATURES: tuple[str, ...] = (
    "Last Activity",
    "Last Notable Activity",
)

#: Written by the sales team *after* working the lead. These leak the outcome and are
#: banned from any shipping model. They appear only in the labelled, unshippable
#: "leakage row" of the results table.
BANNED_LEAKAGE_COLUMNS: tuple[str, ...] = (
    "Tags",
    "Lead Quality",
    "Asymmetrique Activity Index",
    "Asymmetrique Profile Index",
    "Asymmetrique Activity Score",
    "Asymmetrique Profile Score",
)

#: Too sparse to learn from: 74.2% empty once `Select` counts as missing.
DROPPED_SPARSE_COLUMNS: tuple[str, ...] = ("Lead Profile",)

FeatureSet = Literal["shipping", "with_activity", "audit_with_leakage"]


def categorical_features() -> tuple[str, ...]:
    """Admissible categorical columns.

    Excludes the ad and opt-in flags (near-constant, measured in reports/01) and the
    activity columns (contaminated, measured in reports/02).
    """
    return SOURCE_FEATURES + FORM_FEATURES + CONTACT_PREFERENCE_FEATURES


def shipping_features() -> tuple[str, ...]:
    """The 13 columns a shippable model may use. Frozen by T-205 on 2026-11-17.

    Contains no banned column, no near-constant flag and no activity column, by
    construction rather than by convention.
    """
    return NUMERIC_FEATURES + categorical_features()


def with_activity_features() -> tuple[str, ...]:
    """The shipping set plus the two activity columns.

    A labelled comparison only, never shipped: their values mix lead behaviour with
    company actions (`SMS Sent`, `Unreachable`), which cannot exist for a lead that has
    just arrived. Worth +0.032 ROC-AUC and excluded anyway. See reports/02 and D-016.
    """
    return shipping_features() + ACTIVITY_FEATURES


def audit_features() -> tuple[str, ...]:
    """Everything, including the sales-filled columns.

    Used exactly once, to quantify the leakage gap for the README's last row. A model
    trained on this set must never be written to ``models/``.
    """
    return with_activity_features() + BANNED_LEAKAGE_COLUMNS


def feature_set(name: FeatureSet) -> tuple[str, ...]:
    """Resolve a named feature set."""
    resolvers = {
        "shipping": shipping_features,
        "with_activity": with_activity_features,
        "audit_with_leakage": audit_features,
    }
    if name not in resolvers:
        raise ValueError(
            f"unknown feature set {name!r}; expected one of {sorted(resolvers)}"
        )
    return resolvers[name]()


def excluded_columns() -> tuple[str, ...]:
    """Columns that never reach a shippable model, and why each is out.

    Identifiers (no signal), sales-filled (leakage), sparse, near-constant flags, and the
    activity columns (post-contact values).
    """
    return (
        ID_COLUMNS
        + BANNED_LEAKAGE_COLUMNS
        + DROPPED_SPARSE_COLUMNS
        + FLAG_FEATURES
        + ACTIVITY_FEATURES
    )


# ---------------------------------------------------------------------------
# Environment-driven settings
# ---------------------------------------------------------------------------


class Settings(BaseSettings):
    """Runtime configuration. Frozen: load it once, pass it down.

    Every field has a documented default, so the pipeline runs with no ``.env`` file at all.
    A bad value raises on construction rather than failing later in a training run.
    """

    model_config = SettingsConfigDict(
        env_prefix="LEADSCORE_",
        env_file=(".env", ".env.local"),
        env_file_encoding="utf-8",
        extra="ignore",
        frozen=True,
    )

    # --- paths ---
    data_raw: Path = Field(
        default=Path("data/raw/Leads.csv"),
        description="The raw dataset. Git-ignored; downloaded by the user.",
    )
    model_path: Path = Field(default=Path("models/model.joblib"))
    model_card_path: Path = Field(default=Path("models/model_card.json"))
    db_path: Path = Field(default=Path("data/predictions.db"))
    reports_dir: Path = Field(
        default=Path("reports"),
        description=(
            "Where generated reports and figures land. Overridden in tests and CI so a "
            "fixture run can never overwrite a report built from the real data."
        ),
    )

    # --- training ---
    random_seed: int = Field(default=42, ge=0)
    test_size: float = Field(default=0.2, gt=0.0, lt=1.0)
    cv_folds: int = Field(default=5, ge=2)
    optuna_trials: int = Field(
        default=40,
        ge=0,
        description="0 disables tuning and uses LightGBM defaults. First item on the cut list.",
    )
    rare_level_threshold: float = Field(
        default=0.01,
        gt=0.0,
        lt=1.0,
        description="Categorical levels rarer than this are grouped into 'Other'.",
    )
    target_rate_tolerance: float = Field(
        default=0.02,
        gt=0.0,
        lt=1.0,
        description="How far the loaded data's conversion rate may drift before a warning.",
    )

    # --- quality gate (docs/10) ---
    gate_min_roc_auc: float = Field(default=0.80, ge=0.5, le=1.0)
    gate_max_brier: float = Field(default=0.18, gt=0.0, le=1.0)

    # --- grades: a policy in percentiles, not score cut-offs ---
    # The actual 0-100 cut-offs are computed on validation data and written to the model
    # card, because they depend on the fitted score distribution. Nothing hardcodes them.
    grade_a_percentile: float = Field(
        default=0.80, gt=0.0, lt=1.0, description="Top 20% of scores are grade A."
    )
    grade_b_percentile: float = Field(
        default=0.50, gt=0.0, lt=1.0, description="The next 30% are grade B; the rest are C."
    )

    # --- api ---
    api_host: str = Field(default="0.0.0.0")  # noqa: S104  - containerised; see docs/09
    api_port: int = Field(default=8000, ge=1, le=65_535)
    max_batch: int = Field(default=1_000, ge=1)
    api_key: str = Field(
        default="",
        description="Optional shared secret for write endpoints. Empty means open (local only).",
    )
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"

    # --- dashboard ---
    api_base_url: str = Field(default="http://localhost:8000")

    # --- experiment tracking ---
    # MLflow's own variable name, not the LEADSCORE_ prefix, so the standard tooling and
    # this project read the same setting. Matches .env.example.
    mlflow_tracking_uri: str = Field(
        default="sqlite:///mlflow.db",
        validation_alias="MLFLOW_TRACKING_URI",
        description=(
            "Local SQLite file by default: no server, no account. MLflow 3.18 retired the "
            "plain `file:./mlruns` store, so a database backend is the supported local path."
        ),
    )

    @field_validator("data_raw", "model_path", "model_card_path", "db_path", "reports_dir")
    @classmethod
    def _no_empty_path(cls, value: Path) -> Path:
        if str(value).strip() in {"", "."}:
            raise ValueError("path must not be empty")
        return value

    @model_validator(mode="after")
    def _grades_are_ordered(self) -> Settings:
        if self.grade_b_percentile >= self.grade_a_percentile:
            raise ValueError(
                "grade_b_percentile must be below grade_a_percentile "
                f"(got B={self.grade_b_percentile}, A={self.grade_a_percentile})"
            )
        return self

    @property
    def figures_dir(self) -> Path:
        """Charts live beside the reports that embed them."""
        return self.reports_dir / "figures"

    @property
    def grade_band_policy(self) -> dict[str, float]:
        """The band policy, as written into the model card."""
        return {
            "A_from_percentile": self.grade_a_percentile,
            "B_from_percentile": self.grade_b_percentile,
            "C_from_percentile": 0.0,
        }


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """The process-wide settings object. Cached, so two calls return the same instance."""
    return Settings()
