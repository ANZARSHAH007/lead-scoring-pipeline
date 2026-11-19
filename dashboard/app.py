"""The demo surface: three pages over the scoring API (task T-601).

This is a demo, not the product. The pipeline is the product; this exists so that a reviewer
with a browser can watch it work, and so the three flows in docs/04 have somewhere to happen:

1. **Rank leads** - upload a CSV, get it back sorted by score with reasons. The default page.
2. **Try one lead** - one form, one score, change a field and watch the score move.
3. **Model card** - what the model is, what it was measured at, and what it must not be used
   for. Read from `/model-info`; nothing here recomputes a metric.

Every page shows the model version in the footer, so a screenshot is always attributable to
a model (docs/04). Colours and fonts come from the generated design system
(`design-system/lead-scoring-pipeline/MASTER.md`), with chart marks using the validated pair
from `plotting.py` rather than the UI primary - see docs/08 and D-014.

    .venv/Scripts/python -m streamlit run dashboard/app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
# Both paths, and both deliberately: `src` for `leadscore` when it is not pip-installed, and
# the repo root for `dashboard.client`. Streamlit puts the *script's* directory on sys.path,
# not its parent, so without this the import works from the repo root and fails inside the
# container, where the working directory is /app.
for extra in (ROOT, ROOT / "src"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

from dashboard import client  # noqa: E402

from leadscore import config as cfg  # noqa: E402
from leadscore.api import schemas as sc  # noqa: E402

PRIMARY = "#1E40AF"
ACCENT = "#D97706"
INK = "#1E3A8A"
MUTED_INK = "#475569"
SURFACE = "#FFFFFF"
BACKGROUND = "#F8FAFC"
BORDER = "#DBEAFE"

GRADE_COLOURS = {"A": "#15803D", "B": "#B45309", "C": "#64748B"}

PAGES = ("Rank leads", "Try one lead", "Model card")

STYLE = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Fira+Code:wght@400;500;600;700&family=Fira+Sans:wght@300;400;500;600;700&display=swap');
html, body, [class*="css"] {{ font-family: 'Fira Sans', 'Segoe UI', sans-serif; }}
h1, h2, h3 {{ font-family: 'Fira Code', ui-monospace, monospace; color: {INK}; }}
.stApp {{ background: {BACKGROUND}; }}
.grade-chip {{
  display: inline-block; padding: 2px 10px; border-radius: 999px;
  font-weight: 600; font-size: 0.85rem; color: #FFFFFF;
}}
.reason {{
  border-left: 3px solid {ACCENT}; padding: 2px 0 2px 10px; margin: 6px 0;
  color: {INK};
}}
.footer {{
  margin-top: 2.5rem; padding-top: 0.75rem; border-top: 1px solid {BORDER};
  color: {MUTED_INK}; font-family: 'Fira Code', monospace; font-size: 0.78rem;
}}
.kpi {{
  background: {SURFACE}; border: 1px solid {BORDER}; border-radius: 10px;
  padding: 14px 16px;
}}
.kpi-value {{ font-family: 'Fira Code', monospace; font-size: 1.6rem; color: {PRIMARY}; }}
.kpi-label {{ color: {MUTED_INK}; font-size: 0.8rem; }}
</style>
"""


# ---------------------------------------------------------------------------
# shared pieces
# ---------------------------------------------------------------------------


@st.cache_data(ttl=30, show_spinner=False)
def _health(base_url: str) -> dict:
    return client.health(base_url)


@st.cache_data(ttl=60, show_spinner=False)
def _model_info(base_url: str) -> dict:
    return client.model_info(base_url)


def _grade_chip(grade: str) -> str:
    colour = GRADE_COLOURS.get(grade, MUTED_INK)
    return f'<span class="grade-chip" style="background:{colour}">{grade}</span>'


def _kpi(label: str, value: str) -> str:
    return (
        f'<div class="kpi"><div class="kpi-value">{value}</div>'
        f'<div class="kpi-label">{label}</div></div>'
    )


def _service_down(error: Exception) -> None:
    """One message for an unreachable API, on every page (docs/04 flows 1 and 2)."""
    st.error(
        "**The scoring service is not responding.** Nothing was lost - whatever you "
        "uploaded or typed is still here.\n\n"
        f"Tried `{st.session_state.base_url}`. Start the API with:\n\n"
        "```bash\n.venv/Scripts/python -m uvicorn leadscore.api.main:app --port 8000\n```\n\n"
        f"Detail: {error}"
    )


def _footer(version: str | None) -> None:
    st.markdown(
        f'<div class="footer">model <strong>{version or "unavailable"}</strong> &middot; '
        "scores use only what is known when a lead arrives &middot; "
        "a ranking, not a decision</div>",
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# page 1 - rank leads
# ---------------------------------------------------------------------------


def page_rank_leads(base_url: str, max_batch: int) -> None:
    st.title("Rank leads")
    st.caption(
        "Upload a CSV of leads. They come back sorted by score, with the three reasons "
        "behind each one."
    )

    upload = st.file_uploader("Lead CSV", type=["csv"], label_visibility="collapsed")

    if upload is None:
        st.info(
            "**Start with a CSV.** Required columns: "
            f"`{'`, `'.join(client.REQUIRED_COLUMNS)}`. Everything else is optional - "
            "blanks and `Select` are treated as missing, exactly as in training."
        )
        st.download_button(
            "Download a 25-row sample",
            data=client.sample_csv(),
            file_name="sample_leads.csv",
            mime="text/csv",
        )
        st.markdown(
            "The sample is **synthetic**. It carries the shapes the pipeline handles - "
            "blanks, `Select`, missing numbers - so no score from it means anything about a "
            "real lead."
        )
        return

    try:
        frame = client.read_csv(upload.getvalue())
    except Exception as error:
        st.error(f"That file could not be read as CSV: {error}")
        return

    check = client.check_frame(frame, max_batch)
    if not check.ok:
        st.error(check.message)
        if check.missing:
            st.markdown(
                "Rename or add the missing column(s) and upload again. Nothing was sent."
            )
        return
    if check.message:
        st.warning(check.message)
    if check.unknown:
        st.info(
            f"Ignoring {len(check.unknown)} column(s) the model does not use: "
            f"`{'`, `'.join(check.unknown)}`. Columns the sales team fills in after contact "
            "are excluded on purpose - see the model card."
        )

    payloads = client.to_payloads(frame, limit=check.truncated_to)
    with st.spinner(f"Scoring {len(payloads):,} leads..."):
        try:
            status, body = client.score_batch(base_url, payloads)
        except client.ApiUnreachable as error:
            _service_down(error)
            return

    if status != 200:
        st.error(client.describe_error(status, body))
        return

    results = client.results_to_frame(body["results"])
    counts = results["grade"].value_counts()

    columns = st.columns(4)
    with columns[0]:
        st.markdown(_kpi("leads scored", f"{len(results):,}"), unsafe_allow_html=True)
    for index, grade in enumerate("ABC", start=1):
        with columns[index]:
            st.markdown(
                _kpi(f"grade {grade}", f"{int(counts.get(grade, 0)):,}"), unsafe_allow_html=True
            )

    st.markdown("")
    only_a = st.checkbox("Show grade A only", value=False)
    shown = results[results["grade"] == "A"] if only_a else results

    st.dataframe(
        shown,
        hide_index=True,
        width="stretch",
        column_config={
            "score": st.column_config.ProgressColumn(
                "score", min_value=0, max_value=100, format="%d"
            ),
            "probability": st.column_config.NumberColumn("probability", format="%.3f"),
        },
    )

    st.download_button(
        "Download the ranked CSV",
        data=shown.to_csv(index=False),
        file_name="ranked_leads.csv",
        mime="text/csv",
    )


# ---------------------------------------------------------------------------
# page 2 - try one lead
# ---------------------------------------------------------------------------

#: Dataset modes and medians, so the form is never blank (docs/04 surface flow 2).
FORM_DEFAULTS: dict[str, object] = {
    "lead_origin": "Landing Page Submission",
    "lead_source": "Google",
    "total_time_spent_on_website": 248,
    "total_visits": 3,
    "page_views_per_visit": 2.0,
    "current_occupation": "Unemployed",
    "specialization": "Select",
    "what_matters_most": "Better Career Prospects",
    "city": "Mumbai",
    "country": "India",
    "how_did_you_hear": "Select",
}

ORIGINS = ("Landing Page Submission", "API", "Lead Add Form", "Lead Import", "Other")
SOURCES = (
    "Google", "Direct Traffic", "Olark Chat", "Organic Search", "Reference",
    "Welingak Website", "Referral Sites", "Other",
)
OCCUPATIONS = (
    "Select", "Unemployed", "Working Professional", "Student",
    "Businessman", "Housewife", "Other",
)
MOTIVATIONS = ("Select", "Better Career Prospects", "Flexibility & Convenience", "Other")


def page_try_one_lead(base_url: str) -> None:
    st.title("Try one lead")
    st.caption(
        "Every field is one the model is allowed to see. Change one, submit again, and watch "
        "the score move - the cheapest way to see what it is reacting to."
    )

    with st.form("one_lead"):
        left, right = st.columns(2)
        with left:
            lead_origin = st.selectbox("How did the lead arrive?", ORIGINS, index=0)
            lead_source = st.selectbox("Source", SOURCES, index=0)
            time_on_site = st.number_input(
                "Time on website (seconds)", min_value=0, max_value=100_000,
                value=int(FORM_DEFAULTS["total_time_spent_on_website"]), step=30,
                help="The strongest single signal in the data.",
            )
            total_visits = st.number_input(
                "Visits", min_value=0, max_value=500, value=int(FORM_DEFAULTS["total_visits"])
            )
            page_views = st.number_input(
                "Page views per visit", min_value=0.0, max_value=100.0,
                value=float(FORM_DEFAULTS["page_views_per_visit"]), step=0.5,
            )
        with right:
            occupation = st.selectbox("Current occupation", OCCUPATIONS, index=1)
            specialization = st.text_input("Specialisation", value="Select")
            motivation = st.selectbox("What matters most to them", MOTIVATIONS, index=1)
            city = st.text_input("City", value=str(FORM_DEFAULTS["city"]))
            country = st.text_input("Country", value=str(FORM_DEFAULTS["country"]))
            how_heard = st.text_input("How did they hear about the course", value="Select")
            do_not_email = st.checkbox("Opted out of email", value=False)
            do_not_call = st.checkbox("Opted out of calls", value=False)

        submitted = st.form_submit_button("Score this lead", type="primary")

    if not submitted:
        return

    payload = {
        "prospect_id": "dashboard-try-one",
        "lead_origin": lead_origin,
        "lead_source": lead_source,
        "total_time_spent_on_website": int(time_on_site),
        "total_visits": int(total_visits),
        "page_views_per_visit": float(page_views),
        "current_occupation": occupation,
        "specialization": specialization,
        "what_matters_most": motivation,
        "city": city,
        "country": country,
        "how_did_you_hear": how_heard,
        "do_not_email": do_not_email,
        "do_not_call": do_not_call,
    }

    try:
        status, body = client.score_one(base_url, payload)
    except client.ApiUnreachable as error:
        _service_down(error)
        return

    if status != 200:
        st.error(client.describe_error(status, body))
        return

    score = int(body["score"])
    columns = st.columns([1, 1, 2])
    with columns[0]:
        st.markdown(_kpi("score out of 100", str(score)), unsafe_allow_html=True)
    with columns[1]:
        st.markdown(
            _kpi("probability", f"{float(body['probability']):.1%}"), unsafe_allow_html=True
        )
    with columns[2]:
        st.markdown(
            f'<div class="kpi">{_grade_chip(body["grade"])}'
            f'<div class="kpi-label" style="margin-top:8px">grade, from the cut-offs in the '
            "model card</div></div>",
            unsafe_allow_html=True,
        )

    st.subheader("Why")
    for reason in body["reasons"]:
        arrow = "&uarr;" if reason["direction"] == "up" else "&darr;"
        st.markdown(
            f'<div class="reason">{arrow} {reason["text"]} '
            f'<span style="color:{MUTED_INK}">({reason["feature"]})</span></div>',
            unsafe_allow_html=True,
        )
    st.caption(
        "Reasons are feature attributions, not causes. A high score is not advice about what "
        "to change."
    )


# ---------------------------------------------------------------------------
# page 3 - model card
# ---------------------------------------------------------------------------

MUST_NOT = (
    "**Do not discard a lead automatically.** The model ranks; a person decides who to call.",
    "**Do not use it on a different population.** It was fitted on one education provider's "
    "web leads. Another business needs its own measurement.",
    "**Do not read it as a judgement about a person.** A score is a prediction about a "
    "conversion event, and nothing more.",
    "**Do not quote the leakage row as the model's accuracy.** Those columns do not exist "
    "when a lead arrives.",
)


def page_model_card(base_url: str) -> None:
    st.title("Model card")
    st.caption("Everything here is read from the API. No metric on this page is recomputed.")

    try:
        info = _model_info(base_url)
    except client.ApiUnreachable as error:
        _service_down(error)
        return
    except Exception as error:
        st.error(f"The model card could not be read: {error}")
        return

    metrics = info.get("metrics", {})
    columns = st.columns(4)
    for column, (label, key, fmt) in zip(
        columns,
        (
            ("Capture@20%", "capture_at_20", "{:.1%}"),
            ("ROC-AUC", "roc_auc", "{:.3f}"),
            ("PR-AUC", "pr_auc", "{:.3f}"),
            ("Brier", "brier", "{:.3f}"),
        ),
        strict=True,
    ):
        with column:
            st.markdown(
                _kpi(label, fmt.format(float(metrics.get(key, 0.0)))), unsafe_allow_html=True
            )

    st.markdown(
        f"**Version `{info.get('model_version', 'unknown')}`**, trained "
        f"{str(info.get('trained_at', ''))[:10]}. Algorithm: {info.get('algorithm', 'unknown')}."
    )

    st.subheader("Against the baselines")
    baselines = info.get("baselines", {})
    rows = [
        {
            "model": "the shipped model",
            "ROC-AUC": float(metrics.get("roc_auc", 0.0)),
            "Capture@20%": float(metrics.get("capture_at_20", 0.0)),
        }
    ]
    for name, values in baselines.items():
        rows.append(
            {
                "model": name.replace("_", " "),
                "ROC-AUC": float(values.get("roc_auc", 0.0)),
                "Capture@20%": float(values.get("capture_at_20", 0.0)),
            }
        )
    st.dataframe(
        pd.DataFrame(rows),
        hide_index=True,
        width="stretch",
        column_config={
            "Capture@20%": st.column_config.NumberColumn(format="%.1f%%"),
            "ROC-AUC": st.column_config.NumberColumn(format="%.3f"),
        },
    )
    st.caption(
        "The one-rule baseline is a spreadsheet sorted by time on site. It is the bar that "
        "matters, because it is what the alternative actually is."
    )

    st.subheader("Grades")
    bands = info.get("grade_bands", {})
    st.dataframe(
        pd.DataFrame(
            [
                {"grade": grade, "score from": int(bands.get(grade, {}).get("min_score", 0))}
                for grade in ("A", "B", "C")
            ]
        ),
        hide_index=True,
        width="stretch",
    )
    st.caption(
        "Cut-offs were set on validation data and stored in the model card. This page reads "
        "them; it does not know them."
    )

    left, right = st.columns(2)
    with left:
        st.subheader("Features it uses")
        st.markdown(
            "\n".join(f"- `{name}`" for name in info.get("features", []))
            or "_none reported_"
        )
    with right:
        st.subheader("Excluded as leakage")
        st.markdown(
            "\n".join(f"- `{name}`" for name in info.get("excluded_as_leakage", []))
            or "_none reported_"
        )
        st.caption(
            "These are filled in by the sales team **after** contacting a lead. Using them "
            "would score a lead on what is already known about it."
        )

    st.subheader("What this model must not be used for")
    for line in MUST_NOT:
        st.markdown(f"- {line}")

    limits = info.get("limits", [])
    if limits:
        st.subheader("Limitations")
        for limit in limits:
            st.markdown(f"- {limit}")

    figures = ROOT / "reports" / "figures"
    available = [
        ("reliability.png", "Does a score of 70 convert 70% of the time?"),
        ("gains.png", "What the ranking is worth at every calling depth"),
        ("decile_lift.png", "Conversion rate by decile, against the average"),
    ]
    shown = [(figures / name, caption) for name, caption in available if (figures / name).exists()]
    if shown:
        st.subheader("Evidence")
        for path, caption in shown:
            st.image(str(path), caption=caption, width="stretch")


# ---------------------------------------------------------------------------
# shell
# ---------------------------------------------------------------------------


def main() -> None:
    settings = cfg.Settings()
    st.set_page_config(page_title="Lead Scoring", page_icon="📈", layout="wide")
    st.markdown(STYLE, unsafe_allow_html=True)

    st.session_state.setdefault("base_url", settings.api_base_url)

    with st.sidebar:
        st.markdown("### Lead Scoring")
        page = st.radio("Page", PAGES, index=0, label_visibility="collapsed")
        st.markdown("---")
        st.session_state.base_url = st.text_input(
            "Scoring API", value=st.session_state.base_url, help="Where this demo sends requests."
        )

        try:
            status = _health(st.session_state.base_url)
            st.success(f"API up - model `{status.get('model_version', '?')}`")
            version = status.get("model_version")
        except Exception:
            st.warning("API unreachable")
            version = None

        st.markdown("---")
        st.caption(
            f"{len(sc.FIELD_TO_COLUMN)} admissible fields. Columns the sales team fills in "
            "after contact are excluded by design."
        )

    base_url = st.session_state.base_url
    if page == "Rank leads":
        page_rank_leads(base_url, settings.max_batch)
    elif page == "Try one lead":
        page_try_one_lead(base_url)
    else:
        page_model_card(base_url)

    _footer(version)


main()
