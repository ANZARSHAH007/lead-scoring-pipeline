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

import altair as alt
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
SECONDARY = "#3B82F6"
ACCENT = "#D97706"
INK = "#1E3A8A"          # headings, per docs/08 "Foreground"
BODY_INK = "#0F172A"     # body copy: darker than INK, for paragraphs rather than headings
MUTED_INK = "#475569"
SURFACE = "#FFFFFF"
MUTED = "#E9EEF6"
BACKGROUND = "#F8FAFC"
BORDER = "#DBEAFE"

#: Background and foreground per grade, straight from docs/08 §Grade chips. Both halves
#: matter: a chip that sets a background without a matching text colour is how contrast
#: regressions happen.
#:
#: These used to be green/brown/grey, which contradicted the design system in the one place
#: it argues a case: grade A is *not* "success green", because the lead is **likely, not
#: won**, and C is never red, because a C lead is **deprioritised, not an error** (D-035).
GRADE_STYLE = {
    "A": (PRIMARY, "#FFFFFF"),
    "B": (ACCENT, "#000000"),
    "C": (MUTED, MUTED_INK),
}

PAGES = ("Rank leads", "Try one lead", "Model card")

#: One icon per page, drawn from Material Symbols - the icon family Streamlit already ships
#: and renders from its own bundled font, so there is no CDN, no extra dependency and no
#: glyph that renders differently on someone else's machine.
#:
#: NOT emoji. An emoji is a font-dependent picture of a thing, sized and coloured by the
#: platform rather than by this design system; Material Symbols inherit the surrounding text
#: colour and weight and stay one visual family across every surface.
#:
#: Each icon is decorative: every page is still named in words beside it, so nothing is
#: conveyed by the glyph alone and a screen reader loses nothing by skipping it.
PAGE_ICONS = {
    "Rank leads": ":material/leaderboard:",
    "Try one lead": ":material/person_search:",
    "Model card": ":material/fact_check:",
}

STYLE = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Fira+Code:wght@400;500;600;700&family=Fira+Sans:wght@300;400;500;600;700&display=swap');
html, body, [class*="css"] {{ font-family: 'Fira Sans', 'Segoe UI', sans-serif; }}
h1, h2, h3 {{ font-family: 'Fira Code', ui-monospace, monospace; color: {INK}; }}

/* Text colour is stated, never inherited. This page is light by design (docs/08), and the
   CSS used to set a light background while leaving the text to Streamlit's own theme. On a
   viewer whose Streamlit resolved to dark mode that produced near-white text on a near-white
   background - the page rendered and could not be read (D-035). `.streamlit/config.toml`
   pins the theme so this cannot recur; these rules are the second layer, for the case where
   the app runs without that file. */
.stApp {{ background: {BACKGROUND}; color: {BODY_INK}; }}
[data-testid="stMarkdownContainer"] p,
[data-testid="stMarkdownContainer"] li,
[data-testid="stMarkdownContainer"] td,
.stApp label, .stApp .stRadio label {{ color: {BODY_INK}; }}
[data-testid="stCaptionContainer"], .stApp small {{ color: {MUTED_INK}; }}
.stApp a {{ color: {PRIMARY}; }}

/* The spacing scale from design-system/.../MASTER.md, as variables rather than as numbers
   typed into each rule. Everything below spaces itself from these, so "a bit more room
   here" is a token change rather than a new magic number - which is how the stylesheet
   drifted from the design system in the first place. */
:root {{
  --space-sm: 4px; --space-md: 8px; --space-lg: 12px;
  --space-xl: 16px; --space-2xl: 24px; --space-3xl: 32px;
  --radius-control: 8px; --radius-card: 12px;
  --shadow-sm: 0 1px 2px rgba(15, 23, 42, 0.05);
  --shadow-md: 0 4px 6px rgba(15, 23, 42, 0.10);
  --shadow-lg: 0 10px 15px rgba(15, 23, 42, 0.10);
  --ease: 200ms ease;
}}

/* BUTTONS - MASTER.md §Component Specs. The stylesheet had no button rule at all, so every
   control was whatever Streamlit shipped: inconsistent widths, no hierarchy, and a download
   button stretched across a column while the text beside it sat on a different baseline.
   Accent is the call to action, outlined primary is secondary - one CTA per view, per
   docs/08.
   `min-height: 44px` is the touch floor. The web minimum is 24px, but 44 is the figure that
   holds on a phone and a trackpad alike, and these buttons are the only way through the
   page. */
.stButton > button, .stDownloadButton > button, .stFormSubmitButton > button,
[data-testid^="stBaseButton"] {{
  min-height: 44px;
  padding: var(--space-lg) var(--space-2xl);
  border-radius: var(--radius-control);
  font-weight: 600;
  transition: transform var(--ease), box-shadow var(--ease), background var(--ease);
}}
.stButton > button:hover, .stDownloadButton > button:hover,
.stFormSubmitButton > button:hover {{
  transform: translateY(-1px);
  box-shadow: var(--shadow-md);
}}
.stButton > button:active, .stDownloadButton > button:active {{ transform: translateY(0); }}

/* The call to action: accent on black, which docs/08 pairs for 8.4:1 contrast. */
.stFormSubmitButton > button, [data-testid="stBaseButton-primary"] {{
  background: {ACCENT}; color: #000000; border: 1px solid {ACCENT};
}}
.stFormSubmitButton > button:hover, [data-testid="stBaseButton-primary"]:hover {{
  background: #C06A05; border-color: #C06A05; color: #000000;
}}

/* Secondary: outlined, never a second filled button competing with the CTA. */
.stDownloadButton > button, [data-testid="stBaseButton-secondary"] {{
  background: transparent; color: {PRIMARY}; border: 2px solid {PRIMARY};
}}
.stDownloadButton > button:hover, [data-testid="stBaseButton-secondary"]:hover {{
  background: {PRIMARY}; color: #FFFFFF; border-color: {PRIMARY};
}}

/* Focus is never removed, only restyled. Priority 1 of the UI guidance, and the only way
   the page is usable from a keyboard. */
.stApp *:focus-visible {{
  outline: 2px solid {PRIMARY}; outline-offset: 2px; border-radius: var(--radius-control);
}}

/* INPUTS - MASTER.md §Inputs. 16px text, because anything smaller makes iOS Safari zoom the
   whole page on focus. */
.stTextInput input, .stNumberInput input, .stSelectbox div[data-baseweb="select"] > div {{
  border-radius: var(--radius-control) !important;
  font-size: 16px;
  transition: border-color var(--ease), box-shadow var(--ease);
}}
.stTextInput input:focus, .stNumberInput input:focus {{
  border-color: {PRIMARY} !important;
  box-shadow: 0 0 0 3px rgba(30, 64, 175, 0.12) !important;
}}

/* Adjacent controls need room between them: 8px is the floor, and a column gap below that
   turns two buttons into one target. */
[data-testid="stHorizontalBlock"] {{ gap: var(--space-xl); }}

.grade-chip {{
  display: inline-block; padding: var(--space-xs, 2px) 10px; border-radius: 999px;
  font-weight: 600; font-size: 0.85rem;
  transition: transform .15s ease;
}}
.grade-chip:hover {{ transform: scale(1.06); }}
.reason {{
  border-left: 3px solid {ACCENT}; padding: 2px 0 2px 10px; margin: 6px 0;
  color: {BODY_INK};
}}

/* Motion is decoration, so it is small, fast and skippable. Cards rise 6px once on arrival
   and lift on hover; nothing loops, nothing moves while being read. The reduced-motion query
   is not optional politeness - it is the same accessibility floor as the grade letters. */
@keyframes rise {{
  from {{ opacity: 0; transform: translateY(6px); }}
  to   {{ opacity: 1; transform: none; }}
}}
.kpi, .reason {{ animation: rise .28s ease-out both; }}
@media (prefers-reduced-motion: reduce) {{
  .kpi, .reason, .grade-chip {{ animation: none; transition: none; }}
}}
.footer {{
  margin-top: var(--space-3xl); padding-top: var(--space-lg);
  border-top: 1px solid {BORDER};
  color: {MUTED_INK}; font-family: 'Fira Code', monospace; font-size: 0.78rem;
}}
/* CARDS - MASTER.md §Cards: 12px radius, shadow-md, and a 2px lift on hover. They are all
   the same height whatever the number inside, because a row of cards with ragged bottoms is
   the first thing that reads as unfinished. */
.kpi {{
  background: {SURFACE};
  border: 1px solid {BORDER};
  border-radius: var(--radius-card);
  padding: var(--space-xl) var(--space-2xl);
  box-shadow: var(--shadow-sm);
  transition: transform var(--ease), box-shadow var(--ease);
  height: 100%;
  display: flex; flex-direction: column; justify-content: center; gap: var(--space-sm);
}}
.kpi:hover {{ transform: translateY(-2px); box-shadow: var(--shadow-lg); }}
.kpi-value {{
  font-family: 'Fira Code', monospace; font-size: 1.75rem; line-height: 1.1;
  font-weight: 600; color: {PRIMARY};
}}
.kpi-label {{
  color: {MUTED_INK}; font-size: 0.78rem; letter-spacing: 0.02em;
}}
/* Stretch each column so the cards in a row share one height. */
[data-testid="stHorizontalBlock"] > [data-testid="stColumn"] {{ display: flex; }}
[data-testid="stHorizontalBlock"] > [data-testid="stColumn"] > div {{ width: 100%; }}

/* A small caps rule between bands of the page. The data-dense dashboard pattern asks for
   several widgets per screen with minimal padding; without a label per band that reads as
   clutter rather than structure. */
.section {{
  font-family: 'Fira Code', monospace; font-size: 0.72rem; letter-spacing: 0.08em;
  text-transform: uppercase; color: {MUTED_INK};
  margin: var(--space-3xl) 0 var(--space-lg); padding-bottom: var(--space-md);
  border-bottom: 1px solid {BORDER};
}}

/* Density. Streamlit's defaults are built for a wide marketing page; a dashboard wants the
   first row of data above the fold rather than below a screenful of padding. */
.block-container {{ padding-top: 2.2rem; padding-bottom: 3rem; max-width: 1180px; }}
[data-testid="stMetricValue"] {{ font-family: 'Fira Code', monospace; }}

/* NOT a global gap override. Setting `gap` on every stVerticalBlock collapses the spacing
   Streamlit uses to separate a column row from whatever follows it, and the chart ended up
   drawn on top of the KPI cards. Density belongs on the elements this page owns. */
[data-testid="stHorizontalBlock"]:has(.kpi) {{ margin-bottom: var(--space-2xl); }}
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


def _title(name: str) -> None:
    """A page heading with its icon in front of it.

    `st.title`, not a raw `<h1>` in markdown: the raw version looks identical and throws away
    the real heading element, which is what the page's structure, its anchor link and the
    render tests all read. The icon is decoration - the page is still named in words, so
    nothing depends on the glyph rendering (WCAG 1.4.1).
    """
    st.title(f"{PAGE_ICONS.get(name, '')} {name}".strip())


def _grade_chip(grade: str) -> str:
    """One grade's chip: its background and its text colour, never one without the other.

    A chip that sets a background and leaves the foreground to inherit is exactly how the
    page became unreadable in the first place (D-035).
    """
    background, foreground = GRADE_STYLE.get(grade, (MUTED, MUTED_INK))
    return (
        f'<span class="grade-chip" style="background:{background};color:{foreground}">'
        f"{grade}</span>"
    )


def _kpi(label: str, value: str) -> str:
    return (
        f'<div class="kpi"><div class="kpi-value">{value}</div>'
        f'<div class="kpi-label">{label}</div></div>'
    )


def _headline(info: dict) -> None:
    """The four numbers that say whether this model is worth anything, shown on arrival.

    A page whose whole content is "upload a file" tells a visitor nothing about the thing
    they are being asked to use. These come from `/model-info`, which reads the model card,
    so they move when the model does and are never typed into the UI.
    """
    metrics = info.get("metrics") or {}
    bands = info.get("grade_bands") or {}
    cells = (
        ("capture @ top 20%", f"{metrics.get('capture_at_20', 0):.1%}"),
        ("ROC-AUC", f"{metrics.get('roc_auc', 0):.3f}"),
        ("Brier - lower is better", f"{metrics.get('brier', 0):.3f}"),
        ("grade A from", str((bands.get("A") or {}).get("min_score", "-"))),
    )
    for column, (label, value) in zip(st.columns(len(cells)), cells, strict=True):
        with column:
            st.markdown(_kpi(label, value), unsafe_allow_html=True)


#: Baseline keys as `/model-info` returns them, in the words a reader needs.
BASELINE_LABELS = {
    "one_rule": "One rule: time on site",
    "logistic_regression": "Logistic regression",
    "random": "Random order",
}


def capture_frame(info: dict) -> pd.DataFrame:
    """The shipping model and its baselines on the headline metric, ready to plot.

    Split out from the chart so it can be tested without rendering anything.
    """
    metrics = info.get("metrics") or {}
    rows = [
        {
            "approach": "This model",
            "capture": float(metrics.get("capture_at_20") or 0.0),
            "shipping": True,
        }
    ]
    for key, baseline in (info.get("baselines") or {}).items():
        value = (baseline or {}).get("capture_at_20")
        if value is not None:
            rows.append(
                {
                    "approach": BASELINE_LABELS.get(key, key.replace("_", " ")),
                    "capture": float(value),
                    "shipping": False,
                }
            )
    return pd.DataFrame(rows)


def _capture_chart(info: dict) -> None:
    """The project's whole claim, as one chart: this model against what it has to beat.

    Horizontal bars, sorted, with the value written at the end of each one - docs/08 asks
    for value labels rather than a gridline the reader has to measure against. The shipping
    model is the only bar in the primary colour, and it is also the only one labelled "This
    model", so the colour is never the sole encoding (WCAG 1.4.1).
    """
    frame = capture_frame(info)
    if len(frame) < 2:  # nothing to compare against; a single bar says nothing
        return

    base = alt.Chart(frame).encode(
        y=alt.Y("approach:N", sort="-x", title=None, axis=alt.Axis(labelFontSize=13)),
        x=alt.X(
            "capture:Q",
            title="share of all conversions reached by calling the top 20%",
            axis=alt.Axis(format="%", grid=False),
            scale=alt.Scale(domain=[0, max(0.6, float(frame["capture"].max()) * 1.25)]),
        ),
        tooltip=[
            alt.Tooltip("approach:N", title="approach"),
            alt.Tooltip("capture:Q", title="capture @ 20%", format=".1%"),
        ],
    )
    bars = base.mark_bar(height=20, cornerRadiusEnd=3).encode(
        color=alt.condition(alt.datum.shipping, alt.value(PRIMARY), alt.value("#94A3B8"))
    )
    labels = base.mark_text(align="left", dx=6, fontSize=13, color=BODY_INK).encode(
        text=alt.Text("capture:Q", format=".1%")
    )
    st.altair_chart(
        (bars + labels)
        .properties(height=max(150, 46 * len(frame)), padding={"top": 10, "bottom": 6})
        .configure_view(stroke=None)
        .configure_axis(labelColor=MUTED_INK, titleColor=MUTED_INK, domainColor=BORDER),
        width='stretch',
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
    _title("Rank leads")
    st.caption(
        "Upload a CSV of leads. They come back sorted by score, with the three reasons "
        "behind each one."
    )

    upload = st.file_uploader("Lead CSV", type=["csv"], label_visibility="collapsed")

    if upload is None:
        # The landing state is a dashboard, not a prompt. A page whose entire content is
        # "upload a file" says nothing about the model a visitor is being asked to trust,
        # and it is the first thing anyone opening the demo link sees (D-036).
        try:
            info = _model_info(base_url)
        except client.ApiUnreachable as error:
            _service_down(error)
            return

        st.markdown(
            '<div class="section">Measured on held-out leads</div>',
            unsafe_allow_html=True,
        )
        _headline(info)
        _capture_chart(info)

        st.markdown(
            '<div class="section">Score your own leads</div>',
            unsafe_allow_html=True,
        )
        left, right = st.columns([3, 2])
        with left:
            st.markdown(
                "Required columns: "
                f"`{'`, `'.join(client.REQUIRED_COLUMNS)}`. Everything else is optional - "
                "blanks and `Select` are treated as missing, exactly as in training."
            )
        with right:
            # Not stretched across the column. A button sized to its own label sits on the
            # same baseline as the text beside it; one stretched to the full column width
            # does not, and reads as a banner rather than a control.
            st.download_button(
                "Download a 25-row sample",
                data=client.sample_csv(),
                file_name="sample_leads.csv",
                mime="text/csv",
            )
        st.caption(
            "The sample is synthetic. It carries the shapes the pipeline handles - blanks, "
            "`Select`, missing numbers - so no score from it means anything about a real lead."
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
    _title("Try one lead")
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
    _title("Model card")
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
    st.set_page_config(
        page_title="Lead Scoring", page_icon=":material/query_stats:", layout="wide"
    )
    st.markdown(STYLE, unsafe_allow_html=True)

    st.session_state.setdefault("base_url", settings.api_base_url)

    with st.sidebar:
        st.markdown("### Lead Scoring")
        page = st.radio(
            "Page",
            PAGES,
            index=0,
            label_visibility="collapsed",
            format_func=lambda name: f"{PAGE_ICONS.get(name, '')}  {name}",
        )
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
