"""SCRUM-189: Streamlit app for the Day 30 early-warning model.

Run from the repo root:  streamlit run app/streamlit_app.py

Layout, built around a tutor's workflow, in a dark card-based theme:
* Sidebar: app name and purpose, page navigation, the controls every page uses (presentation, module,
  contact capacity) and an Advanced section (training presentations with their in-sample note, flagging
  rule, capacity scope).
* Pages: Overview (who to contact first), Student (why, and what support fits), Modules (bands,
  outcomes and retention by module), About (model, design, limits, data). Only the selected page runs.

* The final model (models/final/) scores students with a calibrated risk score, a risk band
  (Low, Medium, High) and a rank. Students are flagged by capacity ranking or by the wide-net cutoff.
* Data, model, explainer, precomputed explanations and the similar-student pool load once (cached).
* Anything missing shows a message; the app does not crash on it. File names, build commands and
  build dates are kept on the About page.
* The validation presentation is shown by default. A training presentation, whose scores are
  in-sample, is offered only after the user opts in under Advanced, and is labelled wherever shown.
* Displays are aggregated by default; individual students appear only in the priority list and the
  student view. Groups under MIN_CELL students are suppressed in every aggregated table.
"""

import sys
import textwrap
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import plotly.graph_objects as go  # noqa: E402
import plotly.io as pio  # noqa: E402
import streamlit as st  # noqa: E402

from sar.config import CUTOFF_DAY, RANDOM_SEED  # noqa: E402
from sar.data.split import TEST_PRESENTATION, make_split, overlap_report  # noqa: E402
from sar.explain.chart import explanation_table  # noqa: E402
from sar.explain.shap_explain import (  # noqa: E402
    explain_rows,
    explain_student,
    explanation_from_contributions,
    make_explainer,
)
from sar.recommend.interventions import ACTIONABLE_FEATURES  # noqa: E402
from sar.recommend.neighbours import NeighbourPool  # noqa: E402
from sar.recommend.recommender import NOTE, recommend  # noqa: E402
from sar.service.data import (  # noqa: E402
    OPTIONAL_TABLES,
    SHAP_PREFIX,
    Loaded,
    TableStatus,
    load_final_model,
    load_matrix,
    load_shap_cache,
    resolve_paths,
    table_status,
)
from sar.service.figures import band_figure, hidden_bands  # noqa: E402
from sar.service.priority import (  # noqa: E402
    OTHER,
    priority_details,
    reason_counts,
    selected_row,
    short_reason,
    similar_line,
    snapshot,
    support_sentence,
    verdict_sentence,
)
from sar.service.views import (  # noqa: E402
    KEY,
    MIN_CELL,
    band_evidence,
    batch_summary,
    calibration_bins,
    flag_capacity,
    flag_outcome,
    flag_wide_net,
    flagged_by_band,
    example_student,
    is_training_presentation,
    module_rank,
    risk_band_edges,
    risk_band_table,
    score_presentation,
)
from sar.tables.department import ALL_MODULES, BAND_RATES, BAND_SHARES  # noqa: E402
from sar.tables.profiles import PROFILE_FEATURES  # noqa: E402

# Dark theme: one palette for bands, text and lines, used by the CSS and the Plotly template alike.
BAND_COLOURS = {"High": "#ff5d73", "Medium": "#ffb454", "Low": "#35d0ba"}
TEXT, MUTED, GRID, ACCENT, ACCENT_2 = "#e8ecff", "#9aa3c7", "rgba(154,163,199,0.18)", "#7c6cff", "#4f8cff"
STYLE = """<style>
.stApp {background: radial-gradient(circle at 15% 0%, rgba(124,108,255,0.22), transparent 45%),
        radial-gradient(circle at 90% 10%, rgba(79,140,255,0.18), transparent 40%), #0b1020;}
[data-testid="stSidebar"] > div:first-child {background: linear-gradient(180deg, #131a33 0%, #0d1328 100%);}
[data-testid="stMainBlockContainer"] {padding-top: 2rem; padding-bottom: 2rem;}
[data-testid="stMainBlockContainer"] [data-testid="stVerticalBlock"] {gap: 0.8rem;}
[class*="st-key-card-"], [data-testid="stMetric"] {
    border: 1px solid rgba(124,108,255,0.35) !important; border-radius: 14px !important; padding: 1rem 1.1rem;
    background: linear-gradient(160deg, rgba(124,108,255,0.10), rgba(19,26,51,0.65));
    box-shadow: 0 0 22px rgba(124,108,255,0.12);}
[data-testid="stExpander"] details {border: 1px solid rgba(124,108,255,0.25); border-radius: 14px;}
.pill {display: inline-block; padding: 0.15rem 0.7rem; border-radius: 999px; font-size: 0.85rem; font-weight: 600;
       border: 1px solid currentColor;}
.pill-high {color: #ff5d73; background: rgba(255,93,115,0.14);}
.pill-medium {color: #ffb454; background: rgba(255,180,84,0.14);}
.pill-low {color: #35d0ba; background: rgba(53,208,186,0.14);}
.pill-info {color: #b9b2ff; background: rgba(124,108,255,0.16);}
.page-subtitle {color: #9aa3c7; margin-top: -0.6rem;}
.brand {font-size: 1.15rem; font-weight: 700; color: #e8ecff;}
.st-key-nav [role="radiogroup"] {gap: 0.2rem;}
.st-key-nav [role="radiogroup"] label {width: 100%; padding: 0.45rem 0.7rem; border-radius: 10px; margin: 0;}
.st-key-nav [role="radiogroup"] label > div:first-child {display: none;}
.st-key-nav [role="radiogroup"] label:has(input:checked) {background: rgba(124,108,255,0.25);
       box-shadow: inset 3px 0 0 #7c6cff;}
.st-key-nav [role="radiogroup"] label:hover {background: rgba(124,108,255,0.12);}
</style>"""
CHART_HEIGHT = 380
pio.templates["sar_dark"] = go.layout.Template(layout=go.Layout(
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", height=CHART_HEIGHT,
    font={"color": TEXT, "family": "Inter, Segoe UI, sans-serif", "size": 13}, title={"font": {"size": 15}},
    colorway=[ACCENT, ACCENT_2, BAND_COLOURS["Low"], BAND_COLOURS["Medium"], BAND_COLOURS["High"]],
    xaxis={"gridcolor": GRID, "zerolinecolor": GRID, "linecolor": GRID},
    yaxis={"gridcolor": GRID, "zerolinecolor": GRID, "linecolor": GRID},
    legend={"bgcolor": "rgba(0,0,0,0)"}, hoverlabel={"bgcolor": "#131a33", "font": {"color": TEXT}}))
pio.templates.default = "plotly_dark+sar_dark"
BAND_HELP = ("Risk bands come from the calibrated risk score: Low below the wide-net cutoff ({medium:.2f}), "
             "Medium from it, High from the top-20% cutoff ({high:.2f}). Both cutoffs were set on training data.")
TRAINING_NOTE = ("This is a training presentation: the model learned from these students' outcomes, so their scores "
                 f"look better than they would on new students. Use {TEST_PRESENTATION} to see the model on new students.")
CAPACITY, WIDE_NET = "Capacity ranking", "Wide net (fixed cutoff)"
FLAG_VS_BAND = ("High is a fixed calibrated-score cutoff set when the model was built; capacity flags the "
                "highest-risk share of this cohort, which can include Medium students when the share exceeds the "
                "High band's share.")
EVIDENCE_NOTE = ("Observed at-risk rate inside each band, and its lift: the band's rate divided by the overall "
                 "at-risk rate of the same row, that is the module's own rate for a module and the cohort's rate for "
                 "All modules. Bands with fewer than {n} students are shown as n < {n}.")
SCOPE_COHORT, SCOPE_MODULE = "the whole cohort", "each module separately"
APP_NAME = "Student Academic Risk"
APP_PURPOSE = "Day 30 early warning: who to contact first, why, and which support fits."
PAGES = ["Overview", "Student", "Modules", "About"]
PAGE_ICONS = {"Overview": ":material/dashboard:", "Student": ":material/person:", "Modules": ":material/grid_view:",
              "Support": ":material/volunteer_activism:", "About": ":material/info:"}
PAGE_STATE = ["lookup_module", "lookup_student", "funnel_module", "show_3d"]
PAGE_SUBTITLES = {"Overview": "Who to contact first, and why.",
                  "Student": "Why one student is flagged, and which support fits.",
                  "Modules": "Risk bands, outcomes and engagement by module.",
                  "Support": "Suggested support across flagged students.",
                  "About": "The model, its design, its limits and the data."}
CHIP_COLOURS = ["red", "orange", "violet", "blue", "green", "primary"]
SNAPSHOT = {"total_clicks": "Total clicks", "active_days": "Active days",
            "trailing_inactive_days": f"Days inactive just before Day {CUTOFF_DAY}",
            "n_submitted": "Assessments submitted", "n_missed": "Assessments missed"}
LOWER_IS_BETTER = {"trailing_inactive_days", "n_missed"}
SAME_AS_MEDIAN = "same as module median"
WRAP = 30  # characters per line of a factor label
MAX_LIVE_REASONS = 50
OPEN_FOR_DETAILS = "Open the student to see"
WHY_LEGEND = "Red bars raise the risk score and teal bars lower it. A longer bar means a larger effect."
PSEUDONYM_NOTE = "Student IDs are the pseudonymised identifiers published with OULAD."
DESCRIPTIVE = "Similar past students (descriptive, not evidence of effectiveness)"
CALIBRATION_NOTE = ("Each point is a bin of {val} students by calibrated score (width 0.1): its mean calibrated score "
                    "against the share who went on to fail or withdraw, with the number of students beside it. Points "
                    "on the diagonal are well calibrated. Band cutoffs: Medium from {medium:.2f}, High from "
                    "{high:.2f}. Calibrated scores take a limited number of distinct values because isotonic "
                    "calibration maps ranges of the forest's score to the same value.")
DEPT_NOTE = ("One row per module in the chosen presentation: students in the Day 30 population, the observed "
             "at-risk rate (final outcome Fail or Withdrawn) and the share of students in each risk band of the "
             "final model. Groups under {n} students are shown without figures.")
DEPT_TRAINING_NOTE = ("Training presentation: the model learned from these students' outcomes, so the band shares "
                      f"are better separated than they would be for a new intake. {TEST_PRESENTATION} is the "
                      "validation presentation.")
IN_SAMPLE_NOTE = ("Training presentations have in-sample scores: the model learned from these students' outcomes, so "
                  "their scores and band shares look better than they would for a new intake.")
SCORE_SOURCE = {"validation": f"validation ({TEST_PRESENTATION})", "training": "training (in-sample)"}
FUNNEL_NOTE = ("Retrospective, descriptive view. Stages up to Day 30 use data available on Day 30; the last two "
               "stages use final outcomes, which are not model inputs. Each stage counts the students of the previous "
               "stage who also meet its condition. Where no assessment was due by Day 30, the submission stage is not "
               f"applicable and the previous count is carried forward. Counts under {MIN_CELL} are hidden, together "
               "with enough other modules that the total cannot reveal them.")
PROFILE_NOTE = ("Profiles from KMeans on five Day 30 engagement and submission features, without background "
                "characteristics. Clusters were fitted on the training presentations; {val} students are assigned "
                "to the nearest one. The at-risk rate comes from final outcomes and is not a clustering input.")


@st.cache_data(show_spinner="Loading the Day 30 feature matrix...")
def get_matrix(processed_dir: str) -> Loaded:
    return load_matrix(processed_dir)


@st.cache_resource(show_spinner="Loading the model...")
def get_model(models_dir: str) -> Loaded:
    return load_final_model(models_dir)


@st.cache_resource(show_spinner=False)
def get_explainer(models_dir: str, _model):
    return make_explainer(_model)


@st.cache_resource(show_spinner="Loading precomputed explanations...")
def get_shap_cache(models_dir: str, presentation: str, _config) -> Loaded:
    return load_shap_cache(models_dir, presentation, _config)


@st.cache_resource(show_spinner="Building the similar-student pool...")
def get_pool(processed_dir: str, _matrix) -> Loaded:
    try:
        return Loaded(NeighbourPool(make_split(_matrix)[0]))
    except ValueError as e:
        return Loaded(message=f"Similar students are not available: {e}")


@st.cache_data(show_spinner="Scoring students...")
def get_scores(processed_dir: str, models_dir: str, presentation: str, _fitted, _matrix):
    return score_presentation(_fitted, _matrix, presentation)


@st.cache_data(show_spinner="Explaining the score...")
def get_explanation(models_dir: str, presentation: str, module: str, student: int, band: str, _fitted, _config,
                    _row):
    """From the precomputed batch values when present, otherwise computed once for this student."""
    features = _fitted["features"]
    stored = get_shap_cache(models_dir, presentation, _config)
    if stored.ok:
        table, base = stored.value
        hit = table[(table["code_module"] == module) & (table["id_student"] == student)]
        if len(hit):
            contrib = hit.iloc[0][[SHAP_PREFIX + f for f in features]]
            contrib.index = features
            return explanation_from_contributions(_row[features].iloc[0], contrib, base, hit.iloc[0]["raw_score"],
                                                  band)
    return explain_student(_fitted["model"], _row[features], get_explainer(models_dir, _fitted["model"]), band)


@st.cache_data(show_spinner="Preparing reasons and suggestions...")
def get_details(models_dir: str, presentation: str, keys: pd.DataFrame, _fitted, _config, _matrix) -> pd.DataFrame:
    """Main reason and suggested support for the students in `keys` (key columns, in list order). Uses the
    precomputed explanations when present; otherwise explains at most MAX_LIVE_REASONS students, the
    top of the list, and marks the rest to be opened one at a time."""
    features = _fitted["features"]
    out = keys.reset_index(drop=True).assign(main_reason=OPEN_FOR_DETAILS, suggested_support=OPEN_FOR_DETAILS)
    rows = out[KEY].merge(_matrix, on=KEY, how="left")
    stored = get_shap_cache(models_dir, presentation, _config)
    if stored.ok:
        table, _ = stored.value
        shap_cols = [SHAP_PREFIX + f for f in features]
        got = out[["code_module", "id_student"]].merge(table[["code_module", "id_student", *shap_cols]],
                                                       on=["code_module", "id_student"], how="left")
        known = got[shap_cols].notna().all(axis=1).to_numpy()
        contrib = got.loc[known, shap_cols].set_axis(features, axis=1)
    else:
        known = np.arange(len(out)) < MAX_LIVE_REASONS
        live = rows.loc[known, features]
        contrib = (explain_rows(_fitted["model"], live, get_explainer(models_dir, _fitted["model"]))[1]
                   if len(live) else pd.DataFrame(columns=features))
    out = out.assign(main_factor=None, prepared=known)
    if known.any():
        details = priority_details(rows.loc[known, features], contrib.set_axis(rows.index[known]))
        cols = ["main_factor", "main_reason", "suggested_support"]
        out.loc[known, cols] = details[cols].values
    return out


@st.cache_data(ttl=60, show_spinner=False)
def get_table(name: str, processed_dir: str) -> TableStatus:
    return table_status(OPTIONAL_TABLES[name], processed_dir)


def score_source(presentation: str) -> str:
    return SCORE_SOURCE["training" if is_training_presentation(presentation) else "validation"]


CARD_MESSAGES = {"missing": "Not available yet.",
                 "unreadable": "The table could not be read.",
                 "refused": "Not shown: the file contains student-level rows, and only aggregated tables are shown.",
                 "missing_columns": "Not shown: the file does not have the expected columns."}


def status_card(status: TableStatus) -> None:
    """Card for a team table that cannot be shown: what it is and why. Technical details are in About."""
    with st.container(border=True):
        st.markdown(f"#### {status.spec.title}")
        st.write(status.spec.description)
        (st.info if status.state == "missing" else st.warning)(
            f"{CARD_MESSAGES[status.state]} Details are on the About page, under Data details.")


PROFILE_HEADERS = {"mean_total_clicks": ("Mean clicks", "{:,.0f}"),
                   "mean_active_days": ("Mean active days", "{:.1f}"),
                   "mean_trailing_inactive_days": ("Mean days inactive before Day 30", "{:.1f}"),
                   "mean_n_submitted": ("Mean assessments submitted", "{:.2f}"),
                   "mean_n_missed": ("Mean assessments missed", "{:.2f}")}
SMALL = f"n < {MIN_CELL}"
DASH = "\u2013"


def text_cells(df: pd.DataFrame, formats: dict, hidden=None, missing: str = DASH) -> pd.DataFrame:
    """`df` with each column in `formats` turned into display text: values formatted with their pattern,
    missing values as a dash, and, where `hidden` is True, the text "n < MIN_CELL"."""
    out = df.copy()
    hidden = np.zeros(len(df), bool) if hidden is None else np.asarray(hidden, dtype=bool)
    for c, pattern in formats.items():
        out[c] = [SMALL if h else v if isinstance(v, str) else missing if pd.isna(v) else pattern.format(v)
                  for v, h in zip(df[c], hidden)]
    return out


def show_table(df: pd.DataFrame, formats: dict, hidden=None, missing: str = DASH) -> pd.DataFrame:
    """Show `df` as formatted text, with the formatted (numeric) columns right-aligned. Returns what is shown."""
    out = text_cells(df, formats, hidden, missing)
    st.dataframe(out, hide_index=True,
                 column_config={c: st.column_config.TextColumn(c, alignment="right") for c in formats})
    return out


def shown_rows(table: pd.DataFrame) -> pd.DataFrame:
    return table[~table["suppressed"].astype(bool)]


@dataclass
class View:
    """The sidebar selection every page uses."""
    presentation: str
    module: str                # a module code, or ALL_MODULES
    share: float               # contact capacity, 0 to 1
    rule: str                  # CAPACITY or WIDE_NET
    per_module: bool           # capacity applies to each module separately

    @property
    def in_sample(self) -> bool:
        return is_training_presentation(self.presentation)


def presentation_label(p: str) -> str:
    return f"{p} (validation)" if p == TEST_PRESENTATION else f"{p} (training, in-sample)"


def sidebar(presentations: list[str], modules_of) -> View:
    """App name, purpose and the shared controls. Training presentations appear only after the
    Advanced toggle is on; the toggle's state is read first so the presentation list can follow it."""
    sb = st.sidebar
    training = [p for p in presentations if p != TEST_PRESENTATION]
    if TEST_PRESENTATION not in presentations:
        options = presentations
    elif st.session_state.get("show_training") and training:
        options = training
    else:
        options = [TEST_PRESENTATION]
    pres = sb.selectbox("Presentation", options, key="presentation", format_func=presentation_label,
                        help=f"{TEST_PRESENTATION} is the validation presentation. Training presentations are "
                             "available under Advanced.")
    module = sb.selectbox("Module", [ALL_MODULES, *modules_of(pres)], key="module_filter")
    rule = st.session_state.get("flag_rule", CAPACITY)
    pct = sb.slider("Contact capacity", min_value=1, max_value=50, value=20, step=1, format="%d%%", key="capacity",
                    disabled=rule == WIDE_NET,
                    help="Share of the cohort the team can contact, highest risk first.")
    with sb.expander("Advanced"):
        if st.toggle("Show training presentations (in-sample scores)", key="show_training", disabled=not training):
            st.caption(IN_SAMPLE_NOTE)
        rule = st.radio("Flagging rule", [CAPACITY, WIDE_NET], key="flag_rule")
        scope = st.radio("Capacity applies to", [SCOPE_COHORT, SCOPE_MODULE], key="capacity_scope",
                         disabled=rule == WIDE_NET)
    if is_training_presentation(pres):
        sb.warning(TRAINING_NOTE)
    return View(pres, module, pct / 100, rule, scope == SCOPE_MODULE)


def sidebar_nav() -> None:
    """Brand block and page navigation, above the shared controls."""
    st.sidebar.markdown(f'<div class="brand">{APP_NAME}</div>', unsafe_allow_html=True)
    st.sidebar.caption(APP_PURPOSE)
    with st.sidebar.container(key="nav"):
        st.radio("Navigation", PAGES, key="nav_page", format_func=lambda p: f"{PAGE_ICONS[p]}  {p}",
                 label_visibility="collapsed")


def band_pill(band: str) -> str:
    return f'<span class="pill pill-{band.lower()}">{band} risk</span>'


def show_chart(fig) -> None:
    """Every chart uses the app's dark Plotly template rather than Streamlit's chart theme."""
    st.plotly_chart(fig, width="stretch", theme=None)


def page_header(page: str, pres: str | None) -> None:
    """Page title and one-line subtitle, with the presentation on the right."""
    left, right = st.columns([4, 1.4], vertical_alignment="center")
    left.markdown(f"## {page}")
    left.markdown(f'<div class="page-subtitle">{PAGE_SUBTITLES[page]}</div>', unsafe_allow_html=True)
    if pres:
        kind = "validation" if pres == TEST_PRESENTATION else "training, in-sample"
        right.markdown(f'<span class="pill pill-info">Presentation: {pres} ({kind})</span>', unsafe_allow_html=True)


st.set_page_config(page_title="Student Academic Risk: Day 30 early warning", layout="wide")
# Only the selected page runs, and Streamlit drops the state of widgets that are not drawn; writing these
# values back keeps the chosen student and the page selections when the user moves between pages.
for k in PAGE_STATE:
    if k in st.session_state:
        st.session_state[k] = st.session_state[k]
st.markdown(STYLE, unsafe_allow_html=True)
sidebar_nav()

paths = resolve_paths()
processed, models = str(paths["processed"]), str(paths["models"])
matrix_l, model_l = get_matrix(processed), get_model(models)
core_ok = matrix_l.ok and model_l.ok


def core_message() -> None:
    missing = [name for name, loaded in (("the Day 30 data", matrix_l), ("the model", model_l)) if not loaded.ok]
    st.info(f"This view needs {' and '.join(missing)}, which are not available yet. Details are on the About page, "
            "under Data details.")


if core_ok:
    matrix = matrix_l.value
    final = model_l.value
    fitted, cutoffs, metrics = final.fitted, final.fitted["cutoffs"], final.metrics
    edges = dict(risk_band_edges(cutoffs))
    ctl = sidebar(sorted(matrix["code_presentation"].unique()),
                   lambda p: sorted(matrix.loc[matrix["code_presentation"] == p, "code_module"].unique()))
else:
    ctl = None

def show_example(band: str) -> None:
    """Button callback: choose a validation student in `band` (fixed seed), in the current module when possible."""
    scored = get_scores(processed, models, TEST_PRESENTATION, fitted, matrix)
    current = st.session_state.get("lookup_module")
    pick = example_student(scored, band, current, RANDOM_SEED)
    st.session_state["lookup_example"] = band if pick is None else None
    if pick is not None:
        st.session_state["show_training"] = False
        st.session_state["presentation"] = TEST_PRESENTATION
        st.session_state["lookup_module"], st.session_state["lookup_student"] = pick


def open_selected() -> None:
    """Priority-list selection callback: open the chosen student on the Student page."""
    pick = selected_row(st.session_state.get("priority_rows", []), st.session_state.get("priority_table"))
    if pick is not None:
        st.session_state["lookup_module"], st.session_state["lookup_student"] = pick
        st.session_state["nav_page"] = "Student"


def number(v: float) -> str:
    return f"{v:,.0f}" if float(v).is_integer() else f"{v:,.1f}"


def why_chart(expl):
    """The six actionable inputs that move this student's score most, in plain words, coloured by direction.
    Course-level context (the module, its assessment calendar) stays in the full table below the chart."""
    c = expl.contributions[expl.contributions.index.isin(ACTIONABLE_FEATURES)]
    top = c.loc[c["contribution"].abs().sort_values(ascending=False).index[:6]].iloc[::-1]
    labels = ["<br>".join(textwrap.wrap(p, WRAP)) for p in top["phrase"]]
    fig = go.Figure()
    for name, colour, raises in [("Raises risk", BAND_COLOURS["High"], True),
                                 ("Lowers risk", BAND_COLOURS["Low"], False)]:
        part = (top["contribution"] > 0) == raises
        fig.add_trace(go.Bar(x=top.loc[part, "contribution"].tolist(),
                             y=[lab for lab, p in zip(labels, part) if p], orientation="h", name=name,
                             marker_color=colour, customdata=top.loc[part, "phrase"].tolist(),
                             hovertemplate="%{customdata}: %{x:+.3f}<extra></extra>"))
    fig.update_layout(title="What moves this score", height=CHART_HEIGHT, margin={"t": 50, "l": 10, "r": 10},
                      xaxis_title="Effect on risk", legend={"orientation": "h", "y": -0.2},
                      yaxis={"automargin": True, "categoryorder": "array", "categoryarray": labels})
    return fig


def student_view(pres: str, module: str, sid: int) -> None:
    in_pres = matrix[matrix["code_presentation"] == pres]
    row = in_pres[(in_pres["code_module"] == module) & (in_pres["id_student"] == sid)].head(1)
    scored = get_scores(processed, models, pres, fitted, matrix)
    me = scored[(scored["code_module"] == module) & (scored["id_student"] == sid)].iloc[0]
    band = str(me["band"])
    expl = get_explanation(models, pres, module, int(sid), band, fitted, final.config, row)
    rank, n_module = module_rank(scored, module, sid)

    with st.container(border=True):
        head = st.container(horizontal=True, vertical_alignment="center", gap="medium")
        head.markdown(f"### Student {sid}")
        head.markdown(f"Module **{module}**")
        head.markdown(band_pill(band), unsafe_allow_html=True,
                      help=BAND_HELP.format(**{k.lower(): v for k, v in edges.items()}))
        head.markdown(f"ranked {rank:,} of {n_module:,} in module" + (" (in-sample)" if ctl.in_sample else ""))
        st.markdown(verdict_sentence(expl))

    why, support, snap = st.columns(3, border=True)
    with why:
        show_chart(why_chart(expl))
        st.caption(WHY_LEGEND)
        with st.expander("In words, and all inputs"):
            st.write(expl.sentence)
            st.dataframe(explanation_table(expl), hide_index=True)
    with support:
        st.markdown("#### Suggested support")
        pool_l = get_pool(processed, matrix)
        if not pool_l.ok:
            st.info(pool_l.message)
        else:
            support_cards(expl, recommend(expl, row, pool_l.value))
    with snap:
        st.markdown(f"#### Day {CUTOFF_DAY} snapshot")
        values = snapshot(matrix, pres, module, sid, SNAPSHOT)
        for f, r in values.iterrows():
            diff = r["difference"]
            if diff == 0:
                delta, colour = SAME_AS_MEDIAN, "off"
            else:
                delta = f"{'+' if diff > 0 else '-'}{number(abs(diff))} vs module median"
                colour = "inverse" if f in LOWER_IS_BETTER else "normal"
            st.metric(r["label"], number(r["value"]), delta=delta, delta_color=colour,
                      delta_arrow="off" if diff == 0 else "auto",
                      help=f"Module median in {pres}: {number(r['median'])}.")


def support_cards(expl, rec) -> None:
    """One primary suggestion as a card, in a plain sentence; for Low-band students only that one, marked
    optional; otherwise the rest under "Other options", with the descriptive-context note shown once."""
    if not rec.suggestions:
        st.write("No activity, assessment or study-load input raises the score noticeably, so no support type "
                 "is matched.")
    else:
        if rec.optional:
            st.caption("Low risk: support is optional.")
        first, others = rec.suggestions[0], ([] if rec.optional else rec.suggestions[1:])
        with st.container(border=True):
            st.markdown(f"**{first.label}**")
            st.write(support_sentence(first.intervention, expl.contributions.loc[first.drivers[0], "phrase"]))
            st.caption(similar_line(rec.neighbours.n, rec.neighbours.n_at_risk))
        if others:
            with st.expander("Other options"):
                for s in others:
                    st.markdown(f"**{s.label}**: "
                                + support_sentence(s.intervention, expl.contributions.loc[s.drivers[0], "phrase"]))
                st.caption(f"{DESCRIPTIVE}: {rec.neighbours.summary()} {NOTE}")
            return
    with st.expander("About these suggestions"):
        st.caption(f"{DESCRIPTIVE}: {rec.neighbours.summary()} {NOTE}")


def page_student() -> None:
    if not core_ok:
        core_message()
    else:
        pres = ctl.presentation
        in_pres = matrix[matrix["code_presentation"] == pres]
        bar = st.container(horizontal=True, vertical_alignment="bottom")
        bar.button("Show an example High-risk student", key="example_high", on_click=show_example, args=("High",))
        bar.button("Show an example Low-risk student", key="example_low", on_click=show_example, args=("Low",))
        module = bar.selectbox("Module", sorted(in_pres["code_module"].unique()), key="lookup_module")
        ids = sorted(in_pres.loc[in_pres["code_module"] == module, "id_student"].unique())
        if st.session_state.get("lookup_student") not in ids:
            st.session_state.pop("lookup_student", None)
        if "lookup_student" in st.session_state:
            sid = bar.selectbox("Student ID", ids, placeholder="Choose a student", key="lookup_student")
        else:
            sid = bar.selectbox("Student ID", ids, index=None, placeholder="Choose a student", key="lookup_student")
        st.caption(f"Examples are {TEST_PRESENTATION} students in that band, drawn with a fixed seed from the chosen "
                   "module when it has one.")
        if st.session_state.get("lookup_example"):
            st.info(f"No {TEST_PRESENTATION} student is in the {st.session_state['lookup_example']} band.")
        if sid is None:
            st.info("Select a student in the Overview list, choose a module and student ID here, or open an "
                    "example.")
        else:
            student_view(pres, module, sid)

def cohort_cards(pres: str, module: str, dept: pd.DataFrame | None) -> None:
    """Cohort at-risk rate and the shares in the High and Low bands, for the sidebar selection."""
    where = ALL_MODULES if module == ALL_MODULES else module
    rate = high = low = None
    if core_ok:
        scored = get_scores(processed, models, pres, fitted, matrix)
        part = scored if where == ALL_MODULES else scored[scored["code_module"] == where]
        if len(part) >= MIN_CELL:
            rate, high, low = part["at_risk"].mean(), (part["band"] == "High").mean(), (part["band"] == "Low").mean()
    elif dept is not None:
        row = dept[(dept["code_presentation"] == pres) & (dept["code_module"] == where)]
        if len(row) and not bool(row["suppressed"].iloc[0]):
            rate, high, low = row[["at_risk_rate", "share_high", "share_low"]].iloc[0]
    cards = st.columns(3)
    label = "all modules" if where == ALL_MODULES else f"module {where}"
    cards[0].metric("Cohort at-risk rate", DASH if rate is None else f"{rate:.0%}", border=True,
                    help=f"Share of {pres} students in {label} whose final outcome was Fail or Withdrawn.")
    cards[1].metric("Share in the High band", DASH if high is None else f"{high:.0%}", border=True,
                    help="Students at or above the top-20% cutoff set on training data.")
    cards[2].metric("Share in the Low band", DASH if low is None else f"{low:.0%}", border=True,
                    help="Students below the wide-net cutoff set on training data.")


def band_chart(part: pd.DataFrame, pres: str):
    bands = {s: s.removeprefix("share_").capitalize() for s in BAND_SHARES}
    modules = shown_rows(part)
    fig = go.Figure([go.Bar(name=b, x=modules["code_module"].tolist(), y=modules[s].tolist(),
                            marker_color=BAND_COLOURS[b], texttemplate="%{y:.0%}", textposition="inside")
                     for s, b in bands.items()])
    fig.update_layout(barmode="stack", yaxis_title="Share of students", yaxis_tickformat=".0%", height=CHART_HEIGHT,
                      margin={"t": 50}, legend={"orientation": "h"}, xaxis_title="Module",
                      title=f"Share of students in each risk band by module, {pres}")
    return fig


def profile_chart(shown: pd.DataFrame, scope_label: str):
    fig = go.Figure(go.Bar(x=shown["profile"].tolist(), y=shown["at_risk_rate"].tolist(), marker_color=ACCENT,
                           text=[f"{r:.0%} at risk<br>{sh:.0%} of students"
                                 for r, sh in zip(shown["at_risk_rate"], shown["share_of_scope"])],
                           textposition="outside"))
    fig.update_layout(yaxis_title="Observed at-risk rate", yaxis_tickformat=".0%", yaxis_range=[0, 1.1],
                      height=CHART_HEIGHT, margin={"t": 50}, title=f"Observed at-risk rate by profile, {scope_label}")
    return fig


def funnel_view(fun: pd.DataFrame, pres: str, module: str):
    """The funnel chart, its one caption line, and its table for one presentation and module."""
    in_pres = fun[fun["code_presentation"] == pres]
    part = in_pres[in_pres["code_module"] == module].sort_values("stage_order")
    part = part.assign(stage=part["stage"].where(part["applicable"].astype(bool),
                                                 part["stage"] + " (not applicable)"))
    shown = shown_rows(part)
    fig = go.Figure(go.Funnel(y=shown["stage"].tolist(), x=shown["students"].astype(float).tolist(),
                              marker_color=ACCENT_2, texttemplate="%{value:,.0f} (%{percentInitial:.0%})"))
    fig.update_layout(height=CHART_HEIGHT, margin={"t": 50}, separators=".,",
                      title=f"Students at each stage (retrospective), {module}, {pres}")
    notes = ["Retrospective: the last two stages use final outcomes, which are not model inputs."]
    carried = in_pres[~in_pres["applicable"].astype(bool)]
    if module != ALL_MODULES:
        carried = carried[carried["code_module"] == module]
    if len(carried):
        n = carried["students"].dropna().astype(int).sum()
        notes.append(f"{n:,} students in {', '.join(sorted(carried['code_module']))} are carried forward on the "
                     "submission check, because no assessment was due by Day 30 there.")
    if len(shown) < len(part):
        notes.append(f"{len(part) - len(shown)} stages are hidden because of small counts.")
    table = part[["stage", "students", "share_of_registered", "share_of_previous"]].rename(
        columns={"stage": "Stage", "students": "Students", "share_of_registered": "Share of registered",
                 "share_of_previous": "Share of previous stage"})
    table["Students"] = ["not applicable" if not a else SMALL if pd.isna(n) else f"{int(n):,}"
                         for n, a in zip(part["students"], part["applicable"].astype(bool))]
    table.loc[~part["applicable"].astype(bool).to_numpy(), "Share of previous stage"] = float("nan")
    return fig, " ".join(notes), table


def evidence_table(scored: pd.DataFrame) -> pd.DataFrame:
    ev = band_evidence(scored)
    wide = ev.pivot(index="code_module", columns="band", values=["students", "at_risk_rate", "lift"])
    order = [ALL_MODULES, *sorted(set(ev["code_module"]) - {ALL_MODULES})]
    table = pd.DataFrame({f"{b}: {lab}": wide[(col, b)] for b in ["High", "Medium", "Low"]
                          for col, lab in [("students", "students"), ("at_risk_rate", "at-risk rate"),
                                           ("lift", "lift")]}).loc[order]
    table.index.name = "Module"
    return table.reset_index()


def calibration_chart():
    cal = calibration_bins(get_scores(processed, models, TEST_PRESENTATION, fitted, matrix))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Perfect calibration",
                             line={"color": MUTED, "dash": "dash"}))
    fig.add_trace(go.Scatter(x=cal["mean_score"].tolist(), y=cal["observed_rate"].tolist(), mode="markers+lines+text",
                             name="Observed", marker={"color": ACCENT, "size": 9}, line={"color": ACCENT},
                             text=[f"{n:,}" for n in cal["students"]], textposition="top left"))
    fig.update_layout(xaxis_title="Mean calibrated risk score", yaxis_title="Observed at-risk rate",
                      xaxis_range=[0, 1], yaxis_range=[0, 1], xaxis_tickformat=".0%", yaxis_tickformat=".0%",
                      height=CHART_HEIGHT, margin={"t": 50}, legend={"orientation": "h"},
                      title=f"Observed at-risk rate against calibrated score, {TEST_PRESENTATION}")
    return fig


def page_modules() -> None:
    pres = ctl.presentation if ctl else TEST_PRESENTATION
    module = ctl.module if ctl else ALL_MODULES
    in_sample = is_training_presentation(pres)
    st.markdown("### Risk bands by module", help=DEPT_NOTE.format(n=MIN_CELL))
    if in_sample:
        st.warning(DEPT_TRAINING_NOTE)
    dept_status = get_table("department_risk", processed)
    dept = dept_status.table if dept_status.ok else None
    cohort_cards(pres, module, dept)
    dept_part = None
    if dept is None:
        status_card(dept_status)
    else:
        dept_part = dept[dept["code_presentation"] == pres]
        show_chart(band_chart(dept_part, pres))
        if dept_part["suppressed"].astype(bool).any():
            st.caption(f"Modules with fewer than {MIN_CELL} students are not shown.")

    prof_status = get_table("student_profiles", processed)
    st.markdown("### Student profiles", help=PROFILE_NOTE.format(val=TEST_PRESENTATION))
    prof_part = None
    if not prof_status.ok:
        status_card(prof_status)
    else:
        prof = prof_status.table
        scope = "training" if in_sample and "training" in set(prof["scope"]) else "validation"
        scope_label = "training presentations" if scope == "training" else TEST_PRESENTATION
        prof_part = prof[prof["scope"] == scope].sort_values("at_risk_rate", ascending=False, na_position="last")
        shown = shown_rows(prof_part)
        show_chart(profile_chart(shown, scope_label))
        if scope == "training":
            st.caption("Shown for the training presentations together, the rows the clusters were fitted on.")
        with st.expander("What each profile means"):
            for _, row in shown.iterrows():
                st.markdown(f"**{row['profile']}**: {row['description']}")

    fun_status = get_table("retention_funnel", processed)
    st.markdown("### Retention funnel", help=FUNNEL_NOTE)
    funnel_table = None
    if not fun_status.ok:
        status_card(fun_status)
    else:
        fun = fun_status.table
        options = [ALL_MODULES, *sorted(set(fun.loc[fun["code_presentation"] == pres, "code_module"]) - {ALL_MODULES})]
        default = options.index(module) if module in options else 0
        if "funnel_module" in st.session_state:
            funnel_module = st.selectbox("Module", options, key="funnel_module")
        else:
            funnel_module = st.selectbox("Module", options, index=default, key="funnel_module")
        fig, note, funnel_table = funnel_view(fun, pres, funnel_module)
        show_chart(fig)
        st.caption(note)

    with st.expander("View data tables"):
        if dept_part is not None:
            st.markdown(f"**Risk bands by module, {pres}** ({score_source(pres)} scores)")
            bands = {s: s.removeprefix("share_").capitalize() for s in BAND_SHARES}
            view = dept_part[["code_module", "students", "at_risk_rate", *reversed(BAND_SHARES)]].rename(
                columns={"code_module": "Module", "students": "Students", "at_risk_rate": "At-risk rate",
                         **{s: f"{b} band" for s, b in bands.items()}})
            pcts = [c for c in view.columns if c.endswith(("rate", "band"))]
            view = view.astype({c: object for c in pcts})
            view.loc[dept_part["suppressed"].astype(bool).to_numpy(), pcts] = SMALL
            show_table(view, {"Students": "{:,}", **dict.fromkeys(pcts, "{:.0%}")})
        if core_ok:
            st.markdown("**Observed at-risk rate and lift inside each band**", help=EVIDENCE_NOTE.format(n=MIN_CELL))
            if in_sample:
                st.caption(f"Shown for the validation presentation ({TEST_PRESENTATION}) only; training scores are "
                           "in-sample.")
            else:
                ev = evidence_table(get_scores(processed, models, pres, fitted, matrix))
                show_table(ev, {c: "{:,.0f}" if c.endswith("students") else "{:.0%}" if c.endswith("rate")
                                else "{:.2f}" for c in ev.columns if ": " in c}, missing=SMALL)
        if prof_part is not None:
            st.markdown("**Student profiles**")
            means = {f"mean_{f}": PROFILE_HEADERS[f"mean_{f}"] for f in PROFILE_FEATURES}
            view = prof_part[["profile", "students", "share_of_scope", "at_risk_rate", *means]].rename(
                columns={"profile": "Profile", "students": "Students", "share_of_scope": "Share",
                         "at_risk_rate": "At-risk rate", **{m: h for m, (h, _) in means.items()}})
            show_table(view, {"Students": "{:,}", "Share": "{:.0%}", "At-risk rate": "{:.0%}",
                              **{h: f for h, f in means.values()}})
        if funnel_table is not None:
            st.markdown(f"**Retention funnel, {funnel_module}**")
            show_table(funnel_table, {"Students": "{}", "Share of registered": "{:.0%}",
                                      "Share of previous stage": "{:.0%}"})

    if core_ok:
        with st.expander("Model reliability"):
            show_chart(calibration_chart())
            st.caption(CALIBRATION_NOTE.format(val=TEST_PRESENTATION, medium=edges["Medium"], high=edges["High"]))

        with st.expander("Optional 3D view"):
            if st.toggle("Show average risk score by activity and score bands in 3D", key="show_3d"):
                scored = get_scores(processed, models, pres, fitted, matrix)
                bands = risk_band_table(scored, matrix)
                if not bands["shown"].any():
                    st.info(f"No band of {pres} students has at least {MIN_CELL} students, so there is nothing to "
                            f"show in 3D. {hidden_bands(bands)} smaller bands are hidden.")
                else:
                    show_chart(band_figure(bands))
                    st.caption(f"Each marker is a group of {pres} students with the same days active by Day "
                               f"{CUTOFF_DAY} (one horizontal axis) and average assessment score by Day {CUTOFF_DAY} "
                               "(the other horizontal axis). The vertical axis and the colour both show the group's "
                               "average calibrated risk score; the marker size shows how many students it holds. "
                               f"Groups with fewer than {MIN_CELL} students are not shown ({hidden_bands(bands)} "
                               "groups hidden). Drag to rotate; hover for details.")


def band_style(band: str) -> str:
    return f"color: {BAND_COLOURS[band]}; font-weight: 600" if band in BAND_COLOURS else ""


def rule_caption(c: View) -> str:
    where = "all modules" if c.module == ALL_MODULES else f"module {c.module}"
    if c.rule == WIDE_NET:
        cut = cutoffs["wide_net"]["score"]
        return f"Wide net: calibrated risk score of at least {cut:.2f}, {c.presentation}, {where}."
    whose = f"each module in {c.presentation}" if c.per_module else c.presentation
    return f"Top {c.share:.0%} of {whose} by risk, {where}."


def page_overview() -> None:
    if not core_ok:
        core_message()
    else:
        pres = ctl.presentation
        in_sample = ctl.in_sample
        scored = get_scores(processed, models, pres, fitted, matrix)
        if ctl.rule == CAPACITY:
            flagging = flag_capacity(scored, ctl.share, per_module=ctl.per_module)
        else:
            flagging = flag_wide_net(scored, cutoffs)
        scored = scored.assign(flagged=flagging.flagged)
        in_view = scored if ctl.module == ALL_MODULES else scored[scored["code_module"] == ctl.module]
        outcome = flag_outcome(in_view, in_view["flagged"])
        tag = " (in-sample)" if in_sample else ""
        flagged = in_view[in_view["flagged"]].sort_values("rank")
        details = (get_details(models, pres, flagged[KEY].reset_index(drop=True), fitted, final.config, matrix)
                   if len(flagged) else None)
        if details is None:
            st.markdown("## No student in view is flagged")
        else:
            st.markdown(f"## Contact these {len(flagged):,} students first")
            chips = st.container(horizontal=True, gap="small")
            counts = reason_counts(details.loc[details["prepared"], "main_factor"])
            for (kind, n), colour in zip(counts.items(), CHIP_COLOURS):
                chips.badge(f"{kind}: {n:,}", color="gray" if kind == OTHER else colour)
        cards = st.columns(4)
        cards[0].metric("Students in view", f"{outcome.students:,}", border=True,
                        help="Students in the selected presentation and module.")
        cards[1].metric("Flagged for contact", f"{outcome.flagged:,}", border=True,
                        help=(f"{outcome.flagged / outcome.students:.0%} of the students in view, highest risk first."
                              if outcome.students else None))
        cards[2].metric("Precision of the flags" + tag, DASH if outcome.flagged == 0 else f"{outcome.precision:.0%}",
                        border=True, help=f"{outcome.caught:,} of the {outcome.flagged:,} flagged students went on "
                                          "to fail or withdraw.")
        cards[3].metric("Share of at-risk students reached" + tag,
                        DASH if outcome.at_risk == 0 else f"{outcome.recall:.0%}", border=True,
                        help=f"{outcome.caught:,} of the {outcome.at_risk:,} students who went on to fail or withdraw "
                             f"are flagged; {outcome.missed:,} are not.")
        st.caption(rule_caption(ctl))

        if flagged.empty:
            st.info("No student in view is flagged with this rule. Raise the capacity or choose another module.")
        else:
            listing = pd.DataFrame({
                "Rank": range(1, len(flagged) + 1), "Student ID": flagged["id_student"].astype(str).to_numpy(),
                "Module": flagged["code_module"].to_numpy(), "Risk band": flagged["band"].astype(str).to_numpy(),
                "Main reason": [short_reason(r) if p else r for r, p in zip(details["main_reason"],
                                                                            details["prepared"])],
                "Suggested support": details["suggested_support"].to_numpy()})
            st.session_state["priority_rows"] = list(zip(flagged["code_module"], flagged["id_student"].astype(int)))
            st.dataframe(listing.style.map(band_style, subset=["Risk band"]), hide_index=True, key="priority_table",
                         on_select=open_selected, selection_mode="single-row",
                         column_config={"Rank": st.column_config.NumberColumn(width="small"),
                                        "Student ID": st.column_config.TextColumn(width="small"),
                                        "Module": st.column_config.TextColumn(width="small"),
                                        "Risk band": st.column_config.TextColumn(width="small"),
                                        "Main reason": st.column_config.TextColumn(width="medium"),
                                        "Suggested support": st.column_config.TextColumn(width="medium")})
            if not details["prepared"].all():
                st.caption(f"Reasons and suggestions are prepared for the top {MAX_LIVE_REASONS} students of this "
                           "presentation; open any other student to see theirs.")
            st.caption("Select a row to open the student. " + PSEUDONYM_NOTE)
            export = listing.assign(score_source=score_source(pres))
            st.download_button("Download the list (CSV)", export.to_csv(index=False),
                               file_name=f"priority_{pres}{'_in_sample' if in_sample else ''}.csv", mime="text/csv")

        with st.expander("Flags by module and band"):
            by_band = flagged_by_band(scored)
            st.caption("Flagged students by risk band: " + ", ".join(f"{b} {n:,}" for b, n in by_band.items())
                       + ". " + FLAG_VS_BAND)
            summary = batch_summary(scored, outcomes=not in_sample).assign(Scores=score_source(pres)).reset_index()
            show_table(summary, {"Students": "{:,}", "Flagged": "{:,}", "Flagged share": "{:.0%}",
                                 **{b: "{:,}" for b in by_band.index},
                                 **{c: "{:.0%}" for c in ["At-risk rate", "Recall", "Precision"]
                                    if c in summary.columns}})
            if in_sample:
                st.caption(f"Per-module at-risk rate, recall and precision are shown for {TEST_PRESENTATION} only.")
            else:
                st.caption("Recall is the share of the module's at-risk students who are flagged; precision is the "
                           "share of the module's flagged students who are at risk.")

def band_reading(pres: str) -> pd.DataFrame | None:
    """Observed at-risk rate and lift per band for a presentation: from the generated department table when it
    is loaded, otherwise computed from the model's scores. None when neither is available."""
    status = get_table("department_risk", processed)
    if status.ok:
        row = status.table[(status.table["code_presentation"] == pres) & (status.table["code_module"] == ALL_MODULES)]
        if len(row) and not bool(row["suppressed"].iloc[0]):
            row = row.iloc[0]
            return pd.DataFrame({"Band": ["Low", "Medium", "High"],
                                 "rate": [row[r] for r in BAND_RATES],
                                 "lift": [row[r] / row["at_risk_rate"] for r in BAND_RATES]})
    if not core_ok:
        return None
    ev = band_evidence(get_scores(processed, models, pres, fitted, matrix))
    ev = ev[ev["code_module"] == ALL_MODULES].set_index("band").loc[["Low", "Medium", "High"]]
    return pd.DataFrame({"Band": ev.index, "rate": ev["at_risk_rate"].to_numpy(), "lift": ev["lift"].to_numpy()})


def page_about() -> None:
    if core_ok:
        val = ", ".join(metrics["validation_presentation"])
        train_pres = ", ".join(sorted(set(matrix["code_presentation"]) - {val}))
        v = metrics["final"]["validation"]
        st.markdown("### Model")
        st.write(f"A Random Forest scores each student from {len(fitted['features'])} inputs known by Day "
                 f"{CUTOFF_DAY}: the module, VLE activity, assessment submissions and scores so far, registration "
                 "timing, study load and previous attempts. Protected characteristics (gender, region, highest "
                 "education, deprivation band, age band, disability) are not inputs. Isotonic calibration turns the "
                 f"forest's score into a calibrated risk score. Trained on {train_pres}; on {val}: ROC-AUC "
                 f"{v['roc_auc']:.3f}, PR-AUC {v['pr_auc']:.3f}, at-risk rate {v['prevalence']:.0%} among "
                 f"{v['n']:,} students.")
        st.write("OULAD organises teaching into modules (courses), not departments; this application uses modules "
                 "wherever a department view is shown.")

        st.markdown("### Design and validation")
        overlap = overlap_report(matrix, val)
        st.markdown(
            f"- Prediction day: each student is scored from what is known by Day {CUTOFF_DAY} of the module; "
            f"students who left on or before Day {CUTOFF_DAY} are not scored.\n"
            f"- Temporal split by presentation: the model is trained on earlier presentations ({train_pres}) and "
            f"checked on the latest one ({val}), as it would be used on a new intake.\n"
            f"- Students who appear in both: {overlap['students in both']:,} students enrolled in a training "
            f"presentation and in {val}; their {overlap['train rows removed from overlap']:,} training rows are "
            "removed, so no student is seen in both.\n"
            "- Cutoffs: the wide-net and top-20% cutoffs, and so the band edges, are chosen on out-of-fold scores "
            "of the training presentations, never on the validation presentation.\n"
            f"- {val} is a validation presentation, not an untouched test set: it was used to compare model "
            "configurations, so its figures may be slightly optimistic.")

        st.markdown("### Risk bands")
        st.write("Each band starts at a calibrated-score cutoff chosen on training data.")
        show_table(pd.DataFrame({
            "Band": ["Low", "Medium", "High"],
            "Calibrated score": [f"below {edges['Medium']:.2f}", f"{edges['Medium']:.2f} to below {edges['High']:.2f}",
                                 f"{edges['High']:.2f} and above"],
            "How the cutoff was set": [
                "Everyone below the Medium cutoff.",
                "In the training data, 8 in 10 students who went on to fail or withdraw scored at least this.",
                "In the training data, the highest-scoring 1 in 5 students scored at least this."]}), {})

        st.markdown("### How to read the bands")
        reading = band_reading(val)
        if reading is not None:
            show_table(reading.rename(columns={"rate": f"Observed at-risk rate, {val}", "lift": "Lift"}),
                       {f"Observed at-risk rate, {val}": "{:.0%}", "Lift": "{:.2f}"}, missing=SMALL)
            st.caption(f"Lift is the band's at-risk rate divided by the at-risk rate of all {val} students.")
        st.write("The Medium band's at-risk rate is close to the cohort average, so a Medium band is not a warning "
                 "on its own.")

        st.markdown("### Reading the cohort insights")
        st.markdown(f"- Risk bands by module: {DEPT_NOTE.format(n=MIN_CELL)}\n"
                    f"- Band evidence: {EVIDENCE_NOTE.format(n=MIN_CELL)}\n"
                    f"- Student profiles: {PROFILE_NOTE.format(val=TEST_PRESENTATION)}\n"
                    f"- Retention funnel: {FUNNEL_NOTE}")

        st.markdown("### Capacity flagging")
        st.write("The priority list ranks the chosen cohort by calibrated risk (ties broken by the forest's raw score) "
                 "and flags the highest-risk share set on the slider, to match a team with a fixed number of "
                 "contacts; the share can apply to the whole cohort or to each module. Precision, recall and the "
                 "at-risk students caught or missed are computed from that cohort's observed outcomes. The wide-net "
                 "rule instead flags everyone above a fixed cutoff, so the share it flags varies between intakes.")

        st.markdown("### Limitations")
        groups = metrics["final"].get("fairness_validation", {}).get("groups", {})
        gaps = []
        if "gender" in groups:
            g = groups["gender"]
            gaps.append(f"women {g['F']['recall']:.0%} and men {g['M']['recall']:.0%}")
        if "age_band" in groups:
            g = groups["age_band"]
            gaps.append(f"ages 35-55 {g['35-55']['recall']:.0%} and under 35 {g['0-35']['recall']:.0%}")
        no_due = matrix.groupby("code_module")["n_assessments_due"].max()
        no_due = ", ".join(no_due[no_due == 0].index) or "none"
        val_scored = get_scores(processed, models, val, fitted, matrix)
        val_scored = val_scored.assign(flagged=flag_capacity(val_scored, 0.2).flagged)
        recall = batch_summary(val_scored, outcomes=True).drop(index=ALL_MODULES)["Recall"].sort_values()
        lowest = "; ".join(f"{m} {r:.0%}" for m, r in recall.head(2).items())
        st.markdown(
            "- Support suggestions are rule-based: they match the model's risk drivers to support types and have not "
            "been tested as interventions.\n"
            "- OULAD records no interventions or their outcomes, so the effect of contacting a student is unknown.\n"
            f"- In some modules no assessment is due by Day {CUTOFF_DAY} ({no_due}), so the model has no early "
            "scores there.\n"
            f"- With 20% capacity across the whole cohort on {val}, the lowest module recall is {lowest} (share of "
            "the module's at-risk students who are flagged); capacity can apply to each module instead "
            "(sidebar, Advanced).\n"
            + (f"- Recall differs between groups at the top-20% cutoff on {val}: {'; '.join(gaps)}.\n" if gaps else "")
            + f"- Validation figures come from one presentation ({val}); training presentations are in-sample.")
    else:
        st.info("The model description appears here once the Day 30 data and the model are available.")

    st.markdown("### Data source")
    st.write("Open University Learning Analytics Dataset (OULAD). As stated on the dataset page "
             "(https://research.stem.open.ac.uk/ouanalyse/open-dataset-more/):")
    st.markdown('- License: "This dataset is released under CC-BY 4.0 license." '
                "(https://creativecommons.org/licenses/by/4.0/)\n"
                '- Citing the dataset: "When citing the dataset please use the following reference:" Kuzilek J., '
                "Hlosta M., Zdrahal Z. Open University Learning Analytics dataset Sci. Data 4:170171 doi: "
                "10.1038/sdata.2017.171 (2017).")

    with st.expander("Data details"):
        built = model_l.value.config.get("created_utc", "unknown") if model_l.ok else "not built"
        st.markdown(f"- Model built (UTC): {built}\n"
                    f"- Day 30 data: {'loaded' if matrix_l.ok else 'not loaded'}\n"
                    f"- Model: {'loaded' if model_l.ok else 'not loaded'}\n"
                    + "\n".join(f"- {spec.title}: {'loaded' if get_table(name, processed).ok else 'not loaded'}"
                                 for name, spec in OPTIONAL_TABLES.items()))
        st.markdown("Rebuild everything from the repository root:")
        st.code("python scripts/run_baseline.py        # Day 30 feature matrix\n"
                "python scripts/train_final.py         # final model\n"
                "python scripts/precompute_shap.py     # optional: explanations for the validation presentation\n"
                "python scripts/build_team_tables.py   # department, funnel and profile tables", language="bash")


PAGE_VIEWS = {"Overview": page_overview, "Student": page_student, "Modules": page_modules, "About": page_about}
page = st.session_state.get("nav_page", PAGES[0])
page_header(page, ctl.presentation if ctl else None)
PAGE_VIEWS[page]()
