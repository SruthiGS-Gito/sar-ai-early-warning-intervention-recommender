"""SCRUM-189: Streamlit app for the Day 30 early-warning model.

Run from the repo root:  streamlit run app/streamlit_app.py
Tabs: Student Lookup, Department Dashboard, Retention Funnel, Batch Report, About.

* The final model (models/final/) scores students with a calibrated risk score, a risk band
  (Low, Medium, High) and a rank. Students are flagged by capacity ranking or by the wide-net cutoff.
* Data, model, explainer, precomputed explanations and the similar-student pool load once (cached).
* Anything missing shows a message; the app does not crash on it. The main tabs speak to staff; file
  names, build commands, ticket references and build dates are kept in the About tab.
* Model-based band views (Department risk, Batch Report) show the validation presentation by
  default. A training presentation, whose scores are in-sample, appears only after the user opts in,
  with the in-sample note beside the choice and a score-source column in every exportable table.
* Displays are aggregated by default. One student's details appear only after they are chosen,
  and the flagged-student list sits in a closed expander.
* Department, funnel and profile tables (tickets 203, 207 and 210) are aggregated tables built by
  scripts/build_team_tables.py. Each is shown with a simple chart when its CSV exists and passes the
  checks in sar.service.data; otherwise a status card says what it is and how to build it.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
import plotly.graph_objects as go  # noqa: E402
import streamlit as st  # noqa: E402

from sar.data.split import TEST_PRESENTATION, make_split  # noqa: E402
from sar.explain.chart import explanation_table, plot_explanation  # noqa: E402
from sar.explain.shap_explain import explain_student, explanation_from_contributions, make_explainer  # noqa: E402
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
from sar.service.views import (  # noqa: E402
    KEY,
    MIN_CELL,
    band_evidence,
    batch_summary,
    flag_capacity,
    flag_outcome,
    flag_wide_net,
    flagged_by_band,
    is_training_presentation,
    risk_band_edges,
    risk_band_table,
    score_presentation,
)
from sar.tables.department import ALL_MODULES, BAND_RATES, BAND_SHARES  # noqa: E402
from sar.tables.profiles import PROFILE_FEATURES  # noqa: E402
from sar.viz.plots import AT_RISK, NEUTRAL, ON_TRACK  # noqa: E402

MODEL_NOTE = ("Validated on {val}. Students are ranked by calibrated risk and placed in Low, Medium or High risk "
              "bands; the About tab explains the model, the bands and their limits.")
BAND_HELP = ("Risk bands come from the calibrated risk score: Low below the wide-net cutoff ({medium:.2f}), "
             "Medium from it, High from the top-20% cutoff ({high:.2f}). Both cutoffs were set on training data.")
TRAINING_NOTE = ("This is a training presentation: the model learned from these students' outcomes, so their scores "
                 f"look better than they would on new students. Use {TEST_PRESENTATION} to see the model on new students.")
CAPACITY, WIDE_NET = "Capacity ranking", "Wide net (fixed cutoff)"
BAND_COLOURS = {"Low": ON_TRACK, "Medium": NEUTRAL, "High": AT_RISK}
FLAG_VS_BAND = ("High is a fixed calibrated-score cutoff set when the model was built; capacity flags the "
                "highest-risk share of this cohort, which can include Medium students when the share exceeds the "
                "High band's share.")
EVIDENCE_NOTE = ("Observed at-risk rate inside each band, and its lift: the band's rate divided by the cohort's "
                 "overall at-risk rate. Bands with fewer than {n} students are shown as n < {n}.")
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


@st.cache_data(ttl=60, show_spinner=False)
def get_table(name: str, processed_dir: str) -> TableStatus:
    return table_status(OPTIONAL_TABLES[name], processed_dir)


def presentation_picker(matrix, key: str) -> str:
    options = sorted(matrix["code_presentation"].unique())
    default = options.index(TEST_PRESENTATION) if TEST_PRESENTATION in options else len(options) - 1
    return st.selectbox("Presentation", options, index=default, key=key)


def validation_or_training(options, key: str) -> str:
    """The validation presentation by default. A training presentation is offered only after the user
    turns on the labelled option, with the in-sample note shown beside it."""
    if TEST_PRESENTATION not in options:
        return st.selectbox("Presentation", options, key=key)
    training = [p for p in options if p != TEST_PRESENTATION]
    left, right = st.columns([1, 2])
    opt_in = left.toggle("Show a training presentation (in-sample scores)", key=f"{key}_training",
                         disabled=not training)
    right.caption(IN_SAMPLE_NOTE)
    if not opt_in:
        st.caption(f"Showing {TEST_PRESENTATION}, the validation presentation.")
        return TEST_PRESENTATION
    return st.selectbox("Training presentation (in-sample scores)", training, key=key)


def score_source(presentation: str) -> str:
    return SCORE_SOURCE["training" if is_training_presentation(presentation) else "validation"]


STATUS_LABELS = {"missing": "Not available yet", "unreadable": "File could not be read",
                 "refused": "Refused: contains student-level rows", "missing_columns": "Missing required columns",
                 "loaded": "Loaded"}


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
            f"{CARD_MESSAGES[status.state]} Details are in the About tab, under Data details.")


def team_table(name: str, processed_dir: str, heading: bool = True) -> pd.DataFrame | None:
    """The team table when it can be shown (after its title); otherwise its status card."""
    status = get_table(name, processed_dir)
    if not status.ok:
        status_card(status)
        return None
    if heading:
        st.markdown(f"#### {status.spec.title}")
    return status.table


PROFILE_HEADERS = {"mean_total_clicks": ("Mean clicks", "{:,.0f}"),
                   "mean_active_days": ("Mean active days", "{:.1f}"),
                   "mean_trailing_inactive_days": ("Mean days inactive before Day 30", "{:.1f}"),
                   "mean_n_submitted": ("Mean assessments submitted", "{:.2f}"),
                   "mean_n_missed": ("Mean assessments missed", "{:.2f}")}
SMALL = f"n < {MIN_CELL}"
DASH = "\u2013"


def small_cells(table: pd.DataFrame, view: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """`view` with the figures of suppressed groups replaced by the text "n < MIN_CELL"."""
    hidden = table["suppressed"].astype(bool).to_numpy()
    view = view.copy()
    for c in columns:
        view[c] = view[c].astype(object)
        view.loc[hidden, c] = SMALL
    return view


def fmt(pattern: str):
    """Formatter for Styler.format that leaves text (such as "n < 20") as it is and shows missing values as a dash."""
    return lambda v: v if isinstance(v, str) else DASH if pd.isna(v) else pattern.format(v)


def shown_rows(table: pd.DataFrame) -> pd.DataFrame:
    return table[~table["suppressed"].astype(bool)]


st.set_page_config(page_title="SAR-AI early warning", layout="wide")
st.title("Student Academic Risk: Day 30 early warning")

paths = resolve_paths()
processed, models = str(paths["processed"]), str(paths["models"])
matrix_l, model_l = get_matrix(processed), get_model(models)
core_ok = matrix_l.ok and model_l.ok


def core_message() -> None:
    missing = [name for name, loaded in (("the Day 30 data", matrix_l), ("the model", model_l)) if not loaded.ok]
    st.info(f"This view needs {' and '.join(missing)}, which are not available yet. Details are in the About tab, "
            "under Data details.")


if core_ok:
    matrix = matrix_l.value
    final = model_l.value
    fitted, cutoffs, metrics = final.fitted, final.fitted["cutoffs"], final.metrics
    edges = dict(risk_band_edges(cutoffs))
    st.caption(MODEL_NOTE.format(val=", ".join(metrics["validation_presentation"])))

lookup, dashboard, funnel, batch, about = st.tabs(["Student Lookup", "Department Dashboard", "Retention Funnel",
                                                  "Batch Report", "About"])

with lookup:
    st.subheader("Student Lookup")
    if not core_ok:
        core_message()
    else:
        c1, c2, c3 = st.columns(3)
        with c1:
            pres = presentation_picker(matrix, "lookup_presentation")
        in_pres = matrix[matrix["code_presentation"] == pres]
        module = c2.selectbox("Module", sorted(in_pres["code_module"].unique()), key="lookup_module")
        ids = sorted(in_pres.loc[in_pres["code_module"] == module, "id_student"].unique())
        sid = c3.selectbox("Student ID", ids, index=None, placeholder="Choose a student", key="lookup_student")
        if sid is None:
            st.caption(f"{len(ids):,} students in {module} {pres}. Choose one to see why the model scores them "
                       "as it does and which support fits.")
        else:
            row = in_pres[(in_pres["code_module"] == module) & (in_pres["id_student"] == sid)].head(1)
            scored = get_scores(processed, models, pres, fitted, matrix)
            me = scored[(scored["code_module"] == module) & (scored["id_student"] == sid)].iloc[0]
            if is_training_presentation(pres):
                st.info(TRAINING_NOTE)
            expl = get_explanation(models, pres, module, int(sid), str(me["band"]), fitted, final.config, row)
            left, right = st.columns([3, 2])
            with left:
                fig = plot_explanation(expl)
                st.pyplot(fig)
                plt.close(fig)
            with right:
                st.metric("Risk band", str(me["band"]), help=BAND_HELP.format(**{k.lower(): v for k, v in edges.items()}))
                st.caption(f"Rank {int(me['rank']):,} of {len(scored):,} students in {pres} by calibrated risk.")
                st.write(expl.sentence)
            st.markdown("#### Suggested support")
            pool_l = get_pool(processed, matrix)
            if not pool_l.ok:
                st.info(pool_l.message)
            else:
                rec = recommend(expl, row, pool_l.value)
                st.caption(rec.neighbours.summary())
                if rec.optional:
                    st.caption("Low risk band, so these suggestions are optional.")
                if not rec.suggestions:
                    st.write("No activity, assessment or study-load input raises the score noticeably, "
                             "so no support type is matched.")
                for i, s in enumerate(rec.suggestions, 1):
                    st.markdown(f"**{i}. {s.label}**: {s.reason.split(': ', 1)[1]}")
                    st.caption(s.evidence)
                st.caption(NOTE)
            with st.expander("All inputs and their contributions"):
                st.dataframe(explanation_table(expl), hide_index=True)

with dashboard:
    st.subheader("Department Dashboard")
    st.caption("Model-based counts by module are in the Batch Report tab.")
    dept = team_table("department_risk", processed)
    if dept is not None:
        st.caption(DEPT_NOTE.format(n=MIN_CELL))
        pres = validation_or_training(sorted(dept["code_presentation"].unique()), "dept_presentation")
        part = dept[dept["code_presentation"] == pres]
        if (part["score_source"] == "training").any():
            st.info(DEPT_TRAINING_NOTE)
        bands = {s: s.removeprefix("share_").capitalize() for s in BAND_SHARES}
        view = part[["code_module", "code_presentation", "score_source", "students", "at_risk_rate",
                     *reversed(BAND_SHARES)]].assign(score_source=part["score_source"].map(SCORE_SOURCE)).rename(
            columns={"code_module": "Module", "code_presentation": "Presentation", "score_source": "Scores",
                     "students": "Students", "at_risk_rate": "At-risk rate",
                     **{s: f"{b} band" for s, b in bands.items()}})
        pcts = [c for c in view.columns if c.endswith(("rate", "band"))]
        view = small_cells(part, view, pcts)
        st.dataframe(view.style.format({"Students": fmt("{:,}"), **{c: fmt("{:.0%}") for c in pcts}}),
                     hide_index=True)
        st.markdown("##### Observed at-risk rate inside each band")
        st.caption(EVIDENCE_NOTE.format(n=MIN_CELL))
        base = part.loc[part["code_module"] == ALL_MODULES, "at_risk_rate"]
        base = float(base.iloc[0]) if len(base) else float("nan")
        evidence = part[["code_module"]].rename(columns={"code_module": "Module"})
        for r, b in reversed(list(zip(BAND_RATES, BAND_SHARES))):
            name = bands[b]
            evidence[f"{name}: at-risk rate"] = part[r].to_numpy()
            evidence[f"{name}: lift"] = (part[r] / base).to_numpy()
        st.dataframe(evidence.style.format(
            {c: (fmt("{:.0%}") if c.endswith("rate") else fmt("{:.2f}")) for c in evidence.columns if ": " in c},
            na_rep=SMALL), hide_index=True)
        modules = shown_rows(part)
        fig = go.Figure([go.Bar(name=b, x=modules["code_module"], y=modules[s], marker_color=BAND_COLOURS[b],
                                texttemplate="%{y:.0%}", textposition="inside")
                         for s, b in bands.items()])
        fig.update_layout(barmode="stack", yaxis_title="Share of students", yaxis_tickformat=".0%", height=360,
                          margin={"t": 50}, legend={"orientation": "h"}, xaxis_title="Module",
                          title=f"Share of students in each risk band by module, {pres}")
        st.plotly_chart(fig, width="stretch")

    prof = team_table("student_profiles", processed)
    if prof is not None:
        scopes = {"validation": f"{TEST_PRESENTATION} students (validation)",
                  "training": "Training presentations (used to fit the clusters)"}
        scope = st.radio("Students", [s for s in scopes if s in set(prof["scope"])], format_func=scopes.get,
                         horizontal=True, key="profile_scope")
        st.caption(PROFILE_NOTE.format(val=TEST_PRESENTATION))
        part = prof[prof["scope"] == scope].sort_values("at_risk_rate", ascending=False, na_position="last")
        means = {f"mean_{f}": PROFILE_HEADERS[f"mean_{f}"] for f in PROFILE_FEATURES}
        view = part[["profile", "students", "share_of_scope", "at_risk_rate", *means]].rename(
            columns={"profile": "Profile", "students": "Students", "share_of_scope": "Share",
                     "at_risk_rate": "At-risk rate", **{m: h for m, (h, _) in means.items()}})
        view = small_cells(part, view, ["At-risk rate", *(h for h, _ in means.values())])
        st.dataframe(view.style.format({"Students": fmt("{:,}"), "Share": fmt("{:.0%}"), "At-risk rate": fmt("{:.0%}"),
                                        **{h: fmt(f) for h, f in means.values()}}), hide_index=True)
        shown = shown_rows(part)
        with st.expander("What each profile means"):
            for _, row in shown.iterrows():
                st.markdown(f"**{row['profile']}**: {row['description']}")
        fig = go.Figure(go.Bar(x=shown["profile"], y=shown["at_risk_rate"], marker_color=NEUTRAL,
                               text=[f"{r:.0%}" for r in shown["at_risk_rate"]], textposition="outside"))
        fig.update_layout(yaxis_title="Observed at-risk rate", yaxis_tickformat=".0%", yaxis_range=[0, 1],
                          height=340, margin={"t": 50}, title="Observed at-risk rate by profile")
        st.plotly_chart(fig, width="stretch")

with funnel:
    st.subheader("Retention Funnel")
    st.info(FUNNEL_NOTE)
    fun = team_table("retention_funnel", processed, heading=False)
    if fun is not None:
        c1, c2 = st.columns(2)
        options = sorted(fun["code_presentation"].unique())
        pres = c1.selectbox("Presentation", options, key="funnel_presentation",
                            index=options.index(TEST_PRESENTATION) if TEST_PRESENTATION in options else 0)
        in_pres = fun[fun["code_presentation"] == pres]
        module = c2.selectbox("Module", [ALL_MODULES, *sorted(set(in_pres["code_module"]) - {ALL_MODULES})],
                              key="funnel_module")
        part = in_pres[in_pres["code_module"] == module].sort_values("stage_order")
        part = part.assign(stage=part["stage"].where(part["applicable"].astype(bool),
                                                     part["stage"] + " (not applicable)"))
        shown = shown_rows(part)
        fig = go.Figure(go.Funnel(y=shown["stage"], x=shown["students"].astype(float), marker_color=ON_TRACK,
                                  texttemplate="%{value:,.0f} (%{percentInitial:.0%})"))
        fig.update_layout(height=380, margin={"t": 50}, separators=".,",
                          title=f"Students at each stage, {module}, {pres}")
        st.plotly_chart(fig, width="stretch")
        if len(shown) < len(part):
            st.caption(f"{len(part) - len(shown)} stages are hidden because of small counts.")
        carried = in_pres[~in_pres["applicable"].astype(bool)]
        if module != ALL_MODULES:
            carried = carried[carried["code_module"] == module]
        if len(carried):
            n = carried["students"].dropna().astype(int).sum()
            st.caption(f"{n:,} students in {', '.join(sorted(carried['code_module']))} are carried forward on the "
                       "submission check: no assessment was due by Day 30 there, so the submission stage is not "
                       "applicable and the later stages start from the previous stage's count.")
        view = part[["stage", "students", "share_of_registered", "share_of_previous"]].rename(
            columns={"stage": "Stage", "students": "Students", "share_of_registered": "Share of registered",
                     "share_of_previous": "Share of previous stage"})
        view["Students"] = [
            "not applicable" if not a else SMALL if pd.isna(n) else f"{int(n):,}"
            for n, a in zip(part["students"], part["applicable"].astype(bool))]
        view.loc[~part["applicable"].astype(bool).to_numpy(), "Share of previous stage"] = float("nan")
        st.dataframe(view.style.format({"Share of registered": fmt("{:.0%}"),
                                        "Share of previous stage": fmt("{:.0%}")}), hide_index=True)

with batch:
    st.subheader("Batch Report")
    if not core_ok:
        core_message()
    else:
        pres = validation_or_training(sorted(matrix["code_presentation"].unique()), "batch_presentation")
        if is_training_presentation(pres):
            st.info(TRAINING_NOTE)
        in_sample = is_training_presentation(pres)
        scored = get_scores(processed, models, pres, fitted, matrix)
        rule = st.radio("Flagging rule", [CAPACITY, WIDE_NET], horizontal=True, key="flag_rule")
        if rule == CAPACITY:
            pct = st.slider("Contact capacity (share of this cohort)", min_value=1, max_value=50, value=20, step=1,
                            format="%d%%", key="capacity")
            flagging = flag_capacity(scored, pct / 100)
        else:
            flagging = flag_wide_net(scored, cutoffs)
        scored = scored.assign(flagged=flagging.flagged)
        outcome = flag_outcome(scored, scored["flagged"])
        tag = " (in-sample)" if in_sample else ""
        st.caption(f"{flagging.label}. Caught, missed, precision and recall compare the flags with this cohort's "
                   "observed outcomes (Fail or Withdrawn)" + ("; in-sample for a training presentation." if in_sample
                                                             else "."))
        cards = st.columns(6)
        cards[0].metric("Students", f"{outcome.students:,}")
        cards[1].metric("Flagged", f"{outcome.flagged:,}",
                        help=f"{outcome.flagged / outcome.students:.0%} of the cohort" if outcome.students else None)
        cards[2].metric("At-risk caught" + tag, f"{outcome.caught:,}")
        cards[3].metric("At-risk missed" + tag, f"{outcome.missed:,}")
        cards[4].metric("Precision" + tag, "\u2013" if outcome.flagged == 0 else f"{outcome.precision:.0%}")
        cards[5].metric("Recall" + tag, "\u2013" if outcome.at_risk == 0 else f"{outcome.recall:.0%}")
        by_band = flagged_by_band(scored)
        st.caption("Flagged students by risk band: " + ", ".join(f"{b} {n:,}" for b, n in by_band.items()) + ". "
                   + FLAG_VS_BAND)

        st.markdown("#### By module")
        summary = batch_summary(scored, outcomes=not in_sample).assign(Scores=score_source(pres))
        st.dataframe(summary.style.format({"Students": "{:,}", "Flagged": "{:,}", "Flagged share": "{:.0%}",
                                           **{b: "{:,}" for b in by_band.index},
                                           **{c: "{:.0%}" for c in ["At-risk rate", "Recall", "Precision"]
                                              if c in summary.columns}}, na_rep="\u2013"))
        if in_sample:
            st.caption(f"Per-module at-risk rate, recall and precision are shown for {TEST_PRESENTATION} only.")
        else:
            st.caption("Recall is the share of the module's at-risk students who are flagged; precision is the share "
                       "of the module's flagged students who are at risk.")

        st.markdown("#### Observed at-risk rate inside each risk band")
        if in_sample:
            st.caption(f"Shown for the validation presentation ({TEST_PRESENTATION}) only; training scores are "
                       "in-sample.")
        else:
            st.caption(EVIDENCE_NOTE.format(n=MIN_CELL))
            ev = band_evidence(scored)
            table = ev.pivot(index="code_module", columns="band", values=["students", "at_risk_rate", "lift"])
            order = ["All modules", *sorted(set(ev["code_module"]) - {"All modules"})]
            bands_hi = list(by_band.index)
            table = pd.DataFrame({f"{b}: {lab}": table[(col, b)] for b in bands_hi
                                  for col, lab in [("students", "students"), ("at_risk_rate", "at-risk rate"),
                                                   ("lift", "lift")]}).loc[order]
            table.index.name = "Module"
            st.dataframe(table.style.format(
                {c: "{:,.0f}" if c.endswith("students") else "{:.0%}" if c.endswith("rate") else "{:.2f}"
                 for c in table.columns}, na_rep=f"n < {MIN_CELL}"))

        st.markdown("#### How calibrated risk spreads across the bands")
        hist = go.Figure()
        for is_flagged, colour, name in [(False, ON_TRACK, "Not flagged"), (True, AT_RISK, "Flagged")]:
            hist.add_trace(go.Histogram(x=scored.loc[scored["flagged"] == is_flagged, "score"], name=name,
                                        marker_color=colour, xbins={"start": 0, "end": 1, "size": 0.025}))
        for band, edge in edges.items():
            if edge > 0:
                hist.add_vline(x=edge, line_dash="dash", annotation_text=f"{band} from {edge:.2f}")
        hist.update_layout(barmode="overlay", xaxis_title="Calibrated risk score", yaxis_title="Students",
                           margin={"t": 50}, height=380, legend={"orientation": "h"}, separators=".,",
                           title=f"Calibrated risk scores of {pres} students, with the band cutoffs")
        st.plotly_chart(hist, width="stretch")

        with st.expander("Flagged students (individual rows; for staff follow-up only)"):
            flagged = scored[scored["flagged"]].sort_values("rank")[KEY + ["band", "rank"]]
            flagged = flagged.assign(score_source=score_source(pres))
            st.caption("Open a student in Student Lookup to see the reasons and suggested support.")
            st.dataframe(flagged, hide_index=True)
            st.download_button("Download the flagged list (CSV)", flagged.to_csv(index=False),
                               file_name=f"flagged_{pres}{'_in_sample' if is_training_presentation(pres) else ''}.csv",
                               mime="text/csv")

        st.markdown("### Optional 3D view")
        if st.toggle("Show average risk score by activity and score bands in 3D", key="show_3d"):
            bands = risk_band_table(scored, matrix)
            if not bands["shown"].any():
                st.info(f"No band of {pres} students has at least {MIN_CELL} students, so there is nothing to "
                        f"show in 3D. {hidden_bands(bands)} smaller bands are hidden.")
            else:
                st.plotly_chart(band_figure(bands), width="stretch")
                st.caption(f"Each marker is a band of {pres} students: height and colour show the band's average "
                           f"calibrated risk score, size shows how many students it holds. {hidden_bands(bands)} bands "
                           f"with fewer than {MIN_CELL} students are not shown. Drag to rotate; hover for details.")

with about:
    st.subheader("About")
    if core_ok:
        val = ", ".join(metrics["validation_presentation"])
        v = metrics["final"]["validation"]
        st.markdown("#### Model")
        st.write(f"A Random Forest scores each student from {len(fitted['features'])} inputs known by Day 30: the module, VLE "
                 "activity, assessment submissions and scores so far, registration timing, study load and previous "
                 "attempts. Protected characteristics (gender, region, highest education, deprivation band, age band, "
                 "disability) are not inputs. Isotonic calibration turns the forest's score into a calibrated risk "
                 f"score. The model was trained on {', '.join(sorted(set(matrix['code_presentation']) - {val}))} and validated on "
                 f"{val}, which it never saw during training: ROC-AUC {v['roc_auc']:.3f}, PR-AUC {v['pr_auc']:.3f}, "
                 f"at-risk rate {v['prevalence']:.0%} among {v['n']:,} students.")
        st.markdown("#### Risk bands")
        st.write("Each band starts at a calibrated-score cutoff chosen on out-of-fold training scores, never on the "
                 "validation presentation.")
        st.dataframe(pd.DataFrame({
            "Band": ["Low", "Medium", "High"],
            "Calibrated score": [f"below {edges['Medium']:.2f}", f"{edges['Medium']:.2f} to below {edges['High']:.2f}",
                                 f"{edges['High']:.2f} and above"],
            "Cutoff": ["\u2013", "Wide net: 80% of at-risk training students reach it",
                       "Top 20% of training students by risk"]}), hide_index=True)
        st.markdown("#### Capacity flagging")
        st.write("The Batch Report ranks the chosen cohort by calibrated risk (ties broken by the forest's raw score) "
                 "and flags the highest-risk share set on the slider, to match a team with a fixed number of contacts. "
                 "Precision, recall and the at-risk students caught or missed are computed from that cohort's observed "
                 "outcomes. The wide-net rule instead flags everyone above a fixed cutoff, so the share it flags varies "
                 "between intakes.")
        st.markdown("#### Limitations")
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
        st.markdown(
            "- Support suggestions are rule-based: they match the model's risk drivers to support types and have not "
            "been tested as interventions.\n"
            "- OULAD records no interventions or their outcomes, so the effect of contacting a student is unknown.\n"
            f"- In some modules no assessment is due by Day 30 ({no_due}), so the model has no early scores there.\n"
            + (f"- Recall differs between groups at the top-20% cutoff on {val}: {'; '.join(gaps)}.\n" if gaps else "")
            + f"- Validation figures come from one presentation ({val}); training presentations are in-sample.")
    else:
        st.info("The model description appears here once the Day 30 data and the model are available.")

    st.markdown("#### Data source")
    st.write("Open University Learning Analytics Dataset (OULAD). As stated on the dataset page "
             "(https://research.stem.open.ac.uk/ouanalyse/open-dataset-more/):")
    st.markdown('- License: "This dataset is released under CC-BY 4.0 license." '
                "(https://creativecommons.org/licenses/by/4.0/)\n"
                '- Citing the dataset: "When citing the dataset please use the following reference:" Kuzilek J., '
                "Hlosta M., Zdrahal Z. Open University Learning Analytics dataset Sci. Data 4:170171 doi: "
                "10.1038/sdata.2017.171 (2017).")

    with st.expander("Data details"):
        if model_l.ok:
            st.write(f"Model built (UTC): {final.config.get('created_utc', 'unknown')}")
        for loaded in (matrix_l, model_l):
            if not loaded.ok:
                st.info(loaded.message)
        for name, spec in OPTIONAL_TABLES.items():
            status = get_table(name, processed)
            st.markdown(
                f"**{spec.title}** ({spec.ticket}): {STATUS_LABELS[status.state]}"
                + (f", updated {status.updated}" if status.updated else "") + "\n"
                f"- Expected file: `data/processed/{spec.file}`\n"
                f"- Required columns: {', '.join(f'`{c}`' for c in spec.columns)}")
            if not status.ok:
                st.caption(status.message)
        st.markdown("Rebuild everything from the repository root:")
        st.code("python scripts/run_baseline.py        # Day 30 feature matrix\n"
                "python scripts/train_final.py         # final model\n"
                "python scripts/precompute_shap.py     # optional: explanations for the validation presentation\n"
                "python scripts/build_team_tables.py   # department, funnel and profile tables", language="bash")
        st.caption("File formats and display rules: docs/app.md. Model details: docs/final_model.md.")
