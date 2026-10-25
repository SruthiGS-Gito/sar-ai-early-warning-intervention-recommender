"""SCRUM-189: Streamlit app for the Day 30 early-warning model.

Run from the repo root:  streamlit run app/streamlit_app.py
Tabs: Student Lookup, Department Dashboard, Retention Funnel, Batch Report.

* Data, model, explainer and the similar-student pool load once (cached).
* Anything missing shows a message; the app does not crash on it.
* Displays are aggregated by default. One student's details appear only after they are chosen,
  and the flagged-student list sits in a closed expander.
* Department, funnel and profile tables belong to tickets 203, 207 and 210. Each gets a status card
  (what it is, owner ticket, expected file and columns, current status); the table itself is shown
  only if its CSV exists and passes the checks in sar.service.data.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib.pyplot as plt  # noqa: E402
import plotly.graph_objects as go  # noqa: E402
import streamlit as st  # noqa: E402

from sar.data.split import TEST_PRESENTATION, make_split  # noqa: E402
from sar.explain.chart import explanation_table, plot_explanation  # noqa: E402
from sar.explain.shap_explain import explain_student, make_explainer  # noqa: E402
from sar.recommend.neighbours import NeighbourPool  # noqa: E402
from sar.recommend.recommender import NOTE, recommend  # noqa: E402
from sar.service.data import (  # noqa: E402
    OPTIONAL_TABLES,
    Loaded,
    TableStatus,
    load_matrix,
    load_provisional_model,
    resolve_paths,
    table_status,
)
from sar.service.figures import band_figure, hidden_bands  # noqa: E402
from sar.service.views import (  # noqa: E402
    MIN_CELL,
    batch_summary,
    is_training_presentation,
    risk_band_table,
    score_presentation,
)
from sar.viz.plots import AT_RISK, ON_TRACK  # noqa: E402

PROVISIONAL = ("**PROVISIONAL model.** A stand-in retrained from the tuned settings of tickets 212 and 217 "
               "(SCRUM-184, SCRUM-185), with a provisional threshold, until the team model (212, 217) and the "
               "decision threshold (214, SCRUM-186) are in the repo. Scores rank risk; they are not probabilities.")
TRAINING_NOTE = ("This is a training presentation: the model learned from these students' outcomes, so their scores "
                 f"look better than they would on new students. Use {TEST_PRESENTATION} to see the model on new students.")


@st.cache_data(show_spinner="Loading the Day 30 feature matrix...")
def get_matrix(processed_dir: str) -> Loaded:
    return load_matrix(processed_dir)


@st.cache_resource(show_spinner="Loading the provisional model...")
def get_model(models_dir: str) -> Loaded:
    return load_provisional_model(models_dir)


@st.cache_resource(show_spinner=False)
def get_explainer(models_dir: str, _model):
    return make_explainer(_model)


@st.cache_resource(show_spinner="Building the similar-student pool...")
def get_pool(processed_dir: str, _matrix) -> Loaded:
    try:
        return Loaded(NeighbourPool(make_split(_matrix)[0]))
    except ValueError as e:
        return Loaded(message=f"Similar students are not available: {e}")


@st.cache_data(show_spinner="Scoring students...")
def get_scores(processed_dir: str, models_dir: str, presentation: str, features: tuple, threshold: float,
               _model, _matrix):
    return score_presentation(_model, _matrix, list(features), presentation, threshold)


@st.cache_data(ttl=60, show_spinner=False)
def get_table(name: str, processed_dir: str) -> TableStatus:
    return table_status(OPTIONAL_TABLES[name], processed_dir)


def presentation_picker(matrix, key: str) -> str:
    options = sorted(matrix["code_presentation"].unique())
    default = options.index(TEST_PRESENTATION) if TEST_PRESENTATION in options else len(options) - 1
    return st.selectbox("Presentation", options, index=default, key=key)


STATUS_LABELS = {"missing": "Not available yet", "unreadable": "File could not be read",
                 "refused": "Refused: contains student-level rows", "missing_columns": "Missing required columns",
                 "loaded": "Loaded"}


def status_card(name: str, processed_dir: str) -> None:
    """One card per team table: what it is, who owns it, what file is expected, and its status."""
    status = get_table(name, processed_dir)
    spec = status.spec
    with st.container(border=True):
        st.markdown(f"#### {spec.title}")
        st.write(spec.description)
        st.markdown(
            f"- **Owner ticket:** {spec.ticket}\n"
            f"- **Expected file:** `data/processed/{spec.file}`\n"
            f"- **Required columns:** {', '.join(f'`{c}`' for c in spec.columns)}\n"
            f"- **Status:** {STATUS_LABELS[status.state]}"
        )
        if status.state == "loaded":
            st.success(status.message)
            st.dataframe(status.table, hide_index=True)
        elif status.state == "missing":
            st.info(status.message)
        else:
            st.warning(status.message)
        st.caption("File format and display rules: docs/app.md")


st.set_page_config(page_title="SAR-AI early warning", layout="wide")
st.title("Student Academic Risk: Day 30 early warning")

paths = resolve_paths()
processed, models = str(paths["processed"]), str(paths["models"])
matrix_l, model_l = get_matrix(processed), get_model(models)
core_ok = matrix_l.ok and model_l.ok
if model_l.ok:
    st.warning(PROVISIONAL)


def core_message() -> None:
    for loaded in (matrix_l, model_l):
        if not loaded.ok:
            st.info(loaded.message)


if core_ok:
    matrix = matrix_l.value
    model, meta = model_l.value
    features, threshold = meta["features"], meta["threshold"]["threshold"]

lookup, dashboard, funnel, batch = st.tabs(["Student Lookup", "Department Dashboard", "Retention Funnel", "Batch Report"])

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
            if is_training_presentation(pres):
                st.info(TRAINING_NOTE)
            expl = explain_student(model, row[features], threshold, get_explainer(models, model))
            left, right = st.columns([3, 2])
            with left:
                fig = plot_explanation(expl)
                st.pyplot(fig)
                plt.close(fig)
            with right:
                st.metric("Risk score", f"{expl.score:.2f}",
                          help=f"Provisional threshold {threshold:.2f}. The score ranks risk; it is not a probability.")
                st.markdown(f"**{'At Risk' if expl.flagged else 'On Track'}** "
                            f"(provisional threshold {threshold:.2f})")
                st.write(expl.sentence)
            st.markdown("#### Suggested support")
            pool_l = get_pool(processed, matrix)
            if not pool_l.ok:
                st.info(pool_l.message)
            else:
                rec = recommend(expl, row, pool_l.value)
                st.caption(rec.neighbours.summary())
                if rec.flagged is False:
                    st.caption("Below the provisional threshold, so these suggestions are optional.")
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
    st.caption("Model-based counts by module (provisional model) are in the Batch Report tab.")
    status_card("department_risk", processed)
    status_card("student_profiles", processed)

with funnel:
    st.subheader("Retention Funnel")
    status_card("retention_funnel", processed)

with batch:
    st.subheader("Batch Report")
    if not core_ok:
        core_message()
    else:
        pres = presentation_picker(matrix, "batch_presentation")
        if is_training_presentation(pres):
            st.info(TRAINING_NOTE)
        scored = get_scores(processed, models, pres, tuple(features), threshold, model, matrix)
        c1, c2, c3 = st.columns(3)
        c1.metric("Students", f"{len(scored):,}")
        c2.metric("Flagged", f"{int(scored['flagged'].sum()):,}")
        c3.metric("Flagged share", f"{scored['flagged'].mean():.0%}" if len(scored) else "n/a")

        st.markdown("#### By module")
        st.dataframe(batch_summary(scored).style.format({"Flagged share": "{:.0%}", "Average score": "{:.2f}"}))

        st.markdown("#### How scores spread around the threshold")
        hist = go.Figure()
        for is_flagged, colour, name in [(False, ON_TRACK, "Below threshold"), (True, AT_RISK, "Flagged")]:
            hist.add_trace(go.Histogram(x=scored.loc[scored["flagged"] == is_flagged, "score"], name=name,
                                        marker_color=colour, xbins={"start": 0, "end": 1, "size": 0.025}))
        hist.add_vline(x=threshold, line_dash="dash", annotation_text=f"Provisional threshold {threshold:.2f}")
        hist.update_layout(barmode="overlay", xaxis_title="Risk score", yaxis_title="Students",
                           margin={"t": 30}, height=360, legend={"orientation": "h"})
        st.plotly_chart(hist, width="stretch")

        with st.expander("Flagged students (individual rows; for staff follow-up only)"):
            flagged = scored[scored["flagged"]].sort_values("score", ascending=False)
            st.caption("Open a student in Student Lookup to see the reasons and suggested support.")
            st.dataframe(flagged.drop(columns="flagged").round({"score": 3}), hide_index=True)
            st.download_button("Download the flagged list (CSV)", flagged.to_csv(index=False),
                               file_name=f"flagged_{pres}.csv", mime="text/csv")

        st.markdown("### Optional 3D view")
        if st.toggle("Show average risk score by activity and score bands in 3D", key="show_3d"):
            bands = risk_band_table(scored, matrix)
            if not bands["shown"].any():
                st.info(f"No band of {pres} students has at least {MIN_CELL} students, so there is nothing to "
                        f"show in 3D. {hidden_bands(bands)} smaller bands are hidden.")
            else:
                st.plotly_chart(band_figure(bands), width="stretch")
                st.caption(f"Each marker is a band of {pres} students: height and colour show the band's average "
                           f"risk score, size shows how many students it holds. {hidden_bands(bands)} bands with fewer "
                           f"than {MIN_CELL} students are not shown. Drag to rotate; hover for details.")
