"""Run the Streamlit app headless on synthetic data (streamlit.testing.v1.AppTest)."""

import re

import pandas as pd
import pytest

from sar.config import PROJECT_ROOT
from sar.explain.shap_explain import explain_rows
from sar.models.final import save_final
from sar.service.data import load_final_model, save_shap_cache

APP = str(PROJECT_ROOT / "app" / "streamlit_app.py")
AppTest = pytest.importorskip("streamlit.testing.v1").AppTest


def _texts(at) -> str:
    parts = [e.value for e in at.info] + [e.value for e in at.warning] + [e.value for e in at.markdown]
    parts += [e.value for e in at.success] + [e.value for e in at.code]
    parts += [e.value for e in at.caption] + [str(e.value) for e in at.metric]
    return "\n".join(str(p) for p in parts)


@pytest.fixture
def empty_env(tmp_path, monkeypatch):
    monkeypatch.setenv("SAR_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("SAR_MODELS_DIR", str(tmp_path / "models"))
    return tmp_path


@pytest.fixture
def synthetic_env(empty_env, matrix, small_final):
    processed = empty_env / "data" / "processed"
    processed.mkdir(parents=True)
    matrix.to_csv(processed / "day30_feature_matrix.csv", index=False)
    fitted, metrics = small_final
    save_final(fitted, {"seed": 42}, metrics, empty_env / "models" / "final")
    return empty_env


MAIN_TABS = ["Student Lookup", "Department Dashboard", "Retention Funnel", "Batch Report"]
DEVELOPER_TEXT = re.compile(r"SCRUM-|ticket \d|scripts/|\.py\b|docs/|models/|metrics\.json|last updated|\d+ rows")


def _main_tab_texts(at) -> str:
    return "\n".join(_texts(t) for t in at.tabs if t.label in MAIN_TABS)


def _about_text(at) -> str:
    return _texts([t for t in at.tabs if t.label == "About"][0])


def test_app_runs_with_nothing_available(empty_env):
    at = AppTest.from_file(APP, default_timeout=120).run()
    assert not at.exception
    assert [t.label for t in at.tabs] == [*MAIN_TABS, "About"]
    main, about = _main_tab_texts(at), _about_text(at)
    assert "not available yet" in main and "Data details" in main
    assert not DEVELOPER_TEXT.search(main), DEVELOPER_TEXT.search(main)
    assert "run_baseline.py" in about and "train_final.py" in about and "build_team_tables.py" in about
    for ticket in ["ticket 203", "ticket 207", "ticket 210"]:
        assert f"not available yet: {ticket}" in about


def test_status_cards_describe_each_team_table(empty_env):
    from sar.service.data import OPTIONAL_TABLES

    at = AppTest.from_file(APP, default_timeout=120).run()
    assert not at.exception
    main, about = _main_tab_texts(at), _about_text(at)
    for spec in OPTIONAL_TABLES.values():
        assert f"#### {spec.title}" in main and spec.description in main
        assert f"**{spec.title}** ({spec.ticket}): Not available yet" in about
        assert f"`data/processed/{spec.file}`" in about
        assert all(f"`{c}`" in about for c in spec.columns)
    assert main.count("Not available yet. Details are in the About tab") == 3
    assert "Model-based counts by module are in the Batch Report tab." in main


def test_main_tabs_hold_no_developer_text_and_about_holds_the_details(tables_env):
    at = AppTest.from_file(APP, default_timeout=120).run()
    at.toggle(key="show_3d").set_value(True).run()
    assert not at.exception
    main, about = _main_tab_texts(at), _about_text(at)
    assert not DEVELOPER_TEXT.search(main), DEVELOPER_TEXT.search(main)
    assert "24 inputs known by Day 30" in about and "isotonic calibration" in about.lower() and "validated on 2014J" in about
    bands = [t for t in at.tabs if t.label == "About"][0].dataframe[0].value
    assert list(bands["Band"]) == ["Low", "Medium", "High"] and "Capacity flagging" in about
    assert "rule-based" in about and "no interventions" in about
    assert '"This dataset is released under CC-BY 4.0 license."' in about
    assert "doi: 10.1038/sdata.2017.171 (2017)." in about
    assert "Model built (UTC)" in about and "python scripts/build_team_tables.py" in about


def test_app_with_synthetic_data_and_model(synthetic_env, matrix):
    at = AppTest.from_file(APP, default_timeout=120).run()
    assert not at.exception
    text = _texts(at)
    assert "provisional" not in text.lower() and "threshold" not in text.lower()
    assert "Validated on 2014J" in text and "24 inputs known by Day 30" in _about_text(at)
    n_val = int((matrix["code_presentation"] == "2014J").sum())
    assert f"{n_val:,}" in text                                      # batch report student count
    assert at.radio(key="flag_rule").value == "Capacity ranking"     # ranking is the default rule
    assert at.slider(key="capacity").value == 20
    cards = {m.label: m.value for m in at.metric}
    assert cards["Flagged"] == f"{round(0.2 * n_val):,}"               # top 20% flagged
    assert {"Students", "Flagged", "At-risk caught", "At-risk missed", "Precision", "Recall"} <= set(cards)
    val = matrix[matrix["code_presentation"] == "2014J"]
    assert int(cards["At-risk caught"].replace(",", "")) + int(cards["At-risk missed"].replace(",", "")) ==         int(val["at_risk"].sum())
    assert "Flagged students by risk band: High" in text and "can include Medium students" in text
    assert "Observed at-risk rate inside each risk band" in text and "lift" in text
    by_module = [df.value for df in at.dataframe if "Flagged share" in df.value.columns][0]
    assert {"At-risk rate", "Recall", "Precision"} <= set(by_module.columns)
    assert at.selectbox(key="lookup_student").value is None        # no student shown by default
    assert not at.toggle(key="show_3d").value                       # 3D view is opt-in

    module = at.selectbox(key="lookup_module").value
    sid = at.selectbox(key="lookup_student").options[0]
    at.selectbox(key="lookup_student").set_value(int(sid)).run()
    assert not at.exception
    text = _texts(at)
    band = [m for m in at.metric if m.label == "Risk band"]
    assert len(band) == 1 and band[0].value in {"Low", "Medium", "High"}
    assert "Rank " in text and "Suggested support" in text
    assert "most similar students from earlier presentations" in text and f"module {module}" in text
    assert "OULAD records no interventions" in text
    assert "provisional" not in text.lower()

    at.radio(key="flag_rule").set_value("Wide net (fixed cutoff)").run()
    assert not at.exception and "Wide net: calibrated risk score of at least" in _texts(at)
    at.toggle(key="show_3d").set_value(True).run()
    assert not at.exception


def test_lookup_uses_precomputed_explanations_when_present(synthetic_env, matrix):
    models = synthetic_env / "models"
    loaded = load_final_model(models).value
    fitted = loaded.fitted
    part = matrix[matrix["code_presentation"] == "2014J"]
    base, contrib, raw = explain_rows(fitted["model"], part[fitted["features"]])
    save_shap_cache(models, "2014J", part[["code_module", "code_presentation", "id_student"]], contrib, raw, base,
                    loaded.config)
    at = AppTest.from_file(APP, default_timeout=120).run()
    sid = at.selectbox(key="lookup_student").options[0]
    at.selectbox(key="lookup_student").set_value(int(sid)).run()
    assert not at.exception
    assert "Background characteristics are not model inputs." in _texts(at)


def test_training_presentation_is_labelled(synthetic_env):
    at = AppTest.from_file(APP, default_timeout=120).run()
    at.toggle(key="batch_presentation_training").set_value(True).run()
    at.selectbox(key="batch_presentation").set_value("2013J").run()
    assert not at.exception
    assert "training presentation" in _texts(at)


@pytest.fixture
def tables_env(synthetic_env, matrix, small_final):
    """Synthetic environment with the three team tables built by sar.tables."""
    from conftest import synthetic_registrations

    from sar.data.split import make_split
    from sar.tables import department, funnel, profiles

    processed = synthetic_env / "data" / "processed"
    department.department_risk(small_final[0], matrix).to_csv(processed / "department_risk.csv", index=False)
    funnel.retention_funnel(*synthetic_registrations(matrix), matrix).to_csv(
        processed / "retention_funnel.csv", index=False)
    train, val = make_split(matrix)
    table = profiles.student_profiles(profiles.fit_profiles(train), train, val)
    table.to_csv(processed / "student_profiles.csv", index=False)
    return synthetic_env, table


def test_team_tables_are_shown_with_charts(tables_env):
    _, prof = tables_env
    at = AppTest.from_file(APP, default_timeout=120).run()
    assert not at.exception
    text = _texts(at)
    assert "not available yet" not in text and "**Status:**" not in text
    for title in ["Department risk", "Student profiles"]:
        assert f"#### {title}" in text
    assert "#### Retention funnel" not in text
    assert "Retrospective, descriptive view" in text and "not model inputs" in text
    assert "Counts under 20 are hidden" in text and "{" not in text
    validation = prof[prof["scope"] == "validation"]
    for _, row in validation.iterrows():
        assert f"**{row['profile']}**: {row['description']}" in text
    assert len(at.dataframe) >= 3
    assert all(c in text for c in ["Groups under 20 students", "without background characteristics"])


def test_team_table_selectors_and_training_note(tables_env):
    at = AppTest.from_file(APP, default_timeout=120).run()
    at.toggle(key="dept_presentation_training").set_value(True).run()
    at.selectbox(key="dept_presentation").set_value("2013J").run()
    assert not at.exception and "Training presentation: the model learned" in _texts(at)
    at.selectbox(key="funnel_module").set_value("AAA").run()
    at.radio(key="profile_scope").set_value("training").run()
    assert not at.exception


def test_student_level_team_table_is_refused(synthetic_env):
    processed = synthetic_env / "data" / "processed"
    pd.DataFrame({"code_module": ["AAA"], "id_student": [1]}).to_csv(processed / "department_risk.csv", index=False)
    at = AppTest.from_file(APP, default_timeout=120).run()
    assert not at.exception
    main, about = _main_tab_texts(at), _about_text(at)
    assert "the file contains student-level rows" in main
    assert "Refused: contains student-level rows" in about and "Retention funnel is not available yet" in about


def test_missing_columns_are_named_in_the_data_details(synthetic_env):
    processed = synthetic_env / "data" / "processed"
    pd.DataFrame({"stage": ["Registered"]}).to_csv(processed / "retention_funnel.csv", index=False)
    at = AppTest.from_file(APP, default_timeout=120).run()
    assert not at.exception
    assert "the file does not have the expected columns" in _main_tab_texts(at)
    about = _about_text(at)
    assert "Missing required columns" in about
    assert "missing expected columns: code_module, code_presentation, stage_order, stage_key, students" in about


def test_funnel_shows_a_stage_that_does_not_apply_as_not_applicable(synthetic_env, matrix):
    from conftest import synthetic_registrations

    from sar.tables import funnel

    no_due = matrix.assign(n_assessments_due=matrix["n_assessments_due"].where(matrix["code_module"] != "GGG", 0))
    table = funnel.retention_funnel(*synthetic_registrations(matrix), no_due)
    table.to_csv(synthetic_env / "data" / "processed" / "retention_funnel.csv", index=False)
    at = AppTest.from_file(APP, default_timeout=120).run()
    at.selectbox(key="funnel_module").set_value("GGG").run()
    assert not at.exception
    text = _texts(at)
    assert "submission stage is not applicable" in text
    carried = table[(table["code_module"] == "GGG") & (table["code_presentation"] == "2014J")
                    & ~table["applicable"]]["students"].sum()
    assert f"{carried:,} students in GGG are carried forward on the submission check" in text
    funnel_view = [df.value for df in at.dataframe if "Stage" in df.value.columns][0]
    assert "not applicable" in set(funnel_view["Students"])
    assert funnel_view["Stage"].str.endswith("(not applicable)").sum() == 1


def test_band_views_default_to_the_validation_presentation(tables_env):
    at = AppTest.from_file(APP, default_timeout=120).run()
    assert not at.exception
    text = _texts(at)
    keys = {s.key for s in at.selectbox}
    assert "dept_presentation" not in keys and "batch_presentation" not in keys
    assert text.count("Showing 2014J, the validation presentation.") == 2
    assert text.count("Training presentations have in-sample scores") == 2
    dept = [df.value for df in at.dataframe if "Scores" in df.value.columns and "Module" in df.value.columns][0]
    assert set(dept["Presentation"]) == {"2014J"} and set(dept["Scores"]) == {"validation (2014J)"}


def test_training_presentations_appear_only_after_opting_in(tables_env):
    at = AppTest.from_file(APP, default_timeout=120).run()
    at.toggle(key="dept_presentation_training").set_value(True).run()
    at.toggle(key="batch_presentation_training").set_value(True).run()
    assert not at.exception
    for key in ["dept_presentation", "batch_presentation"]:
        assert set(at.selectbox(key=key).options) == {"2013B", "2013J", "2014B"}
    at.toggle(key="batch_presentation_training").set_value(True).run()
    at.selectbox(key="batch_presentation").set_value("2013J").run()
    exported = [df.value for df in at.dataframe if "score_source" in df.value.columns]
    assert exported and set(exported[0]["score_source"]) == {"training (in-sample)"}
    by_module = [df.value for df in at.dataframe if "Flagged share" in df.value.columns][0]
    assert set(by_module["Scores"]) == {"training (in-sample)"}


def test_training_presentation_metrics_are_labelled_in_sample(synthetic_env):
    at = AppTest.from_file(APP, default_timeout=120).run()
    at.toggle(key="batch_presentation_training").set_value(True).run()
    at.selectbox(key="batch_presentation").set_value("2013J").run()
    assert not at.exception
    labels = {m.label for m in at.metric}
    assert {"At-risk caught (in-sample)", "Precision (in-sample)", "Recall (in-sample)"} <= labels
    by_module = [df.value for df in at.dataframe if "Flagged share" in df.value.columns][0]
    assert "Recall" not in by_module.columns
    assert "shown for 2014J only" in _texts(at)


def test_capacity_slider_accepts_any_share(synthetic_env, matrix):
    at = AppTest.from_file(APP, default_timeout=120).run()
    at.slider(key="capacity").set_value(13).run()
    assert not at.exception
    n_val = int((matrix["code_presentation"] == "2014J").sum())
    assert {m.label: m.value for m in at.metric}["Flagged"] == f"{round(0.13 * n_val):,}"


def _chart_specs(at) -> list[dict]:
    import json

    return [json.loads(c.proto.spec) for c in at.get("plotly_chart")]


def test_tables_and_charts_are_formatted_for_reading(tables_env):
    env, prof = tables_env
    processed = env / "data" / "processed"
    dept = pd.read_csv(processed / "department_risk.csv")
    hide = (dept["code_presentation"] == "2014J") & (dept["code_module"] == "AAA")
    rate_cols = ["at_risk_rate", "share_low", "share_medium", "share_high", "rate_low", "rate_medium", "rate_high"]
    dept.loc[hide, rate_cols] = float("nan")
    dept.loc[hide, "suppressed"] = True
    dept.to_csv(processed / "department_risk.csv", index=False)
    at = AppTest.from_file(APP, default_timeout=120).run()
    assert not at.exception
    frames = [df.value for df in at.dataframe]
    view = [f for f in frames if "High band" in f.columns][0]
    assert "Hidden (small group)" not in view.columns
    assert view.loc[view["Module"] == "AAA", "At-risk rate"].iloc[0] == "n < 20"
    profiles = [f for f in frames if "Mean clicks" in f.columns][0]
    assert {"Mean active days", "Mean days inactive before Day 30", "Mean assessments submitted",
            "Mean assessments missed"} <= set(profiles.columns)
    rates = profiles["At-risk rate"].astype(float)
    assert rates.is_monotonic_decreasing
    assert any(e.label == "What each profile means" for e in at.expander)
    specs = _chart_specs(at)
    titles = [s["layout"].get("title", {}).get("text", "") for s in specs]
    assert all(titles), titles
    stacked = [s for s in specs if s["layout"]["title"]["text"].startswith("Share of students in each risk band")][0]
    assert "All modules" in stacked["data"][0]["x"] and stacked["data"][0]["texttemplate"] == "%{y:.0%}"
    bars = [s for s in specs if s["layout"]["title"]["text"] == "Observed at-risk rate by profile"][0]
    assert bars["data"][0]["marker"]["color"] == "#8C8C8C"
