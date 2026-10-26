"""Run the Streamlit app headless on synthetic data (streamlit.testing.v1.AppTest)."""

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
    parts += [e.value for e in at.success]
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


def test_app_runs_with_nothing_available(empty_env):
    at = AppTest.from_file(APP, default_timeout=120).run()
    assert not at.exception
    text = _texts(at)
    assert "run_baseline.py" in text and "train_final.py" in text
    for ticket in ["ticket 203", "ticket 207", "ticket 210"]:
        assert f"not available yet: {ticket}" in text
    assert text.count("Build it with: python scripts/build_team_tables.py") == 3
    assert [t.label for t in at.tabs] == ["Student Lookup", "Department Dashboard", "Retention Funnel", "Batch Report"]


def test_status_cards_describe_each_team_table(empty_env):
    from sar.service.data import OPTIONAL_TABLES

    at = AppTest.from_file(APP, default_timeout=120).run()
    assert not at.exception
    text = _texts(at)
    for spec in OPTIONAL_TABLES.values():
        assert f"#### {spec.title}" in text and spec.description in text
        assert f"**Owner ticket:** {spec.ticket}" in text
        assert f"`data/processed/{spec.file}`" in text
        assert all(f"`{c}`" in text for c in spec.columns)
    assert text.count("**Status:** Not available yet") == 3
    assert text.count("File format and display rules: docs/app.md") == 3
    assert "Model-based counts by module are in the Batch Report tab." in text


def test_app_with_synthetic_data_and_model(synthetic_env, matrix):
    at = AppTest.from_file(APP, default_timeout=120).run()
    assert not at.exception
    text = _texts(at)
    assert "provisional" not in text.lower() and "threshold" not in text.lower()
    assert "Validated on 2014J" in text and "24 Day 30 inputs" in text
    n_val = int((matrix["code_presentation"] == "2014J").sum())
    assert f"{n_val:,}" in text                                      # batch report student count
    assert at.radio(key="flag_rule").value == "Capacity ranking"     # ranking is the default rule
    assert at.select_slider(key="capacity").value == 0.2
    assert f"{round(0.2 * n_val):,}" in [m.value for m in at.metric]  # top 20% flagged
    assert "precision" in text and "recall" in text and "metrics.json" in text
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
    for title in ["Department risk", "Retention funnel", "Student profiles"]:
        assert f"#### {title}" in text and f"{title}: " in text
    assert "Retrospective, descriptive view" in text and "not model inputs" in text
    assert "Counts under 20 are hidden" in text and "{" not in text
    validation = prof[prof["scope"] == "validation"]
    for _, row in validation.iterrows():
        assert f"**{row['profile']}**: {row['description']}" in text
    assert len(at.dataframe) >= 3
    assert all(c in text for c in ["Groups under 20 students", "without background characteristics"])


def test_team_table_selectors_and_training_note(tables_env):
    at = AppTest.from_file(APP, default_timeout=120).run()
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
    text = _texts(at)
    assert "student-level rows" in text and "**Status:** Refused: contains student-level rows" in text
    assert "Retention funnel is not available yet" in text


def test_missing_columns_are_named_on_the_card(synthetic_env):
    processed = synthetic_env / "data" / "processed"
    pd.DataFrame({"stage": ["Registered"]}).to_csv(processed / "retention_funnel.csv", index=False)
    at = AppTest.from_file(APP, default_timeout=120).run()
    assert not at.exception
    text = _texts(at)
    assert "**Status:** Missing required columns" in text
    assert "missing expected columns: code_module, code_presentation, stage_order, stage_key, students" in text


def test_funnel_shows_a_stage_that_does_not_apply_as_not_applicable(synthetic_env, matrix):
    from conftest import synthetic_registrations

    from sar.tables import funnel

    no_due = matrix.assign(n_assessments_due=matrix["n_assessments_due"].where(matrix["code_module"] != "GGG", 0))
    table = funnel.retention_funnel(*synthetic_registrations(matrix), no_due)
    table.to_csv(synthetic_env / "data" / "processed" / "retention_funnel.csv", index=False)
    at = AppTest.from_file(APP, default_timeout=120).run()
    at.selectbox(key="funnel_module").set_value("GGG").run()
    assert not at.exception
    assert "submission stage is not applicable" in _texts(at)
    funnel_view = [df.value for df in at.dataframe if "Stage" in df.value.columns][0]
    assert "not applicable" in set(funnel_view["Students"])
    assert funnel_view["Stage"].str.endswith("(not applicable)").sum() == 1
