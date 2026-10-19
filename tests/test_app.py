"""Run the Streamlit app headless on synthetic data (streamlit.testing.v1.AppTest)."""

import pandas as pd
import pytest
from conftest import FEATURES

from sar.config import PROJECT_ROOT
from sar.models.forest import save_model

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
def synthetic_env(empty_env, matrix, small_forest):
    processed = empty_env / "data" / "processed"
    processed.mkdir(parents=True)
    matrix.to_csv(processed / "day30_feature_matrix.csv", index=False)
    meta = {"features": FEATURES, "threshold": {"threshold": 0.5}}
    save_model(small_forest, meta, empty_env / "models")
    return empty_env


def test_app_runs_with_nothing_available(empty_env):
    at = AppTest.from_file(APP, default_timeout=120).run()
    assert not at.exception
    text = _texts(at)
    assert "run_baseline.py" in text and "train_forest.py" in text
    for ticket in ["ticket 203", "ticket 207", "ticket 210"]:
        assert f"not available yet: {ticket}" in text
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
    assert "Model-based counts by module (provisional model) are in the Batch Report tab." in text


def test_app_with_synthetic_data_and_model(synthetic_env, matrix):
    at = AppTest.from_file(APP, default_timeout=120).run()
    assert not at.exception
    text = _texts(at)
    assert "PROVISIONAL model" in text
    n_val = int((matrix["code_presentation"] == "2014J").sum())
    assert f"{n_val:,}" in text                                      # batch report student count
    assert at.selectbox(key="lookup_student").value is None        # no student shown by default
    assert not at.toggle(key="show_3d").value                       # 3D view is opt-in

    module = at.selectbox(key="lookup_module").value
    sid = at.selectbox(key="lookup_student").options[0]
    at.selectbox(key="lookup_student").set_value(int(sid)).run()
    assert not at.exception
    text = _texts(at)
    assert "Risk score" in text and "Suggested support" in text
    assert "most similar students from earlier presentations" in text and f"module {module}" in text
    assert "OULAD records no interventions" in text

    at.toggle(key="show_3d").set_value(True).run()
    assert not at.exception


def test_training_presentation_is_labelled(synthetic_env):
    at = AppTest.from_file(APP, default_timeout=120).run()
    at.selectbox(key="batch_presentation").set_value("2013J").run()
    assert not at.exception
    assert "training presentation" in _texts(at)


def test_valid_team_table_is_shown_and_student_rows_are_refused(synthetic_env):
    processed = synthetic_env / "data" / "processed"
    pd.DataFrame({"stage": ["Registered", "Passed"], "students": [100, 40]}).to_csv(
        processed / "retention_funnel.csv", index=False)
    pd.DataFrame({"code_module": ["AAA"], "id_student": [1]}).to_csv(processed / "department_risk.csv", index=False)
    at = AppTest.from_file(APP, default_timeout=120).run()
    assert not at.exception
    text = _texts(at)
    assert "Retention funnel is not available yet" not in text
    assert "student-level rows" in text and "**Status:** Refused: contains student-level rows" in text
    assert "**Status:** Loaded" in text and "Retention funnel: 2 rows, last updated" in text
    assert any(len(df.value) == 2 for df in at.dataframe)


def test_missing_columns_are_named_on_the_card(synthetic_env):
    processed = synthetic_env / "data" / "processed"
    pd.DataFrame({"stage": ["Registered"]}).to_csv(processed / "retention_funnel.csv", index=False)
    at = AppTest.from_file(APP, default_timeout=120).run()
    assert not at.exception
    text = _texts(at)
    assert "**Status:** Missing required columns" in text and "missing expected columns: students" in text
