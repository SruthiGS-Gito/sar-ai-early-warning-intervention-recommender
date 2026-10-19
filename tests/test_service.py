import numpy as np
import pandas as pd
import pytest
from conftest import FEATURES, synthetic_matrix

from sar.models.forest import save_model
from sar.service import data, views


def test_missing_matrix_and_model_give_messages_not_errors(tmp_path):
    m = data.load_matrix(tmp_path)
    assert not m.ok and "run_baseline.py" in m.message
    mod = data.load_provisional_model(tmp_path)
    assert not mod.ok and "train_forest.py" in mod.message


def test_unreadable_matrix_gives_a_message(tmp_path):
    (tmp_path / "day30_feature_matrix.csv").write_bytes(b"\x00\xff\x00")
    assert not data.load_matrix(tmp_path).ok


def test_changed_model_file_gives_a_message(tmp_path, small_forest):
    model_path, _ = save_model(small_forest, {}, tmp_path)
    model_path.write_bytes(b"not a model")
    loaded = data.load_provisional_model(tmp_path)
    assert not loaded.ok and "could not be loaded" in loaded.message


def test_model_loads_with_metadata(tmp_path, small_forest):
    save_model(small_forest, {"threshold": {"threshold": 0.4}}, tmp_path)
    loaded = data.load_provisional_model(tmp_path)
    assert loaded.ok and loaded.value[1]["threshold"]["threshold"] == 0.4


@pytest.mark.parametrize("name", list(data.OPTIONAL_TABLES))
def test_optional_tables(tmp_path, name):
    spec = data.OPTIONAL_TABLES[name]
    missing = data.load_optional_table(spec, tmp_path)
    assert not missing.ok and "not available yet" in missing.message and spec.ticket in missing.message

    pd.DataFrame({"other": [1]}).to_csv(tmp_path / spec.file, index=False)
    assert "missing expected columns" in data.load_optional_table(spec, tmp_path).message

    rows = pd.DataFrame({c: ["x"] for c in spec.columns}).assign(id_student=[1])
    rows.to_csv(tmp_path / spec.file, index=False)
    assert "student-level" in data.load_optional_table(spec, tmp_path).message

    pd.DataFrame({c: ["x", "y"] for c in spec.columns}).to_csv(tmp_path / spec.file, index=False)
    ok = data.load_optional_table(spec, tmp_path)
    assert ok.ok and len(ok.value) == 2


def test_resolve_paths_follow_environment(tmp_path, monkeypatch):
    monkeypatch.setenv("SAR_DATA_DIR", str(tmp_path / "d"))
    monkeypatch.setenv("SAR_MODELS_DIR", str(tmp_path / "m"))
    p = data.resolve_paths()
    assert p["processed"] == tmp_path / "d" / "processed" and p["models"] == tmp_path / "m"


def test_scoring_and_batch_summary(matrix, small_forest):
    scored = views.score_presentation(small_forest, matrix, FEATURES, "2014J", 0.5)
    assert len(scored) == (matrix["code_presentation"] == "2014J").sum()
    assert scored["flagged"].equals(scored["score"] >= 0.5)
    s = views.batch_summary(scored)
    assert s.loc["All modules", "Students"] == len(scored)
    assert s.drop(index="All modules")["Flagged"].sum() == scored["flagged"].sum()
    assert s.loc["All modules", "Flagged share"] == pytest.approx(scored["flagged"].mean())


def test_risk_bands_hide_small_cells(small_forest):
    big = synthetic_matrix(n=4000, seed=5)
    scored = views.score_presentation(small_forest, big, FEATURES, "2014J", 0.5)
    grid = views.risk_bands(scored, big)
    m = scored.merge(big[views.KEY + ["active_days", "mean_score"]], on=views.KEY)
    days = pd.cut(m["active_days"], views.DAY_BANDS[0], labels=views.DAY_BANDS[1])
    scores = pd.cut(m["mean_score"], views.SCORE_BANDS[0], labels=views.SCORE_BANDS[1])
    scores = scores.cat.add_categories("none yet").fillna("none yet")
    sizes = m.groupby([days, scores], observed=False).size().unstack()
    values = grid.to_numpy(dtype=float)
    assert (np.isnan(values) == (sizes.to_numpy() < views.MIN_CELL)).all()
    assert np.isnan(values).any() and not np.isnan(values).all()     # some shown, some hidden
    assert np.nanmax(values) <= 1


def test_training_presentation_flag():
    assert views.is_training_presentation("2013J") and not views.is_training_presentation("2014J")
