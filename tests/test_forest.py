import json

import numpy as np
import pytest
from conftest import FEATURES

from sar.models import forest


def test_forest_trains_and_scores(matrix, small_forest):
    proba = small_forest.predict_proba(matrix[FEATURES])[:, 1]
    assert proba.shape == (len(matrix),) and ((proba >= 0) & (proba <= 1)).all()
    assert list(small_forest.feature_names_in_) == FEATURES


def test_oof_threshold_reaches_recall_target_inside_training(matrix):
    train = matrix[matrix["code_presentation"] != "2014J"]
    params = {"n_estimators": 20, "min_samples_leaf": 5}
    out = forest.oof_threshold(train[FEATURES], train["at_risk"], train["code_presentation"], 0.8, params)
    assert out["folds"] == 3                   # one fold per training presentation
    assert out["oof_recall"] >= 0.8
    assert 0 < out["threshold"] < 1


def test_oof_threshold_needs_two_presentations(matrix):
    one = matrix[matrix["code_presentation"] == "2013J"]
    with pytest.raises(ValueError):
        forest.oof_threshold(one[FEATURES], one["at_risk"], one["code_presentation"])


def test_save_and_load_roundtrip(tmp_path, matrix, small_forest):
    model_path, meta_path = forest.save_model(small_forest, {"threshold": {"threshold": 0.4}}, tmp_path)
    meta = json.loads(meta_path.read_text())
    assert meta["provisional"] is True and meta["model_file"] == model_path.name
    loaded, meta2 = forest.load_model(meta_path, tmp_path)
    np.testing.assert_allclose(loaded.predict_proba(matrix[FEATURES]),
                               small_forest.predict_proba(matrix[FEATURES]), atol=1e-12)
    assert meta2["threshold"]["threshold"] == 0.4


def test_load_refuses_a_changed_or_missing_model_file(tmp_path, small_forest):
    model_path, meta_path = forest.save_model(small_forest, {}, tmp_path)
    model_path.write_bytes(model_path.read_bytes() + b"x")
    with pytest.raises(ValueError):
        forest.load_model(meta_path, tmp_path)
    model_path.unlink()
    with pytest.raises(FileNotFoundError):
        forest.load_model(meta_path, tmp_path)
