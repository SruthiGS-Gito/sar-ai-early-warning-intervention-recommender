import json

import numpy as np
import pytest
from conftest import FEATURES, synthetic_matrix

from sar.config import PROCESSED_DIR, PROTECTED_FEATURES
from sar.explain.factors import BACKGROUND
from sar.models import final

SMALL = {"n_estimators": 30, "min_samples_leaf": 5, "class_weight": "balanced_subsample"}
TOP_KEYS = {"validation_presentation", "train_rows", "validation_rows", "final", "comparisons"}
FINAL_KEYS = {"features", "cutoffs_from_oof", "oof_training", "validation", "bootstrap_validation",
              "fairness_validation", "leave_one_presentation_out"}
VALIDATION_KEYS = {"n", "prevalence", "roc_auc", "pr_auc", "roc_auc_calibrated", "pr_auc_calibrated",
                   "raw", "calibrated", "cutoffs", "capacity"}
CUTOFFS = ["wide_net", "top_30", "top_20", "top_10"]          # loosest to strictest
RATES = {"prevalence", "roc_auc", "pr_auc", "roc_auc_calibrated", "pr_auc_calibrated", "brier", "ece",
         "flagged_share", "precision", "recall", "flag_rate", "oof_flagged_share", "oof_recall", "oof_precision",
         "mean_predicted", "observed_rate", "score", "brier_calibrated", "ece_calibrated"}


@pytest.fixture(scope="module")
def result():
    m = synthetic_matrix(n=1600, seed=3)
    fitted, metrics = final.run(m, FEATURES, params=SMALL, n_boot=20)
    return m, fitted, metrics


def _rates(node, path=""):
    """Yield (path, value) for every rate in a nested metrics dict, including CI bounds."""
    if isinstance(node, dict):
        for k, v in node.items():
            if k in RATES and isinstance(v, (int, float)):
                yield f"{path}.{k}", v
            elif k in {"precision_top_20", "recall_top_20", "roc_auc", "pr_auc"} and isinstance(v, dict):
                for b in v["ci95"]:
                    yield f"{path}.{k}.ci95", b
            else:
                yield from _rates(v, f"{path}.{k}")


def test_final_features_leave_out_every_protected_feature():
    feats = final.final_features(FEATURES)
    assert not set(feats) & set(PROTECTED_FEATURES)
    assert len(feats) == len(FEATURES) - len(PROTECTED_FEATURES) == 24
    assert BACKGROUND == PROTECTED_FEATURES                       # one list, used everywhere


def test_cutoffs_get_stricter(result):
    _, fitted, _ = result
    c = fitted["cutoffs"]
    assert list(c) == CUTOFFS
    scores = [c[k]["score"] for k in CUTOFFS]
    shares = [c[k]["oof_flagged_share"] for k in CUTOFFS]
    # Isotonic scores are steps, so on small data two cutoffs can share a step; never looser.
    assert all(a <= b for a, b in zip(scores, scores[1:]))
    assert all(a >= b for a, b in zip(shares, shares[1:]))
    assert scores[0] < scores[-1]
    assert c["wide_net"]["oof_recall"] >= 0.80


def test_calibrator_is_monotonic(result):
    _, fitted, _ = result
    out = fitted["calibrator"].predict(np.linspace(0, 1, 501))
    assert (np.diff(out) >= 0).all() and out.min() >= 0 and out.max() <= 1


def test_saved_model_reloads_with_identical_predictions(tmp_path, result):
    m, fitted, metrics = result
    paths = final.save_final(fitted, {"seed": 42}, metrics, tmp_path)
    loaded, config = final.load_final(tmp_path)
    sample = m.sample(200, random_state=0)
    # One thread for both, so the sum over trees runs in the same order and results are bit-identical.
    for f in (fitted, loaded):
        f["model"].set_params(clf__n_jobs=1)
    for a, b in zip(final.predict(fitted, sample), final.predict(loaded, sample)):
        np.testing.assert_array_equal(a, b)
    assert config["features"] == fitted["features"] and config["cutoffs"] == fitted["cutoffs"]
    paths["calibrator"].write_bytes(paths["calibrator"].read_bytes() + b"x")
    with pytest.raises(ValueError):
        final.load_final(tmp_path)


def test_metrics_have_expected_keys_and_rates_in_range(result):
    _, _, metrics = result
    assert TOP_KEYS <= set(metrics)
    assert FINAL_KEYS <= set(metrics["final"])
    assert set(metrics["comparisons"]) == {"forest_30_features", "logistic_baseline"}
    for v in [metrics["final"]["validation"]] + [c["validation"] for c in metrics["comparisons"].values()]:
        assert VALIDATION_KEYS <= set(v)
        assert list(v["cutoffs"]) == CUTOFFS
        assert list(v["capacity"]) == ["5%", "10%", "15%", "20%", "25%", "30%", "40%", "50%"]
        assert len(v["raw"]["reliability"]) == len(v["calibrated"]["reliability"]) == 10
    assert set(metrics["final"]["leave_one_presentation_out"]) == {"2013B", "2013J", "2014B", "2014J"}
    gaps = metrics["final"]["fairness_validation"]["gaps"]
    assert set(gaps) == {"recall_gap_gender_F_minus_M", "flag_rate_gap_gender_F_minus_M",
                         "recall_gap_age_band_35-55_minus_0-35", "flag_rate_gap_age_band_35-55_minus_0-35"}
    rates = list(_rates(metrics))
    assert len(rates) > 100
    bad = [(p, v) for p, v in rates if not 0 <= v <= 1]
    assert not bad, bad


def test_validation_rows_never_reach_the_fit(result):
    m, fitted, _ = result
    n_train = int((m["code_presentation"] != "2014J").sum() - m.loc[m["code_presentation"] != "2014J", "id_student"]
                  .isin(m.loc[m["code_presentation"] == "2014J", "id_student"]).sum())
    assert len(fitted["oof_y"]) == n_train


@pytest.mark.skipif(not all((final.FINAL_DIR / f).exists() for f in ("model.joblib", "calibrator.joblib")),
                    reason="run python scripts/train_final.py to build the final model")
def test_built_final_model_files():
    loaded, config = final.load_final()
    assert not set(config["features"]) & set(PROTECTED_FEATURES) and len(config["features"]) == 24
    metrics = json.loads((final.FINAL_DIR / "metrics.json").read_text())
    assert TOP_KEYS <= set(metrics) and FINAL_KEYS <= set(metrics["final"])
    assert not [(p, v) for p, v in _rates(metrics) if not 0 <= v <= 1]
    scores = [config["cutoffs"][k]["score"] for k in CUTOFFS]
    assert all(a < b for a, b in zip(scores, scores[1:]))


@pytest.mark.skipif(not all((final.FINAL_DIR / f).exists() for f in ("model.joblib", "calibrator.joblib"))
                    or not (PROCESSED_DIR / "day30_feature_matrix.csv").exists(),
                    reason="run python scripts/run_baseline.py and python scripts/train_final.py first")
def test_app_capacity_flags_reproduce_the_committed_capacity_table():
    """The app's ranking (calibrated score, ties broken by raw score) gives the stored validation figures."""
    import pandas as pd

    from sar.service import views

    fitted, config = final.load_final()
    metrics = json.loads((final.FINAL_DIR / "metrics.json").read_text())
    matrix = pd.read_csv(PROCESSED_DIR / "day30_feature_matrix.csv")
    scored = views.score_presentation(fitted, matrix, config["validation_presentation"][0])
    for key, stored in metrics["final"]["validation"]["capacity"].items():
        o = views.flag_outcome(scored, views.flag_capacity(scored, int(key.rstrip("%")) / 100).flagged)
        assert o.flagged == stored["flagged"]
        assert o.precision == pytest.approx(stored["precision"], abs=1e-12)
        assert o.recall == pytest.approx(stored["recall"], abs=1e-12)
