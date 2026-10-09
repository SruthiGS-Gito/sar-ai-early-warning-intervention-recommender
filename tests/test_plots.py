import numpy as np
import pandas as pd

from sar.viz import plots


def test_charts_are_saved(tmp_path):
    info = pd.DataFrame({
        "code_module": ["AAA", "AAA", "BBB", "BBB"],
        "final_result": ["Pass", "Fail", "Withdrawn", "Distinction"],
    })
    sa = pd.DataFrame({"score": [10.0, 55.0, None, 90.0]})
    eng = pd.DataFrame({"code_module": ["AAA", "BBB", "BBB"], "active_days": [3, 8, 12]})
    plots.plot_final_results(info, out_dir=tmp_path)
    plots.plot_results_by_module(info, out_dir=tmp_path)
    plots.plot_score_distribution(sa, out_dir=tmp_path)
    plots.plot_active_days_by_module(eng, out_dir=tmp_path)
    plots.plot_submission_timing(pd.Series([-3, 0, 4, None]), out_dir=tmp_path)
    assert len(list(tmp_path.glob("*.png"))) == 5


def _day30_data(n=400, seed=0):
    rng = np.random.default_rng(seed)
    key = pd.DataFrame({
        "code_module": rng.choice(["AAA", "BBB", "GGG"], n),
        "code_presentation": np.where(np.arange(n) < 300, "2013J", "2014J"),
        "id_student": np.arange(n),
    })
    active = rng.integers(0, 31, n)
    at_risk = (active + rng.normal(0, 8, n) < 12).astype(int)
    matrix = key.assign(
        at_risk=at_risk,
        active_days=active,
        total_clicks=active * 10 + rng.integers(0, 5, n),
        max_consecutive_inactive_days=np.where(active == 0, 31, 30 - active),
        mean_score=np.where(key["code_module"] == "GGG", np.nan, rng.normal(70, 10, n)),
        region=rng.choice(["North", "South"], n),
    )
    info = key.assign(final_result=np.where(at_risk == 1, "Withdrawn", "Pass"))
    reg = key.assign(date_unregistration=np.where(at_risk == 1, rng.integers(-20, 200, n), np.nan))
    vle = key.assign(sum_click=matrix["total_clicks"] + (1 - at_risk) * 500)
    return matrix, info, reg, vle


def test_day30_population_and_engagement_charts(tmp_path):
    matrix, info, reg, vle = _day30_data()
    figs = [
        plots.plot_day30_funnel(info, reg, 30, out_dir=tmp_path),
        plots.plot_withdrawal_timing(info, reg, 30, out_dir=tmp_path),
        plots.plot_at_risk_by_module(matrix, out_dir=tmp_path),
        plots.plot_at_risk_by_active_days(matrix, out_dir=tmp_path),
        plots.plot_at_risk_by_inactive_streak(matrix, out_dir=tmp_path),
        plots.plot_at_risk_by_click_quartile(matrix, out_dir=tmp_path),
        plots.plot_score_coverage_by_module(matrix, out_dir=tmp_path),
        plots.plot_leakage_check(matrix, vle, 30, out_dir=tmp_path),
    ]
    assert len(list(tmp_path.glob("*.png"))) == len(figs)
    assert all(any(ch.isdigit() for ch in plots.finding(f)) for f in figs)
    stayed = ~(reg["date_unregistration"] <= 30)
    assert f"{stayed.sum():,} of {len(reg):,}" in plots.finding(figs[0])
    no_activity = matrix.loc[matrix["active_days"] == 0, "at_risk"].mean()
    assert f"{no_activity:.0%} at risk" in plots.finding(figs[3])
    assert "0% in GGG" in plots.finding(figs[6])


def test_model_charts(tmp_path):
    from sar.models.baseline import train_baseline

    matrix, *_ = _day30_data()
    features = ["code_module", "active_days", "total_clicks", "region"]
    train, val = matrix[matrix["code_presentation"] == "2013J"], matrix[matrix["code_presentation"] == "2014J"]
    model = train_baseline(train[features], train["at_risk"])
    proba = model.predict_proba(val[features])[:, 1]
    figs = [
        plots.plot_roc_pr_curves(val["at_risk"], proba, "2014J", out_dir=tmp_path),
        plots.plot_confusion_matrix(val["at_risk"], proba, out_dir=tmp_path),
        plots.plot_auc_by_module(val["code_module"], val["at_risk"], proba, out_dir=tmp_path),
        plots.plot_top_coefficients(model, n=4, out_dir=tmp_path),
    ]
    assert len(list(tmp_path.glob("*.png"))) == len(figs)
    caught = ((proba >= 0.5) & (val["at_risk"] == 1)).sum()
    assert f"({caught:,} of {val['at_risk'].sum():,})" in plots.finding(figs[1])
    assert "2014J" in plots.finding(figs[0])


def test_feature_label():
    assert plots.feature_label("num__active_days") == "Active days"
    assert plots.feature_label("cat__code_module_FFF") == "Module FFF"
    assert plots.feature_label("cat__imd_band_0-10%") == "Deprivation band: 0-10%"
