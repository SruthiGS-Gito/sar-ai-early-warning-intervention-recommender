import numpy as np
import pandas as pd
import pytest
from conftest import synthetic_matrix

from sar.config import RISK_BANDS
from sar.explain.shap_explain import explain_rows
from sar.models import final
from sar.service import data, views


@pytest.fixture
def saved_final(tmp_path, small_final):
    fitted, metrics = small_final
    final.save_final(fitted, {"seed": 42}, metrics, tmp_path / "final")
    return tmp_path


def test_missing_matrix_and_model_give_messages_not_errors(tmp_path):
    m = data.load_matrix(tmp_path)
    assert not m.ok and "run_baseline.py" in m.message
    mod = data.load_final_model(tmp_path)
    assert not mod.ok and "train_final.py" in mod.message


def test_unreadable_matrix_gives_a_message(tmp_path):
    (tmp_path / "day30_feature_matrix.csv").write_bytes(b"\x00\xff\x00")
    assert not data.load_matrix(tmp_path).ok


def test_changed_model_file_gives_a_message(saved_final):
    (saved_final / "final" / "model.joblib").write_bytes(b"not a model")
    loaded = data.load_final_model(saved_final)
    assert not loaded.ok and "could not be loaded" in loaded.message


def test_final_model_loads_with_config_and_metrics(saved_final, small_final):
    loaded = data.load_final_model(saved_final)
    assert loaded.ok
    assert loaded.value.fitted["features"] == small_final[0]["features"]
    assert loaded.value.config["cutoffs"] == small_final[0]["cutoffs"]
    assert loaded.value.metrics == small_final[1]


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


def test_table_status_states(tmp_path):
    spec = data.OPTIONAL_TABLES["retention_funnel"]
    st = data.table_status(spec, tmp_path)
    assert st.state == "missing" and st.updated is None and st.path == tmp_path / spec.file
    assert "python scripts/build_team_tables.py" in st.message
    (tmp_path / spec.file).write_bytes(b"\x00\xff")
    assert data.table_status(spec, tmp_path).state in {"unreadable", "missing_columns"}
    pd.DataFrame({"stage": ["a"]}).to_csv(tmp_path / spec.file, index=False)
    st = data.table_status(spec, tmp_path)
    assert st.state == "missing_columns" and st.missing_columns == tuple(c for c in spec.columns if c != "stage")
    assert st.updated and "build_team_tables.py" not in st.message
    valid = pd.DataFrame({c: [1, 2, 3] for c in spec.columns})
    valid.assign(id_student=9).to_csv(tmp_path / spec.file, index=False)
    assert data.table_status(spec, tmp_path).state == "refused"
    valid.to_csv(tmp_path / spec.file, index=False)
    st = data.table_status(spec, tmp_path)
    assert st.ok and len(st.table) == 3 and "3 rows, last updated" in st.message
    assert all(s.description for s in data.OPTIONAL_TABLES.values())


def test_resolve_paths_follow_environment(tmp_path, monkeypatch):
    monkeypatch.setenv("SAR_DATA_DIR", str(tmp_path / "d"))
    monkeypatch.setenv("SAR_MODELS_DIR", str(tmp_path / "m"))
    p = data.resolve_paths()
    assert p["processed"] == tmp_path / "d" / "processed" and p["models"] == tmp_path / "m"


def test_scoring_gives_calibrated_score_band_and_rank(matrix, small_final):
    fitted, _ = small_final
    scored = views.score_presentation(fitted, matrix, "2014J")
    n = int((matrix["code_presentation"] == "2014J").sum())
    assert len(scored) == n and list(scored.columns) == views.KEY + ["score", "raw_score", "band", "rank", "at_risk"]
    raw, cal = final.predict(fitted, matrix[matrix["code_presentation"] == "2014J"])
    np.testing.assert_allclose(scored["score"], cal, atol=1e-12)   # parallel tree sums differ in the last digits
    assert sorted(scored["rank"]) == list(range(1, n + 1))
    by_rank = scored.sort_values("rank")
    assert by_rank["raw_score"].is_monotonic_decreasing and by_rank["score"].is_monotonic_decreasing
    assert set(scored["band"]) <= set(views.BAND_NAMES)


def test_band_edges_are_ordered_and_bands_follow_them(small_final):
    cutoffs = small_final[0]["cutoffs"]
    edges = views.risk_band_edges(cutoffs)
    assert [b for b, _ in edges] == [b for b, _ in RISK_BANDS] == ["Low", "Medium", "High"]
    lows = [e for _, e in edges]
    assert lows[0] == 0 and all(a < b for a, b in zip(lows, lows[1:]))
    assert lows[1] == cutoffs["wide_net"]["score"] and lows[2] == cutoffs["top_20"]["score"]
    probe = [0.0, lows[1] - 1e-9, lows[1], lows[2] - 1e-9, lows[2], 1.0]
    assert list(views.assign_risk_band(probe, cutoffs)) == ["Low", "Low", "Medium", "Medium", "High", "High"]
    bad = {**cutoffs, "top_20": {**cutoffs["top_20"], "score": cutoffs["wide_net"]["score"]}}
    with pytest.raises(ValueError):
        views.risk_band_edges(bad)


def test_risk_order_breaks_calibrated_ties_by_raw_score_then_row_order():
    cal = [0.5, 0.9, 0.5, 0.5, 0.9]
    raw = [0.2, 0.7, 0.4, 0.2, 0.8]
    assert list(views.risk_order(cal, raw)) == [4, 1, 2, 0, 3]


def test_capacity_ranking_flags_the_right_number_of_students(matrix, small_final):
    fitted, _ = small_final
    scored = views.score_presentation(fitted, matrix, "2014J")
    for share in [0.01, 0.05, 0.13, 0.2, 0.5]:
        f = views.flag_capacity(scored, share)
        k = int(round(share * len(scored)))
        assert f.flagged.sum() == k
        assert set(scored.loc[f.flagged, "rank"]) == set(range(1, k + 1))       # exactly the top k
    for bad in [-0.1, 1.5]:
        with pytest.raises(ValueError):
            views.flag_capacity(scored, bad)


def test_capacity_outcomes_match_the_capacity_table(matrix, small_final):
    """Computed from the cohort's outcomes, the figures equal those final.evaluate stored for each share."""
    fitted, metrics = small_final
    scored = views.score_presentation(fitted, matrix, "2014J")
    for key, stored in metrics["final"]["validation"]["capacity"].items():
        o = views.flag_outcome(scored, views.flag_capacity(scored, int(key.rstrip("%")) / 100).flagged)
        assert o.flagged == stored["flagged"]
        assert o.precision == pytest.approx(stored["precision"]) and o.recall == pytest.approx(stored["recall"])


def test_flag_outcome_counts_caught_and_missed():
    scored = pd.DataFrame({"at_risk": [1, 1, 0, 1, 0]})
    o = views.flag_outcome(scored, [True, False, True, True, False])
    assert (o.students, o.flagged, o.at_risk, o.caught, o.missed) == (5, 3, 3, 2, 1)
    assert o.precision == pytest.approx(2 / 3) and o.recall == pytest.approx(2 / 3)
    none = views.flag_outcome(scored, [False] * 5)
    assert none.caught == 0 and np.isnan(none.precision) and none.recall == 0


def test_wide_net_uses_the_config_cutoff(matrix, small_final):
    fitted, _ = small_final
    scored = views.score_presentation(fitted, matrix, "2014J")
    f = views.flag_wide_net(scored, fitted["cutoffs"])
    assert f.flagged.equals(scored["score"] >= fitted["cutoffs"]["wide_net"]["score"])


def test_flagged_by_band_adds_up_to_the_flagged_count(matrix, small_final):
    fitted, _ = small_final
    scored = views.score_presentation(fitted, matrix, "2014J")
    scored = scored.assign(flagged=views.flag_capacity(scored, 0.3).flagged)
    counts = views.flagged_by_band(scored)
    assert list(counts.index) == ["High", "Medium", "Low"] and counts.sum() == scored["flagged"].sum()


def test_band_evidence_rates_and_lifts(matrix, small_final):
    fitted, _ = small_final
    scored = views.score_presentation(fitted, matrix, "2014J")
    ev = views.band_evidence(scored, min_cell=5)
    overall = ev[ev["code_module"] == "All modules"].set_index("band")
    assert overall["students"].sum() == len(scored)
    base = scored["at_risk"].mean()
    for band, row in overall.iterrows():
        cell = scored.loc[scored["band"] == band, "at_risk"]
        if len(cell) >= 5:
            assert row["at_risk_rate"] == pytest.approx(cell.mean()) and row["lift"] == pytest.approx(cell.mean() / base)
        else:
            assert np.isnan(row["at_risk_rate"]) and np.isnan(row["lift"])
    modules = ev[ev["code_module"] != "All modules"]
    assert modules["students"].sum() == len(scored)
    hidden = views.band_evidence(scored, min_cell=10_000)
    assert hidden["at_risk_rate"].isna().all()


def test_batch_summary_counts_flags_and_bands(matrix, small_final):
    fitted, _ = small_final
    scored = views.score_presentation(fitted, matrix, "2014J")
    scored = scored.assign(flagged=views.flag_capacity(scored, 0.2).flagged)
    s = views.batch_summary(scored)
    assert s.loc["All modules", "Students"] == len(scored)
    assert s.drop(index="All modules")["Flagged"].sum() == scored["flagged"].sum()
    assert s.loc["All modules", "Flagged share"] == pytest.approx(scored["flagged"].mean())
    assert s.loc["All modules", ["High", "Medium", "Low"]].sum() == len(scored)
    assert "Average score" not in s.columns and "Recall" not in s.columns
    o = views.batch_summary(scored, outcomes=True)
    for module, g in [*scored.groupby("code_module"), ("All modules", scored)]:
        y, f = g["at_risk"] == 1, g["flagged"]
        assert o.loc[module, "At-risk rate"] == pytest.approx(y.mean())
        assert o.loc[module, "Recall"] == pytest.approx((y & f).sum() / y.sum())
        assert o.loc[module, "Precision"] == pytest.approx((y & f).sum() / f.sum())


def test_shap_cache_roundtrip_and_stale_hash(saved_final, matrix, small_final):
    fitted, _ = small_final
    loaded = data.load_final_model(saved_final).value
    part = matrix[matrix["code_presentation"] == "2014J"]
    assert "precompute_shap.py" in data.load_shap_cache(saved_final, "2014J", loaded.config).message
    base, contrib, raw = explain_rows(fitted["model"], part[fitted["features"]])
    data.save_shap_cache(saved_final, "2014J", part[views.KEY], contrib, raw, base, loaded.config)
    got = data.load_shap_cache(saved_final, "2014J", loaded.config)
    assert got.ok
    table, b = got.value
    assert b == pytest.approx(base) and len(table) == len(part)
    np.testing.assert_allclose(table[[data.SHAP_PREFIX + f for f in fitted["features"]]].to_numpy(),
                               contrib.to_numpy(), atol=1e-12)
    other = {**loaded.config, "model_sha256": "0" * 64}
    stale = data.load_shap_cache(saved_final, "2014J", other)
    assert not stale.ok and "another model file" in stale.message


def test_risk_bands_hide_small_cells(small_final):
    big = synthetic_matrix(n=4000, seed=5)
    scored = views.score_presentation(small_final[0], big, "2014J")
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


def test_capacity_per_module_flags_the_top_share_inside_each_module(matrix, small_final):
    fitted, _ = small_final
    scored = views.score_presentation(fitted, matrix, "2014J")
    f = views.flag_capacity(scored, 0.2, per_module=True).flagged
    for _, g in scored.groupby("code_module"):
        k = int(round(0.2 * len(g)))
        assert f[g.index].sum() == k
        top = set(g.index[views.risk_order(g["score"], g["raw_score"])[:k]])
        assert set(g.index[f[g.index]]) == top


def test_band_evidence_lift_uses_each_rows_own_rate(matrix, small_final):
    fitted, _ = small_final
    scored = views.score_presentation(fitted, matrix, "2014J")
    ev = views.band_evidence(scored, min_cell=1)
    for module, g in [*scored.groupby("code_module"), ("All modules", scored)]:
        base = g["at_risk"].mean()
        rows = ev[ev["code_module"] == module].set_index("band")
        for band, row in rows.iterrows():
            cell = g.loc[g["band"] == band, "at_risk"]
            if len(cell):
                assert row["lift"] == pytest.approx(cell.mean() / base)


def test_calibration_bins_follow_the_reliability_table(matrix, small_final):
    from sar.models.final import reliability

    fitted, _ = small_final
    scored = views.score_presentation(fitted, matrix, "2014J")
    cal = views.calibration_bins(scored)
    rel = [r for r in reliability(scored["at_risk"].to_numpy(), scored["score"].to_numpy())["reliability"]
           if r["count"]]
    assert cal["students"].sum() == len(scored) and list(cal["students"]) == [r["count"] for r in rel]
    assert np.allclose(cal["observed_rate"], [r["observed_rate"] for r in rel])
    assert np.allclose(cal["mean_score"], [r["mean_predicted"] for r in rel])


def test_module_rank_orders_students_within_their_module(matrix, small_final):
    fitted, _ = small_final
    scored = views.score_presentation(fitted, matrix, "2014J")
    module = scored["code_module"].iloc[0]
    part = scored[scored["code_module"] == module]
    ranks = [views.module_rank(scored, module, sid)[0] for sid in part["id_student"]]
    assert sorted(ranks) == list(range(1, len(part) + 1))
    top = part.iloc[views.risk_order(part["score"], part["raw_score"])[0]]
    assert views.module_rank(scored, module, top["id_student"]) == (1, len(part))


def test_example_student_is_fixed_by_the_seed_and_prefers_the_module(matrix, small_final):
    fitted, _ = small_final
    scored = views.score_presentation(fitted, matrix, "2014J")
    for band in ["High", "Low"]:
        module, sid = views.example_student(scored, band, "BBB", 42)
        row = scored[(scored["code_module"] == module) & (scored["id_student"] == sid)].iloc[0]
        assert row["band"] == band
        if ((scored["band"] == band) & (scored["code_module"] == "BBB")).any():
            assert module == "BBB"
        assert views.example_student(scored, band, "BBB", 42) == (module, sid)
    assert views.example_student(scored.assign(band="Low"), "High", None, 42) is None


def test_at_risk_by_presentation_matches_observed_outcomes(matrix):
    table = views.at_risk_by_presentation(matrix)
    assert list(table["code_presentation"]) == sorted(matrix["code_presentation"].unique())
    for _, row in table.iterrows():
        part = matrix[matrix["code_presentation"] == row["code_presentation"]]
        assert row["students"] == len(part) and row["at_risk_rate"] == pytest.approx(part["at_risk"].mean())
    one = views.at_risk_by_presentation(matrix, "AAA")
    assert one["students"].sum() == (matrix["code_module"] == "AAA").sum()


def test_at_risk_by_activity_uses_the_day_buckets_and_hides_small_ones(matrix):
    table = views.at_risk_by_activity(matrix, "2014J")
    assert list(table["day_band"].astype(str)) == ["0", "1-5", "6-10", "11-15", "16-20", "21+"]
    part = matrix[matrix["code_presentation"] == "2014J"]
    assert table["students"].sum() == len(part)
    zero = part[part["active_days"] == 0]
    assert table.loc[0, "students"] == len(zero)
    small = table["students"] < views.MIN_CELL
    assert table.loc[small, "at_risk_rate"].isna().all() and table.loc[small, "suppressed"].all()
    if not small.iloc[0]:
        assert table.loc[0, "at_risk_rate"] == pytest.approx(zero["at_risk"].mean())


def test_band_outcome_uses_the_module_and_falls_back_when_small(matrix, small_final):
    scored = views.score_presentation(small_final[0], matrix, "2014J")
    ev = views.band_evidence(scored)
    for module in sorted(scored["code_module"].unique()):
        for band in ["Low", "Medium", "High"]:
            rate, n, scope = views.band_outcome(scored, module, band)
            cell = ev[(ev["code_module"] == module) & (ev["band"] == band)].iloc[0]
            if cell["students"] >= views.MIN_CELL:
                assert (scope, n) == (module, cell["students"]) and rate == pytest.approx(cell["at_risk_rate"])
            else:
                whole = ev[(ev["code_module"] == "All modules") & (ev["band"] == band)].iloc[0]
                assert scope == "All modules" and n == whole["students"]
    tiny = scored[scored["code_module"] == "AAA"].head(5)
    rate, n, scope = views.band_outcome(pd.concat([tiny, scored[scored["code_module"] != "AAA"]]), "AAA",
                                        str(tiny["band"].iloc[0]))
    assert scope == "All modules" and n >= views.MIN_CELL
