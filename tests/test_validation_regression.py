"""Regression checks on the real 2014J validation presentation, recomputed independently of the app.

They need the git-ignored Day 30 feature matrix, the final model and the precomputed 2014J explanations, and
are skipped when any of them is absent (as in test_final.py). Checks run over every 2014J student where they can
be vectorised; per-student service calls run on stratified samples.
"""

import json
import re
from decimal import ROUND_HALF_UP, Decimal

import numpy as np
import pandas as pd
import pytest

from sar.config import MODELS_DIR, PROCESSED_DIR, PROJECT_ROOT, PROTECTED_FEATURES
from sar.data.split import TEST_PRESENTATION, make_split
from sar.explain.shap_explain import ADDITIVITY_TOLERANCE, explanation_from_contributions
from sar.recommend.neighbours import K, NeighbourPool, _prepare
from sar.recommend.recommender import recommend
from sar.service import priority, views
from sar.service.data import SHAP_PREFIX, load_final_model, load_shap_cache, shap_cache_path

MATRIX = PROCESSED_DIR / "day30_feature_matrix.csv"
NEEDED = [MATRIX, MODELS_DIR / "final" / "model.joblib", MODELS_DIR / "final" / "calibrator.joblib",
          shap_cache_path(MODELS_DIR, TEST_PRESENTATION), shap_cache_path(MODELS_DIR, TEST_PRESENTATION).with_suffix(".json")]
pytestmark = pytest.mark.skipif(not all(p.exists() for p in NEEDED),
                                reason="needs the Day 30 matrix, the final model and the 2014J SHAP cache")
APP = str(PROJECT_ROOT / "app" / "streamlit_app.py")
EXPORT_COLUMNS = ["Rank", "Student ID", "Module", "Risk band", "Calibrated score", "Day 30 snapshot", "Main reason",
                  "Suggested support", "Reason for support", "score_source"]


def half_up(v: float, digits: int) -> Decimal:
    """Independent of sar.rounding: the shortest decimal form, rounded half up."""
    return Decimal(repr(float(v))).quantize(Decimal(1).scaleb(-digits), rounding=ROUND_HALF_UP)


@pytest.fixture(scope="module")
def real():
    matrix = pd.read_csv(MATRIX)
    final = load_final_model(MODELS_DIR).value
    fitted = final.fitted
    scored = views.score_presentation(fitted, matrix, TEST_PRESENTATION)
    cache, base = load_shap_cache(MODELS_DIR, TEST_PRESENTATION, final.config).value
    val = matrix[matrix["code_presentation"] == TEST_PRESENTATION].reset_index(drop=True)
    cfg = json.loads((MODELS_DIR / "final" / "config.json").read_text())
    return {"matrix": matrix, "fitted": fitted, "config": final.config, "scored": scored, "cache": cache,
            "base": base, "val": val, "features": fitted["features"], "cutoffs": cfg["cutoffs"]}


def contributions_for(real, keys: pd.DataFrame) -> pd.DataFrame:
    """SHAP contributions from the cache for the (code_module, id_student) rows of `keys`, in order."""
    cols = [SHAP_PREFIX + f for f in real["features"]]
    got = keys[["code_module", "id_student"]].merge(real["cache"], on=["code_module", "id_student"], how="left")
    return got[cols].set_axis(real["features"], axis=1).set_axis(keys.index)


def test_every_validation_student_has_a_score_from_0_to_1_and_the_band_its_cutoffs_give(real):
    scored, cut = real["scored"], real["cutoffs"]
    assert len(scored) == len(real["val"]) == 9203
    s = scored["score"].to_numpy()
    assert np.isfinite(s).all() and ((s >= 0) & (s <= 1)).all()
    expected = np.where(s >= cut["top_20"]["score"], "High", np.where(s >= cut["wide_net"]["score"], "Medium", "Low"))
    assert (scored["band"].astype(str).to_numpy() == expected).all()


def test_rank_within_module_follows_the_score_then_the_raw_score(real):
    scored = real["scored"].assign(pos=np.arange(len(real["scored"])))
    expected = {}
    for module, g in scored.groupby("code_module"):
        order = np.lexsort((g["pos"].to_numpy(), -g["raw_score"].to_numpy(), -g["score"].to_numpy()))
        expected.update({(module, sid): (r + 1, len(g)) for r, sid in enumerate(g["id_student"].to_numpy()[order])})
        assert sorted(r for (m, _), (r, _) in expected.items() if m == module) == list(range(1, len(g) + 1))
    sample = pd.concat([g.sort_values("score").iloc[[0, -1]] for _, g in scored.groupby("code_module")]
                       + [scored.groupby("code_module").sample(40, random_state=0)])
    for module, sid in zip(sample["code_module"], sample["id_student"]):
        assert views.module_rank(real["scored"], module, sid) == expected[(module, sid)]


def test_shap_cache_covers_every_student_and_adds_up_to_the_raw_score(real):
    cache, scored = real["cache"], real["scored"]
    keys = set(zip(cache["code_module"], cache["id_student"]))
    assert len(cache) == len(keys) == len(scored) and keys == set(zip(scored["code_module"], scored["id_student"]))
    merged = scored[["code_module", "id_student", "raw_score"]].merge(
        cache, on=["code_module", "id_student"], suffixes=("_model", "_cache"))
    total = real["base"] + merged[[SHAP_PREFIX + f for f in real["features"]]].sum(axis=1)
    assert (total - merged["raw_score_model"]).abs().max() <= ADDITIVITY_TOLERANCE
    assert (merged["raw_score_cache"] - merged["raw_score_model"]).abs().max() <= 1e-12


def test_ring_is_the_observed_rate_of_the_band_in_the_module_or_across_modules(real):
    scored = real["scored"]
    whole = scored.groupby("band", observed=True)["at_risk"].agg(["mean", "size"])
    for (module, band), g in scored.groupby(["code_module", "band"], observed=True):
        rate, n, scope = views.band_outcome(scored, module, str(band))
        if len(g) >= views.MIN_CELL:
            assert (rate, n, scope) == (pytest.approx(g["at_risk"].mean()), len(g), module)
        else:
            assert (rate, n, scope) == (pytest.approx(whole.loc[band, "mean"]), whole.loc[band, "size"], "All modules")


def all_neighbour_counts(real, pool: NeighbourPool) -> pd.DataFrame:
    """For every 2014J student, the number of similar students and how many ended at risk: the K nearest
    training students of the same module on the scaled inputs, the student's own records left out."""
    val, data = real["val"], pool.data
    z_val = pool.scaler.transform(_prepare(val))
    out = []
    for module, idx in val.groupby("code_module").groups.items():
        cand = np.flatnonzero((data["code_module"] == module).to_numpy())
        assert len(cand) > K
        d = ((z_val[idx][:, None, :] - pool.Z[cand][None, :, :]) ** 2).sum(axis=2)
        d[val.loc[idx, "id_student"].to_numpy()[:, None] == data["id_student"].to_numpy()[cand][None, :]] = np.inf
        nearest = cand[np.argsort(d, axis=1, kind="stable")[:, :K]]
        out.append(pd.DataFrame({"row": idx, "n": K, "at_risk": data["at_risk"].to_numpy()[nearest].sum(axis=1)}))
    return pd.concat(out).set_index("row").sort_index()


def test_similar_students_say_all_n_none_or_x_of_the_n(real):
    pool = NeighbourPool(make_split(real["matrix"])[0])
    counts = all_neighbour_counts(real, pool)
    cases = {"all": counts.index[counts["at_risk"] == counts["n"]], "none": counts.index[counts["at_risk"] == 0],
             "some": counts.index[(counts["at_risk"] > 0) & (counts["at_risk"] < counts["n"])]}
    assert all(len(rows) for rows in cases.values()), {k: len(v) for k, v in cases.items()}
    for case, rows in cases.items():
        for i in rows[:: max(1, len(rows) // 15)][:15]:
            nb = pool.query(real["val"].iloc[[i]])
            assert (nb.n, nb.n_at_risk) == (counts.at[i, "n"], counts.at[i, "at_risk"])
            text = priority.similar_definition(nb)
            start = {"all": f"All {nb.n:,} ended at risk", "none": "None of them ended at risk",
                     "some": f"{nb.n_at_risk:,} of the {nb.n:,} ended at risk"}[case]
            assert f"left out. {start} (Fail or Withdrawn)." in text, text
            assert not re.search(r"\b(\d+) of them\b", text)


@pytest.mark.parametrize("share", [0.2, 0.5])
def test_download_applies_the_low_band_rule_with_capacity_per_module(real, share):
    """Flagged Low-band students get the Student page's primary suggestion in the export, or "No specific
    support suggested at Day 30." when the page shows none; the reason for support is the primary's own."""
    scored, val, features = real["scored"], real["val"], real["features"]
    flagged = scored[views.flag_capacity(scored, share, per_module=True).flagged].reset_index(drop=True)
    rows = flagged[views.KEY].merge(val, on=views.KEY, how="left")
    details = priority.priority_details(rows[features], contributions_for(real, flagged), flagged["band"])
    low = flagged.index[flagged["band"].astype(str) == "Low"]
    assert share < 0.5 or len(low)                                       # Low students are flagged at 50%
    assert (details.loc[flagged.index.difference(low), "suggested_support"] != priority.NO_LOW_SUGGESTION).all()
    pool = NeighbourPool(make_split(real["matrix"])[0])
    contrib = contributions_for(real, flagged)
    for i in low:
        expl = priority.for_reasons(explanation_from_contributions(
            rows.loc[i, features], contrib.loc[i], real["base"], flagged.at[i, "raw_score"], "Low"))
        rec = recommend(expl, rows.loc[[i]], pool)
        first = priority.primary_for(rec)
        if first is None:
            expected = (priority.NO_LOW_SUGGESTION if rec.suggestions else priority.NO_SUPPORT, "")
        else:
            expected = (first.label, priority.support_sentence(first.intervention,
                                                               expl.contributions.loc[first.drivers[0], "phrase"]))
        assert tuple(details.loc[i, ["suggested_support", "support_reason"]]) == expected


@pytest.fixture(scope="module")
def app_session(real):
    """The app on the real data, default settings: the on-screen priority list and its details."""
    AppTest = pytest.importorskip("streamlit.testing.v1").AppTest
    at = AppTest.from_file(APP, default_timeout=300).run()
    assert not at.exception
    listing = [df.value for df in at.dataframe if "Main reason" in df.value.columns][0]
    return at, listing


def test_download_matches_the_on_screen_list_and_has_the_expected_structure(real, app_session):
    _, listing = app_session
    scored, val, features = real["scored"], real["val"], real["features"]
    flagged = scored[views.flag_capacity(scored, 0.2).flagged].sort_values("rank").reset_index(drop=True)
    assert list(listing["Student ID"]) == flagged["id_student"].astype(str).tolist()
    assert list(listing["Module"]) == flagged["code_module"].tolist()
    assert list(listing["Risk band"]) == flagged["band"].astype(str).tolist()
    rows = flagged[views.KEY].merge(val, on=views.KEY, how="left")
    details = priority.priority_details(rows[features], contributions_for(real, flagged), flagged["band"])
    assert list(listing["Suggested support"]) == details["suggested_support"].tolist()
    assert list(listing["Main reason"]) == [priority.short_reason(r) for r in details["main_reason"]]
    snap = [f"{a:.0f} active day{'' if a == 1 else 's'}, {m:.0f} missed, {n:.0f} submitted"
            for a, m, n in zip(rows["active_days"], rows["n_missed"], rows["n_submitted"])]   # whole counts
    assert list(listing["Day 30 snapshot"]) == snap

    export = views.priority_export(listing, flagged["score"], "validation (2014J)", details["support_reason"])
    assert list(export.columns) == EXPORT_COLUMNS and not set(PROTECTED_FEATURES) & set(export.columns)
    assert len(export) == len(flagged) and list(export["Rank"]) == list(range(1, len(flagged) + 1))
    assert export.drop(columns=["Calibrated score", "Reason for support", "score_source"]).equals(listing)
    assert [Decimal(repr(v)) for v in export["Calibrated score"]] == [half_up(s, 3) for s in flagged["score"]]
    has_type = ~export["Suggested support"].isin(priority.NOT_A_TYPE)
    assert (export.loc[has_type, "Reason for support"].str.len() > 0).all()
    assert (export.loc[~has_type, "Reason for support"] == "").all()


def test_all_inputs_table_equals_the_models_shap_to_three_decimals(real, app_session):
    at, _ = app_session
    from sar.explain.chart import short_label

    val = real["val"]
    weightless = val[val["weighted_score"].isna() & val["mean_score"].notna()]
    picks = [weightless.iloc[0], val.loc[val["code_module"] == "CCC"].iloc[0]]
    cache = real["cache"].set_index(["code_module", "id_student"])
    at.radio(key="nav_page").set_value("Student").run()
    for row in picks:
        at.selectbox(key="lookup_module").set_value(row["code_module"]).run()
        at.selectbox(key="lookup_student").set_value(int(row["id_student"])).run()
        assert not at.exception
        table = [df.value for df in at.dataframe if "Change in raw forest score" in df.value.columns][0]
        shown = dict(zip(table["Input"], table["Change in raw forest score"]))
        assert len(shown) == len(real["features"])
        for f in real["features"]:
            value = cache.at[(row["code_module"], row["id_student"]), SHAP_PREFIX + f]
            assert Decimal(shown[short_label(f)]) == half_up(value, 3), f
    assert cache.at[(picks[0]["code_module"], picks[0]["id_student"]), SHAP_PREFIX + "weighted_score"] != 0
