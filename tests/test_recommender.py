import pandas as pd
import pytest
from conftest import FEATURES

from sar.config import PROTECTED_FEATURES
from sar.explain import factors
from sar.explain.shap_explain import Explanation, explain_student, make_explainer
from sar.recommend.interventions import ACTIONABLE_FEATURES
from sar.recommend.neighbours import NeighbourPool
from sar.recommend.recommender import driver_scores, recommend

BANNED = ["because of", "caused", "benefit", "led to", "due to", "helped", "improve", "effective", "works"]


@pytest.fixture(scope="module")
def setup(matrix, small_final):
    model = small_final[0]["model"]
    train = matrix[matrix["code_presentation"] != "2014J"]
    val = matrix[matrix["code_presentation"] == "2014J"]
    return model, make_explainer(model), NeighbourPool(train, k=30), val


def _fake(contribs: dict, band=None) -> Explanation:
    table = pd.DataFrame({
        "value": 1.0,
        "contribution": [contribs.get(f, 0.0) for f in FEATURES],
        "group": [factors.group_of(f) for f in FEATURES],
        "phrase": [f"phrase for {f}" for f in FEATURES],
    }, index=pd.Index(FEATURES, name="feature"))
    return Explanation(0.7, 0.5, band, table)


def test_background_and_course_never_drive_a_suggestion():
    expl = _fake({f: 0.2 for f in factors.BACKGROUND + factors.COURSE})
    assert driver_scores(expl).empty
    expl2 = _fake({"imd_band": 0.5, "max_consecutive_inactive_days": 0.04})
    assert set(driver_scores(expl2)["feature"]) == {"max_consecutive_inactive_days"}


def test_scores_follow_the_mapping():
    expl = _fake({"mean_score": 0.10, "studied_credits": 0.02})
    s = driver_scores(expl).groupby("intervention")["score"].sum()
    assert s["tutoring"] == pytest.approx(0.06) and s["peer_tutoring"] == pytest.approx(0.04)
    assert s["financial_aid_referral"] == pytest.approx(0.01) and s["mentoring"] == pytest.approx(0.01)


def test_inputs_that_lower_the_score_are_ignored():
    assert driver_scores(_fake({"active_days": -0.3, "n_missed": 0.001})).empty


def test_recommendation_for_real_explanations(setup):
    model, explainer, pool, val = setup
    for i in range(15):
        row = val.iloc[[i]]
        expl = explain_student(model, row[list(model.feature_names_in_)], explainer, band="High")
        rec = recommend(expl, row, pool)
        assert len(rec.suggestions) <= 3
        assert [s.score for s in rec.suggestions] == sorted([s.score for s in rec.suggestions], reverse=True)
        assert len({s.intervention for s in rec.suggestions}) == len(rec.suggestions)
        if len(rec.suggestions) >= 2:      # the second suggestion leads with a new driver when it has one
            first, second = rec.suggestions[0], rec.suggestions[1]
            scores = driver_scores(expl)
            options = set(scores.loc[scores["intervention"] == second.intervention, "feature"])
            if options - {factors.family_of(f) for f in first.drivers} - set(first.drivers):
                assert second.drivers[0] not in first.drivers
        text = rec.text()
        assert "30 most similar students" in text and "OULAD records no interventions" in text
        assert not any(b in text.lower() for b in BANNED)
        assert not set(expl.contributions.index) & set(PROTECTED_FEATURES)
        for s in rec.suggestions:
            assert all(d in ACTIONABLE_FEATURES for d in s.drivers)
            assert expl.contributions.loc[s.drivers[0], "phrase"] in s.reason
            assert "n = " in s.evidence or "Too few" in s.evidence
            assert "median" in s.evidence.lower() or "Too few" in s.evidence


def test_no_suggestion_when_nothing_raises_the_score(setup):
    _, _, pool, val = setup
    rec = recommend(_fake({"active_days": -0.2}), val.iloc[[0]], pool)
    assert rec.suggestions == [] and "no support type is matched" in rec.text()


def test_low_band_is_marked_optional(setup):
    _, _, pool, val = setup
    rec = recommend(_fake({"n_missed": 0.1}, band="Low"), val.iloc[[0]], pool)
    lines = rec.text().splitlines()
    assert rec.optional and lines[0].startswith("The 30 most similar")
    assert lines[1] == "Low risk band, so these suggestions are optional."
    high = recommend(_fake({"n_missed": 0.1}, band="High"), val.iloc[[0]], pool)
    assert not high.optional and "optional" not in high.text()


def test_previous_attempts_never_drive_a_suggestion():
    assert driver_scores(_fake({"num_of_prev_attempts": 0.3})).empty
