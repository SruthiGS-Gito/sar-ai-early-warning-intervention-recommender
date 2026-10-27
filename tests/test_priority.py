"""The priority list's batch reasons and suggestions agree with the single-student explanation and
recommender, on synthetic data."""

import pandas as pd
import pytest

from sar.config import PROTECTED_FEATURES
from sar.explain.shap_explain import MIN_CONTRIBUTION, explain_rows, explanation_from_contributions
from sar.recommend.interventions import ACTIONABLE_FEATURES
from sar.recommend.neighbours import NeighbourPool
from sar.recommend.recommender import recommend
from sar.service import priority


@pytest.fixture(scope="module")
def explained(matrix, small_final):
    model = small_final[0]["model"]
    features = list(model.feature_names_in_)
    part = matrix[matrix["code_presentation"] == "2014J"].head(60)
    base, contrib, raw = explain_rows(model, part[features])
    return part, features, base, contrib, raw


def test_batch_details_match_the_single_student_explanation_and_recommendation(matrix, explained):
    part, features, base, contrib, raw = explained
    details = priority.priority_details(part[features], contrib)
    pool = NeighbourPool(matrix[matrix["code_presentation"] != "2014J"], k=30)
    assert list(details.index) == list(part.index)
    for i in part.index:
        expl = explanation_from_contributions(part.loc[i, features], contrib.loc[i], base, raw.loc[i], "High")
        c = expl.contributions
        raising = c[c.index.isin(ACTIONABLE_FEATURES) & (c["contribution"] >= MIN_CONTRIBUTION)]
        if len(raising):
            top = raising["contribution"].idxmax()
            assert details.loc[i, "main_factor"] == top
            assert details.loc[i, "main_reason"] == c.loc[top, "phrase"]
        else:
            assert details.loc[i, "main_reason"] == priority.NO_REASON
        rec = recommend(expl, part.loc[[i]], pool)
        expected = rec.suggestions[0].label if rec.suggestions else priority.NO_SUPPORT
        assert details.loc[i, "suggested_support"] == expected


def test_main_reason_never_uses_a_protected_or_non_actionable_input(explained):
    part, features, _, contrib, _ = explained
    factors = set(priority.priority_details(part[features], contrib)["main_factor"].dropna())
    assert factors <= set(ACTIONABLE_FEATURES) and not factors & set(PROTECTED_FEATURES)


def test_no_raising_input_gives_no_reason_and_no_support(explained):
    part, features, _, contrib, _ = explained
    details = priority.priority_details(part[features], contrib * 0 - 1.0)
    assert set(details["main_reason"]) == {priority.NO_REASON}
    assert set(details["suggested_support"]) == {priority.NO_SUPPORT}


def test_support_sentence_names_the_fact_then_the_support():
    from sar.recommend.interventions import INTERVENTIONS

    assert set(priority.SUPPORT_ACTIONS) == set(INTERVENTIONS)
    assert (priority.support_sentence("weekly_checkins", "1 assessment missed")
            == "1 assessment missed; a short weekly check-in can help them keep up with activity and deadlines.")
    assert priority.support_sentence("tutoring", "an average score of 41").startswith("An average score of 41; ")
    assert priority.similar_line(50, 1) == "Among 50 similar earlier students, 1 ended at risk."


def test_selected_row_reads_a_single_row_selection():
    rows = [("AAA", 1), ("BBB", 2)]
    assert priority.selected_row(rows, {"selection": {"rows": [1], "columns": []}}) == ("BBB", 2)
    assert priority.selected_row(rows, {"selection": {"rows": []}}) is None
    assert priority.selected_row(rows, None) is None
    assert priority.selected_row(rows, {"selection": {"rows": [5]}}) is None


def test_snapshot_compares_with_the_module_median(matrix):
    sid = int(matrix.loc[matrix["code_presentation"] == "2014J", "id_student"].iloc[0])
    row = matrix[(matrix["code_presentation"] == "2014J") & (matrix["id_student"] == sid)].iloc[0]
    snap = priority.snapshot(matrix, "2014J", row["code_module"], sid, {"active_days": "Active days"})
    part = matrix[(matrix["code_presentation"] == "2014J") & (matrix["code_module"] == row["code_module"])]
    assert snap.loc["active_days", "value"] == row["active_days"]
    assert snap.loc["active_days", "median"] == pytest.approx(part["active_days"].median())
    assert snap.loc["active_days", "difference"] == pytest.approx(row["active_days"] - part["active_days"].median())
    assert isinstance(snap, pd.DataFrame)
