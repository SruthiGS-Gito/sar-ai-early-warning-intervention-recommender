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


def test_chart_factors_and_verdict_name_only_shown_actionable_factors(explained):
    part, features, base, contrib, raw = explained
    for i in part.index[:20]:
        expl = explanation_from_contributions(part.loc[i, features], contrib.loc[i], base, raw.loc[i], "High")
        raising, lowering = priority.chart_factors(expl)
        assert len(raising) <= 4 and len(lowering) <= 2
        assert set(raising.index) | set(lowering.index) <= set(ACTIONABLE_FEATURES)
        assert (raising["contribution"] >= MIN_CONTRIBUTION).all()
        assert (lowering["contribution"] <= -MIN_CONTRIBUTION).all()
        c = expl.contributions[expl.contributions.index.isin(ACTIONABLE_FEATURES)]
        if (c["contribution"] <= -MIN_CONTRIBUTION).any():
            assert len(lowering) and lowering.index[0] == c["contribution"].idxmin()
        text = priority.verdict_sentence(raising, lowering)
        named = [p for p in c["phrase"] if p in text]
        assert set(named) <= set(raising["phrase"]) | set(lowering["phrase"])
        assert ("It is lowered most by" in text) == bool(len(lowering))
        if not len(lowering):
            assert text.endswith("No activity, assessment or study-load input lowers it noticeably.")


def test_a_weightless_score_is_not_a_reason_or_a_trigger(explained):
    part, features, base, contrib, raw = explained
    i = part.index[0]
    values = part.loc[i, features].copy()
    values["mean_score"], values["weighted_score"] = 100.0, float("nan")
    boosted = contrib.loc[i].copy()
    boosted["weighted_score"] = 0.5                                  # the gap would dominate if it were used
    expl = priority.for_reasons(explanation_from_contributions(values, boosted, base, raw.loc[i], "Low"))
    assert expl.contributions.loc["weighted_score", "contribution"] == 0.0
    assert expl.contributions.loc["weighted_score", "phrase"] == priority.WEIGHTLESS
    raising, _ = priority.chart_factors(expl)
    assert "weighted_score" not in raising.index
    details = priority.priority_details(pd.DataFrame([values]), pd.DataFrame([boosted]))
    assert details["main_factor"].iloc[0] != "weighted_score"
    assert "weighted" not in details["suggested_support"].iloc[0]


def test_low_band_gets_a_suggestion_only_for_missed_work_or_inactivity():
    from types import SimpleNamespace

    def rec(band_low, driver):
        s = SimpleNamespace(drivers=[driver], label="Tutoring")
        return SimpleNamespace(suggestions=[s], optional=band_low)

    assert priority.primary_for(rec(True, "mean_score")) is None
    assert priority.primary_for(rec(True, "n_missed")) is not None
    assert priority.primary_for(rec(True, "trailing_inactive_days")) is not None
    assert priority.primary_for(rec(False, "mean_score")) is not None
    assert priority.primary_for(SimpleNamespace(suggestions=[], optional=False)) is None


def test_reason_counts_group_kinds_and_put_small_ones_in_other():
    factors = pd.Series(["mean_score"] * 50 + ["weighted_score"] * 10 + ["n_missed"] * 30 + ["n_late"] * 3
                        + [None] * 7)
    counts = priority.reason_counts(factors)
    assert list(counts.items()) == [("Assessment scores", 60), ("Assessments missed", 30),
                                    (priority.NO_KIND, 7), (priority.OTHER, 3)]
    assert set(priority.REASON_KINDS) <= set(ACTIONABLE_FEATURES)


def test_short_reason_keeps_the_fact_on_one_line():
    short = priority.short_reason
    assert short("12 days without activity just before Day 30") == "12 days without activity before Day 30"
    assert short("first active 3 days before the module started") == "First active 3 days before the start"
    assert priority.short_reason(priority.NO_REASON) == priority.NO_KIND


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


def test_support_counts_keep_every_type_in_order():
    from sar.recommend.interventions import INTERVENTIONS, label

    counts = priority.support_counts(pd.Series(["Tutoring", "Weekly check-ins", "Tutoring", priority.NO_SUPPORT]))
    assert list(counts.index) == [label(n) for n in INTERVENTIONS] + [priority.NO_SUPPORT]
    assert counts["Tutoring"] == 2 and counts["Weekly check-ins"] == 1 and counts["Mentoring"] == 0
    assert counts.sum() == 4


def test_never_first_names_the_mapped_inputs_and_the_sharing_types():
    from sar.recommend.interventions import FEATURE_TO_INTERVENTIONS, INTERVENTIONS, label

    for name in INTERVENTIONS:
        text = priority.never_first(name)
        own = [f for f, w in FEATURE_TO_INTERVENTIONS.items() if name in w]
        sharing = {label(o) for f in own for o in FEATURE_TO_INTERVENTIONS[f] if o != name}
        assert all(t in text for t in sharing) and label(name) not in text.split(", which")[1]
    assert "study load and registration timing" in priority.never_first("financial_aid_referral")
