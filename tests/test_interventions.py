import pytest

from sar.config import PROTECTED_FEATURES
from sar.explain import factors
from sar.recommend import interventions


def test_every_actionable_input_is_mapped_and_nothing_else():
    # Previous attempts is a study input in explanations but is fixed history, so it is not actionable.
    mapped = set(interventions.FEATURE_TO_INTERVENTIONS)
    assert mapped == set(interventions.ACTIONABLE_FEATURES)
    assert set(interventions.ACTIONABLE) == {"engagement", "submissions", "timing", "workload"}
    assert all(factors.group_of(f) in factors.REASON_GROUPS for f in mapped)
    assert "num_of_prev_attempts" not in mapped
    assert mapped.isdisjoint(PROTECTED_FEATURES)
    assert mapped.isdisjoint(factors.COURSE)


def test_weights_are_positive_and_sum_to_one():
    for weights in interventions.FEATURE_TO_INTERVENTIONS.values():
        assert all(w > 0 for w in weights.values())
        assert sum(weights.values()) == pytest.approx(1)
        assert set(weights) <= set(interventions.INTERVENTIONS)


def test_all_five_support_types_are_reachable():
    reachable = {name for w in interventions.FEATURE_TO_INTERVENTIONS.values() for name in w}
    assert reachable == {"tutoring", "peer_tutoring", "mentoring", "weekly_checkins", "financial_aid_referral"}


@pytest.mark.parametrize("feature", ["imd_band", "num_of_prev_attempts", "code_module"])
def test_mapping_a_non_actionable_input_is_refused(monkeypatch, feature):
    monkeypatch.setitem(interventions.FEATURE_TO_INTERVENTIONS, feature, {"financial_aid_referral": 1.0})
    with pytest.raises(ValueError):
        interventions._check_mapping()
