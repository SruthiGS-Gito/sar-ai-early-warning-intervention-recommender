import pytest

from sar.explain import factors
from sar.recommend import interventions


def test_every_reason_input_is_mapped_and_nothing_else():
    reason_inputs = {f for f, g in factors.GROUPS.items() if g in factors.REASON_GROUPS}
    assert set(interventions.FEATURE_TO_INTERVENTIONS) == reason_inputs
    mapped = set(interventions.FEATURE_TO_INTERVENTIONS)
    assert mapped.isdisjoint(factors.BACKGROUND)
    assert mapped.isdisjoint(factors.COURSE)


def test_weights_are_positive_and_sum_to_one():
    for weights in interventions.FEATURE_TO_INTERVENTIONS.values():
        assert all(w > 0 for w in weights.values())
        assert sum(weights.values()) == pytest.approx(1)
        assert set(weights) <= set(interventions.INTERVENTIONS)


def test_all_five_support_types_are_reachable():
    reachable = {name for w in interventions.FEATURE_TO_INTERVENTIONS.values() for name in w}
    assert reachable == {"tutoring", "peer_tutoring", "mentoring", "weekly_checkins", "financial_aid_referral"}


def test_mapping_a_background_input_is_refused(monkeypatch):
    monkeypatch.setitem(interventions.FEATURE_TO_INTERVENTIONS, "imd_band", {"financial_aid_referral": 1.0})
    with pytest.raises(ValueError):
        interventions._check_mapping()
