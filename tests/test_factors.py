import pytest
from conftest import FEATURES

from sar.explain import factors


def test_every_model_input_has_a_group_and_a_phrase(matrix):
    for f in FEATURES:
        assert factors.group_of(f) in {"behaviour", "study", "course", "background"}
        assert factors.phrase(f, matrix[f].iloc[0])
        assert factors.phrase(f, float("nan"))            # missing values get wording too
    assert set(factors.BACKGROUND) == {"gender", "region", "highest_education", "imd_band",
                                       "age_band", "disability"}


def test_unknown_feature_is_an_error_not_a_silent_gap():
    with pytest.raises(KeyError):
        factors.phrase("new_feature", 1)
    with pytest.raises(KeyError):
        factors.group_of("new_feature")


@pytest.mark.parametrize("feature,value,expected", [
    ("max_consecutive_inactive_days", 1.0, "a longest run of 1 day without activity"),
    ("max_consecutive_inactive_days", 12.0, "a longest run of 12 days without activity"),
    ("n_missed", 2, "2 assessments missed"),
    ("n_missed", 0, "no assessments missed"),
    ("max_consecutive_inactive_days", 0.0, "activity on every day up to Day 30"),
    ("avg_clicks_per_active_day", 0.0, "no clicks on any active day"),
    ("mean_score", float("nan"), "no assessment score yet"),
    ("first_active_day", 31.0, "no activity on the course site by Day 30"),
    ("first_active_day", -3.0, "first active 3 days before the module started"),
    ("mean_delay_days", -1.0, "submitting 1 day early on average"),
    ("click_trend", -40, "weekly clicks down by 40 clicks in the last week"),
    ("num_of_prev_attempts", 0, "a first attempt at this module"),
    ("code_module", "FFF", "being on module FFF"),
])
def test_phrases(feature, value, expected):
    assert factors.phrase(feature, value) == expected
