import pandas as pd

from sar.features.registration import registration_features


def _reg():
    df = pd.DataFrame({"id_student": [1, 2, 3, 4], "date_registration": [-60.0, 5.0, 40.0, None],
                       "date_unregistration": [None, 12.0, None, None]})
    df["code_module"], df["code_presentation"] = "AAA", "2013J"
    return df


def test_features_and_no_outcome_columns():
    out = registration_features(_reg(), cutoff_day=30)
    assert out.loc[("AAA", "2013J", 1), "reg_lead_days"] == 60
    assert out.loc[("AAA", "2013J", 2), "registered_late"] == 1
    assert out.loc[("AAA", "2013J", 3), "registered_after_cutoff"] == 1
    assert out.loc[("AAA", "2013J", 4), "reg_date_missing"] == 1
    assert "date_unregistration" not in out.columns


def test_unregistration_does_not_change_output():
    a = registration_features(_reg())
    b = registration_features(_reg().assign(date_unregistration=999.0))
    pd.testing.assert_frame_equal(a, b)


def test_carries_engagement_columns():
    eng = pd.DataFrame({"max_consecutive_inactive_days": [3, 5], "trailing_inactive_days": [1, 2]},
                       index=pd.MultiIndex.from_tuples([("AAA", "2013J", 1), ("AAA", "2013J", 2)],
                                                       names=["code_module", "code_presentation", "id_student"]))
    out = registration_features(_reg(), eng)
    assert out.loc[("AAA", "2013J", 2), "max_consecutive_inactive_days"] == 5
