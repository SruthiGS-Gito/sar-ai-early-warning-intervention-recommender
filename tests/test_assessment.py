import pandas as pd

from sar.features.assessment import assessment_features


def _data():
    a = pd.DataFrame({
        "id_assessment": [1, 2, 3, 4],
        "assessment_type": ["TMA", "TMA", "TMA", "Exam"],
        "date": [10, 25, 90, None],
        "weight": [10, 20, 30, 100],
    })
    a["code_module"], a["code_presentation"] = "AAA", "2013J"
    sa = pd.DataFrame({
        "id_student": [1, 1, 1, 1, 2],
        "id_assessment": [1, 2, 3, 4, 1],
        "date_submitted": [8, 29, 31, 100, 10],   # 3 and 4 are after day 30
        "is_banked": [0, 0, 0, 0, 0],
        "score": [60.0, 80.0, 99.0, 99.0, 50.0],
    })
    return sa, a


def _students():
    df = pd.DataFrame({"id_student": [1, 2, 3]})
    df["code_module"], df["code_presentation"] = "AAA", "2013J"
    return df


def test_only_known_by_cutoff():
    sa, a = _data()
    r = assessment_features(sa, a, 30, _students()).loc[("AAA", "2013J", 1)]
    assert r["n_submitted"] == 2
    assert r["mean_score"] == 70.0
    assert r["score_trend"] == 20.0
    assert r["weighted_score"] == (60 * 10 + 80 * 20) / 30
    assert r["mean_delay_days"] == (-2 + 4) / 2
    assert r["n_late"] == 1
    assert r["n_assessments_due"] == 2 and r["n_missed"] == 0


def test_missing_students_and_future_changes_nothing():
    sa, a = _data()
    res = assessment_features(sa, a, 30, _students())
    r3 = res.loc[("AAA", "2013J", 3)]
    assert r3["n_submitted"] == 0 and r3["n_missed"] == 2
    assert res.loc[("AAA", "2013J", 2), "n_missed"] == 1
    sa2 = sa.copy()
    sa2.loc[sa2["date_submitted"] > 30, "score"] = 0.0
    pd.testing.assert_frame_equal(res, assessment_features(sa2, a, 30, _students()))
