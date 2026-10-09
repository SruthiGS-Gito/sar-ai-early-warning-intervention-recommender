import pandas as pd

from sar.features.engagement import KEY, consecutive_inactive_days, engagement_metrics


def _vle():
    rows = [
        # student 1: clicks on days -5, 0, 1, 10, 28, 29 (+ one after cutoff + one exact duplicate)
        (1, -5, 3), (1, 0, 2), (1, 1, 4), (1, 10, 5), (1, 28, 1), (1, 29, 6),
        (1, 29, 6),   # exact repeat -> kept, it is a separate record
        (1, 31, 99),  # after cutoff -> must be ignored
        # student 2: only one click, day 5
        (2, 5, 7),
    ]
    df = pd.DataFrame(rows, columns=["id_student", "date", "sum_click"])
    df["code_module"], df["code_presentation"], df["id_site"] = "AAA", "2013J", 1
    return df


def _students():
    df = pd.DataFrame({"id_student": [1, 2, 3]})
    df["code_module"], df["code_presentation"] = "AAA", "2013J"
    return df


def test_totals_ignore_future_and_duplicates():
    res = engagement_metrics(_vle(), 30, _students())
    s1 = res.loc[("AAA", "2013J", 1)]
    assert s1["total_clicks"] == 3 + 2 + 4 + 5 + 1 + 6 + 6  # no 99; the repeated 6 counts
    assert s1["active_days"] == 6
    assert s1["clicks_last_7d"] == 1 + 6 + 6   # days 24..30
    assert s1["click_trend"] == (1 + 6 + 6) - 0


def test_student_without_clicks_gets_zeros():
    res = engagement_metrics(_vle(), 30, _students())
    s3 = res.loc[("AAA", "2013J", 3)]
    assert s3["total_clicks"] == 0 and s3["active_days"] == 0
    assert s3["first_active_day"] == 31


def test_drop_duplicates_option():
    res = engagement_metrics(_vle(), 30, _students(), drop_duplicates=True)
    assert res.loc[("AAA", "2013J", 1), "total_clicks"] == 3 + 2 + 4 + 5 + 1 + 6


def test_future_clicks_change_nothing():
    base = engagement_metrics(_vle(), 30, _students())
    extra = pd.concat([_vle(), _vle().assign(date=40, sum_click=1000)])
    pd.testing.assert_frame_equal(base, engagement_metrics(extra, 30, _students()))


def test_inactive_streaks():
    res = consecutive_inactive_days(_vle(), 30, _students())
    s1 = res.loc[("AAA", "2013J", 1)]
    # active window days: 0,1,10,28,29 -> longest gap is days 11..27 = 17
    assert s1["max_consecutive_inactive_days"] == 17
    assert s1["trailing_inactive_days"] == 1  # day 30 silent
    s2 = res.loc[("AAA", "2013J", 2)]
    assert s2["trailing_inactive_days"] == 25
    assert s2["max_consecutive_inactive_days"] == 25
    s3 = res.loc[("AAA", "2013J", 3)]
    assert s3["max_consecutive_inactive_days"] == 31
    assert list(res.index.names) == KEY
