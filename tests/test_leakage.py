"""SCRUM-171: no feature may use data dated after the cutoff day.

Each builder is run twice, once on the data and once on the same data with every
post-cutoff row altered. The outputs must be identical.
"""

import pandas as pd
import pytest

from sar.config import CUTOFF_DAY
from sar.features.assessment import assessment_features
from sar.features.engagement import consecutive_inactive_days, engagement_metrics

FORBIDDEN = {"final_result", "date_unregistration", "at_risk"}


def _vle():
    rows = [(1, d, c) for d, c in [(-3, 2), (0, 1), (5, 4), (20, 3), (29, 2), (35, 9), (120, 50)]]
    rows += [(2, d, c) for d, c in [(2, 5), (31, 7), (200, 1)]]
    df = pd.DataFrame(rows, columns=["id_student", "date", "sum_click"])
    df["code_module"], df["code_presentation"], df["id_site"] = "AAA", "2013J", 1
    return df


def _alter_future(df, col):
    out = df.copy()
    out.loc[out[col] > CUTOFF_DAY, "sum_click" if "sum_click" in out else "score"] += 1000
    return out


def _students():
    df = pd.DataFrame({"id_student": [1, 2, 3]})
    df["code_module"], df["code_presentation"] = "AAA", "2013J"
    return df


@pytest.mark.parametrize("builder", [engagement_metrics, consecutive_inactive_days])
def test_engagement_ignores_future_rows(builder):
    base = builder(_vle(), CUTOFF_DAY, _students())
    changed = builder(_alter_future(_vle(), "date"), CUTOFF_DAY, _students())
    pd.testing.assert_frame_equal(base, changed)
    assert not FORBIDDEN & set(base.columns)


def test_assessment_ignores_future_rows():
    a = pd.DataFrame({"id_assessment": [1, 2], "assessment_type": ["TMA", "TMA"],
                      "date": [10, 90], "weight": [10, 30]})
    a["code_module"], a["code_presentation"] = "AAA", "2013J"
    sa = pd.DataFrame({"id_student": [1, 1], "id_assessment": [1, 2],
                       "date_submitted": [9, 95], "is_banked": [0, 0], "score": [60.0, 80.0]})
    base = assessment_features(sa, a, CUTOFF_DAY, _students())
    sa2 = sa.copy()
    sa2.loc[sa2["date_submitted"] > CUTOFF_DAY, "score"] = 0.0
    pd.testing.assert_frame_equal(base, assessment_features(sa2, a, CUTOFF_DAY, _students()))
    assert not FORBIDDEN & set(base.columns)
