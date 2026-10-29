"""Displayed numbers are rounded half up through one helper (sar.rounding), not half to even."""

import pytest

from sar.explain.chart import _fmt
from sar.explain.factors import phrase
from sar.rounding import fixed, round_half_up


@pytest.mark.parametrize("value, digits, expected", [
    (0.5, 0, 1), (1.5, 0, 2), (2.5, 0, 3), (-2.5, 0, -3), (2.4999, 0, 2),
    (10.25, 1, 10.3), (0.0065, 3, 0.007), (-0.0065, 3, -0.007), (0.1234, 3, 0.123),
])
def test_round_half_up(value, digits, expected):
    assert round_half_up(value, digits) == expected
    assert isinstance(round_half_up(value, digits), int if digits == 0 else float)


def test_fixed_writes_the_rounded_value_with_sign_and_grouping():
    assert fixed(2.5) == "3" and fixed(1234.5, grouping=True) == "1,235"
    assert fixed(0.0065, 3, sign=True) == "+0.007" and fixed(-0.0125, 3, sign=True) == "-0.013"
    assert fixed(10.25, 1) == "10.3" and fixed(0.125, 2) == "0.13"


def test_phrases_round_ties_up():
    assert phrase("mean_score", 64.5) == "an average score of 65"
    assert phrase("weighted_score", 72.5) == "a weighted average score of 73"
    assert phrase("n_missed", 0.5) == "1 assessment missed"
    assert phrase("trailing_inactive_days", 2.5) == "3 days without activity just before Day 30"
    assert phrase("avg_clicks_per_active_day", 0.5) == "about 1 click per active day"
    assert phrase("mean_delay_days", 0.5) == "submitting 1 day late on average"
    assert phrase("first_active_day", 4.5) == "first active on day 5"


def test_table_values_round_ties_up():
    assert _fmt(10.25) == "10.3" and _fmt(2.5) == "2.5" and _fmt(150.5) == "151" and _fmt(1234.0) == "1,234"
