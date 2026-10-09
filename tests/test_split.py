import pandas as pd
import pytest

from sar.data.split import make_split, overlap_report


def _df():
    return pd.DataFrame({
        "id_student": [1, 2, 3, 1, 4],
        "code_presentation": ["2013J", "2013J", "2014B", "2014J", "2014J"],
        "at_risk": [0, 1, 0, 1, 0],
    })


def test_temporal_split_removes_overlap():
    train, val = make_split(_df())
    assert set(val["code_presentation"]) == {"2014J"}
    assert 1 not in set(train["id_student"])          # student 1 is in validation
    assert set(train["id_student"]) == {2, 3}
    assert set(train["id_student"]).isdisjoint(set(val["id_student"]))


def test_keep_overlap_option_and_report():
    train, _ = make_split(_df(), drop_overlap=False)
    assert 1 in set(train["id_student"])
    rep = overlap_report(_df())
    assert rep["students in both"] == 1 and rep["validation rows"] == 2


def test_unknown_presentation():
    with pytest.raises(ValueError):
        make_split(_df(), test_presentation="1999X")
