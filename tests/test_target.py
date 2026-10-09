import pandas as pd
import pytest

from sar.data.target import build_target, day30_population, early_leavers


def _tables():
    info = pd.DataFrame({
        "id_student": [1, 2, 3, 4, 5],
        "final_result": ["Pass", "Fail", "Withdrawn", "Distinction", "Withdrawn"],
    })
    info["code_module"], info["code_presentation"] = "AAA", "2013J"
    reg = info[["code_module", "code_presentation", "id_student"]].copy()
    reg["date_unregistration"] = [None, None, 20.0, None, 60.0]
    return {"studentInfo": info, "studentRegistration": reg}


def test_target_values():
    y = build_target(_tables()["studentInfo"])
    assert y.tolist() == [0, 1, 1, 0, 1]
    assert y.name == "at_risk"


def test_target_rejects_unknown_label():
    info = _tables()["studentInfo"].assign(final_result="Maybe")
    with pytest.raises(ValueError):
        build_target(info)


def test_population_drops_only_early_leavers():
    t = _tables()
    assert day30_population(t, 30)["id_student"].tolist() == [1, 2, 4, 5]  # 5 left on day 60: stays
    assert early_leavers(t, 30)["id_student"].tolist() == [3]
    assert day30_population(t, 20)["id_student"].tolist() == [1, 2, 4, 5]  # day 20 counts as left
    assert day30_population(t, 19)["id_student"].tolist() == [1, 2, 3, 4, 5]
