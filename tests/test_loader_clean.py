import pandas as pd
import pytest

from sar.data.clean import (check_join_keys, clean_all, clean_student_info,
                            clean_student_vle, save_interim)
from sar.data.loader import load_all, load_table


def _write_all(d):
    k = {"code_module": ["AAA"], "code_presentation": ["2013J"], "id_student": [1]}
    frames = {
        "courses": pd.DataFrame({"code_module": ["AAA"], "code_presentation": ["2013J"], "module_presentation_length": [268]}),
        "assessments": pd.DataFrame({"code_module": ["AAA"], "code_presentation": ["2013J"], "id_assessment": [1], "assessment_type": ["TMA"], "date": [19], "weight": [10]}),
        "vle": pd.DataFrame({"id_site": [5], "code_module": ["AAA"], "code_presentation": ["2013J"], "activity_type": ["resource"], "week_from": [None], "week_to": [None]}),
        "studentInfo": pd.DataFrame({**k, "gender": ["M"], "region": ["Wales"], "highest_education": ["A Level or Equivalent"], "imd_band": ["10-20"], "age_band": ["0-35"], "num_of_prev_attempts": [0], "studied_credits": [60], "disability": ["N"], "final_result": ["Pass"]}),
        "studentRegistration": pd.DataFrame({**k, "date_registration": [-30], "date_unregistration": [None]}),
        "studentAssessment": pd.DataFrame({"id_assessment": [1], "id_student": [1], "date_submitted": [18], "is_banked": [0], "score": [78]}),
        "studentVle": pd.DataFrame({**k, "id_site": [5], "date": [3], "sum_click": [4]}),
    }
    for n, f in frames.items():
        f.to_csv(d / f"{n}.csv", index=False)
    return frames


def test_loader_reads_question_mark_as_nan(tmp_path):
    (tmp_path / "studentInfo.csv").write_text("id_student,imd_band\n1,?\n2,10-20\n")
    df = load_table("studentInfo", tmp_path)
    assert df["imd_band"].isna().tolist() == [True, False]


def test_loader_errors(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_table("studentInfo", tmp_path)
    with pytest.raises(ValueError):
        load_table("nope", tmp_path)


def test_clean_and_checks_and_save(tmp_path):
    raw = tmp_path / "raw"; raw.mkdir()
    _write_all(raw)
    tables = load_all(raw)
    rep = check_join_keys(tables)
    assert rep["studentInfo key unique"] and rep["info missing in registration"] == 0
    assert rep["assessments with no submission"] == 0
    tables["studentVle"] = pd.concat([tables["studentVle"]] * 2, ignore_index=True)
    cleaned = clean_all(tables)
    assert len(cleaned["studentVle"]) == 2          # kept by default
    assert len(clean_all(tables, drop_vle_duplicates=True)["studentVle"]) == 1
    assert cleaned["studentInfo"]["imd_band"].tolist() == ["10-20%"]
    paths = save_interim(cleaned, tmp_path / "interim")
    assert len(paths) == 7 and all(p.exists() for p in paths)


def test_imd_fill_and_vle_keep():
    info = pd.DataFrame({"imd_band": [None, " 30-40% "], "gender": ["M", "F"]})
    assert clean_student_info(info)["imd_band"].tolist() == ["Unknown", "30-40%"]
    v = pd.DataFrame({"a": [1, 1]})
    assert len(clean_student_vle(v)) == 2
    assert len(clean_student_vle(v, drop_duplicates=True)) == 1
