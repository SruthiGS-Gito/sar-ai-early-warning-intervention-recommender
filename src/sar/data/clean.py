"""SCRUM-161: cleaning rules and join-key integrity checks.

Rules come from the Part 1 data-understanding notebook and the team notebooks:
  * studentInfo: imd_band label "10-20" is written "10-20%"; missing imd_band
    becomes "Unknown"; text columns are stripped
  * studentVle: rows are kept as they are (drop_duplicates=False). Part 2 showed that
    several records per student, site and day exist, so exact repeats are real records
  * nothing else is changed here; features and the target are built elsewhere
"""

from pathlib import Path

import pandas as pd

from sar.config import INTERIM_DIR

KEY = ["code_module", "code_presentation", "id_student"]


def clean_student_info(df: pd.DataFrame, imd_fill: str = "Unknown") -> pd.DataFrame:
    out = df.copy()
    for col in out.select_dtypes(exclude="number").columns:
        out[col] = out[col].str.strip()
    out["imd_band"] = out["imd_band"].replace({"10-20": "10-20%"}).fillna(imd_fill)
    return out


def clean_student_vle(df: pd.DataFrame, drop_duplicates: bool = False) -> pd.DataFrame:
    return df.drop_duplicates().reset_index(drop=True) if drop_duplicates else df.copy()


def _missing_keys(a: pd.DataFrame, b: pd.DataFrame, on: list[str]) -> int:
    m = a[on].drop_duplicates().merge(b[on].drop_duplicates(), on=on, how="left", indicator=True)
    return int((m["_merge"] == "left_only").sum())


def check_join_keys(tables: dict[str, pd.DataFrame]) -> dict:
    """Report missing or duplicated keys across tables. All values should be 0 / True on OULAD."""
    info, reg = tables["studentInfo"], tables["studentRegistration"]
    return {
        "studentInfo key unique": not info.duplicated(KEY).any(),
        "studentRegistration key unique": not reg.duplicated(KEY).any(),
        "assessments id unique": bool(tables["assessments"]["id_assessment"].is_unique),
        "vle id_site unique": bool(tables["vle"]["id_site"].is_unique),
        "info missing in registration": _missing_keys(info, reg, KEY),
        "registration missing in info": _missing_keys(reg, info, KEY),
        "studentVle enrolments missing in info": _missing_keys(tables["studentVle"], info, KEY),
        "studentVle sites missing in vle": _missing_keys(tables["studentVle"], tables["vle"], ["id_site"]),
        "studentAssessment unknown id_assessment": int(
            (~tables["studentAssessment"]["id_assessment"].isin(tables["assessments"]["id_assessment"])).sum()
        ),
        "assessments with no submission": int(
            (~tables["assessments"]["id_assessment"].isin(tables["studentAssessment"]["id_assessment"])).sum()
        ),
    }


def clean_all(tables: dict[str, pd.DataFrame], drop_vle_duplicates: bool = False) -> dict[str, pd.DataFrame]:
    out = dict(tables)
    out["studentInfo"] = clean_student_info(tables["studentInfo"])
    out["studentVle"] = clean_student_vle(tables["studentVle"], drop_vle_duplicates)
    return out


def save_interim(tables: dict[str, pd.DataFrame], out_dir: Path | None = None) -> list[Path]:
    """SCRUM-163: write cleaned tables to data/interim as parquet (git-ignored)."""
    out = Path(out_dir or INTERIM_DIR)
    out.mkdir(parents=True, exist_ok=True)
    paths = []
    for name, df in tables.items():
        p = out / f"{name}.parquet"
        df.to_parquet(p, index=False)
        paths.append(p)
    return paths
