"""SCRUM-172: target definition and the Day 30 model population.

Target  at_risk = 1 if final_result in {Fail, Withdrawn}, else 0 (Pass, Distinction).
Population  students who had NOT unregistered on or before the cutoff day.
            Early leavers are returned separately for the retention funnel.
final_result and date_unregistration must never be used as features.
"""

import pandas as pd

from sar.config import CUTOFF_DAY

KEY = ["code_module", "code_presentation", "id_student"]
AT_RISK = {"Fail", "Withdrawn"}
ON_TRACK = {"Pass", "Distinction"}


def build_target(student_info: pd.DataFrame) -> pd.Series:
    """Return a 0/1 `at_risk` Series indexed by the 3 key columns."""
    unknown = set(student_info["final_result"].dropna().unique()) - AT_RISK - ON_TRACK
    if unknown:
        raise ValueError(f"Unexpected final_result values: {sorted(unknown)}")
    if student_info["final_result"].isna().any():
        raise ValueError("final_result has missing values")
    y = student_info["final_result"].isin(AT_RISK).astype(int)
    y.index = pd.MultiIndex.from_frame(student_info[KEY])
    y.name = "at_risk"
    if y.index.has_duplicates:
        raise ValueError("studentInfo key is not unique")
    return y


def _unregistered_by_cutoff(registration: pd.DataFrame, cutoff_day: int) -> pd.DataFrame:
    gone = registration["date_unregistration"].notna() & (
        registration["date_unregistration"] <= cutoff_day
    )
    return registration.loc[gone, KEY]


def day30_population(
    tables: dict[str, pd.DataFrame], cutoff_day: int = CUTOFF_DAY
) -> pd.DataFrame:
    """Students still registered at the cutoff. Returns the 3 key columns only."""
    info = tables["studentInfo"][KEY].drop_duplicates()
    gone = _unregistered_by_cutoff(tables["studentRegistration"], cutoff_day)
    merged = info.merge(gone, on=KEY, how="left", indicator=True)
    return merged.loc[merged["_merge"] == "left_only", KEY].reset_index(drop=True)


def early_leavers(
    tables: dict[str, pd.DataFrame], cutoff_day: int = CUTOFF_DAY
) -> pd.DataFrame:
    """Students who unregistered on or before the cutoff (for the retention funnel)."""
    info = tables["studentInfo"][KEY].drop_duplicates()
    gone = _unregistered_by_cutoff(tables["studentRegistration"], cutoff_day)
    return info.merge(gone, on=KEY, how="inner").reset_index(drop=True)
