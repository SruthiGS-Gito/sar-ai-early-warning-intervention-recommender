"""SCRUM-166 / SCRUM-167: VLE engagement features, using ONLY days <= cutoff.

Input is the raw studentVle table (columns: code_module, code_presentation,
id_student, id_site, date, sum_click). `date` is relative to module start and
can be negative (activity before the module began).

Rules (see docs/leakage_checklist.md):
  * rows with date > cutoff_day are dropped before anything is computed
  * rows are used as they are. Several records per student, site and day exist in
    OULAD, so exact repeats are treated as real records; pass drop_duplicates=True to drop them
  * pass `students` (a frame with the 3 key columns) to get one row for every
    student, with zero activity filled in, not only students who clicked
"""

import pandas as pd

KEY = ["code_module", "code_presentation", "id_student"]


def _prepare(student_vle: pd.DataFrame, cutoff_day: int, drop_duplicates: bool) -> pd.DataFrame:
    df = student_vle.drop_duplicates() if drop_duplicates else student_vle
    return df[df["date"] <= cutoff_day]


def _align(result: pd.DataFrame, students: pd.DataFrame | None) -> pd.DataFrame:
    if students is None:
        return result
    idx = pd.MultiIndex.from_frame(students[KEY].drop_duplicates())
    return result.reindex(idx)


def engagement_metrics(
    student_vle: pd.DataFrame,
    cutoff_day: int,
    students: pd.DataFrame | None = None,
    drop_duplicates: bool = False,
) -> pd.DataFrame:
    """Per-student click features up to the cutoff.

    Columns: total_clicks, active_days, avg_clicks_per_active_day,
    clicks_last_7d, clicks_prev_7d, click_trend (last7 - prev7),
    first_active_day (cutoff+1 if never active).
    """
    df = _prepare(student_vle, cutoff_day, drop_duplicates)
    daily = df.groupby(KEY + ["date"], as_index=False)["sum_click"].sum()

    res = daily.groupby(KEY).agg(
        total_clicks=("sum_click", "sum"),
        active_days=("date", "nunique"),
        first_active_day=("date", "min"),
    )
    last7 = daily[daily["date"] > cutoff_day - 7].groupby(KEY)["sum_click"].sum()
    prev7 = daily[
        (daily["date"] > cutoff_day - 14) & (daily["date"] <= cutoff_day - 7)
    ].groupby(KEY)["sum_click"].sum()
    res["clicks_last_7d"] = last7
    res["clicks_prev_7d"] = prev7

    res = _align(res, students)
    for col in ["total_clicks", "active_days", "clicks_last_7d", "clicks_prev_7d"]:
        res[col] = res[col].fillna(0).astype(int)
    res["first_active_day"] = res["first_active_day"].fillna(cutoff_day + 1)
    res["avg_clicks_per_active_day"] = (
        res["total_clicks"] / res["active_days"].where(res["active_days"] > 0)
    ).fillna(0.0)
    res["click_trend"] = res["clicks_last_7d"] - res["clicks_prev_7d"]
    return res[
        [
            "total_clicks", "active_days", "avg_clicks_per_active_day",
            "clicks_last_7d", "clicks_prev_7d", "click_trend", "first_active_day",
        ]
    ]


def consecutive_inactive_days(
    student_vle: pd.DataFrame,
    cutoff_day: int,
    students: pd.DataFrame | None = None,
    drop_duplicates: bool = False,
) -> pd.DataFrame:
    """Inactivity streaks inside the window day 0..cutoff.

    Columns: max_consecutive_inactive_days (longest silent stretch) and
    trailing_inactive_days (silent days right before the cutoff).
    A student with no click in the window gets cutoff_day + 1 for both.
    """
    df = _prepare(student_vle, cutoff_day, drop_duplicates)
    days = (
        df[df["date"] >= 0][KEY + ["date"]]
        .drop_duplicates()
        .sort_values(KEY + ["date"])
        .reset_index(drop=True)
    )
    prev = days.groupby(KEY)["date"].shift(1).fillna(-1)
    days["gap"] = days["date"] - prev - 1

    res = days.groupby(KEY).agg(longest_gap=("gap", "max"), last_day=("date", "max"))
    res["trailing_inactive_days"] = cutoff_day - res["last_day"]
    res["max_consecutive_inactive_days"] = res[["longest_gap", "trailing_inactive_days"]].max(axis=1)
    res = res[["max_consecutive_inactive_days", "trailing_inactive_days"]]

    res = _align(res, students)
    return res.fillna(cutoff_day + 1).astype(int)
