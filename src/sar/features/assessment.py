"""SCRUM-173: assessment features (score trend, delays), using only what is known by the cutoff.

Inputs are the raw studentAssessment and assessments tables.
Rules (see docs/leakage_checklist.md):
  * only submissions with date_submitted <= cutoff_day are used
  * Exams are ignored (they happen at the end of the module)
  * banked (transferred) results are ignored for delays
  * `n_missed` counts assessments DUE by the cutoff that the student has not submitted;
    it needs `students` (key columns) so students with no submission still get a row
"""

import pandas as pd

KEY = ["code_module", "code_presentation", "id_student"]
COURSE = ["code_module", "code_presentation"]


def assessment_features(
    student_assessment: pd.DataFrame,
    assessments: pd.DataFrame,
    cutoff_day: int,
    students: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """One row per student (indexed by KEY).

    Columns: n_submitted, mean_score, weighted_score, score_trend (last - first score),
    mean_delay_days (negative = early), n_late, n_assessments_due, n_missed.
    """
    a = assessments[assessments["assessment_type"] != "Exam"]
    sub = student_assessment.merge(
        a[["id_assessment", "code_module", "code_presentation", "date", "weight"]],
        on="id_assessment",
        how="inner",
    )
    sub = sub[sub["date_submitted"] <= cutoff_day]
    sub = sub.sort_values(KEY + ["date_submitted"])

    sub["delay"] = sub["date_submitted"] - sub["date"]
    sub.loc[sub["is_banked"] == 1, "delay"] = float("nan")
    sub["is_late"] = (sub["delay"] > 0).astype(int)
    sub["w_score"] = sub["score"] * sub["weight"]
    sub["w_used"] = sub["weight"].where(sub["score"].notna())

    g = sub.groupby(KEY)
    res = g.agg(
        n_submitted=("id_assessment", "nunique"),
        mean_score=("score", "mean"),
        mean_delay_days=("delay", "mean"),
        n_late=("is_late", "sum"),
        first_score=("score", "first"),
        last_score=("score", "last"),
    )
    wsum = g["w_score"].sum()
    wtot = g["w_used"].sum()
    res["weighted_score"] = (wsum / wtot.where(wtot > 0))
    res["score_trend"] = res["last_score"] - res["first_score"]
    res = res.drop(columns=["first_score", "last_score"])

    # Assessments that were due by the cutoff, per course.
    due = a[a["date"].notna() & (a["date"] <= cutoff_day)]
    due_n = due.groupby(COURSE).size().rename("n_assessments_due")
    done_due = (
        sub[sub["id_assessment"].isin(due["id_assessment"])]
        .groupby(KEY)["id_assessment"].nunique()
        .rename("done_due")
    )

    if students is not None:
        idx = pd.MultiIndex.from_frame(students[KEY].drop_duplicates())
        res = res.reindex(idx)
    res = res.join(done_due)
    res = res.join(due_n, on=COURSE)
    res["n_assessments_due"] = res["n_assessments_due"].fillna(0).astype(int)
    res["n_missed"] = (res["n_assessments_due"] - res["done_due"].fillna(0)).clip(lower=0).astype(int)
    res = res.drop(columns=["done_due"])
    for col in ["n_submitted", "n_late"]:
        res[col] = res[col].fillna(0).astype(int)
    return res
