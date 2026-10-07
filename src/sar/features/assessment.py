"""Ticket 211: cumulative GPA trend and submission delays."""

import pandas as pd


def assessment_features(
    student_assessment: pd.DataFrame, assessments: pd.DataFrame, cutoff_day: int
) -> pd.DataFrame:
    raise NotImplementedError("Ticket 211")
