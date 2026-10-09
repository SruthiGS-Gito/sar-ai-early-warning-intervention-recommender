"""SCRUM-170: validation split.

Rule: train on earlier presentations, validate on the latest one (2014J).
This mirrors deployment, where a new presentation is scored by a model trained on past ones.
Students who appear in both sets are removed from the training set, so no student
is seen in training and validation. Rule and reason: docs/cutoff_and_validation.md.
"""

import pandas as pd

TEST_PRESENTATION = "2014J"


def make_split(
    df: pd.DataFrame,
    test_presentation: str = TEST_PRESENTATION,
    drop_overlap: bool = True,
):
    """Return (train, validation). `df` needs code_presentation and id_student columns."""
    if test_presentation not in set(df["code_presentation"]):
        raise ValueError(f"{test_presentation!r} not found in code_presentation")
    is_val = df["code_presentation"] == test_presentation
    train, val = df[~is_val], df[is_val]
    if drop_overlap:
        train = train[~train["id_student"].isin(val["id_student"])]
    return train.reset_index(drop=True), val.reset_index(drop=True)


def overlap_report(df: pd.DataFrame, test_presentation: str = TEST_PRESENTATION) -> dict:
    """Counts used in the validation notes."""
    val = df[df["code_presentation"] == test_presentation]
    train = df[df["code_presentation"] != test_presentation]
    shared = set(val["id_student"]) & set(train["id_student"])
    return {
        "train rows": len(train),
        "validation rows": len(val),
        "students in both": len(shared),
        "train rows removed from overlap": int(train["id_student"].isin(shared).sum()),
    }
