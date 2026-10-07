"""Ticket 111: target definition and Day 30 model population."""

import pandas as pd


def build_target(student_info: pd.DataFrame) -> pd.Series:
    """Return the at-risk label per student registration."""
    raise NotImplementedError("Ticket 111")


def day30_population(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Students still registered at the cutoff day."""
    raise NotImplementedError("Ticket 111")
