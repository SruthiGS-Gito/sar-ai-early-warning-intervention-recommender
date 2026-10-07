"""Ticket 104: clean student data and check join-key integrity."""

import pandas as pd


def clean_student_info(df: pd.DataFrame) -> pd.DataFrame:
    raise NotImplementedError("Ticket 104")


def check_join_keys(tables: dict[str, pd.DataFrame]) -> dict:
    """Report missing or duplicated keys across tables."""
    raise NotImplementedError("Ticket 104")
