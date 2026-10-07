"""Ticket 103: reproducible loader for the seven OULAD tables."""

import pandas as pd

from sar.config import OULAD_TABLES, RAW_DIR  # noqa: F401


def load_table(name: str) -> pd.DataFrame:
    """Load one OULAD table from RAW_DIR."""
    raise NotImplementedError("Ticket 103")


def load_all() -> dict[str, pd.DataFrame]:
    """Load every table in OULAD_TABLES into a dict keyed by table name."""
    raise NotImplementedError("Ticket 103")
