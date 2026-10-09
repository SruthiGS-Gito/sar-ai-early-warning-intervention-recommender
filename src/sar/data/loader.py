"""SCRUM-160: reproducible loader for the seven OULAD tables.

Missing values are written as "?" in the OULAD files, so "?" is read as NaN.
Files are read from RAW_DIR (data/raw, or SAR_DATA_DIR/raw).
"""

from pathlib import Path

import pandas as pd

from sar.config import OULAD_TABLES, RAW_DIR


def load_table(name: str, raw_dir: Path | None = None) -> pd.DataFrame:
    """Load one OULAD table by name, for example "studentInfo"."""
    if name not in OULAD_TABLES:
        raise ValueError(f"Unknown table {name!r}. Expected one of {OULAD_TABLES}")
    path = Path(raw_dir or RAW_DIR) / f"{name}.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Put the seven OULAD CSV files in {Path(raw_dir or RAW_DIR)}"
        )
    return pd.read_csv(path, na_values=["?"])


def load_all(raw_dir: Path | None = None) -> dict[str, pd.DataFrame]:
    """Load every table in OULAD_TABLES into a dict keyed by table name."""
    return {name: load_table(name, raw_dir) for name in OULAD_TABLES}
