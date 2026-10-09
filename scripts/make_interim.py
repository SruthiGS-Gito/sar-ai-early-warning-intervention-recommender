"""SCRUM-160 / 161 / 163: raw CSVs -> checked, cleaned parquet tables in data/interim.

Run from the repo root:  python scripts/make_interim.py
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sar.data.clean import check_join_keys, clean_all, save_interim  # noqa: E402
from sar.data.loader import load_all  # noqa: E402

tables = load_all()
report = check_join_keys(tables)
print(json.dumps(report, indent=2, default=str))
cleaned = clean_all(tables)
for p in save_interim(cleaned):
    print("saved", p)
print("rows:", {k: len(v) for k, v in cleaned.items()})
