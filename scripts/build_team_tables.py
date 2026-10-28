"""Build the three aggregated team tables for the app (tickets 203, 207 and 210).

Run from the repo root, after scripts/run_baseline.py and scripts/train_final.py:
    python scripts/build_team_tables.py
Writes to data/processed/ (git-ignored, like all generated data):
    department_risk.csv   ticket 203: risk bands and at-risk rate per module and presentation
    retention_funnel.csv  ticket 207: six retention stages per module and presentation
    student_profiles.csv  ticket 210: KMeans engagement profiles, fitted on training presentations
Every row is a group of students; groups below MIN_CELL are marked suppressed with blank figures.
It also writes reports/team_tables.json (tracked), the summary quoted in docs/app.md, and
regenerates that section of docs/app.md from it.
"""

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd  # noqa: E402

from sar.config import MODELS_DIR, PROCESSED_DIR, PROJECT_ROOT  # noqa: E402
from sar.data.clean import clean_student_info  # noqa: E402
from sar.data.loader import load_table  # noqa: E402
from sar.data.split import make_split  # noqa: E402
from sar.models.final import load_final  # noqa: E402
from sar.service.data import final_dir  # noqa: E402
from sar.tables.department import department_risk  # noqa: E402
from sar.tables.funnel import retention_funnel  # noqa: E402
from sar.tables.profiles import fit_profiles, student_profiles  # noqa: E402
from sar.tables.summary import render_block, replace_block, summarise  # noqa: E402

start = time.perf_counter()
matrix = pd.read_csv(PROCESSED_DIR / "day30_feature_matrix.csv")
fitted, _ = load_final(final_dir(MODELS_DIR))

tables = {"department_risk": department_risk(fitted, matrix)}
info = clean_student_info(load_table("studentInfo"))
tables["retention_funnel"] = retention_funnel(info, load_table("studentRegistration"), matrix)
train, val = make_split(matrix)
profiles = fit_profiles(train)
tables["student_profiles"] = student_profiles(profiles, train, val)

for name, table in tables.items():
    if "id_student" in table.columns:
        raise ValueError(f"{name} has a student-level column")
    path = PROCESSED_DIR / f"{name}.csv"
    table.to_csv(path, index=False)
    print(f"{path.name}: {len(table):,} rows, {int(table['suppressed'].sum())} suppressed")

summary = summarise(*tables.values(), profiles["selection"], profiles["k"])
summary_path = PROJECT_ROOT / "reports" / "team_tables.json"
summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
docs = PROJECT_ROOT / "docs" / "app.md"
docs.write_text(replace_block(docs.read_text(encoding="utf-8"), render_block(summary)), encoding="utf-8")
print(f"saved {summary_path.relative_to(PROJECT_ROOT)} and updated {docs.relative_to(PROJECT_ROOT)}")

print("profile k candidates:")
print(profiles["selection"].round(4).to_string(index=False))
print(f"chosen k = {profiles['k']}")
print(f"done in {time.perf_counter() - start:.0f} s")
