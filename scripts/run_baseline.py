"""SCRUM-175: raw CSVs -> Day 30 feature matrix -> baseline model metrics.

Run from the repo root:  python scripts/run_baseline.py
Writes data/processed/day30_feature_matrix.csv and reports/metrics_baseline.json.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sar.config import CUTOFF_DAY, PROCESSED_DIR, PROJECT_ROOT  # noqa: E402
from sar.data.clean import clean_all  # noqa: E402
from sar.data.loader import load_all  # noqa: E402
from sar.data.split import make_split, overlap_report  # noqa: E402
from sar.data.target import build_target, day30_population  # noqa: E402
from sar.features.assessment import assessment_features  # noqa: E402
from sar.features.engagement import consecutive_inactive_days, engagement_metrics  # noqa: E402
from sar.features.registration import registration_features  # noqa: E402
from sar.models.baseline import evaluate, save_metrics, train_baseline  # noqa: E402

KEY = ["code_module", "code_presentation", "id_student"]
ENROLMENT = ["gender", "region", "highest_education", "imd_band", "age_band",
             "num_of_prev_attempts", "studied_credits", "disability"]
FORBIDDEN = {"final_result", "date_unregistration", "date_registration"}
# Columns of the matrix that are not model inputs. code_module stays in as an input.
NOT_FEATURES = ["code_presentation", "id_student", "at_risk"]

tables = clean_all(load_all())
pop = day30_population(tables, CUTOFF_DAY)
target = build_target(tables["studentInfo"])

engagement = engagement_metrics(tables["studentVle"], CUTOFF_DAY, pop)
inactivity = consecutive_inactive_days(tables["studentVle"], CUTOFF_DAY, pop)
assessment = assessment_features(tables["studentAssessment"], tables["assessments"], CUTOFF_DAY, pop)
registration = registration_features(tables["studentRegistration"], inactivity, CUTOFF_DAY)
enrolment = tables["studentInfo"].set_index(KEY)[ENROLMENT]

matrix = (
    pop.join(target, on=KEY)
    .join(engagement, on=KEY)
    .join(assessment, on=KEY)
    .join(registration, on=KEY)
    .join(enrolment, on=KEY)
)
leaked = FORBIDDEN & set(matrix.columns)
if leaked:
    raise ValueError(f"Outcome or raw date columns in the matrix: {sorted(leaked)}")

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
matrix_path = PROCESSED_DIR / "day30_feature_matrix.csv"
matrix.to_csv(matrix_path, index=False)

train, val = make_split(matrix)
features = [c for c in matrix.columns if c not in NOT_FEATURES]
model = train_baseline(train[features], train["at_risk"])
metrics = {
    "cutoff_day": CUTOFF_DAY,
    "population": len(matrix),
    "at_risk_share": float(matrix["at_risk"].mean()),
    "features": features,
    "overlap": overlap_report(matrix),
    "train": len(train),
    "validation": len(val),
    "validation_metrics": evaluate(model, val[features], val["at_risk"]),
}
metrics_path = save_metrics(metrics, PROJECT_ROOT / "reports" / "metrics_baseline.json")

print("saved", matrix_path)
print("saved", metrics_path)
print(json.dumps(metrics, indent=2))
