"""Shared settings. Everything else reads paths and constants from here."""

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Set SAR_DATA_DIR to keep data outside the repository if you prefer.
DATA_DIR = Path(os.environ.get("SAR_DATA_DIR", PROJECT_ROOT / "data"))
RAW_DIR = DATA_DIR / "raw"
INTERIM_DIR = DATA_DIR / "interim"
PROCESSED_DIR = DATA_DIR / "processed"
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"
# Trained model files. Binaries are git-ignored: rebuild the final model with scripts/train_final.py.
MODELS_DIR = PROJECT_ROOT / "models"

OULAD_TABLES = (
    "assessments",
    "courses",
    "studentAssessment",
    "studentInfo",
    "studentRegistration",
    "studentVle",
    "vle",
)

# Prediction is made from activity up to this day of the module.
# Confirmed in ticket 109 (see docs/cutoff_and_validation.md).
CUTOFF_DAY = 30

RANDOM_SEED = 42

# Protected (background) attributes. The final model leaves them out as inputs; they are
# still read from the matrix for fairness checks and shown separately in explanations.
PROTECTED_FEATURES = ("gender", "region", "highest_education", "imd_band", "age_band", "disability")

# Risk bands on the final model's calibrated risk score, lowest first. Each band starts at a named
# cutoff in models/final/config.json; those cutoffs are chosen on out-of-fold training scores, never
# on the validation presentation. Low: below the "wide net" cutoff (80% out-of-fold recall).
# Medium: from the wide-net cutoff up to the top-20% cutoff. High: from the top-20% cutoff up.
RISK_BANDS = (("Low", None), ("Medium", "wide_net"), ("High", "top_20"))
