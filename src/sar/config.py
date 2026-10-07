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
