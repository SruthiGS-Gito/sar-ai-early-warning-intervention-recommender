"""SCRUM-189: loading for the Streamlit app. Nothing here raises on missing or broken files.

Every loader returns a Loaded(value, message): value is None and message says what to do when
something is missing or unreadable, so the app can show the message instead of crashing.

Optional team tables (department risk, retention funnel, student profiles) belong to other
tickets. The app only reads them if their CSV exists in data/processed, has the expected columns,
and holds no student-level column; otherwise it says "not available yet". It never builds them.
"""

import os
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from sar.config import MODELS_DIR, PROCESSED_DIR
from sar.models.forest import MODEL_NAME, load_model

STUDENT_LEVEL_COLUMNS = {"id_student"}


@dataclass
class Loaded:
    value: object = None
    message: str | None = None

    @property
    def ok(self) -> bool:
        return self.message is None


@dataclass(frozen=True)
class TableSpec:
    title: str
    file: str
    ticket: str
    columns: tuple[str, ...]


OPTIONAL_TABLES = {
    "department_risk": TableSpec("Department risk", "department_risk.csv", "ticket 203 (SCRUM-178)", ("code_module",)),
    "retention_funnel": TableSpec("Retention funnel", "retention_funnel.csv", "ticket 207 (SCRUM-177)",
                                  ("stage", "students")),
    "student_profiles": TableSpec("Student profiles", "student_profiles.csv", "ticket 210 (SCRUM-183)", ("profile",)),
}


def resolve_paths() -> dict[str, Path]:
    """Folders read at call time, so SAR_DATA_DIR / SAR_MODELS_DIR can point the app elsewhere."""
    data = os.environ.get("SAR_DATA_DIR")
    models = os.environ.get("SAR_MODELS_DIR")
    return {
        "processed": Path(data) / "processed" if data else PROCESSED_DIR,
        "models": Path(models) if models else MODELS_DIR,
    }


def load_matrix(processed_dir: Path) -> Loaded:
    path = Path(processed_dir) / "day30_feature_matrix.csv"
    if not path.exists():
        return Loaded(message=f"The Day 30 feature matrix is not available yet. Run: python scripts/run_baseline.py "
                              f"(expected at {path}).")
    try:
        return Loaded(pd.read_csv(path))
    except Exception as e:  # unreadable file: report it, do not crash the app
        return Loaded(message=f"Could not read {path.name}: {type(e).__name__}.")


def load_provisional_model(models_dir: Path) -> Loaded:
    """Loaded.value is (model, metadata)."""
    meta = Path(models_dir) / f"{MODEL_NAME}.json"
    if not meta.exists():
        return Loaded(message="The provisional model is not available yet. Run: python scripts/train_forest.py")
    try:
        return Loaded(load_model(meta, models_dir))
    except (FileNotFoundError, ValueError) as e:
        return Loaded(message=f"The provisional model could not be loaded: {e}")
    except Exception as e:
        return Loaded(message=f"The provisional model could not be loaded ({type(e).__name__}). "
                              "Re-run: python scripts/train_forest.py")


def load_optional_table(spec: TableSpec, processed_dir: Path) -> Loaded:
    path = Path(processed_dir) / spec.file
    if not path.exists():
        return Loaded(message=f"{spec.title} is not available yet: {spec.ticket}.")
    try:
        df = pd.read_csv(path)
    except Exception as e:
        return Loaded(message=f"{spec.title} could not be read ({type(e).__name__}).")
    if STUDENT_LEVEL_COLUMNS & set(df.columns):
        return Loaded(message=f"{spec.title} contains student-level rows, so it is not shown. "
                              "The app shows aggregated tables only.")
    missing = [c for c in spec.columns if c not in df.columns]
    if missing:
        return Loaded(message=f"{spec.title} is missing expected columns: {', '.join(missing)}.")
    return Loaded(df)
