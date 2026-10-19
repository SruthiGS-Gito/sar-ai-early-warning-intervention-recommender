"""SCRUM-189: loading for the Streamlit app. Nothing here raises on missing or broken files.

Every loader returns a Loaded(value, message): value is None and message says what to do when
something is missing or unreadable, so the app can show the message instead of crashing.

Optional team tables (department risk, retention funnel, student profiles) belong to other
tickets. The app only reads them if their CSV exists in data/processed, has the expected columns,
and holds no student-level column; otherwise it says "not available yet". It never builds them.
`table_status` reports which of these cases applies, for the status cards in the app.
"""

import os
from dataclasses import dataclass
from datetime import datetime
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
    description: str = ""


OPTIONAL_TABLES = {
    "department_risk": TableSpec("Department risk", "department_risk.csv", "ticket 203 (SCRUM-178)", ("code_module",),
                                 "Aggregated risk figures per department (module) for the department dashboard."),
    "retention_funnel": TableSpec("Retention funnel", "retention_funnel.csv", "ticket 207 (SCRUM-177)",
                                  ("stage", "students"), "Number of students at each retention funnel stage."),
    "student_profiles": TableSpec("Student profiles", "student_profiles.csv", "ticket 210 (SCRUM-183)", ("profile",),
                                  "Engagement profiles with aggregated figures per profile."),
}


@dataclass
class TableStatus:
    """state: missing, unreadable, refused, missing_columns or loaded."""
    spec: TableSpec
    path: Path
    state: str
    message: str
    table: pd.DataFrame | None = None
    missing_columns: tuple[str, ...] = ()
    updated: str | None = None

    @property
    def ok(self) -> bool:
        return self.state == "loaded"


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


def table_status(spec: TableSpec, processed_dir: Path) -> TableStatus:
    """Which state an optional team table is in. Never raises."""
    path = Path(processed_dir) / spec.file
    if not path.exists():
        return TableStatus(spec, path, "missing", f"{spec.title} is not available yet: {spec.ticket}.")
    updated = datetime.fromtimestamp(path.stat().st_mtime).strftime("%Y-%m-%d %H:%M")
    try:
        df = pd.read_csv(path)
    except Exception as e:  # unreadable file: report it, do not crash the app
        return TableStatus(spec, path, "unreadable", f"{spec.title} could not be read ({type(e).__name__}).",
                           updated=updated)
    if STUDENT_LEVEL_COLUMNS & set(df.columns):
        return TableStatus(spec, path, "refused", f"{spec.title} contains student-level rows, so it is not shown. "
                           "The app shows aggregated tables only.", updated=updated)
    missing = tuple(c for c in spec.columns if c not in df.columns)
    if missing:
        return TableStatus(spec, path, "missing_columns",
                           f"{spec.title} is missing expected columns: {', '.join(missing)}.",
                           missing_columns=missing, updated=updated)
    return TableStatus(spec, path, "loaded", f"{spec.title}: {len(df):,} rows, last updated {updated}.",
                       table=df, updated=updated)


def load_optional_table(spec: TableSpec, processed_dir: Path) -> Loaded:
    status = table_status(spec, processed_dir)
    return Loaded(status.table) if status.ok else Loaded(message=status.message)
