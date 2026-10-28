"""SCRUM-189: loading for the Streamlit app. Nothing here raises on missing or broken files.

Every loader returns a Loaded(value, message): value is None and message says what to do when
something is missing or unreadable, so the app can show the message instead of crashing.

The model is the final model in models/final/ (sar.models.final), loaded through its hash-checked
loader together with config.json and metrics.json. Batch SHAP values precomputed by
scripts/precompute_shap.py are stored next to it (models/final/shap/, git-ignored, student-level)
and are only used when they were computed for the same model file.

Team tables (department risk, retention funnel, student profiles; tickets 203, 207 and 210) are
built by scripts/build_team_tables.py (sar.tables) into data/processed. The app only reads them:
a table is shown if its CSV exists, has the columns listed in sar.tables, and holds no
student-level column; otherwise its status card says why. `table_status` reports which case applies.
"""

import json
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import pandas as pd

from sar.config import MODELS_DIR, PROCESSED_DIR
from sar.models.final import load_final
from sar.tables import department, funnel, profiles

FINAL_SUBDIR = "final"
SHAP_SUBDIR = "shap"
SHAP_PREFIX = "shap_"
TRAIN_HINT = "Run: python scripts/train_final.py"
TABLES_HINT = "Build it with: python scripts/build_team_tables.py"

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
    "department_risk": TableSpec("Department risk", "department_risk.csv", "ticket 203 (SCRUM-178)",
                                 department.COLUMNS, "Students, observed at-risk rate and risk band shares per "
                                 "module and presentation."),
    "retention_funnel": TableSpec("Retention funnel", "retention_funnel.csv", "ticket 207 (SCRUM-177)",
                                  funnel.COLUMNS, "Students at each of six retention stages per module and "
                                  "presentation."),
    "student_profiles": TableSpec("Student profiles", "student_profiles.csv", "ticket 210 (SCRUM-183)",
                                  profiles.COLUMNS, "Day 30 engagement profiles with their size, mean values and "
                                  "observed at-risk rate."),
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


@dataclass
class FinalModel:
    fitted: dict    # model, calibrator, features, cutoffs (sar.models.final.load_final)
    config: dict    # models/final/config.json
    metrics: dict   # models/final/metrics.json


def final_dir(models_dir: Path) -> Path:
    return Path(models_dir) / FINAL_SUBDIR


def load_final_model(models_dir: Path) -> Loaded:
    """Loaded.value is a FinalModel."""
    folder = final_dir(models_dir)
    if not all((folder / f).exists() for f in ("config.json", "metrics.json", "model.joblib", "calibrator.joblib")):
        return Loaded(message=f"The model is not available yet. {TRAIN_HINT}")
    try:
        fitted, config = load_final(folder)
        metrics = json.loads((folder / "metrics.json").read_text())
        return Loaded(FinalModel(fitted, config, metrics))
    except (FileNotFoundError, ValueError) as e:
        return Loaded(message=f"The model could not be loaded: {e}")
    except Exception as e:
        return Loaded(message=f"The model could not be loaded ({type(e).__name__}). {TRAIN_HINT}")


def shap_cache_path(models_dir: Path, presentation: str) -> Path:
    return final_dir(models_dir) / SHAP_SUBDIR / f"{presentation}.parquet"


def save_shap_cache(models_dir: Path, presentation: str, keys: pd.DataFrame, contributions: pd.DataFrame,
                    raw_scores: pd.Series, base_score: float, config: dict) -> Path:
    """Store one presentation's SHAP contributions, tied to the model file's hash."""
    path = shap_cache_path(models_dir, presentation)
    path.parent.mkdir(parents=True, exist_ok=True)
    table = keys.reset_index(drop=True).copy()
    table["raw_score"] = raw_scores.to_numpy()
    shap_cols = contributions.add_prefix(SHAP_PREFIX).reset_index(drop=True)
    pd.concat([table, shap_cols], axis=1).to_parquet(path, index=False)
    path.with_suffix(".json").write_text(json.dumps({
        "presentation": presentation, "rows": len(table), "base_score": float(base_score),
        "features": list(contributions.columns), "model_sha256": config["model_sha256"]}, indent=2))
    return path


def load_shap_cache(models_dir: Path, presentation: str, config: dict) -> Loaded:
    """Loaded.value is (table, base_score) for a presentation, or a message saying how to build it."""
    path = shap_cache_path(models_dir, presentation)
    hint = f"Run: python scripts/precompute_shap.py --presentation {presentation}"
    if not path.exists() or not path.with_suffix(".json").exists():
        return Loaded(message=f"No precomputed explanations for {presentation}. {hint}")
    try:
        meta = json.loads(path.with_suffix(".json").read_text())
        if meta["model_sha256"] != config["model_sha256"]:
            return Loaded(message=f"Precomputed explanations for {presentation} belong to another model file. {hint}")
        return Loaded((pd.read_parquet(path), meta["base_score"]))
    except Exception as e:  # unreadable file: report it, do not crash the app
        return Loaded(message=f"Precomputed explanations for {presentation} could not be read ({type(e).__name__}). "
                              f"{hint}")


def table_status(spec: TableSpec, processed_dir: Path) -> TableStatus:
    """Which state an optional team table is in. Never raises."""
    path = Path(processed_dir) / spec.file
    if not path.exists():
        return TableStatus(spec, path, "missing", f"{spec.title} is not available yet: {spec.ticket}. {TABLES_HINT}.")
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
