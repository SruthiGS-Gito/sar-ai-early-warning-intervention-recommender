"""PROVISIONAL Random Forest and threshold for the Day 30 at-risk model.

The team model (tickets 212 and 217, SCRUM-184 and SCRUM-185) exists only as a notebook, with no saved
model file, and the decision threshold (ticket 214, SCRUM-186) is not in the repo yet.
This module retrains its tuned settings (RandomizedSearchCV result, cell 69 of
notebooks/xgboost-random-forest (2).ipynb, commit c4c3ba6) on the repo's Day 30 feature
matrix, so explanations (SCRUM-187) and recommendations (SCRUM-188) have a model to use.
Replace the model when 212/217 deliver a saved model, and the threshold when 214 lands.

Leakage rules:
  * the model is fitted on the training presentations only (sar.data.split.make_split)
  * the threshold comes from out-of-fold scores on the training presentations
    (GroupKFold by presentation); the validation presentation is never used to choose it
"""

import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import precision_recall_curve
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from sar.config import MODELS_DIR, RANDOM_SEED

PROVISIONAL = True
MODEL_NAME = "forest_provisional"
RECALL_TARGET = 0.80
FOREST_PARAMS = {
    "class_weight": "balanced_subsample",
    "max_depth": None,
    "max_features": "sqrt",
    "min_samples_leaf": 5,
    "min_samples_split": 20,
    "n_estimators": 234,
}


def build_forest(X: pd.DataFrame, params: dict | None = None) -> Pipeline:
    """Median-impute numbers, one-hot encode text columns, then a Random Forest.

    No scaling: trees do not need it, and SHAP values stay in the original units.
    Step names "num" and "cat" match sar.viz.plots.feature_label.
    """
    num_cols = X.select_dtypes(include="number").columns.tolist()
    cat_cols = [c for c in X.columns if c not in num_cols]
    pre = ColumnTransformer(
        [
            ("num", SimpleImputer(strategy="median"), num_cols),
            (
                "cat",
                Pipeline([
                    ("imp", SimpleImputer(strategy="most_frequent")),
                    ("oh", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                ]),
                cat_cols,
            ),
        ]
    )
    clf = RandomForestClassifier(random_state=RANDOM_SEED, n_jobs=-1, **(params or FOREST_PARAMS))
    return Pipeline([("pre", pre), ("clf", clf)])


def train_forest(X_train: pd.DataFrame, y_train: pd.Series, params: dict | None = None) -> Pipeline:
    model = build_forest(X_train, params)
    model.fit(X_train, y_train)
    return model


def oof_scores(estimator: Pipeline, X_train: pd.DataFrame, y_train: pd.Series, groups: pd.Series) -> np.ndarray:
    """Out-of-fold at-risk scores for an unfitted `estimator`.

    Each training presentation is scored by a model fitted on the other training
    presentations (GroupKFold, one fold per presentation), so no score comes from a model
    that saw that student's presentation.
    """
    n_groups = pd.Series(groups).nunique()
    if n_groups < 2:
        raise ValueError("oof_threshold needs at least two training presentations")
    return cross_val_predict(
        estimator, X_train, y_train,
        groups=groups, cv=GroupKFold(n_splits=n_groups), method="predict_proba",
    )[:, 1]


def recall_cutoff(y: pd.Series, scores: np.ndarray, recall_target: float = RECALL_TARGET) -> dict:
    """Highest cutoff on `scores` whose recall reaches `recall_target`."""
    precision, recall, thresholds = precision_recall_curve(y, scores)
    i = int(np.where(recall[:-1] >= recall_target)[0][-1])
    return {"threshold": float(thresholds[i]), "recall": float(recall[i]), "precision": float(precision[i]),
            "flagged_share": float((scores >= thresholds[i]).mean())}


def oof_threshold(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    groups: pd.Series,
    recall_target: float = RECALL_TARGET,
    params: dict | None = None,
) -> dict:
    """Highest threshold whose out-of-fold recall reaches `recall_target` (folds as in oof_scores)."""
    oof = oof_scores(build_forest(X_train, params), X_train, y_train, groups)
    cut = recall_cutoff(y_train, oof, recall_target)
    return {
        "threshold": cut["threshold"],
        "recall_target": recall_target,
        "oof_recall": cut["recall"],
        "oof_precision": cut["precision"],
        "oof_flagged_share": cut["flagged_share"],
        "folds": int(pd.Series(groups).nunique()),
    }


def file_sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save_model(model: Pipeline, metadata: dict, models_dir: Path | None = None,
               metadata_path: Path | None = None) -> tuple[Path, Path]:
    """Write the model (git-ignored folder) and its metadata JSON (committed, no student data)."""
    models_dir = Path(models_dir or MODELS_DIR)
    models_dir.mkdir(parents=True, exist_ok=True)
    model_path = models_dir / f"{MODEL_NAME}.joblib"
    joblib.dump(model, model_path)
    metadata = {**metadata, "provisional": PROVISIONAL, "model_file": model_path.name,
                "model_sha256": file_sha256(model_path)}
    metadata_path = Path(metadata_path or models_dir / f"{MODEL_NAME}.json")
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(json.dumps(metadata, indent=2))
    return model_path, metadata_path


def load_model(metadata_path: Path, models_dir: Path | None = None) -> tuple[Pipeline, dict]:
    """Load the model named in the metadata JSON and check it is the same file.

    joblib files can run code when loaded, so only load a file you built yourself with
    scripts/train_forest.py; the hash check stops a swapped or stale file from loading.
    """
    metadata = json.loads(Path(metadata_path).read_text())
    model_path = Path(models_dir or MODELS_DIR) / metadata["model_file"]
    return load_checked(model_path, metadata["model_sha256"], Path(metadata_path).name), metadata


def load_checked(path: Path, sha256: str, metadata_name: str, script: str = "scripts/train_forest.py"):
    """joblib.load `path` only if it exists and its hash matches the one recorded in the metadata.

    joblib can run code on load: use this only for files built locally by this repo's scripts.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run: python {script}")
    if file_sha256(path) != sha256:
        raise ValueError(f"{path.name} does not match {metadata_name}. Re-run {script}")
    return joblib.load(path)
