"""SCRUM-175: baseline Logistic Regression for the Day 30 at-risk model.

Input X must contain ONLY cutoff-safe features (never final_result or date_unregistration).
Numbers are median-imputed and scaled, text columns are most-frequent-imputed and
one-hot encoded, and class_weight="balanced" handles the imbalanced target.
"""

import json
from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from sar.config import RANDOM_SEED


def build_pipeline(X: pd.DataFrame) -> Pipeline:
    num_cols = X.select_dtypes(include="number").columns.tolist()
    cat_cols = [c for c in X.columns if c not in num_cols]
    pre = ColumnTransformer(
        [
            ("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]), num_cols),
            (
                "cat",
                Pipeline([
                    ("imp", SimpleImputer(strategy="most_frequent")),
                    ("oh", OneHotEncoder(handle_unknown="ignore")),
                ]),
                cat_cols,
            ),
        ]
    )
    clf = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=RANDOM_SEED)
    return Pipeline([("pre", pre), ("clf", clf)])


def train_baseline(X_train: pd.DataFrame, y_train: pd.Series) -> Pipeline:
    model = build_pipeline(X_train)
    model.fit(X_train, y_train)
    return model


def evaluate(model: Pipeline, X: pd.DataFrame, y: pd.Series, threshold: float = 0.5) -> dict:
    """Recall and PR-AUC matter most (missing an at-risk student is the costly error)."""
    proba = model.predict_proba(X)[:, 1]
    pred = (proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "recall": float(recall_score(y, pred, zero_division=0)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "pr_auc": float(average_precision_score(y, proba)),
        "roc_auc": float(roc_auc_score(y, proba)),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "positive_rate": float(y.mean()),
        "n": int(len(y)),
    }


def save_metrics(metrics: dict, path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metrics, indent=2))
    return path
