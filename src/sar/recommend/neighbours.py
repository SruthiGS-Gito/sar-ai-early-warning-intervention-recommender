"""SCRUM-188: the most similar students from the training presentations.

Similarity uses activity, assessment and study-load inputs only (sar.explain.factors BEHAVIOUR and
STUDY). Background factors are not part of the distance, and the module is used as a filter, not
an input: neighbours come from the same module, unless it has fewer than k earlier students, in
which case all modules are used and the result says so.

Click counts are log-scaled so a few very active students do not dominate. Missing scores are
filled with the pool median and flagged with an indicator column. Imputer and scaler are fitted
on the pool only.

Leakage: the pool must not contain the validation presentation (checked). Only pool outcomes are
stored, so a validation student's outcome can never reach a recommendation.
Neighbour rows stay in memory: report counts and medians, never the rows themselves.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from sar.data.split import TEST_PRESENTATION
from sar.explain.factors import BEHAVIOUR, STUDY

KEY = ["code_module", "code_presentation", "id_student"]
NEIGHBOUR_FEATURES = BEHAVIOUR + STUDY
LOG_COLUMNS = ("total_clicks", "avg_clicks_per_active_day", "clicks_last_7d", "clicks_prev_7d", "click_trend")
K = 50


def _prepare(X: pd.DataFrame) -> pd.DataFrame:
    X = X[list(NEIGHBOUR_FEATURES)].astype(float)
    for c in LOG_COLUMNS:
        X[c] = np.sign(X[c]) * np.log1p(X[c].abs())
    return X


@dataclass
class Neighbours:
    frame: pd.DataFrame  # neighbour inputs and at_risk; in memory only, never print or save it
    module: str
    same_module: bool
    module_candidates: int
    presentations: list[str]
    note: str

    @property
    def n(self) -> int:
        return len(self.frame)

    @property
    def n_at_risk(self) -> int:
        return int(self.frame["at_risk"].sum())

    @property
    def n_on_track(self) -> int:
        return self.n - self.n_at_risk

    @property
    def at_risk_share(self) -> float:
        return self.n_at_risk / self.n if self.n else float("nan")

    def summary(self) -> str:
        where = f"module {self.module}" if self.same_module else "all modules"
        text = (f"The {self.n} most similar students from earlier presentations ({', '.join(self.presentations)}; "
                f"{where}): {self.n_at_risk} ended At Risk ({self.at_risk_share:.0%}) and {self.n_on_track} On Track.")
        return f"{text} {self.note}" if self.note else text


class NeighbourPool:
    """Training-presentation students to compare against. Build once, query per student."""

    def __init__(self, train: pd.DataFrame, k: int = K, test_presentation: str = TEST_PRESENTATION):
        if (train["code_presentation"] == test_presentation).any():
            raise ValueError(f"The neighbour pool must not contain {test_presentation} (the validation presentation)")
        self.k = k
        self.data = train[KEY + list(NEIGHBOUR_FEATURES) + ["at_risk"]].reset_index(drop=True)
        self.presentations = sorted(self.data["code_presentation"].unique())
        self.scaler = Pipeline([
            ("imp", SimpleImputer(strategy="median", add_indicator=True)),
            ("sc", StandardScaler()),
        ])
        self.Z = self.scaler.fit_transform(_prepare(self.data))

    def query(self, row, same_module: bool = True) -> Neighbours:
        """`row` holds one student's inputs (Series or one-row DataFrame) including code_module.
        If it has id_student, that student's own records are left out of the pool."""
        X = row.to_frame().T if isinstance(row, pd.Series) else row
        if len(X) != 1:
            raise ValueError(f"query takes one row, got {len(X)}")
        z = self.scaler.transform(_prepare(X))[0]
        module = X["code_module"].iloc[0]
        allowed = np.ones(len(self.data), dtype=bool)
        if "id_student" in X.columns:
            allowed &= (self.data["id_student"] != X["id_student"].iloc[0]).to_numpy()
        in_module = allowed & (self.data["code_module"] == module).to_numpy()
        n_module = int(in_module.sum())
        if same_module and n_module >= self.k:
            candidates, used_module, note = in_module, True, ""
        else:
            candidates, used_module = allowed, False
            note = (f"Module {module} has only {n_module} earlier students to compare with, so the {self.k} "
                    "most similar were taken from all modules." if same_module else "")
        idx = np.flatnonzero(candidates)
        dist = ((self.Z[idx] - z) ** 2).sum(axis=1)
        nearest = idx[np.argsort(dist, kind="stable")[: self.k]]
        return Neighbours(self.data.iloc[nearest], module, used_module, n_module, self.presentations, note)
