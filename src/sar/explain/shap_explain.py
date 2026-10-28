"""SCRUM-187: per-student SHAP explanations for a tree model pipeline.

`explain_student(model, row, band=band)` returns an Explanation: the model score, each input's
contribution to it (SHAP value, one-hot columns added back into their original input), the
student's risk band (Low, Medium, High; see sar.config.RISK_BANDS) and a plain-language sentence.
`explain_batch` computes contributions for many rows at once (optionally in parallel), and
`explanation_from_contributions` builds an Explanation from stored batch values without recomputing.

How the contributions are computed
  * shap.TreeExplainer with its default tree_path_dependent mode: the expected score comes from
    the training data stored in the trees, so no other data (and no validation data) is used
  * contributions plus the starting score add up to the model's score; this is checked
  * background inputs (sar.explain.factors.BACKGROUND) are never named as a main reason; the final
    model does not use them as inputs at all, so they have no contribution to show

Wording: contributions describe the model's raw score, not causes. Sentences say what "raises the
score", never what caused an outcome, and report the risk band rather than a number.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
import shap
from joblib import Parallel, delayed
from sklearn.pipeline import Pipeline

from sar.explain.factors import REASON_GROUPS, family_of, group_of, phrase

ADDITIVITY_TOLERANCE = 1e-6
# Smallest change in score worth naming in a sentence (half a point on a 0-100 scale).
MIN_CONTRIBUTION = 0.005


@dataclass
class Explanation:
    score: float                 # raw model score; the SHAP contributions add up to it
    base_score: float
    band: str | None             # risk band from the calibrated score, if known
    contributions: pd.DataFrame  # index: feature; columns: value, contribution, group, phrase

    def _top(self, n: int, raises: bool) -> pd.DataFrame:
        """Largest contributions in one direction, behaviour and study groups only,
        one input per family (e.g. average score and weighted score count once)."""
        c = self.contributions
        sign = c["contribution"] >= MIN_CONTRIBUTION if raises else c["contribution"] <= -MIN_CONTRIBUTION
        eligible = c[c["group"].isin(REASON_GROUPS) & sign]
        eligible = eligible.sort_values("contribution", ascending=not raises)
        families = eligible.index.map(family_of)
        return eligible[~families.duplicated()].head(n)

    def reasons(self, n: int = 3) -> pd.DataFrame:
        """Inputs that raise the score most."""
        return self._top(n, raises=True)

    def in_favour(self, n: int = 2) -> pd.DataFrame:
        """Inputs that lower the score most."""
        return self._top(n, raises=False)

    def background(self) -> pd.DataFrame:
        return self.contributions[self.contributions["group"] == "background"]

    @property
    def sentence(self) -> str:
        return build_sentence(self)


def make_explainer(model: Pipeline) -> shap.TreeExplainer:
    return shap.TreeExplainer(model.named_steps["clf"])


def _source_columns(pre) -> list[str]:
    """The original input behind every column the preprocessor outputs."""
    source = []
    for name, trans, cols in pre.transformers_:
        if name == "remainder" or trans == "drop":
            continue
        if isinstance(trans, Pipeline) and "oh" in trans.named_steps:
            oh = trans.named_steps["oh"]
            source += [c for c, cats in zip(cols, oh.categories_) for _ in cats]
        else:
            source += list(trans.get_feature_names_out(cols))
    return source


def explain_rows(model: Pipeline, X: pd.DataFrame, explainer: shap.TreeExplainer | None = None):
    """Return (base_score, contributions, scores) for every row of X.

    contributions is a rows x inputs DataFrame in the model's input order.
    """
    features = list(model.feature_names_in_)
    pre, clf = model.named_steps["pre"], model.named_steps["clf"]
    explainer = explainer or make_explainer(model)
    Xt = pre.transform(X[features])
    source = _source_columns(pre)
    if len(source) != Xt.shape[1]:
        raise ValueError(f"Could not map {Xt.shape[1]} model columns back to inputs ({len(source)} mapped)")

    values = np.asarray(explainer.shap_values(Xt))
    positive = list(clf.classes_).index(1)
    values = values[..., positive] if values.ndim == 3 else values
    base = float(np.ravel(explainer.expected_value)[positive])
    contrib = pd.DataFrame(values, columns=source, index=X.index).T.groupby(level=0, sort=False).sum().T
    contrib = contrib.reindex(columns=features, fill_value=0.0)

    scores = pd.Series(clf.predict_proba(Xt)[:, positive], index=X.index)
    gap = np.abs(base + contrib.sum(axis=1) - scores).max()
    if gap > ADDITIVITY_TOLERANCE:
        raise ValueError(f"SHAP values do not add up to the score (gap {gap:.2e})")
    return base, contrib, scores


def _explain_chunk(model: Pipeline, X: pd.DataFrame):
    return explain_rows(model, X)


def explain_batch(model: Pipeline, X: pd.DataFrame, n_jobs: int = 1, chunk_size: int = 200):
    """explain_rows for many rows, split into chunks and run on `n_jobs` workers.

    Returns (base_score, contributions, scores) in the row order of X.
    """
    X = X[list(model.feature_names_in_)]
    if n_jobs == 1 or len(X) <= chunk_size:
        return explain_rows(model, X)
    chunks = [X.iloc[i:i + chunk_size] for i in range(0, len(X), chunk_size)]
    parts = Parallel(n_jobs=n_jobs)(delayed(_explain_chunk)(model, c) for c in chunks)
    return parts[0][0], pd.concat([p[1] for p in parts]), pd.concat([p[2] for p in parts])


def explanation_from_contributions(values: pd.Series, contributions: pd.Series, base_score: float,
                                   score: float, band: str | None = None) -> Explanation:
    """Build an Explanation from one row of inputs and its precomputed contributions."""
    table = pd.DataFrame({
        "value": values,
        "contribution": contributions.reindex(values.index).astype(float),
        "group": [group_of(f) for f in values.index],
        "phrase": [phrase(f, v) for f, v in values.items()],
    })
    table.index.name = "feature"
    return Explanation(float(score), float(base_score), band, table)


def explain_student(model: Pipeline, row, explainer: shap.TreeExplainer | None = None,
                    band: str | None = None) -> Explanation:
    """Explain one student. `row` is a one-row DataFrame or a Series of model inputs."""
    X = row.to_frame().T if isinstance(row, pd.Series) else row
    if len(X) != 1:
        raise ValueError(f"explain_student takes one row, got {len(X)}")
    X = X[list(model.feature_names_in_)].infer_objects()
    base, contrib, scores = explain_rows(model, X, explainer)
    return explanation_from_contributions(X.iloc[0], contrib.iloc[0], base, scores.iloc[0], band)


def _join(items: list[str]) -> str:
    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + " and " + items[-1]


def build_sentence(expl: Explanation, n_reasons: int = 3, course_min: float = 0.01) -> str:
    """Plain-language summary. Background inputs never appear in it."""
    parts = [f"Risk band: {expl.band}."] if expl.band else []
    reasons = expl.reasons(n_reasons)
    parts.append(f"Raising it most: {_join(list(reasons['phrase']))}." if len(reasons)
                 else "No activity, assessment or study-load input raises it noticeably.")
    favour = expl.in_favour()
    if len(favour):
        parts.append(f"Lowering it most: {_join(list(favour['phrase']))}.")
    course = expl.contributions[expl.contributions["group"] == "course"]
    course = course[course["contribution"].abs() >= course_min]
    clauses = []
    for way, rows in [("raise", course[course["contribution"] > 0]), ("lower", course[course["contribution"] < 0])]:
        if len(rows):
            clauses.append(f"{_join(list(rows['phrase']))} {way + ('s' if len(rows) == 1 else '')} it")
    if clauses:
        parts.append(f"Course-level patterns, not about the student: {' and '.join(clauses)}.")
    if len(expl.background()):
        parts.append("Background factors are shown separately and are not used as reasons.")
    else:
        parts.append("Background characteristics are not model inputs.")
    return " ".join(parts)
