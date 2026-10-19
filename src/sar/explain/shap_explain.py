"""SCRUM-187: per-student SHAP explanations for a tree model pipeline.

`explain_student(model, row, threshold)` returns an Explanation: the risk score, each input's
contribution to it (SHAP value, one-hot columns added back into their original input), and
a plain-language sentence.

How the contributions are computed
  * shap.TreeExplainer with its default tree_path_dependent mode: the expected score comes from
    the training data stored in the trees, so no other data (and no validation data) is used
  * contributions plus the starting score add up to the model's score; this is checked
  * background inputs (sar.explain.factors.BACKGROUND) are kept in the table but never named
    as a main reason in the sentence

Wording: the score is a model output, not a probability, and contributions describe the model,
not causes. Sentences say what "raises the score", never what caused an outcome.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
import shap
from sklearn.pipeline import Pipeline

from sar.explain.factors import REASON_GROUPS, family_of, group_of, phrase

ADDITIVITY_TOLERANCE = 1e-6
# Smallest change in score worth naming in a sentence (half a point on a 0-100 scale).
MIN_CONTRIBUTION = 0.005


@dataclass
class Explanation:
    score: float
    base_score: float
    threshold: float | None
    contributions: pd.DataFrame  # index: feature; columns: value, contribution, group, phrase

    @property
    def flagged(self) -> bool | None:
        return None if self.threshold is None else self.score >= self.threshold

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


def explain_student(model: Pipeline, row, threshold: float | None = None,
                    explainer: shap.TreeExplainer | None = None) -> Explanation:
    """Explain one student. `row` is a one-row DataFrame or a Series of model inputs."""
    X = row.to_frame().T if isinstance(row, pd.Series) else row
    if len(X) != 1:
        raise ValueError(f"explain_student takes one row, got {len(X)}")
    X = X[list(model.feature_names_in_)].infer_objects()
    base, contrib, scores = explain_rows(model, X, explainer)
    values = X.iloc[0]
    table = pd.DataFrame({
        "value": values,
        "contribution": contrib.iloc[0],
        "group": [group_of(f) for f in values.index],
        "phrase": [phrase(f, v) for f, v in values.items()],
    })
    table.index.name = "feature"
    return Explanation(float(scores.iloc[0]), base, threshold, table)


def _join(items: list[str]) -> str:
    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + " and " + items[-1]


def build_sentence(expl: Explanation, n_reasons: int = 3, course_min: float = 0.01) -> str:
    """Plain-language summary. Background inputs never appear in it."""
    if expl.threshold is None:
        parts = [f"Risk score {expl.score:.2f}."]
    else:
        group = "At Risk" if expl.flagged else "On Track"
        parts = [f"Risk score {expl.score:.2f} (provisional threshold {expl.threshold:.2f}): {group}."]
    reasons = expl.reasons(n_reasons)
    parts.append(f"Raising it most: {_join(list(reasons['phrase']))}." if len(reasons)
                 else "No activity, assessment or study-load input raises it noticeably.")
    favour = expl.in_favour()
    if len(favour):
        parts.append(f"Lowering it most: {_join(list(favour['phrase']))}.")
    course = expl.contributions[expl.contributions["group"] == "course"]
    course = course[course["contribution"].abs() >= course_min]
    for way, rows in [("raise", course[course["contribution"] > 0]), ("lower", course[course["contribution"] < 0])]:
        if len(rows):
            verb = way + ("s" if len(rows) == 1 else "")
            parts.append(f"Course-level patterns, not about the student: {_join(list(rows['phrase']))} "
                         f"also {verb} it.")
    parts.append("Background factors are shown separately and are not used as reasons.")
    return " ".join(parts)
