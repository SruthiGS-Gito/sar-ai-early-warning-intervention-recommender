"""SCRUM-189: the app's priority list: main reason and suggested support for many students at once.

The same rules as the single-student views, applied to a table of SHAP contributions:
  * main reason: the actionable input (sar.recommend.interventions.ACTIONABLE_FEATURES) that raises
    the score most, by at least MIN_CONTRIBUTION, in the plain phrase of sar.explain.factors
  * suggested support: the support type that recommend() would list first, that is the one with the
    largest sum of weight x contribution over the actionable inputs that raise the score
Neither needs the similar-student pool, so the whole list can be filled from precomputed values.
"""

import numpy as np
import pandas as pd

from sar.explain.factors import phrase
from sar.explain.shap_explain import MIN_CONTRIBUTION
from sar.recommend.interventions import ACTIONABLE_FEATURES, FEATURE_TO_INTERVENTIONS, label

NO_REASON = "No activity, assessment or study-load input raises the score noticeably"
NO_SUPPORT = "None matched"


def priority_details(values: pd.DataFrame, contributions: pd.DataFrame) -> pd.DataFrame:
    """One row per row of `contributions` (same index): main_factor, main_reason and suggested_support.
    `values` holds the students' inputs with the same index."""
    feats = [f for f in contributions.columns if f in ACTIONABLE_FEATURES]     # model input order, as in idxmax
    c = contributions[feats].to_numpy(dtype=float)
    raising = np.where(c >= MIN_CONTRIBUTION, c, 0.0)

    has_reason = raising.max(axis=1, initial=0.0) > 0
    top = raising.argmax(axis=1)
    factors = [feats[j] if ok else None for j, ok in zip(top, has_reason)]
    reasons = [phrase(f, values.at[i, f]) if f else NO_REASON for i, f in zip(contributions.index, factors)]

    # Scores built and summed in the order recommend() uses, and ranked with the same pandas sort,
    # so that exact ties between support types resolve the same way.
    parts = []
    for f, weights in FEATURE_TO_INTERVENTIONS.items():
        if f not in contributions.columns:
            continue
        cf = contributions[f].to_numpy(dtype=float)
        rows = np.flatnonzero(cf >= MIN_CONTRIBUTION)
        for name, w in weights.items():
            parts.append(pd.DataFrame({"row": rows, "intervention": name, "score": w * cf[rows]}))
    support = [NO_SUPPORT] * len(contributions)
    if parts:
        totals = pd.concat(parts, ignore_index=True).groupby(["row", "intervention"])["score"].sum()
        for r, t in totals.groupby(level="row"):
            support[r] = label(t.droplevel("row").sort_values(ascending=False).index[0])
    return pd.DataFrame({"main_factor": factors, "main_reason": reasons, "suggested_support": support},
                        index=contributions.index)


def selected_row(rows: list, selection: dict | None):
    """The item of `rows` picked in a single-row dataframe selection, or None."""
    picked = (selection or {}).get("selection", {}).get("rows", [])
    return rows[picked[0]] if picked and 0 <= picked[0] < len(rows) else None


def snapshot(matrix: pd.DataFrame, presentation: str, module: str, student: int, features: dict) -> pd.DataFrame:
    """The student's value and the module median (same presentation) for each feature in `features`
    (feature -> label), with their difference."""
    part = matrix[(matrix["code_presentation"] == presentation) & (matrix["code_module"] == module)]
    row = part[part["id_student"] == student].iloc[0]
    out = pd.DataFrame({"label": list(features.values()),
                        "value": [float(row[f]) for f in features],
                        "median": [float(part[f].median()) for f in features]}, index=list(features))
    out["difference"] = out["value"] - out["median"]
    return out
