"""SCRUM-189: the app's priority list: main reason and suggested support for many students at once.

The same rules as the single-student views, applied to a table of SHAP contributions:
  * main reason: the actionable input (sar.recommend.interventions.ACTIONABLE_FEATURES) that raises
    the score most, by at least MIN_CONTRIBUTION, in the plain phrase of sar.explain.factors
  * suggested support: the support type that recommend() would list first, that is the one with the
    largest sum of weight x contribution over the actionable inputs that raise the score
Neither needs the similar-student pool, so the whole list can be filled from precomputed values.

The student view's wording also lives here: one plain sentence per suggested support type, built from
the Day 30 fact that matched it, and a short line on similar earlier students.
"""

import numpy as np
import pandas as pd

from sar.explain.factors import phrase
from sar.explain.shap_explain import MIN_CONTRIBUTION
from sar.recommend.interventions import ACTIONABLE_FEATURES, FEATURE_TO_INTERVENTIONS, label

NO_REASON = "No activity, assessment or study-load input raises the score noticeably"
NO_SUPPORT = "None matched"
# What each support type offers, written to follow the Day 30 fact that matched it.
SUPPORT_ACTIONS = {
    "tutoring": "a tutor can go over the module content and the next assessment with them.",
    "peer_tutoring": "study sessions with an experienced student on the module can help them keep up.",
    "mentoring": "a mentor can help them plan their study time and workload.",
    "weekly_checkins": "a short weekly check-in can help them keep up with activity and deadlines.",
    "financial_aid_referral": "the student funding and support team can advise if study load or registration "
                              "is a barrier.",
}


def _capital(text: str) -> str:
    return text[:1].upper() + text[1:]


def support_sentence(intervention: str, fact: str) -> str:
    """One plain sentence: the Day 30 fact that matched a support type, then what that support offers."""
    return f"{_capital(fact)}; {SUPPORT_ACTIONS[intervention]}"


def similar_line(n: int, n_at_risk: int) -> str:
    return f"Among {n:,} similar earlier students, {n_at_risk:,} ended at risk."


def _join(items: list[str]) -> str:
    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + " and " + items[-1]


def verdict_sentence(expl) -> str:
    """One plain sentence on the main drivers, from the explanation's reasons and inputs in favour."""
    raising, lowering = list(expl.reasons(2)["phrase"]), list(expl.in_favour(1)["phrase"])
    if raising:
        text = f"The risk score is raised most by {_join(raising)}"
        return text + (f", and lowered most by {lowering[0]}." if lowering else ".")
    text = "No activity, assessment or study-load input raises the risk score noticeably"
    return text + (f"; it is lowered most by {lowering[0]}." if lowering else ".")


# Kinds of main reason, for counting the flagged students: a main reason names one input, and the
# inputs that describe the same kind of difficulty are counted together.
REASON_KINDS = {
    **dict.fromkeys(["mean_score", "weighted_score", "score_trend"], "Assessment scores"),
    **dict.fromkeys(["n_missed", "n_submitted"], "Assessments missed"),
    **dict.fromkeys(["max_consecutive_inactive_days", "trailing_inactive_days"], "Inactive spells"),
    **dict.fromkeys(["total_clicks", "active_days", "avg_clicks_per_active_day", "clicks_last_7d", "clicks_prev_7d",
                     "click_trend", "first_active_day"], "Site activity"),
    **dict.fromkeys(["n_late", "mean_delay_days"], "Submission timing"),
    **dict.fromkeys(["reg_lead_days", "registered_late", "registered_after_cutoff", "reg_date_missing"],
                    "Registration timing"),
    "studied_credits": "Study load",
}
NO_KIND = "No single input stands out"
OTHER = "Other"


def reason_counts(factors: pd.Series, min_share: float = 0.05) -> pd.Series:
    """Students per kind of main reason, largest first, from main_factor values (missing: no input stands
    out). Kinds under `min_share` of the students are counted together as "Other", shown last."""
    kinds = factors.map(lambda f: REASON_KINDS.get(f, OTHER) if isinstance(f, str) else NO_KIND)
    counts = kinds.value_counts()
    small = (counts < min_share * len(factors)) | (counts.index == OTHER)
    out = counts[~small]
    if small.any():
        out = pd.concat([out, pd.Series({OTHER: int(counts[small].sum())})])
    return out


# Shorter wording of the longer phrases, for one-line rows in the priority table.
_SHORTER = [(" on the course site", ""), (" just before Day", " before Day"),
            (" before the module started", " before the start"), (" in the last week before Day 30", " last week")]


def short_reason(text: str) -> str:
    if text == NO_REASON:
        return NO_KIND
    for long, short in _SHORTER:
        text = text.replace(long, short)
    return _capital(text)


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
