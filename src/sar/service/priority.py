"""SCRUM-189: the app's priority list: main reason and suggested support for many students at once.

The same rules as the single-student views, applied to a table of SHAP contributions:
  * main reason: the actionable input (sar.recommend.interventions.ACTIONABLE_FEATURES) that raises
    the score most, by at least MIN_CONTRIBUTION, in the plain phrase of sar.explain.factors
  * suggested support: the support type that recommend() would list first, that is the one with the
    largest sum of weight x contribution over the actionable inputs that raise the score
Neither needs the similar-student pool, so the whole list can be filled from precomputed values.

The student view's wording also lives here: one plain sentence per suggested support type, built from
the Day 30 fact that matched it, the factors the chart shows and the sentence that names them, and the
definition of the similar-student group.

Recording artefacts are not reasons. In some modules the only assessment due by Day 30 carries zero
weight, so a student with an average score has no weighted score. That gap says nothing about the
student, so it gets no contribution in reasons, charts and suggestions, and is described as "submitted
work carries no weight yet".
"""

import dataclasses

import numpy as np
import pandas as pd

from sar.explain.factors import family_of, phrase
from sar.explain.shap_explain import MIN_CONTRIBUTION
from sar.recommend.interventions import ACTIONABLE_FEATURES, FEATURE_TO_INTERVENTIONS, INTERVENTIONS, label

NO_REASON = "No activity, assessment or study-load input raises the score noticeably"
NO_SUPPORT = "None matched"
WEIGHTLESS = "submitted work carries no weight yet"
# At Day 30 a Low-band student gets a suggestion only for a concrete problem: a missed assessment or
# a spell without activity.
LOW_BAND_DRIVERS = ("n_missed", "max_consecutive_inactive_days", "trailing_inactive_days")
NO_LOW_SUGGESTION = "No specific support suggested at Day 30."
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


def similar_definition(neighbours) -> str:
    """How the similar-student group is built, and what it found, in plain words."""
    where = f"in module {neighbours.module}" if neighbours.same_module else "across all modules"
    text = (f"Similar earlier students: the {neighbours.n:,} students {where} from the training presentations "
            f"({', '.join(neighbours.presentations)}) whose Day 30 activity, assessment and study-load inputs are "
            "closest to this student's. Background characteristics are not used, and students who also appear in "
            f"the validation presentation are left out. {neighbours.n_at_risk:,} of them ended at risk (Fail or "
            "Withdrawn). They are matched on inputs, not on the model's score, so this share can differ from the "
            "band's observed outcome.")
    return f"{text} {neighbours.note}" if neighbours.note else text


def _join(items: list[str]) -> str:
    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + " and " + items[-1]


def artefacts(values: pd.Series) -> list[str]:
    """Inputs whose value is a recording artefact for this student, not a fact about them."""
    weightless = pd.isna(values.get("weighted_score")) and pd.notna(values.get("mean_score"))
    return ["weighted_score"] if weightless else []


def for_reasons(expl):
    """A copy of the explanation for reasons, charts and suggestions: artefacts get no contribution and a
    phrase that says what the gap means. The full table keeps the model's own values."""
    c = expl.contributions.copy()
    for f in artefacts(c["value"]):
        c.loc[f, "contribution"] = 0.0
        c.loc[f, "phrase"] = WEIGHTLESS
    return dataclasses.replace(expl, contributions=c)


def chart_factors(expl, n_raise: int = 4, n_lower: int = 2) -> tuple[pd.DataFrame, pd.DataFrame]:
    """The actionable inputs the factor chart shows: up to `n_raise` that raise the score and up to `n_lower`
    that lower it, each by at least MIN_CONTRIBUTION, strongest first. Course-level inputs are left out."""
    c = expl.contributions[expl.contributions.index.isin(ACTIONABLE_FEATURES)]
    raising = c[c["contribution"] >= MIN_CONTRIBUTION].sort_values("contribution", ascending=False).head(n_raise)
    lowering = c[c["contribution"] <= -MIN_CONTRIBUTION].sort_values("contribution").head(n_lower)
    return raising, lowering


def verdict_sentence(raising: pd.DataFrame, lowering: pd.DataFrame) -> str:
    """Plain sentences on the main drivers, naming only factors the chart shows (one per family)."""
    up = list(raising[~raising.index.map(family_of).duplicated()]["phrase"].head(2))
    down = list(lowering["phrase"].head(1))
    first = (f"The risk score is raised most by {_join(up)}." if up
             else "No activity, assessment or study-load input raises the risk score noticeably.")
    second = (f"It is lowered most by {down[0]}." if down
              else "No activity, assessment or study-load input lowers it noticeably.")
    return f"{first} {second}"


def primary_for(rec):
    """The suggestion to show first, or None. For a Low-band student only a missed assessment or a spell
    without activity leads to a suggestion."""
    if not rec.suggestions:
        return None
    first = rec.suggestions[0]
    if rec.optional and first.drivers[0] not in LOW_BAND_DRIVERS:
        return None
    return first


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
    contributions = contributions.copy()
    if {"weighted_score", "mean_score"} <= set(values.columns) and "weighted_score" in contributions.columns:
        weightless = (values["weighted_score"].isna() & values["mean_score"].notna()).reindex(contributions.index)
        contributions.loc[weightless.to_numpy(), "weighted_score"] = 0.0          # see artefacts()
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


def support_counts(supports: pd.Series) -> pd.Series:
    """Flagged students per support type (by label, in the order of INTERVENTIONS, zeros kept) from the
    suggested_support column; students with no matched type are counted under NO_SUPPORT, last."""
    order = [label(name) for name in INTERVENTIONS] + [NO_SUPPORT]
    return supports.value_counts().reindex(order, fill_value=0).astype(int)


def _kinds(features) -> list[str]:
    """The kinds of input (REASON_KINDS) behind `features`, in order of first appearance, in lower case."""
    out = []
    for f in features:
        kind = REASON_KINDS.get(f, f).lower()
        if kind not in out:
            out.append(kind)
    return out


def never_first(intervention: str) -> str:
    """Why a support type can be suggested by the rules yet be nobody's primary suggestion: the inputs it is
    matched from, and the types that share those inputs, from FEATURE_TO_INTERVENTIONS."""
    own = [f for f, w in FEATURE_TO_INTERVENTIONS.items() if intervention in w]
    others = sorted({label(o) for f in own for o in FEATURE_TO_INTERVENTIONS[f] if o != intervention})
    return (f"Matched only from {_join(_kinds(own))}, which also point to {_join(others)}. For every flagged "
            "student in view another type has the larger total, so it is never listed first.")
