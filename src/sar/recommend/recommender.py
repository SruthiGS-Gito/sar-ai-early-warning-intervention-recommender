"""SCRUM-188: suggest up to three support types for a student, with a reason and context.

The recommender is rule-based: it matches the model's risk drivers to support types through the
fixed, reviewable mapping in sar.recommend.interventions. Nothing is learned from outcomes.

`recommend(expl, row, pool)` takes the student's SHAP Explanation (SCRUM-187), their inputs and a
NeighbourPool, and returns a Recommendation:
  * each support type is scored from the actionable inputs (engagement, submissions, timing,
    workload) that raise the student's risk score; protected factors, course-level inputs and
    previous attempts are never used
  * each suggestion names its linked drivers in plain language, preferring drivers that a
    higher-ranked suggestion has not already named
  * context comes from the most similar students in the training presentations: how many ended
    At Risk or On Track, and the median of the main driver in each group, with group sizes

OULAD contains no record of interventions. The similar-student context is therefore descriptive:
it shows patterns among similar past students and is not evidence that any support is effective.
Nothing here says a support type works or what it would change.
"""

from dataclasses import dataclass

import pandas as pd

from sar.explain.factors import family_of, phrase
from sar.explain.shap_explain import MIN_CONTRIBUTION
from sar.recommend.interventions import FEATURE_TO_INTERVENTIONS, label
from sar.recommend.neighbours import Neighbours, NeighbourPool

N_SUGGESTIONS = 3
MIN_GROUP = 10  # smallest outcome group whose median is shown
NOTE = ("Suggestions are rule-based: they match the model's risk drivers to support types. Figures for similar "
        "past students are descriptive patterns and do not show whether any support changes outcomes: OULAD "
        "records no interventions. Background characteristics are never used.")
LOW_BAND = "Low"


@dataclass
class Suggestion:
    intervention: str
    label: str
    score: float
    drivers: list[str]
    reason: str
    evidence: str


@dataclass
class Recommendation:
    suggestions: list[Suggestion]
    neighbours: Neighbours
    band: str | None

    @property
    def optional(self) -> bool:
        return self.band == LOW_BAND

    def text(self) -> str:
        lines = [self.neighbours.summary()]
        if self.optional:
            lines.append("Low risk band, so these suggestions are optional.")
        if not self.suggestions:
            lines.append("No activity, assessment or study-load input raises the score noticeably, "
                         "so no support type is matched.")
        for i, s in enumerate(self.suggestions, 1):
            lines.append(f"{i}. {s.reason} {s.evidence}")
        lines.append(NOTE)
        return "\n".join(lines)


def driver_scores(expl) -> pd.DataFrame:
    """One row per (support type, input): weight x contribution for inputs that raise the score."""
    c = expl.contributions
    rows = [
        {"intervention": name, "feature": f, "score": w * c.loc[f, "contribution"]}
        for f, weights in FEATURE_TO_INTERVENTIONS.items()
        if f in c.index and c.loc[f, "contribution"] >= MIN_CONTRIBUTION
        for name, w in weights.items()
    ]
    return pd.DataFrame(rows, columns=["intervention", "feature", "score"])


def _evidence(feature: str, neighbours: Neighbours) -> str:
    """Median of one driver among similar students who ended On Track and At Risk, in plain words."""
    g = neighbours.frame
    on = g.loc[g["at_risk"] == 0, feature].dropna()
    risk = g.loc[g["at_risk"] == 1, feature].dropna()
    if len(on) < MIN_GROUP or len(risk) < MIN_GROUP:
        return (f"Too few similar students in one outcome group to compare this driver "
                f"(On Track {len(on)}, At Risk {len(risk)}).")
    p_on, p_risk = phrase(feature, round(on.median())), phrase(feature, round(risk.median()))
    if p_on == p_risk:
        return (f"Similar students had the same median in both groups ({p_on}; On Track n = {len(on)}, "
                f"At Risk n = {len(risk)}), so this driver does not separate them here.")
    return f"Medians among similar students: On Track (n = {len(on)}) {p_on}; At Risk (n = {len(risk)}) {p_risk}."


def recommend(expl, row, pool: NeighbourPool, n: int = N_SUGGESTIONS, same_module: bool = True) -> Recommendation:
    neighbours = pool.query(row, same_module=same_module)
    scores = driver_scores(expl)
    suggestions, used = [], set()
    if len(scores):
        totals = scores.groupby("intervention")["score"].sum().sort_values(ascending=False)
        for name, total in totals.head(n).items():
            part = scores[scores["intervention"] == name].sort_values("score", ascending=False)
            part = part[~part["feature"].map(family_of).duplicated()]
            fresh = ~part["feature"].map(family_of).isin(used)
            drivers = pd.concat([part[fresh], part[~fresh]])["feature"].head(2).tolist()
            used.update(family_of(f) for f in drivers)
            phrases = [expl.contributions.loc[f, "phrase"] for f in drivers]
            linked = phrases[0] if len(phrases) == 1 else " and ".join(phrases)
            suggestions.append(Suggestion(
                intervention=name, label=label(name), score=float(total), drivers=drivers,
                reason=f"{label(name)}: linked to {linked}.", evidence=_evidence(drivers[0], neighbours),
            ))
    return Recommendation(suggestions, neighbours, expl.band)
