"""SCRUM-188: support types and which risk drivers point to each.

The recommender is rule-based. Five support types; each actionable input maps to one or two of
them, with weights that sum to 1. A support type's match score for a student is the sum of
weight x SHAP contribution over that student's inputs that raise the risk score.

Actionable inputs (ACTIONABLE) are the model inputs a support contact can respond to, in four
categories: engagement (site activity and inactivity), submissions (work submitted, missed and its
scores), timing (submission and registration timing) and workload (credits studied).

Rules, checked when the module is imported
  * only actionable inputs are mapped
  * protected (background) factors, course-level inputs and fixed history such as previous
    attempts are never mapped, so they never drive a suggestion
  * financial aid referral is matched from workload and registration timing only, never from
    the deprivation band

The mapping is an editable assumption for staff to review. It is not learned from data: OULAD
contains no record of interventions.
"""

from sar.config import PROTECTED_FEATURES
from sar.explain.factors import GROUPS, REASON_GROUPS

INTERVENTIONS = {
    "tutoring": ("Tutoring", "Subject help from a tutor on the module content and assessments."),
    "peer_tutoring": ("Peer tutoring", "Study sessions with a more experienced student on the same module."),
    "mentoring": ("Mentoring", "Regular sessions with a mentor on study planning, workload and progress."),
    "weekly_checkins": ("Weekly check-ins", "A short weekly contact to keep up with activity and deadlines."),
    "financial_aid_referral": ("Financial aid referral",
                               "Signposting to the student funding and support team."),
}

_SCORES = {"tutoring": 0.6, "peer_tutoring": 0.4}
_SUBMISSION = {"weekly_checkins": 0.5, "tutoring": 0.5}
_TIMING = {"weekly_checkins": 0.6, "mentoring": 0.4}
_ACTIVITY = {"weekly_checkins": 0.6, "peer_tutoring": 0.4}
_INACTIVITY = {"weekly_checkins": 0.6, "mentoring": 0.4}
_REGISTRATION = {"financial_aid_referral": 0.5, "mentoring": 0.5}

ACTIONABLE = {
    "engagement": ("total_clicks", "active_days", "avg_clicks_per_active_day", "clicks_last_7d", "clicks_prev_7d",
                   "click_trend", "first_active_day", "max_consecutive_inactive_days", "trailing_inactive_days"),
    "submissions": ("n_submitted", "n_missed", "mean_score", "weighted_score", "score_trend"),
    "timing": ("n_late", "mean_delay_days", "reg_lead_days", "registered_late", "registered_after_cutoff",
               "reg_date_missing"),
    "workload": ("studied_credits",),
}
ACTIONABLE_FEATURES = tuple(f for features in ACTIONABLE.values() for f in features)

FEATURE_TO_INTERVENTIONS = {
    "mean_score": _SCORES,
    "weighted_score": _SCORES,
    "score_trend": _SCORES,
    "n_submitted": _SUBMISSION,
    "n_missed": _SUBMISSION,
    "n_late": _TIMING,
    "mean_delay_days": _TIMING,
    "total_clicks": _ACTIVITY,
    "active_days": _ACTIVITY,
    "avg_clicks_per_active_day": _ACTIVITY,
    "clicks_last_7d": _ACTIVITY,
    "clicks_prev_7d": _ACTIVITY,
    "click_trend": _ACTIVITY,
    "first_active_day": _INACTIVITY,
    "max_consecutive_inactive_days": _INACTIVITY,
    "trailing_inactive_days": _INACTIVITY,
    "studied_credits": {"mentoring": 0.5, "financial_aid_referral": 0.5},
    "reg_lead_days": _REGISTRATION,
    "registered_late": _REGISTRATION,
    "registered_after_cutoff": _REGISTRATION,
    "reg_date_missing": _REGISTRATION,
}


def label(intervention: str) -> str:
    return INTERVENTIONS[intervention][0]


def _check_mapping() -> None:
    if set(ACTIONABLE_FEATURES) & set(PROTECTED_FEATURES):
        raise ValueError("A protected factor is listed as actionable")
    for feature, weights in FEATURE_TO_INTERVENTIONS.items():
        if feature not in ACTIONABLE_FEATURES:
            raise ValueError(f"{feature!r} is not an actionable input and must not be mapped")
        if GROUPS.get(feature) not in REASON_GROUPS:
            raise ValueError(f"{feature!r} is not an activity, assessment or study-load input and must not be mapped")
        if set(weights) - set(INTERVENTIONS) or abs(sum(weights.values()) - 1) > 1e-9:
            raise ValueError(f"Weights for {feature!r} must name known interventions and sum to 1")


_check_mapping()
