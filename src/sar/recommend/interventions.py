"""SCRUM-188: support types and which risk drivers point to each.

Five support types. Each activity, assessment or study-load input maps to one or two of them,
with weights that sum to 1. A support type's match score for a student is the sum of
weight x SHAP contribution over that student's inputs that raise the risk score.

Rules
  * only behaviour and study inputs are mapped (sar.explain.factors.REASON_GROUPS)
  * background factors and course-level inputs are never mapped, so they never drive a suggestion;
    this is checked when the module is imported
  * financial aid referral is matched from study load and registration timing only, never from
    the deprivation band

The mapping is an editable assumption for staff to review; it is not learned from data
(OULAD records no interventions).
"""

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
    "num_of_prev_attempts": {"mentoring": 0.6, "tutoring": 0.4},
    "studied_credits": {"mentoring": 0.5, "financial_aid_referral": 0.5},
    "reg_lead_days": _REGISTRATION,
    "registered_late": _REGISTRATION,
    "registered_after_cutoff": _REGISTRATION,
    "reg_date_missing": _REGISTRATION,
}


def label(intervention: str) -> str:
    return INTERVENTIONS[intervention][0]


def _check_mapping() -> None:
    for feature, weights in FEATURE_TO_INTERVENTIONS.items():
        if GROUPS.get(feature) not in REASON_GROUPS:
            raise ValueError(f"{feature!r} is not an activity, assessment or study-load input and must not be mapped")
        if set(weights) - set(INTERVENTIONS) or abs(sum(weights.values()) - 1) > 1e-9:
            raise ValueError(f"Weights for {feature!r} must name known interventions and sum to 1")


_check_mapping()
