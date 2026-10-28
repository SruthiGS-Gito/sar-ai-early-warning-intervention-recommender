"""SCRUM-187: which group each model input belongs to, and a plain-language phrase for it.

Groups
  * behaviour  - site activity and assessments up to the cutoff (can be a main reason)
  * study      - study load and previous attempts (can be a main reason)
  * course     - the module and its assessment calendar (course-level patterns; mentioned
                 separately, never a main reason)
  * background - gender, region, highest education, deprivation band, age band, disability.
                 Shown separately for transparency. Never a main reason, never drives an
                 intervention.

`phrase(feature, value)` describes the student's value for one input, e.g.
"a longest run of 12 days without activity". Missing values get their own wording.
"""

import math

from sar.config import CUTOFF_DAY, PROTECTED_FEATURES

BACKGROUND = PROTECTED_FEATURES
COURSE = ("code_module", "n_assessments_due")
STUDY = ("num_of_prev_attempts", "studied_credits")
BEHAVIOUR = (
    "total_clicks", "active_days", "avg_clicks_per_active_day", "clicks_last_7d", "clicks_prev_7d",
    "click_trend", "first_active_day", "max_consecutive_inactive_days", "trailing_inactive_days",
    "n_submitted", "mean_score", "weighted_score", "score_trend", "mean_delay_days", "n_late",
    "n_missed", "reg_lead_days", "registered_late", "registered_after_cutoff",
    "reg_date_missing",
)
GROUPS = {
    **{f: "behaviour" for f in BEHAVIOUR},
    **{f: "study" for f in STUDY},
    **{f: "course" for f in COURSE},
    **{f: "background" for f in BACKGROUND},
}
REASON_GROUPS = ("behaviour", "study")
# Inputs that describe the same idea; a sentence names at most one input per family.
FAMILIES = {
    "mean_score": "score", "weighted_score": "score",
    "max_consecutive_inactive_days": "inactivity", "trailing_inactive_days": "inactivity",
    "reg_lead_days": "registration", "registered_late": "registration",
    "registered_after_cutoff": "registration", "reg_date_missing": "registration",
}


def family_of(feature: str) -> str:
    return FAMILIES.get(feature, feature)


def group_of(feature: str) -> str:
    if feature not in GROUPS:
        raise KeyError(f"No group for feature {feature!r}; add it to sar.explain.factors")
    return GROUPS[feature]


def _missing(v) -> bool:
    return v is None or (isinstance(v, float) and math.isnan(v))


def _n(v, one: str, many: str | None = None) -> str:
    """'no days', '1 day', '3 days', '1,204 clicks'."""
    v = round(float(v))
    word = one if v == 1 else (many or one + "s")
    return f"{v:,} {word}" if v else f"no {word}"


def _signed(v, unit: str, up: str, down: str, same: str) -> str:
    """`up`/`down` hold one {} for the size of the change, e.g. 'scores up by {}'."""
    v = float(v)
    if round(v) == 0:
        return same
    return (up if v > 0 else down).format(_n(abs(v), unit))


def _first_active(v, c):
    if v > c:
        return f"no activity on the course site by Day {c}"
    if v < 0:
        return f"first active {_n(-v, 'day')} before the module started"
    return f"first active on day {v:.0f}"


def _reg_lead(v):
    return _signed(v, "day", "registered {} before the start", "registered {} after the start",
                   "registered on the start day")


_P = {
    "total_clicks": lambda v, c: f"{_n(v, 'click')} on the course site by Day {c}",
    "active_days": lambda v, c: f"{_n(v, 'active day')} by Day {c}",
    "avg_clicks_per_active_day": lambda v, c: (f"about {_n(v, 'click')} per active day" if round(v)
                                               else "no clicks on any active day"),
    "clicks_last_7d": lambda v, c: f"{_n(v, 'click')} in the last week before Day {c}",
    "clicks_prev_7d": lambda v, c: f"{_n(v, 'click')} in the week before that",
    "click_trend": lambda v, c: _signed(v, "click", "weekly clicks up by {} in the last week",
                                        "weekly clicks down by {} in the last week",
                                        "no change in weekly clicks"),
    "first_active_day": _first_active,
    "max_consecutive_inactive_days": lambda v, c: (f"a longest run of {_n(v, 'day')} without activity"
                                                   if round(v) else f"activity on every day up to Day {c}"),
    "trailing_inactive_days": lambda v, c: f"{_n(v, 'day')} without activity just before Day {c}",
    "n_submitted": lambda v, c: f"{_n(v, 'assessment')} submitted by Day {c}",
    "mean_score": lambda v, c: f"an average score of {v:.0f}",
    "weighted_score": lambda v, c: f"a weighted average score of {v:.0f}",
    "score_trend": lambda v, c: _signed(v, "point", "scores up by {}", "scores down by {}",
                                        "no change in score"),
    "mean_delay_days": lambda v, c: _signed(v, "day", "submitting {} late on average",
                                            "submitting {} early on average",
                                            "submitting on the due date on average"),
    "n_late": lambda v, c: _n(v, "late submission"),
    "n_assessments_due": lambda v, c: f"{_n(v, 'assessment')} due by Day {c}",
    "n_missed": lambda v, c: f"{_n(v, 'assessment')} missed",
    "reg_lead_days": lambda v, c: _reg_lead(v),
    "registered_late": lambda v, c: ("registered after the module started" if v
                                     else "registered before the module started"),
    "registered_after_cutoff": lambda v, c: f"registered after Day {c}" if v else f"registered by Day {c}",
    "reg_date_missing": lambda v, c: "no registration date on record" if v else "a registration date on record",
    "num_of_prev_attempts": lambda v, c: ("a first attempt at this module" if v == 0
                                          else f"{_n(v, 'previous attempt')} at this module"),
    "studied_credits": lambda v, c: f"{_n(v, 'credit')} being studied",
    "code_module": lambda v, c: f"being on module {v}",
    "gender": lambda v, c: f"gender ({v})",
    "region": lambda v, c: f"region ({v})",
    "highest_education": lambda v, c: f"highest education ({v})",
    "imd_band": lambda v, c: f"deprivation band ({v})",
    "age_band": lambda v, c: f"age band ({v})",
    "disability": lambda v, c: f"disability recorded ({v})",
}

_MISSING = {
    "mean_score": "no assessment score yet",
    "weighted_score": "no weighted score yet",
    "score_trend": "no score trend yet",
    "mean_delay_days": "no submission timing yet",
    "reg_lead_days": "no registration date on record",
}


def phrase(feature: str, value, cutoff_day: int = CUTOFF_DAY) -> str:
    if feature not in _P:
        raise KeyError(f"No phrase for feature {feature!r}; add it to sar.explain.factors")
    if _missing(value):
        return _MISSING.get(feature, f"no value recorded for {feature.replace('_', ' ')}")
    return _P[feature](value, cutoff_day)
