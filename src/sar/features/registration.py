"""SCRUM-174 (part 1): registration features known at or before the cutoff.

Uses date_registration only. date_unregistration is an outcome field and is never used.
When an engagement frame (indexed by the 3 key columns) is passed, its inactivity
columns are carried into the output so one table holds all engagement-style features.
"""

import pandas as pd

from sar.config import CUTOFF_DAY

KEY = ["code_module", "code_presentation", "id_student"]
CARRY = ["max_consecutive_inactive_days", "trailing_inactive_days"]


def registration_features(
    student_registration: pd.DataFrame,
    engagement: pd.DataFrame | None = None,
    cutoff_day: int = CUTOFF_DAY,
) -> pd.DataFrame:
    """Columns: reg_lead_days (days registered before start), registered_late (after Day 0),
    registered_after_cutoff, reg_date_missing. Indexed by the 3 key columns."""
    reg = student_registration[KEY + ["date_registration"]].drop_duplicates(KEY).set_index(KEY)
    d = reg["date_registration"]
    out = pd.DataFrame(index=reg.index)
    out["reg_lead_days"] = -d
    out["registered_late"] = (d > 0).astype(int)
    out["registered_after_cutoff"] = (d > cutoff_day).astype(int)
    out["reg_date_missing"] = d.isna().astype(int)
    if engagement is not None:
        cols = [c for c in CARRY if c in engagement.columns]
        out = out.join(engagement[cols], how="left")
    return out
