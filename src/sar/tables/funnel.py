"""Ticket 207 (SCRUM-177): retention funnel.

A retrospective, descriptive view of each module and presentation. The stages are nested: each
counts the students of the previous stage who also meet its condition.

  1. registered       every enrolment in studentInfo
  2. still registered not unregistered on or before Day 30 (the Day 30 model population)
  3. active in VLE    at least one day of VLE activity by Day 30
  4. submission check at least one non-exam assessment submitted by Day 30, or none was due
  5. not withdrawn    final outcome is not Withdrawn
  6. passed           final outcome is Pass or Distinction

Stage 4 does not apply to a module presentation with no assessment due by Day 30 (as recorded in
the matrix's n_assessments_due): there it would measure the course schedule, not the student. Its
row is marked `applicable` = False and carries the stage 3 count forward, so stages 5 and 6 are
not shrunk by the schedule. "All modules" rows add up the module rows, carried counts included.

Stages 5 and 6 use final outcomes. They describe what happened after Day 30 and are not model
inputs. Counts from 1 to MIN_CELL - 1 are blanked and marked `suppressed`, together with enough
other modules that the "All modules" total cannot reveal them (see _suppress); zero is shown.
"""

import pandas as pd

from sar.config import CUTOFF_DAY
from sar.data.target import KEY, ON_TRACK, day30_population
from sar.service.views import MIN_CELL

ALL_MODULES = "All modules"
STAGES = (
    ("registered", "Registered"),
    ("still_registered", f"Still registered after Day {CUTOFF_DAY}"),
    ("active_vle", f"Active in the VLE by Day {CUTOFF_DAY}"),
    ("submitted", f"Met the Day {CUTOFF_DAY} submission check (submitted, or none was due)"),
    ("not_withdrawn", "Did not withdraw"),
    ("passed", "Passed"),
)
SCHEDULE_STAGE = "submitted"
COLUMNS = ("code_module", "code_presentation", "stage_order", "stage_key", "stage", "students",
           "share_of_registered", "share_of_previous", "applicable", "suppressed")


def stage_flags(student_info: pd.DataFrame, registration: pd.DataFrame, matrix: pd.DataFrame,
                cutoff_day: int = CUTOFF_DAY) -> pd.DataFrame:
    """One row per enrolment with a boolean column per stage (nested), and `submission_due`: whether
    any assessment was due by the cutoff in that module presentation. Kept in memory only."""
    info = student_info[KEY + ["final_result"]].drop_duplicates(KEY)
    pop = day30_population({"studentInfo": info, "studentRegistration": registration}, cutoff_day)
    df = info.merge(pop.assign(still_registered=True), on=KEY, how="left")
    df = df.merge(matrix[KEY + ["active_days", "n_submitted"]], on=KEY, how="left")
    due = matrix.groupby(["code_module", "code_presentation"])["n_assessments_due"].max().gt(0)
    df = df.merge(due.rename("submission_due").reset_index(), on=["code_module", "code_presentation"], how="left")
    df["submission_due"] = df["submission_due"].fillna(True).astype(bool)
    df["registered"] = True
    df["still_registered"] = df["still_registered"].fillna(False).astype(bool)
    df["active_vle"] = df["still_registered"] & (df["active_days"].fillna(0) > 0)
    df["submitted"] = df["active_vle"] & ((df["n_submitted"].fillna(0) > 0) | ~df["submission_due"])
    df["not_withdrawn"] = df["submitted"] & (df["final_result"] != "Withdrawn")
    df["passed"] = df["not_withdrawn"] & df["final_result"].isin(ON_TRACK)
    return df[KEY + [k for k, _ in STAGES] + ["submission_due"]]


def _suppress(counts: pd.DataFrame, min_cell: int) -> pd.Series:
    """Which counts to hide. Primary: counts from 1 to min_cell - 1, and every stage of a group with
    fewer than min_cell registrations. Complementary: within a presentation and stage, the "All
    modules" total minus the shown modules would reveal the hidden ones, so further modules are
    hidden, smallest non-zero count first, until at least two are hidden and they hold at least
    min_cell students together."""
    group = [counts["code_presentation"], counts["code_module"]]
    registered = counts["students"].groupby(group).transform(lambda s: s.iloc[0])
    hidden = counts["students"].between(1, min_cell - 1) | registered.between(1, min_cell - 1)
    modules = counts["code_module"] != ALL_MODULES
    for _, cell in counts[modules].groupby(["code_presentation", "stage_order"]):
        if not hidden[cell.index].any():
            continue
        for i in cell[~hidden[cell.index] & (cell["students"] > 0)].sort_values("students").index:
            h = cell.index[hidden[cell.index]]
            if len(h) >= 2 and counts.loc[h, "students"].sum() >= min_cell:
                break
            hidden[i] = True
    return hidden


def retention_funnel(student_info: pd.DataFrame, registration: pd.DataFrame, matrix: pd.DataFrame,
                     min_cell: int = MIN_CELL, cutoff_day: int = CUTOFF_DAY) -> pd.DataFrame:
    """Six stage rows per module and presentation, plus "All modules" rows per presentation."""
    flags = stage_flags(student_info, registration, matrix, cutoff_day)
    rows = []
    for pres, p in flags.groupby("code_presentation"):
        for module, g in [*p.groupby("code_module"), (ALL_MODULES, p)]:
            applies = module == ALL_MODULES or bool(g["submission_due"].any())
            for i, (key, label) in enumerate(STAGES, 1):
                rows.append({"code_module": module, "code_presentation": pres, "stage_order": i, "stage_key": key,
                             "stage": label, "students": int(g[key].sum()),
                             "applicable": applies or key != SCHEDULE_STAGE})
    out = pd.DataFrame(rows)
    out["suppressed"] = _suppress(out, min_cell)
    shown = out["students"].where(~out["suppressed"]).astype(float)
    group = [out["code_presentation"], out["code_module"]]
    registered = shown.groupby(group).transform(lambda s: s.iloc[0])
    previous = shown.groupby(group).shift(1)
    out["students"] = shown.astype("Int64")
    out["share_of_registered"] = shown / registered
    out["share_of_previous"] = (shown / previous).where(previous > 0)
    return out[list(COLUMNS)]
