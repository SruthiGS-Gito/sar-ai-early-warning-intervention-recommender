"""Ticket 203 (SCRUM-178): department risk table.

A department is one module in one presentation (code_module, code_presentation). For each, the
table gives the number of students in the Day 30 population, the observed at-risk rate (final
outcome Fail or Withdrawn) and the share of students in each risk band (sar.config.RISK_BANDS) of
the final model's calibrated score. An "All modules" row per presentation sums its modules.

Scores of training presentations are in-sample: the model learned from those students' outcomes,
so their band shares look better separated than they would for a new intake. The `score_source`
column says which presentations are validation and which are training.
"""

import numpy as np
import pandas as pd

from sar.data.split import TEST_PRESENTATION
from sar.service.views import BAND_NAMES, MIN_CELL, score_presentation

ALL_MODULES = "All modules"
BAND_SHARES = tuple(f"share_{b.lower()}" for b in BAND_NAMES)
COLUMNS = ("code_module", "code_presentation", "score_source", "students", "at_risk_rate", *BAND_SHARES,
           "suppressed")


def _row(group: pd.DataFrame, min_cell: int) -> dict:
    n = len(group)
    if n < min_cell:
        return {"students": n, "at_risk_rate": np.nan, **dict.fromkeys(BAND_SHARES, np.nan), "suppressed": True}
    shares = group["band"].value_counts(normalize=True)
    return {"students": n, "at_risk_rate": group["at_risk"].mean(),
            **{s: float(shares.get(b, 0.0)) for s, b in zip(BAND_SHARES, BAND_NAMES)}, "suppressed": False}


def department_risk(fitted: dict, matrix: pd.DataFrame, min_cell: int = MIN_CELL,
                    validation: str = TEST_PRESENTATION) -> pd.DataFrame:
    """One row per module and presentation, plus an "All modules" row per presentation."""
    rows = []
    for pres in sorted(matrix["code_presentation"].unique()):
        scored = score_presentation(fitted, matrix, pres)
        scored = scored.merge(matrix[["code_module", "code_presentation", "id_student", "at_risk"]],
                              on=["code_module", "code_presentation", "id_student"], how="left")
        source = "validation" if pres == validation else "training"
        groups = [(m, g) for m, g in scored.groupby("code_module")] + [(ALL_MODULES, scored)]
        for module, g in groups:
            rows.append({"code_module": module, "code_presentation": pres, "score_source": source,
                         **_row(g, min_cell)})
    out = pd.DataFrame(rows, columns=list(COLUMNS))
    out["students"] = out["students"].astype(int)
    return out
