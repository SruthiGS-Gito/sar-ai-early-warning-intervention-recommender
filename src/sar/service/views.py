"""SCRUM-189: tables behind the app's Batch Report and Student Lookup tabs.

Scores stay in memory. Summaries are aggregates; the risk-band grid hides bands with fewer than
MIN_CELL students.
"""

import pandas as pd

from sar.data.split import TEST_PRESENTATION

KEY = ["code_module", "code_presentation", "id_student"]
MIN_CELL = 20
DAY_BANDS = ([-1, 0, 5, 10, 15, 20, 10_000], ["0", "1-5", "6-10", "11-15", "16-20", "21+"])
SCORE_BANDS = ([-1, 40, 60, 80, 101], ["<40", "40-59", "60-79", "80+"])


def score_presentation(model, matrix: pd.DataFrame, features: list[str], presentation: str,
                       threshold: float) -> pd.DataFrame:
    """Key columns, code_module, risk score and flag for every student in one presentation."""
    part = matrix[matrix["code_presentation"] == presentation]
    out = part[KEY].copy()
    out["score"] = model.predict_proba(part[features])[:, 1] if len(part) else []
    out["flagged"] = out["score"] >= threshold
    return out.reset_index(drop=True)


def batch_summary(scored: pd.DataFrame) -> pd.DataFrame:
    """Per module: students, flagged, flagged share and average score, plus an 'All modules' row."""
    def agg(g):
        return pd.Series({"Students": len(g), "Flagged": int(g["flagged"].sum()),
                          "Flagged share": g["flagged"].mean(), "Average score": g["score"].mean()})

    by_module = scored.groupby("code_module").apply(agg, include_groups=False)
    total = agg(scored).to_frame("All modules").T
    out = pd.concat([by_module, total])
    out[["Students", "Flagged"]] = out[["Students", "Flagged"]].astype(int)
    out.index.name = "Module"
    return out


def risk_band_table(scored: pd.DataFrame, matrix: pd.DataFrame) -> pd.DataFrame:
    """One row per (active-days band, average-score band): students, average risk score, and whether the
    band is large enough to show (at least MIN_CELL students). Average score is NaN for hidden bands."""
    m = scored.merge(matrix[KEY + ["active_days", "mean_score"]], on=KEY, how="left")
    days = pd.cut(m["active_days"], DAY_BANDS[0], labels=DAY_BANDS[1]).rename("day_band")
    scores = pd.cut(m["mean_score"], SCORE_BANDS[0], labels=SCORE_BANDS[1])
    scores = scores.cat.add_categories("none yet").fillna("none yet").rename("score_band")
    cells = m.groupby([days, scores], observed=False)["score"].agg(students="size", average_score="mean")
    cells = cells.reset_index()
    cells["shown"] = cells["students"] >= MIN_CELL
    cells.loc[~cells["shown"], "average_score"] = float("nan")
    return cells


def risk_bands(scored: pd.DataFrame, matrix: pd.DataFrame) -> pd.DataFrame:
    """Average score by active-days band (rows) and average-score band (columns); small bands are NaN."""
    t = risk_band_table(scored, matrix)
    return t.pivot(index="day_band", columns="score_band", values="average_score")


def is_training_presentation(presentation: str) -> bool:
    return presentation != TEST_PRESENTATION
