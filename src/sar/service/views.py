"""SCRUM-189: tables behind the app's Batch Report and Student Lookup tabs.

Scoring uses the final model (sar.models.final): calibrated risk score, risk band and rank.
Students are flagged by ranking, not by one fixed threshold:
  * capacity ranking: the top share of the current cohort by risk (e.g. the top 20%)
  * wide net: everyone at or above the wide-net cutoff from models/final/config.json
Each flagging rule carries the precision and recall measured for it on the validation presentation,
read from models/final/metrics.json.

Risk bands (Low, Medium, High) come from the calibrated score and the cutoffs named in
sar.config.RISK_BANDS. The activity and score bands of the 3D view (DAY_BANDS, SCORE_BANDS) are a
separate grid. Scores stay in memory. Summaries are aggregates; the activity-score grid hides
cells with fewer than MIN_CELL students.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from sar.config import RISK_BANDS
from sar.data.split import TEST_PRESENTATION
from sar.models.final import predict

KEY = ["code_module", "code_presentation", "id_student"]
MIN_CELL = 20
DAY_BANDS = ([-1, 0, 5, 10, 15, 20, 10_000], ["0", "1-5", "6-10", "11-15", "16-20", "21+"])
SCORE_BANDS = ([-1, 40, 60, 80, 101], ["<40", "40-59", "60-79", "80+"])
BAND_NAMES = [name for name, _ in RISK_BANDS]


def risk_band_edges(cutoffs: dict) -> list[tuple[str, float]]:
    """(band, lower edge on the calibrated score), lowest band first. The lowest band starts at 0."""
    edges = [(name, 0.0 if cut is None else float(cutoffs[cut]["score"])) for name, cut in RISK_BANDS]
    lows = [e for _, e in edges]
    if any(b <= a for a, b in zip(lows, lows[1:])):
        raise ValueError(f"Risk band edges must increase: {edges}")
    return edges


def assign_risk_band(scores, cutoffs: dict) -> pd.Categorical:
    """Risk band for each calibrated score: the highest band whose lower edge it reaches."""
    edges = risk_band_edges(cutoffs)
    idx = np.searchsorted([e for _, e in edges], np.asarray(scores, dtype=float), side="right") - 1
    return pd.Categorical([edges[i][0] for i in idx], categories=BAND_NAMES, ordered=True)


def score_presentation(fitted: dict, matrix: pd.DataFrame, presentation: str) -> pd.DataFrame:
    """Key columns, calibrated risk score, raw model score, risk band and rank for one presentation.

    Rank 1 is the highest risk. Ranking uses the raw score: the calibration is monotonic, so this
    gives the calibrated order with its ties broken, as in the capacity table of metrics.json.
    """
    part = matrix[matrix["code_presentation"] == presentation]
    out = part[KEY].copy().reset_index(drop=True)
    raw, cal = predict(fitted, part) if len(part) else (np.array([]), np.array([]))
    out["score"], out["raw_score"] = cal, raw
    out["band"] = assign_risk_band(cal, fitted["cutoffs"])
    rank = np.empty(len(out), dtype=int)
    rank[np.argsort(-raw, kind="stable")] = np.arange(1, len(out) + 1)
    out["rank"] = rank
    return out


@dataclass
class Flagging:
    rule: str                 # "capacity" or "wide_net"
    label: str
    flagged: pd.Series        # bool, aligned with the scored table
    validation: dict          # precision, recall and flagged share on the validation presentation

    def describe(self) -> str:
        v = self.validation
        return (f"{self.label}. On the validation presentation ({TEST_PRESENTATION}) this rule flagged "
                f"{v['flagged_share']:.0%} of students, with precision {v['precision']:.2f} and recall "
                f"{v['recall']:.2f} (models/final/metrics.json).")


def capacity_shares(metrics: dict) -> list[float]:
    """Capacity shares with validation figures in metrics.json, e.g. [0.05, 0.1, ..., 0.5]."""
    return [int(k.rstrip("%")) / 100 for k in metrics["final"]["validation"]["capacity"]]


def flag_capacity(scored: pd.DataFrame, share: float, metrics: dict) -> Flagging:
    """Flag the top `share` of this cohort by risk; precision and recall come from the capacity table."""
    key = f"{round(share * 100)}%"
    table = metrics["final"]["validation"]["capacity"]
    if key not in table:
        raise ValueError(f"No validation figures for a {key} capacity; choose one of {', '.join(table)}")
    k = int(round(share * len(scored)))
    return Flagging("capacity", f"Top {key} of students by risk ({k:,} students)", scored["rank"] <= k, table[key])


def flag_wide_net(scored: pd.DataFrame, cutoffs: dict, metrics: dict) -> Flagging:
    """Flag everyone at or above the wide-net cutoff, a calibrated score fixed before the cohort arrived."""
    cut = cutoffs["wide_net"]["score"]
    return Flagging("wide_net", f"Wide net: calibrated risk score of at least {cut:.2f}", scored["score"] >= cut,
                    metrics["final"]["validation"]["cutoffs"]["wide_net"])


def batch_summary(scored: pd.DataFrame) -> pd.DataFrame:
    """Per module: students, flagged, flagged share and students per risk band, plus an 'All modules' row.
    `scored` needs a boolean 'flagged' column from one of the flagging rules."""
    def agg(g):
        counts = g["band"].value_counts()
        return pd.Series({"Students": len(g), "Flagged": int(g["flagged"].sum()), "Flagged share": g["flagged"].mean(),
                          **{b: int(counts.get(b, 0)) for b in reversed(BAND_NAMES)}})

    by_module = scored.groupby("code_module").apply(agg, include_groups=False)
    total = agg(scored).to_frame("All modules").T
    out = pd.concat([by_module, total])
    ints = ["Students", "Flagged", *BAND_NAMES]
    out[ints] = out[ints].astype(int)
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
