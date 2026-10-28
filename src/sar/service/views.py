"""SCRUM-189: tables behind the app's Batch Report and Student Lookup tabs.

Scoring uses the final model (sar.models.final): calibrated risk score, risk band and rank.
Students are flagged by ranking, not by one fixed threshold:
  * capacity ranking: the top share of the current cohort by risk (e.g. the top 20%)
  * wide net: everyone at or above the wide-net cutoff from models/final/config.json
Ranking orders students by calibrated score, breaks its ties (isotonic calibration has few distinct
values) by the raw forest score, and then by row order. Precision, recall and the at-risk students
caught or missed are computed from the cohort's own outcomes for whichever rule and share is chosen.

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
from sar.models.final import predict, reliability

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


def risk_order(calibrated, raw) -> np.ndarray:
    """Row positions from highest to lowest risk: by calibrated score, ties broken by the raw score,
    then by row order. Because the calibration is monotonic, this is the raw-score order used for the
    capacity table in metrics.json."""
    calibrated, raw = np.asarray(calibrated, dtype=float), np.asarray(raw, dtype=float)
    return np.lexsort((np.arange(len(raw)), -raw, -calibrated))


def score_presentation(fitted: dict, matrix: pd.DataFrame, presentation: str) -> pd.DataFrame:
    """Key columns, calibrated risk score, raw model score, risk band and rank for one presentation,
    plus the observed outcome (`at_risk`) when the matrix has it. Rank 1 is the highest risk."""
    part = matrix[matrix["code_presentation"] == presentation]
    out = part[KEY].copy().reset_index(drop=True)
    raw, cal = predict(fitted, part) if len(part) else (np.array([]), np.array([]))
    out["score"], out["raw_score"] = cal, raw
    out["band"] = assign_risk_band(cal, fitted["cutoffs"])
    rank = np.empty(len(out), dtype=int)
    rank[risk_order(cal, raw)] = np.arange(1, len(out) + 1)
    out["rank"] = rank
    if "at_risk" in part:
        out["at_risk"] = part["at_risk"].to_numpy()
    return out


def module_rank(scored: pd.DataFrame, module: str, student: int) -> tuple[int, int]:
    """(rank, students) of one student within their module by risk_order; rank 1 is the highest risk."""
    part = scored[scored["code_module"] == module].reset_index(drop=True)
    rank = np.empty(len(part), dtype=int)
    rank[risk_order(part["score"], part["raw_score"])] = np.arange(1, len(part) + 1)
    return int(rank[part.index[part["id_student"] == student][0]]), len(part)


def example_student(scored: pd.DataFrame, band: str, module: str | None, seed: int) -> tuple[str, int] | None:
    """(module, student) drawn with a fixed seed from the students in `band`: from `module` when it has
    any, otherwise from the whole cohort. None when no student is in the band."""
    pool = scored[scored["band"] == band]
    if module is not None and (pool["code_module"] == module).any():
        pool = pool[pool["code_module"] == module]
    if pool.empty:
        return None
    pick = pool.sort_values(["code_module", "id_student"]).iloc[np.random.default_rng(seed).integers(len(pool))]
    return str(pick["code_module"]), int(pick["id_student"])


@dataclass
class Flagging:
    rule: str                 # "capacity" or "wide_net"
    label: str
    flagged: pd.Series        # bool, aligned with the scored table


@dataclass
class FlagOutcome:
    """Flagging results against the cohort's observed outcomes (Fail or Withdrawn = at risk)."""
    students: int
    flagged: int
    at_risk: int
    caught: int               # flagged and at risk (true positives)

    @property
    def missed(self) -> int:
        return self.at_risk - self.caught

    @property
    def precision(self) -> float:
        return self.caught / self.flagged if self.flagged else float("nan")

    @property
    def recall(self) -> float:
        return self.caught / self.at_risk if self.at_risk else float("nan")


def flag_outcome(scored: pd.DataFrame, flagged) -> FlagOutcome:
    """Precision, recall, caught and missed for a boolean flag, from the `at_risk` column of `scored`."""
    flagged = np.asarray(flagged, dtype=bool)
    y = scored["at_risk"].to_numpy() == 1
    return FlagOutcome(len(scored), int(flagged.sum()), int(y.sum()), int((flagged & y).sum()))


def flag_capacity(scored: pd.DataFrame, share: float, per_module: bool = False) -> Flagging:
    """Flag the top `share` of this cohort by risk (rank 1 first), for any share from 0 to 1.

    With `per_module`, the share applies inside every module separately: each module's students are
    ordered by risk_order (calibrated score, then raw score, then row order) and its top share is flagged.
    """
    if not 0 <= share <= 1:
        raise ValueError(f"Capacity share must be between 0 and 1, not {share}")
    if not per_module:
        k = int(round(share * len(scored)))
        return Flagging("capacity", f"Top {share:.0%} of students by risk ({k:,} students)", scored["rank"] <= k)
    flagged = pd.Series(False, index=scored.index)
    for _, g in scored.groupby("code_module"):
        k = int(round(share * len(g)))
        flagged[g.index[risk_order(g["score"], g["raw_score"])[:k]]] = True
    return Flagging("capacity", f"Top {share:.0%} of students by risk within each module "
                                f"({int(flagged.sum()):,} students)", flagged)


def flag_wide_net(scored: pd.DataFrame, cutoffs: dict) -> Flagging:
    """Flag everyone at or above the wide-net cutoff, a calibrated score fixed before the cohort arrived."""
    cut = cutoffs["wide_net"]["score"]
    return Flagging("wide_net", f"Wide net: calibrated risk score of at least {cut:.2f}", scored["score"] >= cut)


def flagged_by_band(scored: pd.DataFrame) -> pd.Series:
    """Number of flagged students in each risk band, highest band first. Needs a 'flagged' column."""
    counts = scored.loc[scored["flagged"], "band"].value_counts()
    return pd.Series({b: int(counts.get(b, 0)) for b in reversed(BAND_NAMES)})


def band_evidence(scored: pd.DataFrame, min_cell: int = MIN_CELL) -> pd.DataFrame:
    """Observed at-risk rate and lift inside each risk band, for the cohort ("All modules") and per module.

    Lift is the band's at-risk rate divided by the overall at-risk rate of the same group: the module's
    own rate for a module row, the cohort's rate for the "All modules" row. Cells with fewer than
    `min_cell` students keep their count but have no rate or lift.
    """
    rows = []
    for module, g in [("All modules", scored), *scored.groupby("code_module")]:
        base = g["at_risk"].mean()
        for band in reversed(BAND_NAMES):
            cell = g.loc[g["band"] == band, "at_risk"]
            rate = cell.mean() if len(cell) >= min_cell else np.nan
            rows.append({"code_module": module, "band": band, "students": len(cell), "at_risk_rate": rate,
                         "lift": rate / base if base else np.nan})
    return pd.DataFrame(rows)


def batch_summary(scored: pd.DataFrame, outcomes: bool = False) -> pd.DataFrame:
    """Per module: students, flagged, flagged share and students per risk band, plus an 'All modules' row.
    `scored` needs a boolean 'flagged' column from one of the flagging rules. With `outcomes`, also the
    observed at-risk rate, the recall (share of the module's at-risk students who are flagged) and the
    precision, from the `at_risk` column."""
    def agg(g):
        counts = g["band"].value_counts()
        row = {"Students": len(g), "Flagged": int(g["flagged"].sum()), "Flagged share": g["flagged"].mean(),
               **{b: int(counts.get(b, 0)) for b in reversed(BAND_NAMES)}}
        if outcomes:
            o = flag_outcome(g, g["flagged"])
            row.update({"At-risk rate": g["at_risk"].mean(), "Recall": o.recall, "Precision": o.precision})
        return pd.Series(row)

    by_module = scored.groupby("code_module").apply(agg, include_groups=False)
    total = agg(scored).to_frame("All modules").T
    out = pd.concat([by_module, total])
    ints = ["Students", "Flagged", *BAND_NAMES]
    out[ints] = out[ints].astype(int)
    out.index.name = "Module"
    return out


def calibration_bins(scored: pd.DataFrame, bins: int = 10) -> pd.DataFrame:
    """Observed at-risk rate against mean calibrated score in equal-width score bins (empty bins dropped),
    computed as in the reliability table of metrics.json."""
    rel = reliability(scored["at_risk"].to_numpy(), scored["score"].to_numpy(), bins)
    rows = [r for r in rel["reliability"] if r["count"]]
    return pd.DataFrame({"bin": [r["bin"] for r in rows], "students": [r["count"] for r in rows],
                         "mean_score": [r["mean_predicted"] for r in rows],
                         "observed_rate": [r["observed_rate"] for r in rows]})


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


def _rates(groups) -> pd.DataFrame:
    """Students and observed at-risk rate per group; the rate is NaN for groups under MIN_CELL students."""
    out = groups["at_risk"].agg(students="size", at_risk_rate="mean").reset_index()
    out.loc[out["students"] < MIN_CELL, "at_risk_rate"] = np.nan
    out["suppressed"] = out["students"] < MIN_CELL
    return out


def at_risk_by_presentation(matrix: pd.DataFrame, module: str | None = None) -> pd.DataFrame:
    """Observed at-risk rate per presentation (intake), in presentation order, for one module or all."""
    part = matrix if module is None else matrix[matrix["code_module"] == module]
    return _rates(part.groupby("code_presentation")).sort_values("code_presentation").reset_index(drop=True)


def at_risk_by_activity(matrix: pd.DataFrame, presentation: str, module: str | None = None) -> pd.DataFrame:
    """Observed at-risk rate by days active by Day 30, in the DAY_BANDS buckets (0, 1-5, ..., 21+), for one
    presentation and one module or all. Every bucket is listed, with zero students where none fall in it."""
    part = matrix[matrix["code_presentation"] == presentation]
    if module is not None:
        part = part[part["code_module"] == module]
    days = pd.cut(part["active_days"], DAY_BANDS[0], labels=DAY_BANDS[1]).rename("day_band")
    return _rates(part.groupby(days, observed=False))


def band_outcome(scored: pd.DataFrame, module: str, band: str, min_cell: int = MIN_CELL) -> tuple[float, int, str]:
    """(observed at-risk rate, students, scope) of one band in one module, from band_evidence. When the module
    has fewer than `min_cell` students in the band, the band's rate across all modules is used instead and the
    scope is "All modules"."""
    ev = band_evidence(scored, min_cell)
    for scope in (module, "All modules"):
        row = ev[(ev["code_module"] == scope) & (ev["band"] == band)]
        if len(row) and row["students"].iloc[0] >= min_cell:
            return float(row["at_risk_rate"].iloc[0]), int(row["students"].iloc[0]), scope
    return float("nan"), 0, "All modules"
