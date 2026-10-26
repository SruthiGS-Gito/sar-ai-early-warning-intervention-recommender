"""Shared synthetic data. Every value is generated here; no OULAD rows are used."""

import json

import numpy as np
import pandas as pd
import pytest

from sar.config import PROJECT_ROOT

FEATURES = json.loads((PROJECT_ROOT / "reports" / "metrics_baseline.json").read_text())["features"]


def synthetic_matrix(n: int = 600, seed: int = 0) -> pd.DataFrame:
    """A made-up Day 30 matrix with the real column names and types.

    Risk depends on activity and missed work, plus a deliberately strong gender effect so tests
    can check that background inputs are kept out of reasons even when the model uses them.
    """
    rng = np.random.default_rng(seed)
    active = rng.integers(0, 31, n)
    scored = rng.random(n) < 0.75
    score = np.where(scored, rng.normal(70, 15, n).clip(0, 100), np.nan)
    missed = rng.integers(0, 3, n)
    gender = rng.choice(["F", "M"], n)
    risk = -0.15 * active + 1.2 * missed - 0.03 * np.nan_to_num(score, nan=60) + 2.5 * (gender == "M")
    at_risk = (risk + rng.normal(0, 0.8, n) > np.median(risk)).astype(int)
    df = pd.DataFrame({
        "code_module": rng.choice(["AAA", "BBB", "GGG"], n),
        "code_presentation": rng.choice(["2013B", "2013J", "2014B", "2014J"], n),
        "id_student": np.arange(n),
        "at_risk": at_risk,
        "total_clicks": active * 15 + rng.integers(0, 40, n),
        "active_days": active,
        "avg_clicks_per_active_day": np.where(active > 0, 15.0, 0.0),
        "clicks_last_7d": rng.integers(0, 80, n),
        "clicks_prev_7d": rng.integers(0, 80, n),
        "first_active_day": np.where(active == 0, 31.0, rng.integers(-10, 10, n)),
        "n_submitted": scored.astype(int),
        "mean_score": score,
        "mean_delay_days": np.where(scored, rng.integers(-5, 5, n), np.nan),
        "n_late": rng.integers(0, 2, n),
        "weighted_score": score,
        "score_trend": np.where(scored, 0.0, np.nan),
        "n_assessments_due": np.full(n, 2),
        "n_missed": missed,
        "reg_lead_days": rng.integers(-10, 100, n).astype(float),
        "registered_late": rng.integers(0, 2, n),
        "registered_after_cutoff": np.zeros(n, dtype=int),
        "reg_date_missing": np.zeros(n, dtype=int),
        "max_consecutive_inactive_days": (30 - active).astype(float),
        "trailing_inactive_days": rng.integers(0, 10, n).astype(float),
        "gender": gender,
        "region": rng.choice(["North Region", "Wales"], n),
        "highest_education": rng.choice(["A Level or Equivalent", "HE Qualification"], n),
        "imd_band": rng.choice(["0-10%", "50-60%"], n),
        "age_band": rng.choice(["0-35", "35-55"], n),
        "num_of_prev_attempts": rng.integers(0, 2, n),
        "studied_credits": rng.choice([60, 120], n),
        "disability": rng.choice(["N", "Y"], n),
    })
    df["click_trend"] = df["clicks_last_7d"] - df["clicks_prev_7d"]
    return df


@pytest.fixture(scope="session")
def matrix():
    return synthetic_matrix()


def synthetic_registrations(matrix: pd.DataFrame, n_early: int = 40, seed: int = 1):
    """Made-up studentInfo and studentRegistration rows for the matrix students, plus `n_early`
    students who unregistered before the cutoff (so they are not in the matrix)."""
    rng = np.random.default_rng(seed)
    key = ["code_module", "code_presentation", "id_student"]
    early = pd.DataFrame({"code_module": rng.choice(["AAA", "BBB", "GGG"], n_early),
                          "code_presentation": rng.choice(["2013B", "2013J", "2014B", "2014J"], n_early),
                          "id_student": np.arange(len(matrix), len(matrix) + n_early)})
    risk = matrix["at_risk"].to_numpy()
    odd = matrix["id_student"].to_numpy() % 2 == 1
    result = np.where(risk == 1, np.where(odd, "Withdrawn", "Fail"), np.where(odd, "Distinction", "Pass"))
    info = pd.concat([matrix[key].assign(final_result=result), early.assign(final_result="Withdrawn")],
                     ignore_index=True)
    unreg = np.where(info["final_result"] == "Withdrawn", 100.0, np.nan)
    unreg[len(matrix):] = 10.0
    registration = info[key].assign(date_registration=-20.0, date_unregistration=unreg)
    return info, registration


SMALL_PARAMS = {"n_estimators": 30, "min_samples_leaf": 5, "class_weight": "balanced_subsample"}


@pytest.fixture(scope="session")
def small_forest(matrix):
    """30-input forest (background factors included), for tests of generic explanation behaviour."""
    from sar.models.forest import train_forest

    train = matrix[matrix["code_presentation"] != "2014J"]
    return train_forest(train[FEATURES], train["at_risk"], SMALL_PARAMS)


@pytest.fixture(scope="session")
def small_final(matrix):
    """A small final-configuration model (24 inputs, calibrator, cutoffs) and metrics in the
    metrics.json layout, built with sar.models.final on the synthetic matrix."""
    from sar.data.split import make_split
    from sar.models import final

    train, val = make_split(matrix)
    fitted = final.fit(train, final.final_features(FEATURES), final.forest_builder(SMALL_PARAMS))
    metrics = {"validation_presentation": ["2014J"],
               "final": {"cutoffs_from_oof": fitted["cutoffs"], "validation": final.evaluate(fitted, val)}}
    return fitted, metrics
