"""Schema and privacy checks for the three team tables (sar.tables), on synthetic data."""

import numpy as np
import pandas as pd
import pytest
from conftest import synthetic_registrations

from sar.config import PROTECTED_FEATURES
from sar.data.split import make_split
from sar.service.views import MIN_CELL
from sar.tables import department, funnel, profiles

GROUP = ["code_presentation", "code_module"]


@pytest.fixture(scope="module")
def dept(matrix, small_final):
    return department.department_risk(small_final[0], matrix)


@pytest.fixture(scope="module")
def registrations(matrix):
    return synthetic_registrations(matrix)


@pytest.fixture(scope="module")
def fun(matrix, registrations):
    return funnel.retention_funnel(*registrations, matrix)


@pytest.fixture(scope="module")
def split(matrix):
    return make_split(matrix)


@pytest.fixture(scope="module")
def fitted_profiles(split):
    return profiles.fit_profiles(split[0])


@pytest.fixture(scope="module")
def prof(fitted_profiles, split):
    return profiles.student_profiles(fitted_profiles, *split)


def _modules(t):
    return t[t["code_module"] != department.ALL_MODULES]


# Department risk (ticket 203)

def test_department_columns_and_no_student_column(dept):
    assert tuple(dept.columns) == department.COLUMNS
    assert "id_student" not in dept.columns


def test_department_shares_are_proportions_that_sum_to_one(dept):
    shown = dept[~dept["suppressed"]]
    shares = shown[list(department.BAND_SHARES)]
    assert shares.ge(0).all().all() and shares.le(1).all().all()
    assert np.allclose(shares.sum(axis=1), 1.0)
    assert shown["at_risk_rate"].between(0, 1).all()


def test_department_counts_add_up(dept, matrix):
    totals = dept[dept["code_module"] == department.ALL_MODULES].set_index("code_presentation")["students"]
    assert (_modules(dept).groupby("code_presentation")["students"].sum() == totals).all()
    assert (matrix.groupby("code_presentation").size() == totals).all()
    assert set(dept.loc[dept["code_presentation"] == "2014J", "score_source"]) == {"validation"}
    assert set(dept.loc[dept["code_presentation"] != "2014J", "score_source"]) == {"training"}


@pytest.mark.parametrize("min_cell", [MIN_CELL, 60])
def test_department_hides_groups_below_min_cell(matrix, small_final, min_cell):
    t = department.department_risk(small_final[0], matrix, min_cell=min_cell)
    small = t["students"] < min_cell
    assert (t["suppressed"] == small).all()
    assert t.loc[small, ["at_risk_rate", *department.BAND_SHARES]].isna().all().all()
    if min_cell == 60:
        assert small.any()


# Retention funnel (ticket 207)

def test_funnel_columns_and_stage_order(fun):
    assert tuple(fun.columns) == funnel.COLUMNS
    for _, g in fun.groupby(GROUP):
        assert list(g["stage_order"]) == list(range(1, len(funnel.STAGES) + 1))
        assert list(g["stage_key"]) == [k for k, _ in funnel.STAGES]


def test_funnel_counts_add_up(matrix, registrations):
    info, _ = registrations
    true = funnel.retention_funnel(*registrations, matrix, min_cell=1)
    by_stage = true.set_index(GROUP + ["stage_key"])["students"]
    registered = true[true["stage_key"] == "registered"]
    total = registered[registered["code_module"] == funnel.ALL_MODULES].set_index("code_presentation")["students"]
    assert (info.groupby("code_presentation").size() == total).all()
    still = true[(true["stage_key"] == "still_registered") & (true["code_module"] == funnel.ALL_MODULES)]
    assert (matrix.groupby("code_presentation").size() == still.set_index("code_presentation")["students"]).all()
    modules = _modules(true).groupby(["code_presentation", "stage_key"])["students"].sum()
    alls = true[true["code_module"] == funnel.ALL_MODULES].set_index(["code_presentation", "stage_key"])["students"]
    assert (modules == alls.loc[modules.index]).all()
    for _, g in true.groupby(GROUP):
        assert g["students"].is_monotonic_decreasing
    assert by_stage.notna().all()


def test_funnel_shares_are_proportions(fun):
    shown = fun[~fun["suppressed"]]
    assert shown["share_of_registered"].between(0, 1).all()
    assert shown["share_of_previous"].dropna().between(0, 1).all()


@pytest.mark.parametrize("min_cell", [MIN_CELL, 40])
def test_funnel_hides_small_counts_and_their_complements(matrix, registrations, min_cell):
    t = funnel.retention_funnel(*registrations, matrix, min_cell=min_cell)
    true = funnel.retention_funnel(*registrations, matrix, min_cell=1)["students"]
    shown = t["students"].dropna()
    assert not shown.between(1, min_cell - 1).any()
    assert t.loc[t["suppressed"], ["students", "share_of_registered", "share_of_previous"]].isna().all().all()
    assert (t.loc[~t["suppressed"], "students"] == true[~t["suppressed"]]).all()
    for _, cell in _modules(t).groupby(["code_presentation", "stage_order"]):
        hidden = cell.index[cell["suppressed"]]
        if len(hidden):
            assert len(hidden) >= 2 and true[hidden].sum() >= min_cell
    if min_cell == 40:
        assert t["suppressed"].any()


# Student profiles (ticket 210)

def test_profile_features_exclude_every_protected_feature(fitted_profiles):
    assert not set(profiles.PROFILE_FEATURES) & set(PROTECTED_FEATURES)
    assert not set(fitted_profiles["scaler"].feature_names_in_) & set(PROTECTED_FEATURES)
    assert list(fitted_profiles["scaler"].feature_names_in_) == list(profiles.PROFILE_FEATURES)
    for banned in [*PROTECTED_FEATURES, "at_risk"]:
        with pytest.raises(ValueError):
            profiles.check_features([*profiles.PROFILE_FEATURES, banned])


def test_profile_columns_and_counts_add_up(prof, split, fitted_profiles):
    train, val = split
    assert tuple(prof.columns) == profiles.COLUMNS
    assert "id_student" not in prof.columns
    for scope, rows in [("validation", val), ("training", train)]:
        part = prof[prof["scope"] == scope]
        assert part["students"].sum() == len(rows)
        assert np.isclose(part["share_of_scope"].sum(), 1.0)
        assert part["profile"].is_unique and len(part) == fitted_profiles["k"]
    shown = prof[~prof["suppressed"]]
    assert shown["at_risk_rate"].between(0, 1).all()


def test_profile_k_is_chosen_by_silhouette_within_range(fitted_profiles):
    sel = fitted_profiles["selection"]
    assert list(sel["k"]) == list(profiles.K_RANGE)
    assert fitted_profiles["k"] == sel.loc[sel["silhouette"].idxmax(), "k"]
    assert min(profiles.K_RANGE) == 3 and max(profiles.K_RANGE) == 5


def test_profile_clusters_are_deterministic_for_a_fixed_seed(split, fitted_profiles):
    train, val = split
    again = profiles.fit_profiles(train, seed=fitted_profiles["seed"])
    assert again["k"] == fitted_profiles["k"] and again["names"] == fitted_profiles["names"]
    assert np.array_equal(profiles.assign(again, val), profiles.assign(fitted_profiles, val))
    assert np.allclose(again["kmeans"].cluster_centers_, fitted_profiles["kmeans"].cluster_centers_)
    pd.testing.assert_frame_equal(again["selection"], fitted_profiles["selection"])


def test_profile_names_follow_the_centroid_rules():
    centroids = pd.DataFrame({"total_clicks": [200, 120, 800, 10], "active_days": [10, 8, 27, 1],
                              "trailing_inactive_days": [4, 9, 1, 27], "n_submitted": [0.9, 0.1, 1.3, 0.4],
                              "n_missed": [0.0, 1.1, 0.0, 0.3]})
    assert profiles.name_clusters(centroids) == {0: "Moderately engaged", 1: "Missed early assessment",
                                                 2: "Highly engaged", 3: "Little VLE activity"}


def test_profiles_hide_groups_below_min_cell(fitted_profiles, split):
    t = profiles.student_profiles(fitted_profiles, *split, min_cell=10_000)
    assert t["suppressed"].all()
    assert t[["at_risk_rate", *(f"mean_{f}" for f in profiles.PROFILE_FEATURES)]].isna().all().all()
    assert (t["description"] == "").all()
