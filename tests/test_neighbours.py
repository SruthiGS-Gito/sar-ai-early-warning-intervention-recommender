import numpy as np
import pandas as pd
import pytest
from conftest import FEATURES, synthetic_matrix

from sar.recommend.neighbours import NeighbourPool


@pytest.fixture(scope="module")
def split(matrix):
    return matrix[matrix["code_presentation"] != "2014J"], matrix[matrix["code_presentation"] == "2014J"]


def test_pool_refuses_the_validation_presentation(matrix):
    with pytest.raises(ValueError):
        NeighbourPool(matrix)


def test_neighbours_come_from_training_presentations_and_the_same_module(split):
    train, val = split
    pool = NeighbourPool(train, k=20)
    row = val.iloc[[0]]
    nb = pool.query(row)
    assert nb.n == 20 and nb.same_module and nb.note == ""
    assert set(nb.frame["code_presentation"]) <= {"2013B", "2013J", "2014B"}
    assert set(nb.frame["code_module"]) == {row["code_module"].iloc[0]}
    assert nb.n_at_risk + nb.n_on_track == 20
    assert f"{nb.n_at_risk} ended At Risk" in nb.summary()


def test_small_module_falls_back_to_all_modules_and_says_so(split):
    train, val = split
    few = train.copy()
    few.loc[few.index[:5], "code_module"] = "ZZZ"          # a module with only 5 earlier students
    pool = NeighbourPool(few, k=20)
    row = val.iloc[[0]].assign(code_module="ZZZ")
    nb = pool.query(row)
    assert not nb.same_module and nb.n == 20 and nb.module_candidates == 5
    assert "only 5 earlier students" in nb.summary() and "all modules" in nb.summary()


def test_a_training_student_is_not_their_own_neighbour(split):
    train, _ = split
    pool = NeighbourPool(train, k=10)
    row = train.iloc[[3]]
    nb = pool.query(row)
    assert row["id_student"].iloc[0] not in set(nb.frame["id_student"])


def test_background_and_outcome_do_not_change_the_neighbours(split):
    train, val = split
    pool = NeighbourPool(train, k=15)
    row = val.iloc[[1]][["id_student"] + FEATURES]
    other = row.assign(gender="F" if row["gender"].iloc[0] == "M" else "M", region="Wales",
                       highest_education="HE Qualification", imd_band="0-10%", age_band="35-55", disability="Y")
    a, b = pool.query(row).frame.index, pool.query(other).frame.index
    assert list(a) == list(b)
    with_outcome = val.iloc[[1]]
    assert list(pool.query(with_outcome).frame.index) == list(a)


def test_missing_scores_and_extreme_clicks_are_handled(split):
    train, val = split
    pool = NeighbourPool(train, k=10)
    row = val.iloc[[2]].copy()
    row[["mean_score", "weighted_score", "score_trend", "mean_delay_days"]] = np.nan
    row["total_clicks"] = 10**6
    assert pool.query(row).n == 10


def test_rejects_several_rows(split):
    train, val = split
    with pytest.raises(ValueError):
        NeighbourPool(train, k=10).query(val.head(2))


def test_more_similar_rows_rank_first():
    df = synthetic_matrix(n=400, seed=3)
    train = df[df["code_presentation"] != "2014J"]
    pool = NeighbourPool(train, k=5)
    target = train.iloc[[0]].drop(columns="id_student")    # a copy of a pool row, without its id
    assert pool.query(target).frame.index[0] == 0
    assert isinstance(pool.query(target).frame, pd.DataFrame)
