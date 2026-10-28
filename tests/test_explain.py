import math
import re

import numpy as np
import pandas as pd
import pytest
from conftest import FEATURES

from sar.explain import factors
from sar.explain.chart import explanation_table, plot_explanation
from sar.config import PROTECTED_FEATURES
from sar.explain.shap_explain import (
    MIN_CONTRIBUTION,
    explain_batch,
    explain_rows,
    explain_student,
    explanation_from_contributions,
    make_explainer,
)

BANNED = ["because of", "caused", "benefit", "led to", "due to", "will fail", "will pass"]


def test_contributions_add_up_to_the_score(matrix, small_forest):
    X = matrix[FEATURES].head(50)
    base, contrib, scores = explain_rows(small_forest, X)
    assert list(contrib.columns) == FEATURES          # one column per input, one-hot columns summed
    np.testing.assert_allclose(base + contrib.sum(axis=1), scores, atol=1e-9)
    np.testing.assert_allclose(scores, small_forest.predict_proba(X)[:, 1], atol=1e-12)


def test_works_for_any_student_including_missing_and_unseen_values(matrix, small_forest):
    explainer = make_explainer(small_forest)
    for presentation in ["2013B", "2014J"]:           # a training and a validation presentation
        row = matrix.loc[matrix["code_presentation"] == presentation, FEATURES].head(1)
        assert explain_student(small_forest, row, explainer).sentence
    odd = matrix[FEATURES].head(1).copy()
    odd[["mean_score", "weighted_score", "score_trend", "mean_delay_days"]] = np.nan
    odd["code_module"], odd["region"] = "ZZZ", "Atlantis"    # categories never seen in training
    expl = explain_student(small_forest, odd.iloc[0], explainer)
    assert math.isfinite(expl.score)
    assert "no assessment score yet" in set(expl.contributions["phrase"])


def test_background_never_appears_in_the_sentence(matrix, small_forest):
    explainer = make_explainer(small_forest)
    X = matrix[FEATURES].head(60)
    _, contrib, _ = explain_rows(small_forest, X, explainer)
    assert contrib["gender"].abs().mean() > contrib.drop(columns="gender").abs().mean().median()  # model uses it
    for i in range(len(X)):
        expl = explain_student(small_forest, X.iloc[[i]], explainer)
        s = expl.sentence
        for f in factors.BACKGROUND:
            assert expl.contributions.loc[f, "phrase"] not in s
        assert not expl.reasons()["group"].isin(["background", "course"]).any()
        assert not expl.reasons().index.map(factors.family_of).duplicated().any()
        assert (expl.reasons()["contribution"] >= MIN_CONTRIBUTION).all()
        assert not any(b in s.lower() for b in BANNED)
        assert s.endswith("are not used as reasons.")


def test_sentence_states_the_band_not_a_number(matrix, small_forest):
    row = matrix[FEATURES].head(1)
    high = explain_student(small_forest, row, band="High")
    assert high.sentence.startswith("Risk band: High.") and high.band == "High"
    no_band = explain_student(small_forest, row)
    assert "Risk band" not in no_band.sentence and no_band.band is None
    assert not re.search(r"\d\.\d", high.sentence)              # no raw score or percentage in the text


def test_explain_student_rejects_several_rows(matrix, small_forest):
    with pytest.raises(ValueError):
        explain_student(small_forest, matrix[FEATURES].head(2))


def test_chart_keeps_background_in_its_own_panel(tmp_path, matrix, small_forest):
    expl = explain_student(small_forest, matrix[FEATURES].head(1), band="Medium")
    fig = plot_explanation(expl, n=6, save=True, out_dir=tmp_path)
    top, bottom = fig.axes
    top_labels = [t.get_text() for t in top.get_yticklabels()]
    bottom_labels = [t.get_text() for t in bottom.get_yticklabels()]
    assert len(bottom_labels) == len(factors.BACKGROUND)
    assert not any(lbl.startswith(("Gender", "Region", "Education", "Deprivation", "Age", "Disability"))
                   for lbl in top_labels)
    assert top_labels[-1].startswith("Other inputs")
    assert fig._suptitle.get_text() == "Risk band: Medium"
    assert "PROVISIONAL" not in " ".join(t.get_text() for t in fig.texts)
    assert (tmp_path / "student_explanation.png").exists()
    total = sum(p.get_width() for p in top.patches) + sum(p.get_width() for p in bottom.patches)
    assert total == pytest.approx(expl.score - expl.base_score, abs=1e-9)
    table = explanation_table(expl)
    assert len(table) == len(FEATURES) and isinstance(table, pd.DataFrame)


def test_final_model_explanations_have_no_protected_inputs(tmp_path, matrix, small_final):
    fitted, _ = small_final
    model = fitted["model"]
    X = matrix[fitted["features"]].head(40)
    explainer = make_explainer(model)
    for i in range(len(X)):
        expl = explain_student(model, X.iloc[[i]], explainer, band="Low")
        assert not set(expl.contributions.index) & set(PROTECTED_FEATURES)
        assert expl.background().empty
        assert expl.sentence.endswith("Background characteristics are not model inputs.")
    fig = plot_explanation(expl)
    assert len(fig.axes) == 1                                     # no background panel


def test_batch_and_stored_contributions_match_single_explanations(matrix, small_final):
    fitted, _ = small_final
    model = fitted["model"]
    X = matrix[fitted["features"]].head(30)
    base, contrib, scores = explain_rows(model, X)
    b2, c2, s2 = explain_batch(model, X, n_jobs=2, chunk_size=8)
    assert b2 == pytest.approx(base)
    np.testing.assert_allclose(c2.to_numpy(), contrib.to_numpy(), atol=1e-12)
    np.testing.assert_allclose(s2.to_numpy(), scores.to_numpy(), atol=1e-12)
    direct = explain_student(model, X.iloc[[3]], band="High")
    stored = explanation_from_contributions(X.iloc[3], contrib.iloc[3], base, scores.iloc[3], "High")
    assert stored.sentence == direct.sentence
    np.testing.assert_allclose(stored.contributions["contribution"], direct.contributions["contribution"], atol=1e-12)
