"""No protected (background) attribute may appear in any explanation or recommendation output of the
final model: not as an input, a driver, a phrase, a chart label or a word in the text."""

import matplotlib.pyplot as plt
import pytest

from sar.config import PROTECTED_FEATURES
from sar.explain.chart import explanation_table, plot_explanation
from sar.explain.shap_explain import explain_student, make_explainer
from sar.recommend.neighbours import NeighbourPool
from sar.recommend.recommender import recommend

# Feature names and the words used for them in labels and phrases (sar.explain.factors, sar.viz.plots).
PROTECTED_WORDS = {f.replace("_", " ") for f in PROTECTED_FEATURES} | set(PROTECTED_FEATURES) | {
    "deprivation", "imd", "education", "age band", "disability", "gender", "region"}


@pytest.fixture(scope="module")
def outputs(matrix, small_final):
    model = small_final[0]["model"]
    features = list(model.feature_names_in_)
    pool = NeighbourPool(matrix[matrix["code_presentation"] != "2014J"], k=30)
    explainer = make_explainer(model)
    sample = matrix[matrix["code_presentation"] == "2014J"].sample(25, random_state=0)
    out = []
    for i in range(len(sample)):
        row = sample.iloc[[i]]
        expl = explain_student(model, row[features], explainer, band="High")
        rec = recommend(expl, row, pool)
        fig = plot_explanation(expl)
        chart_text = " ".join([t.get_text() for t in fig.texts] + [fig._suptitle.get_text()]
                              + [lbl.get_text() for ax in fig.axes for lbl in ax.get_yticklabels()]
                              + [ax.get_title() for ax in fig.axes])
        plt.close(fig)
        out.append((expl, rec, chart_text))
    return out


def test_no_protected_input_or_driver(outputs):
    for expl, rec, _ in outputs:
        assert not set(expl.contributions.index) & set(PROTECTED_FEATURES)
        assert not {d for s in rec.suggestions for d in s.drivers} & set(PROTECTED_FEATURES)


def test_no_protected_word_in_any_text(outputs):
    for expl, rec, chart_text in outputs:
        table = explanation_table(expl)
        texts = [expl.sentence, rec.text(), chart_text, " ".join(table["Input"]), " ".join(table["Value"])]
        texts += [s.reason + " " + s.evidence for s in rec.suggestions]
        blob = " ".join(texts).lower()
        found = sorted(w for w in PROTECTED_WORDS if w in blob)
        assert not found, found
