"""The model documentation matches the committed config.json and metrics.json."""

import json

import pytest

from sar.config import PROJECT_ROOT
from sar.models.final import FINAL_DIR
from sar.models.summary import HEADER, render_summary

DOCS = PROJECT_ROOT / "docs"
MODEL_DOCS = ["app.md", "explanations.md", "recommendations.md", "leakage_checklist.md", "final_model.md"]


def test_final_model_page_is_generated_from_the_committed_json():
    config = json.loads((FINAL_DIR / "config.json").read_text())
    metrics = json.loads((FINAL_DIR / "metrics.json").read_text())
    page = (DOCS / "final_model.md").read_text(encoding="utf-8")
    assert page.startswith(HEADER)
    assert page == render_summary(config, metrics), "Re-run: python scripts/write_model_summary.py"


@pytest.mark.parametrize("name", MODEL_DOCS)
def test_model_docs_do_not_describe_a_provisional_model(name):
    text = (DOCS / name).read_text(encoding="utf-8").lower()
    assert "provisional" not in text and "train_forest" not in text
