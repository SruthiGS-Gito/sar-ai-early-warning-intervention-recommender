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


def _team_summary() -> dict:
    return json.loads((PROJECT_ROOT / "reports" / "team_tables.json").read_text(encoding="utf-8"))


def test_team_table_figures_are_generated_from_the_tracked_summary():
    from sar.tables.summary import current_block, render_block

    text = (DOCS / "app.md").read_text(encoding="utf-8")
    assert current_block(text) == render_block(_team_summary()), "Re-run: python scripts/build_team_tables.py"


def test_team_table_settings_quoted_in_the_docs_match_the_code():
    from sar.config import RANDOM_SEED
    from sar.service.views import MIN_CELL
    from sar.tables.profiles import K_RANGE, PROFILE_FEATURES, SILHOUETTE_SAMPLE

    text = (DOCS / "app.md").read_text(encoding="utf-8")
    assert f"`MIN_CELL` ({MIN_CELL})" in text and f"from 1 to {MIN_CELL - 1}" in text
    assert f"{SILHOUETTE_SAMPLE:,} rows, seed {RANDOM_SEED}" in text
    assert f"k from {min(K_RANGE)} to {max(K_RANGE)}" in text
    assert all(f"`{f}`" in text for f in PROFILE_FEATURES)
    profiles = _team_summary()["student_profiles"]
    assert profiles["features"] == list(PROFILE_FEATURES)
    best = max(profiles["selection"], key=lambda r: r["silhouette"])
    assert profiles["k"] == best["k"] and profiles["k"] in K_RANGE


def test_team_table_summary_holds_aggregates_only():
    summary = _team_summary()
    assert "id_student" not in json.dumps(summary)
    min_cell = summary["min_cell"]
    groups = summary["department_risk"]["validation"] + summary["student_profiles"]["profiles"]
    assert all(r["students"] is None or r["students"] >= min_cell for r in groups)
    counts = [r["students"] for r in summary["retention_funnel"]["all_modules"]]
    assert all(n is None or n == 0 or n >= min_cell for n in counts)


OULAD_LICENCE = '"This dataset is released under CC-BY 4.0 license."'
OULAD_REFERENCE = ("Kuzilek J., Hlosta M., Zdrahal Z. Open University Learning Analytics dataset Sci. Data 4:170171 "
                   "doi: 10.1038/sdata.2017.171 (2017).")


@pytest.mark.parametrize("path", [PROJECT_ROOT / "README.md", DOCS / "app.md"])
def test_dataset_licence_and_reference_are_quoted_from_the_source_page(path):
    text = path.read_text(encoding="utf-8")
    assert OULAD_LICENCE in text and OULAD_REFERENCE in text
    assert "https://research.stem.open.ac.uk/ouanalyse/open-dataset-more/" in text
    assert "CC BY 4.0" not in text
