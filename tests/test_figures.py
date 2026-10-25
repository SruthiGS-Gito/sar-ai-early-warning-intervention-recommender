import numpy as np
import pandas as pd
import pytest
from conftest import synthetic_matrix

from sar.service import figures, views


@pytest.fixture(scope="module")
def bands(small_final):
    big = synthetic_matrix(n=4000, seed=5)
    scored = views.score_presentation(small_final[0], big, "2014J")
    return scored, views.risk_band_table(scored, big)


def test_band_table_counts_every_student_and_hides_small_bands(bands):
    scored, table = bands
    assert table["students"].sum() == len(scored)
    assert (table["shown"] == (table["students"] >= views.MIN_CELL)).all()
    assert table.loc[~table["shown"], "average_score"].isna().all()
    assert table.loc[table["shown"], "average_score"].between(0, 1).all()
    assert table["shown"].any() and not table["shown"].all()


def test_figure_plots_only_shown_bands_with_readable_encoding(bands):
    _, table = bands
    fig = figures.band_figure(table)
    stems, markers = fig.data
    shown = table[table["shown"]]
    assert len(markers.x) == len(shown)
    assert markers.marker.colorscale[0][1].lower() == "#440154"          # Viridis, sequential
    assert markers.marker.cmin == 0 and markers.marker.cmax == 1
    assert "risk score" in markers.marker.colorbar.title.text
    assert np.allclose(sorted(markers.z), sorted(shown["average_score"]))
    for text in markers.text:
        assert "Days active by Day 30:" in text and "Average assessment score:" in text
        assert "Students:" in text and "Average risk score:" in text
    hidden_counts = set(table.loc[(~table["shown"]) & (table["students"] > 0), "students"])
    shown_counts = {int(t.split("Students: ")[1].split("<br>")[0].replace(",", "")) for t in markers.text}
    assert shown_counts.isdisjoint(hidden_counts - set(shown["students"]))
    scene = fig.layout.scene
    assert scene.xaxis.title.text == "Average assessment score by Day 30"
    assert scene.yaxis.title.text == "Days active by Day 30"
    assert tuple(scene.zaxis.range) == (0, 1) and scene.camera.eye.x == figures.CAMERA["eye"]["x"]
    assert stems.mode == "lines" and stems.hoverinfo == "skip"


def test_marker_size_grows_with_student_count():
    sizes = figures.marker_sizes(pd.Series([20, 80, 500]))
    assert sizes[0] < sizes[1] < sizes[2]
    assert np.allclose(figures.marker_sizes(pd.Series([50, 50])), figures.marker_sizes(pd.Series([50, 50]))[0])


def test_hidden_band_count(bands):
    _, table = bands
    expected = int(((table["students"] > 0) & (table["students"] < views.MIN_CELL)).sum())
    assert figures.hidden_bands(table) == expected


def test_figure_with_no_shown_bands_does_not_crash():
    table = pd.DataFrame({"day_band": ["0"], "score_band": ["<40"], "students": [3],
                          "average_score": [np.nan], "shown": [False]})
    fig = figures.band_figure(table)
    assert len(fig.data[1].x) == 0
