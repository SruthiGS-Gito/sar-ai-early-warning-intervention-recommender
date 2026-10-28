"""SCRUM-189: the app's optional 3D view of average risk score by activity and score bands.

One marker per band that has at least MIN_CELL students: position = (score band, active-days band,
average risk score), colour = average risk score on the sequential Viridis scale (readable on light
and dark themes), size = number of students. Grey drop lines link each marker to the floor so its
height is easy to read. Hover shows the band names, student count and average score.
Bands with fewer than MIN_CELL students are not plotted.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from sar.service.views import DAY_BANDS, MIN_CELL, SCORE_BANDS

SCORE_ORDER = SCORE_BANDS[1] + ["none yet"]
DAY_ORDER = DAY_BANDS[1]
CAMERA = {"eye": {"x": 1.6, "y": -1.6, "z": 0.9}}


def marker_sizes(students: pd.Series, smallest: float = 8, largest: float = 34) -> np.ndarray:
    """Marker diameter grows with the square root of the student count, so area tracks count."""
    root = np.sqrt(students.to_numpy(dtype=float))
    if root.size == 0:
        return root
    if root.max() == root.min():
        return np.full(len(root), (smallest + largest) / 2)
    return smallest + (largest - smallest) * (root - root.min()) / (root.max() - root.min())


def band_figure(table: pd.DataFrame) -> go.Figure:
    """`table` is the output of sar.service.views.risk_band_table. With no shown bands the figure keeps its
    axes and has no markers."""
    shown = table[table["shown"]].reset_index(drop=True)
    x = shown["score_band"].astype(str).map(SCORE_ORDER.index).to_numpy()
    y = shown["day_band"].astype(str).map(DAY_ORDER.index).to_numpy()
    z = shown["average_score"].to_numpy(dtype=float)
    hover = [f"Days active by Day 30: {d}<br>Average assessment score: {s}<br>Students: {n:,}<br>"
             f"Average risk score: {r:.2f}"
             for d, s, n, r in zip(shown["day_band"], shown["score_band"], shown["students"], z)]

    stems_x, stems_y, stems_z = [], [], []
    for xi, yi, zi in zip(x, y, z):
        stems_x += [xi, xi, None]
        stems_y += [yi, yi, None]
        stems_z += [0, zi, None]

    fig = go.Figure([
        go.Scatter3d(x=stems_x, y=stems_y, z=stems_z, mode="lines", hoverinfo="skip", showlegend=False,
                     line={"color": "rgba(140,140,140,0.7)", "width": 3}),
        go.Scatter3d(
            x=x, y=y, z=z, mode="markers", text=hover, hovertemplate="%{text}<extra></extra>", showlegend=False,
            marker={"size": marker_sizes(shown["students"]), "color": z, "colorscale": "Viridis",
                    "cmin": 0, "cmax": 1, "opacity": 0.95, "line": {"color": "rgba(140,140,140,0.9)", "width": 1},
                    "colorbar": {"title": {"text": "Average<br>risk score"}, "thickness": 18, "len": 0.75}},
        ),
    ])
    fig.update_layout(
        height=580, margin={"l": 0, "r": 0, "t": 10, "b": 0},
        scene={
            "xaxis": {"title": {"text": "Average assessment score by Day 30"}, "tickvals": list(range(len(SCORE_ORDER))),
                      "ticktext": SCORE_ORDER, "range": [-0.5, len(SCORE_ORDER) - 0.5]},
            "yaxis": {"title": {"text": "Days active by Day 30"}, "tickvals": list(range(len(DAY_ORDER))),
                      "ticktext": DAY_ORDER, "range": [-0.5, len(DAY_ORDER) - 0.5]},
            "zaxis": {"title": {"text": "Average risk score"}, "range": [0, 1]},
            "camera": CAMERA,
            "aspectmode": "manual", "aspectratio": {"x": 1.1, "y": 1.3, "z": 0.8},
        },
    )
    return fig


def hidden_bands(table: pd.DataFrame) -> int:
    """Bands with at least one student but fewer than MIN_CELL, left out of the figure."""
    return int(((table["students"] > 0) & ~table["shown"]).sum())


__all__ = ["band_figure", "hidden_bands", "marker_sizes", "MIN_CELL"]
