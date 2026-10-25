"""SCRUM-187: SHAP bar chart for one student's Explanation.

Main panel: the inputs that move this student's score most, red when they raise the score and
blue when they lower it; course-level inputs (module, assessment calendar) are orange, and the
rest are summed into one grey bar. For a model that uses background factors as inputs, a second
panel shows them in grey on the same scale, for transparency only; they are never used as reasons.
The final model has no background inputs, so its chart has the main panel only.
"""

import math

import matplotlib.pyplot as plt
import pandas as pd

from sar.viz.plots import AT_RISK, CATEGORY_LABELS, FEATURE_LABELS, NEUTRAL, ON_TRACK, _clean, save_fig

COURSE_COLOUR = "#DD8452"


def short_label(feature: str) -> str:
    return FEATURE_LABELS.get(feature) or CATEGORY_LABELS.get(feature, feature).rstrip(":")


def _fmt(v) -> str:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "none yet"
    if isinstance(v, (int, float)):
        return f"{v:,.0f}" if float(v).is_integer() or abs(v) >= 100 else f"{v:,.1f}"
    return str(v)


def _bars(ax, labels, values, colors):
    bars = ax.barh(labels, values, color=colors)
    ax.bar_label(bars, labels=[f"{v:+.3f}" for v in values], padding=3, fontsize=8)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.invert_yaxis()
    _clean(ax)


def plot_explanation(expl, n: int = 8, save: bool = False, name: str = "student_explanation", out_dir=None):
    """Draw the explanation chart and return the Figure."""
    c = expl.contributions
    main = c[c["group"] != "background"]
    top = main.reindex(main["contribution"].abs().sort_values(ascending=False).index[:n])
    rest = main.drop(top.index)
    labels = [f"{short_label(f)}: {_fmt(v)}" for f, v in top["value"].items()]
    values = list(top["contribution"])
    colors = [COURSE_COLOUR if g == "course" else AT_RISK if v > 0 else ON_TRACK
              for g, v in zip(top["group"], values)]
    if len(rest):
        labels.append(f"Other inputs ({len(rest)})")
        values.append(float(rest["contribution"].sum()))
        colors.append(NEUTRAL)

    bg = c[c["group"] == "background"]
    bg_labels = [f"{short_label(f)}: {_fmt(v)}" for f, v in bg["value"].items()]

    if len(bg):
        fig, (ax, ax_bg) = plt.subplots(
            2, 1, figsize=(8, 0.42 * (len(labels) + len(bg_labels)) + 2.6), sharex=True,
            gridspec_kw={"height_ratios": [max(len(labels), 1), len(bg_labels)]},
        )
    else:
        fig, ax = plt.subplots(figsize=(8, 0.42 * len(labels) + 2.2))
        ax_bg = ax
    _bars(ax, labels, values, colors)
    ax.set_title("What moves this student's score (red raises it, blue lowers it, orange is course-level)",
                 loc="left", fontsize=10)
    if len(bg):
        _bars(ax_bg, bg_labels, list(bg["contribution"]), [NEUTRAL] * len(bg))
        ax_bg.set_title(f"Background factors, shown for transparency only and never used as reasons "
                        f"(together {bg['contribution'].sum():+.3f})", loc="left", fontsize=10)
    ax_bg.set_xlabel("Change in model score (SHAP value)")
    lo, hi = min(values + list(bg["contribution"]) + [0]), max(values + list(bg["contribution"]) + [0])
    pad = 0.25 * (hi - lo or 0.01)
    ax.set_xlim(lo - pad, hi + pad)

    title = f"Risk band: {expl.band}" if expl.band else "What moves this student's score"
    fig.suptitle(title, x=0.02, ha="left", fontsize=12, fontweight="bold")
    fig.tight_layout()
    caption = (f"Bars start from the model's starting score ({expl.base_score:.2f}, set in training) and add "
               "up to this student's model score. The risk band comes from the calibrated score.")
    fig.text(0.02, -0.01, caption, ha="left", va="top", fontsize=8, color=NEUTRAL, wrap=True)
    if save:
        save_fig(fig, name, out_dir)
    return fig


def explanation_table(expl) -> pd.DataFrame:
    """The contributions as a table for display: label, value, contribution, group."""
    c = expl.contributions
    return pd.DataFrame({
        "Input": [short_label(f) for f in c.index],
        "Value": [_fmt(v) for v in c["value"]],
        "Change in score": c["contribution"].round(4).values,
        "Group": c["group"].values,
    }).sort_values("Change in score", key=abs, ascending=False, ignore_index=True)
