"""SCRUM-187: SHAP bar chart for one student's Explanation.

Top panel: the inputs that move this student's score most (activity, assessments, study load and
module), red when they raise the score and blue when they lower it, with the rest summed into
one bar. Bottom panel: the background factors, in grey on the same scale, shown for
transparency only; they are never used as reasons.
"""

import math

import matplotlib.pyplot as plt
import pandas as pd

from sar.viz.plots import AT_RISK, CATEGORY_LABELS, FEATURE_LABELS, NEUTRAL, ON_TRACK, _clean, save_fig


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


def plot_explanation(expl, n: int = 8, save: bool = False, name: str = "student_explanation",
                     out_dir=None, provisional: bool = True):
    """Draw the explanation chart and return the Figure."""
    c = expl.contributions
    main = c[c["group"] != "background"]
    top = main.reindex(main["contribution"].abs().sort_values(ascending=False).index[:n])
    rest = main.drop(top.index)
    labels = [f"{short_label(f)}: {_fmt(v)}" for f, v in top["value"].items()]
    values = list(top["contribution"])
    colors = [AT_RISK if v > 0 else ON_TRACK for v in values]
    if len(rest):
        labels.append(f"Other inputs ({len(rest)})")
        values.append(float(rest["contribution"].sum()))
        colors.append(NEUTRAL)

    bg = c[c["group"] == "background"]
    bg_labels = [f"{short_label(f)}: {_fmt(v)}" for f, v in bg["value"].items()]

    fig, (ax, ax_bg) = plt.subplots(
        2, 1, figsize=(8, 0.42 * (len(labels) + len(bg_labels)) + 2.6), sharex=True,
        gridspec_kw={"height_ratios": [max(len(labels), 1), max(len(bg_labels), 1)]},
    )
    _bars(ax, labels, values, colors)
    ax.set_title("What moves this student's score (red raises it, blue lowers it)", loc="left", fontsize=10)
    if len(bg):
        _bars(ax_bg, bg_labels, list(bg["contribution"]), [NEUTRAL] * len(bg))
        ax_bg.set_title(f"Background factors, shown for transparency only and never used as reasons "
                        f"(together {bg['contribution'].sum():+.3f})", loc="left", fontsize=10)
    ax_bg.set_xlabel("Change in risk score (SHAP value)")
    lo, hi = min(values + list(bg["contribution"]) + [0]), max(values + list(bg["contribution"]) + [0])
    pad = 0.25 * (hi - lo or 0.01)
    ax.set_xlim(lo - pad, hi + pad)

    status = ""
    if expl.threshold is not None:
        side = "above" if expl.flagged else "below"
        status = f", {side} the {'provisional ' if provisional else ''}threshold of {expl.threshold:.2f}"
    fig.suptitle(f"Risk score {expl.score:.2f}{status}", x=0.02, ha="left", fontsize=12, fontweight="bold")
    fig.tight_layout()
    caption = (f"Bars start from the model's starting score ({expl.base_score:.2f}, set in training) and add "
               "up to this student's score. The score ranks risk; it is not a probability.")
    if provisional:
        caption += " PROVISIONAL model until ticket 214 delivers the reviewed one."
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
