"""Shared chart helpers for the Review 1 notebooks.

SCRUM-164: grade charts   SCRUM-165: attendance / assignment charts.

Every function takes a DataFrame, draws one chart, saves a PNG into
reports/figures/ (unless save=False) and returns the matplotlib Figure.
Charts only aggregate; they never print individual student rows.

The Day 30 insight charts share one palette (AT_RISK, ON_TRACK, NEUTRAL) and
put the finding, with its number, in the title. `finding(fig)` returns that title.
"""

import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # safe on machines without a display
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.ticker import PercentFormatter, StrMethodFormatter  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

from sar.config import CUTOFF_DAY, FIGURES_DIR  # noqa: E402

RESULT_ORDER = ["Distinction", "Pass", "Fail", "Withdrawn"]
KEY = ["code_module", "code_presentation", "id_student"]

AT_RISK = "#C44E52"
ON_TRACK = "#4C72B0"
NEUTRAL = "#8C8C8C"

FEATURE_LABELS = {
    "total_clicks": "Total clicks",
    "active_days": "Active days",
    "avg_clicks_per_active_day": "Clicks per active day",
    "clicks_last_7d": "Clicks in the last 7 days",
    "clicks_prev_7d": "Clicks in the 7 days before that",
    "click_trend": "Change in weekly clicks",
    "first_active_day": "First active day",
    "max_consecutive_inactive_days": "Longest inactive streak",
    "trailing_inactive_days": "Inactive days before the cutoff",
    "n_submitted": "Assessments submitted",
    "mean_score": "Average score",
    "weighted_score": "Weighted average score",
    "score_trend": "Change in score",
    "mean_delay_days": "Average submission delay",
    "n_late": "Late submissions",
    "n_assessments_due": "Assessments due",
    "n_missed": "Assessments missed",
    "reg_lead_days": "Days registered before start",
    "registered_late": "Registered after the start",
    "registered_after_cutoff": "Registered after the cutoff",
    "reg_date_missing": "Registration date missing",
    "num_of_prev_attempts": "Previous attempts",
    "studied_credits": "Credits studied",
}
CATEGORY_LABELS = {
    "code_module": "Module",
    "gender": "Gender:",
    "region": "Region:",
    "highest_education": "Education:",
    "imd_band": "Deprivation band:",
    "age_band": "Age:",
    "disability": "Disability:",
}


def save_fig(fig, name: str, out_dir: Path | None = None) -> Path:
    out = Path(out_dir) if out_dir else FIGURES_DIR
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{name}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    return path


def _finish(fig, ax, title, xlabel, ylabel, name, save, out_dir):
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    fig.tight_layout()
    if save:
        save_fig(fig, name, out_dir)
    return fig


def plot_final_results(student_info, save=True, out_dir=None):
    """Bar chart: number of students per final_result."""
    counts = student_info["final_result"].value_counts().reindex(RESULT_ORDER).dropna()
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(counts.index, counts.values)
    return _finish(fig, ax, "Final result", "", "Students", "final_results", save, out_dir)


def plot_score_distribution(student_assessment, save=True, out_dir=None):
    """Histogram of assessment scores (0-100)."""
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(student_assessment["score"].dropna(), bins=20)
    return _finish(fig, ax, "Assessment scores", "Score", "Submissions", "score_distribution", save, out_dir)


def plot_results_by_module(student_info, save=True, out_dir=None):
    """Stacked bars: share of each final_result inside every module."""
    share = (
        student_info.groupby("code_module")["final_result"]
        .value_counts(normalize=True)
        .unstack(fill_value=0)
        .reindex(columns=RESULT_ORDER, fill_value=0)
    )
    fig, ax = plt.subplots(figsize=(7, 4))
    share.plot(kind="bar", stacked=True, ax=ax)
    ax.legend(title="Result", bbox_to_anchor=(1.02, 1), loc="upper left")
    return _finish(fig, ax, "Results by module", "Module", "Share of students", "results_by_module", save, out_dir)


def plot_active_days_by_module(engagement, save=True, out_dir=None):
    """Boxplot of active days per module. `engagement` needs code_module and active_days columns."""
    modules = sorted(engagement["code_module"].unique())
    data = [engagement.loc[engagement["code_module"] == m, "active_days"] for m in modules]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.boxplot(data, tick_labels=modules)
    return _finish(fig, ax, "Active days by module", "Module", "Active days", "active_days_by_module", save, out_dir)


def plot_submission_timing(delay_days, save=True, out_dir=None):
    """Histogram of (date_submitted - due date). Negative = early, positive = late."""
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(delay_days.dropna(), bins=30)
    ax.axvline(0, linestyle="--")
    return _finish(fig, ax, "Submission timing vs due date", "Days (negative = early)", "Submissions", "submission_timing", save, out_dir)


# --- Day 30 insight charts -------------------------------------------------


def finding(fig) -> str:
    """The finding stated in a chart's title, on one line."""
    title = fig.get_suptitle() or fig.axes[0].get_title(loc="left")
    return " ".join(title.split())


def feature_label(name: str) -> str:
    """Plain-English name for a model input such as "num__active_days" or "cat__code_module_FFF"."""
    name = name.split("__", 1)[-1]
    if name in FEATURE_LABELS:
        return FEATURE_LABELS[name]
    for column, label in CATEGORY_LABELS.items():
        if name.startswith(column + "_"):
            return f"{label} {name[len(column) + 1:]}"
    return name


def _clean(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(False)


def _line_label(ax, y, text):
    """Label a horizontal reference line in the right margin."""
    ax.text(1.01, y, text, transform=ax.get_yaxis_transform(), color=NEUTRAL, va="center", ha="left")


def _headline(fig, ax, title, caption, name, save, out_dir, suptitle=False):
    title = textwrap.fill(title, 68)
    if suptitle:
        fig.suptitle(title, x=0.02, ha="left", fontsize=12, fontweight="bold")
    else:
        ax.set_title(title, loc="left", fontsize=12, fontweight="bold", pad=12)
    fig.tight_layout()
    if caption:
        fig.text(0.02, -0.02, caption, ha="left", va="top", fontsize=9, color=NEUTRAL)
    if save:
        save_fig(fig, name, out_dir)
    return fig


def _rate_bars(rates, counts, xlabel, overall=None, overall_label="Overall", figsize=(7.5, 4.2)):
    """Bars of at-risk rate per group, labelled with the rate and the group size."""
    fig, ax = plt.subplots(figsize=figsize)
    labels = [str(i) for i in rates.index]
    bars = ax.bar(labels, rates.values, color=AT_RISK, width=0.65)
    ax.bar_label(bars, labels=[f"{r:.0%}" for r in rates.values], padding=3)
    ax.set_xticks(range(len(labels)), [f"{g}\nn = {n:,}" for g, n in zip(labels, counts.values)])
    if overall is not None:
        ax.axhline(overall, color=NEUTRAL, linestyle="--", linewidth=1.2)
        _line_label(ax, overall, f"{overall_label} {overall:.0%}")
    ax.set_ylim(0, min(1.0, rates.max() * 1.25) if rates.max() > 0 else 1)
    ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Students at risk")
    _clean(ax)
    return fig, ax


def _grouped_rate(matrix, groups):
    g = matrix.groupby(groups, observed=True)["at_risk"]
    return g.mean(), g.size()


def _registrations(student_info, student_registration):
    reg = student_registration[KEY + ["date_unregistration"]].merge(
        student_info[KEY + ["final_result"]], on=KEY, how="left"
    )
    reg["at_risk"] = reg["final_result"].isin(["Fail", "Withdrawn"]).astype(int)
    return reg


def plot_day30_funnel(student_info, student_registration, cutoff_day=CUTOFF_DAY, save=True, out_dir=None):
    """Registrations, early leavers and the cutoff-day population, split into at risk and on track."""
    reg = _registrations(student_info, student_registration)
    left = reg["date_unregistration"] <= cutoff_day
    stages = {
        "All registrations": reg,
        f"Left by Day {cutoff_day}": reg[left],
        f"Day {cutoff_day} population": reg[~left],
    }
    names = list(stages)[::-1]
    risk = np.array([stages[n]["at_risk"].sum() for n in names])
    total = np.array([len(stages[n]) for n in names])

    fig, ax = plt.subplots(figsize=(8, 3.6))
    ax.barh(names, risk, color=AT_RISK, label="At Risk")
    ax.barh(names, total - risk, left=risk, color=ON_TRACK, label="On Track")
    for i, (r, t) in enumerate(zip(risk, total)):
        ax.text(t, i, f"  {t:,} ({r / t:.0%} at risk)", va="center")
    ax.set_xlim(0, total.max() * 1.35)
    ax.set_xlabel("Registrations")
    ax.xaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))
    ax.legend(frameon=False, loc="center right")
    _clean(ax)
    pop = stages[names[0]]
    title = (f"{len(pop):,} of {len(reg):,} registrations are still enrolled on Day {cutoff_day}, "
             f"and {pop['at_risk'].mean():.0%} of them end at risk")
    caption = f"At risk = Fail or Withdrawn. Every student who left by Day {cutoff_day} is at risk by definition."
    return _headline(fig, ax, title, caption, "day30_funnel", save, out_dir)


def plot_withdrawal_timing(student_info, student_registration, cutoff_day=CUTOFF_DAY, save=True, out_dir=None):
    """Withdrawn students by the day they unregistered, with a marker at the cutoff day."""
    reg = _registrations(student_info, student_registration)
    withdrawn = reg[reg["final_result"] == "Withdrawn"]
    days = withdrawn["date_unregistration"].dropna()
    edges = [-np.inf, -1, cutoff_day, 60, 90, 180, np.inf]
    labels = ["Before Day 0", f"0 to {cutoff_day}", f"{cutoff_day + 1} to 60", "61 to 90", "91 to 180", "After 180"]
    counts = pd.cut(days, edges, labels=labels).value_counts().reindex(labels)
    after = (days > cutoff_day).mean()

    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    bars = ax.bar(labels, counts.values, color=[NEUTRAL, NEUTRAL] + [AT_RISK] * 4, width=0.65)
    ax.bar_label(bars, labels=[f"{c:,}" for c in counts.values], padding=3)
    ax.axvline(1.5, color="black", linestyle="--", linewidth=1.2)
    ax.text(1.55, counts.max() * 1.1, f"Day {cutoff_day}", va="top")
    ax.set_ylim(0, counts.max() * 1.15)
    ax.set_xlabel("Day of withdrawal, relative to module start")
    ax.set_ylabel("Withdrawn students")
    ax.yaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))
    _clean(ax)
    title = f"{after:.0%} of withdrawals happen after Day {cutoff_day}, when a warning can still help"
    caption = (f"{len(days):,} Withdrawn students with a recorded date; "
               f"{len(withdrawn) - len(days):,} more have no date.")
    return _headline(fig, ax, title, caption, "withdrawal_timing", save, out_dir)


def plot_at_risk_by_module(matrix, save=True, out_dir=None):
    """At-risk rate per module, sorted, with the overall rate as a line."""
    rates, counts = _grouped_rate(matrix, "code_module")
    rates = rates.sort_values()
    overall = matrix["at_risk"].mean()
    fig, ax = _rate_bars(rates, counts.reindex(rates.index), "Module", overall)
    title = (f"At-risk rate runs from {rates.iloc[0]:.0%} in {rates.index[0]} "
             f"to {rates.iloc[-1]:.0%} in {rates.index[-1]}")
    caption = f"{len(matrix):,} registrations in the Day 30 population."
    return _headline(fig, ax, title, caption, "at_risk_by_module", save, out_dir)


def plot_at_risk_by_active_days(matrix, cutoff_day=CUTOFF_DAY, save=True, out_dir=None):
    """At-risk rate by number of days with at least one click up to the cutoff."""
    labels = ["0", "1 to 5", "6 to 10", "11 to 20", "21 or more"]
    bins = pd.cut(matrix["active_days"], [-1, 0, 5, 10, 20, np.inf], labels=labels)
    rates, counts = _grouped_rate(matrix, bins)
    fig, ax = _rate_bars(rates, counts, f"Days with activity by Day {cutoff_day}", matrix["at_risk"].mean())
    title = (f"Students with no activity by Day {cutoff_day} are {rates.iloc[0]:.0%} at risk, "
             f"against {rates.iloc[-1]:.0%} for those active on {rates.index[-1]} days")
    return _headline(fig, ax, title, None, "at_risk_by_active_days", save, out_dir)


def plot_at_risk_by_inactive_streak(matrix, cutoff_day=CUTOFF_DAY, save=True, out_dir=None):
    """At-risk rate by the longest run of days without a click between Day 0 and the cutoff."""
    labels = ["0 to 3", "4 to 7", "8 to 14", f"15 to {cutoff_day}", "No activity"]
    bins = pd.cut(matrix["max_consecutive_inactive_days"], [-1, 3, 7, 14, cutoff_day, np.inf], labels=labels)
    rates, counts = _grouped_rate(matrix, bins)
    fig, ax = _rate_bars(rates, counts, "Longest run of days without activity", matrix["at_risk"].mean())
    title = (f"At-risk rate is {rates.iloc[0]:.0%} when the longest silent streak is {rates.index[0]} days "
             f"and {rates.iloc[-2]:.0%} when it is {rates.index[-2]} days")
    return _headline(fig, ax, title, None, "at_risk_by_inactive_streak", save, out_dir)


def plot_at_risk_by_click_quartile(matrix, cutoff_day=CUTOFF_DAY, save=True, out_dir=None):
    """At-risk rate by quartile of total clicks up to the cutoff."""
    labels = ["Lowest quarter", "Second", "Third", "Highest quarter"]
    bins = pd.qcut(matrix["total_clicks"].rank(method="first"), 4, labels=labels)
    rates, counts = _grouped_rate(matrix, bins)
    fig, ax = _rate_bars(rates, counts, f"Total clicks by Day {cutoff_day}", matrix["at_risk"].mean())
    title = (f"The least active quarter by clicks is {rates.iloc[0]:.0%} at risk, "
             f"the most active quarter {rates.iloc[-1]:.0%}")
    cuts = matrix["total_clicks"].quantile([0.25, 0.5, 0.75])
    caption = "Quarter boundaries: " + ", ".join(f"{c:,.0f}" for c in cuts) + " clicks."
    return _headline(fig, ax, title, caption, "at_risk_by_click_quartile", save, out_dir)


def plot_score_coverage_by_module(matrix, cutoff_day=CUTOFF_DAY, save=True, out_dir=None):
    """Share of students per module with at least one assessment score by the cutoff."""
    share = matrix["mean_score"].notna().groupby(matrix["code_module"]).mean()
    low = share[share < 0.5]
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    bars = ax.bar(share.index, share.values, color=NEUTRAL, width=0.65)
    ax.bar_label(bars, labels=[f"{s:.0%}" for s in share.values], padding=3)
    ax.set_ylim(0, 1.1)
    ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
    ax.set_xlabel("Module")
    ax.set_ylabel(f"Students with a score by Day {cutoff_day}")
    _clean(ax)
    if len(low):
        gap = " and ".join(f"{s:.0%} in {m}" for m, s in low.items())
        title = (f"Only {gap} have a score by Day {cutoff_day}; "
                 f"elsewhere it is {share.drop(low.index).min():.0%} or more")
    else:
        title = f"At least {share.min():.0%} of students in every module have a score by Day {cutoff_day}"
    caption = f"{matrix['mean_score'].isna().sum():,} of {len(matrix):,} students have no score by Day {cutoff_day}."
    return _headline(fig, ax, title, caption, "score_coverage_by_module", save, out_dir)


def _auc_strength(y, x):
    auc = roc_auc_score(y, x)
    return max(auc, 1 - auc)


def plot_leakage_check(matrix, student_vle, cutoff_day=CUTOFF_DAY, save=True, out_dir=None):
    """ROC-AUC of clicks up to the cutoff against clicks over the whole course, each used alone."""
    whole = student_vle.groupby(KEY)["sum_click"].sum().rename("whole_course_clicks")
    both = matrix[KEY + ["at_risk", "total_clicks"]].join(whole, on=KEY).fillna({"whole_course_clicks": 0})
    early = _auc_strength(both["at_risk"], both["total_clicks"])
    full = _auc_strength(both["at_risk"], both["whole_course_clicks"])

    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    bars = ax.bar([f"Clicks by Day {cutoff_day}\n(used)", f"Whole-course clicks\n(not known on Day {cutoff_day})"],
                  [early, full], color=[ON_TRACK, NEUTRAL], width=0.55)
    ax.bar_label(bars, labels=[f"{early:.2f}", f"{full:.2f}"], padding=3)
    ax.axhline(0.5, color=NEUTRAL, linestyle="--", linewidth=1.2)
    _line_label(ax, 0.5, "No signal 0.50")
    ax.set_ylim(0, 1)
    ax.set_ylabel("ROC-AUC of the feature alone")
    _clean(ax)
    title = (f"Whole-course clicks would score {full:.2f} ROC-AUC; "
             f"the clicks known by Day {cutoff_day} score {early:.2f}")
    caption = "The gap is information from after the cutoff. Using it would overstate what an early warning can do."
    return _headline(fig, ax, title, caption, "leakage_check", save, out_dir)


def plot_roc_pr_curves(y_true, proba, label="validation", save=True, out_dir=None):
    """ROC and precision-recall curves, with the at-risk share drawn as the precision baseline."""
    y_true, proba = np.asarray(y_true), np.asarray(proba)
    base = y_true.mean()
    roc, pr = roc_auc_score(y_true, proba), average_precision_score(y_true, proba)
    fpr, tpr, _ = roc_curve(y_true, proba)
    precision, recall, _ = precision_recall_curve(y_true, proba)

    fig, (a, b) = plt.subplots(1, 2, figsize=(10, 4.4))
    a.plot(fpr, tpr, color=AT_RISK, linewidth=2)
    a.plot([0, 1], [0, 1], color=NEUTRAL, linestyle="--", linewidth=1.2)
    a.text(0.6, 0.55, "Chance", color=NEUTRAL)
    a.set_xlabel("On-track students flagged by mistake")
    a.set_ylabel("At-risk students caught")
    a.set_title(f"ROC curve, AUC {roc:.2f}", loc="left", fontsize=10)
    b.plot(recall, precision, color=AT_RISK, linewidth=2)
    b.axhline(base, color=NEUTRAL, linestyle="--", linewidth=1.2)
    _line_label(b, base, f"At-risk share {base:.2f}")
    b.set_xlabel("At-risk students caught (recall)")
    b.set_ylabel("Flags that are correct (precision)")
    b.set_title(f"Precision-recall curve, AUC {pr:.2f}", loc="left", fontsize=10)
    for ax in (a, b):
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.02)
        _clean(ax)
    title = (f"On {label} the model reaches ROC-AUC {roc:.2f} and PR-AUC {pr:.2f}, "
             f"against a PR-AUC baseline of {base:.2f}")
    return _headline(fig, a, title, f"{len(y_true):,} registrations in {label}.", "model_roc_pr_curves",
                     save, out_dir, suptitle=True)


def plot_confusion_matrix(y_true, proba, threshold=0.5, save=True, out_dir=None):
    """Confusion matrix at one threshold, with counts and row percentages."""
    y_true = np.asarray(y_true)
    pred = (np.asarray(proba) >= threshold).astype(int)
    cm = confusion_matrix(y_true, pred, labels=[0, 1])
    share = cm / cm.sum(axis=1, keepdims=True)

    fig, ax = plt.subplots(figsize=(5.6, 4.4))
    ax.imshow(share, cmap=LinearSegmentedColormap.from_list("grey", ["white", NEUTRAL]), vmin=0, vmax=1, aspect="auto")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i, j]:,}\n{share[i, j]:.0%}", ha="center", va="center", fontsize=12)
    ax.set_xticks([0, 1], ["Predicted On Track", "Predicted At Risk"])
    ax.set_yticks([0, 1], ["On Track", "At Risk"])
    ax.set_ylabel("Actual outcome")
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    title = (f"At threshold {threshold}, the model catches {share[1, 1]:.0%} of at-risk students "
             f"({cm[1, 1]:,} of {cm[1].sum():,})")
    caption = f"Percentages are shares of each row. {cm[0, 1]:,} on-track students are flagged by mistake."
    return _headline(fig, ax, title, caption, "model_confusion_matrix", save, out_dir)


def plot_auc_by_module(modules, y_true, proba, save=True, out_dir=None):
    """ROC-AUC inside each module, labelled with the module's at-risk share."""
    df = pd.DataFrame({"module": np.asarray(modules), "y": np.asarray(y_true), "p": np.asarray(proba)})
    overall = roc_auc_score(df["y"], df["p"])
    by = df.groupby("module").apply(
        lambda g: pd.Series({"auc": roc_auc_score(g["y"], g["p"]), "share": g["y"].mean()}),
        include_groups=False,
    ).sort_values("auc")

    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    bars = ax.bar(by.index, by["auc"], color=ON_TRACK, width=0.65)
    ax.bar_label(bars, labels=[f"{a:.2f}" for a in by["auc"]], padding=3)
    ax.set_xticks(range(len(by)), [f"{m}\n{s:.0%}" for m, s in by["share"].items()])
    ax.axhline(overall, color=NEUTRAL, linestyle="--", linewidth=1.2)
    _line_label(ax, overall, f"All modules {overall:.2f}")
    ax.set_ylim(0.5, 1)
    ax.set_xlabel("Module, with its share of students at risk")
    ax.set_ylabel("ROC-AUC (0.5 = chance)")
    _clean(ax)
    title = (f"The model ranks students best in {by.index[-1]} (ROC-AUC {by['auc'].iloc[-1]:.2f}) "
             f"and worst in {by.index[0]} ({by['auc'].iloc[0]:.2f})")
    return _headline(fig, ax, title, None, "model_auc_by_module", save, out_dir)


def plot_top_coefficients(model, n=10, save=True, out_dir=None):
    """The n largest logistic regression coefficients, coloured by direction."""
    names = model.named_steps["pre"].get_feature_names_out()
    coef = pd.Series(model.named_steps["clf"].coef_[0], index=[feature_label(c) for c in names])
    top = coef.reindex(coef.abs().sort_values(ascending=False).index[:n]).sort_values()

    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    bars = ax.barh(top.index, top.values, color=[AT_RISK if v > 0 else ON_TRACK for v in top.values])
    ax.bar_label(bars, labels=[f"{v:+.2f}" for v in top.values], padding=3)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlim(top.min() * 1.25 if top.min() < 0 else 0, top.max() * 1.25 if top.max() > 0 else 0)
    ax.set_xlabel("Coefficient (red raises risk, blue lowers it)")
    _clean(ax)
    title = (f"{top.index[-1]} raises predicted risk most ({top.iloc[-1]:+.2f}); "
             f"{top.index[0]} lowers it most ({top.iloc[0]:+.2f})")
    caption = "Logistic regression coefficients. Numeric inputs are standardised; categories are one-hot encoded."
    return _headline(fig, ax, title, caption, "model_top_coefficients", save, out_dir)
