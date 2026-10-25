"""Final Day 30 at-risk model: 24-feature Random Forest with isotonic calibration.

Configuration
  * features: the 30 Day 30 inputs minus sar.config.PROTECTED_FEATURES (24 inputs)
  * model: sar.models.forest.build_forest with the tuned FOREST_PARAMS and RANDOM_SEED
  * calibration: isotonic regression fitted on out-of-fold training scores only
  * cutoffs: chosen on out-of-fold calibrated training scores only:
    the scores at the top 10%, 20% and 30%, and the "wide net" cutoff with 80% recall

Leakage rules (as sar.models.forest):
  * the model is fitted on the training presentations only (sar.data.split.make_split)
  * out-of-fold scores come from GroupKFold by code_presentation (sar.models.forest.oof_scores)
  * the validation presentation (2014J) is never used to fit, calibrate or choose a cutoff;
    it is only scored for the metrics
Protected attributes are read from the matrix only for the fairness checks, never as inputs.
"""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import average_precision_score, roc_auc_score

from sar.config import MODELS_DIR, PROTECTED_FEATURES, RANDOM_SEED
from sar.data.split import make_split
from sar.models.baseline import build_pipeline
from sar.models.forest import FOREST_PARAMS, RECALL_TARGET, build_forest, file_sha256, load_checked, oof_scores, recall_cutoff

FINAL_DIR = MODELS_DIR / "final"
CUTOFF_SHARES = (0.10, 0.20, 0.30)
CAPACITY_SHARES = (0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50)
FAIRNESS_CUTOFF = "top_20"
FAIRNESS_GROUPS = {"gender": ("F", "M"), "age_band": ("35-55", "0-35")}   # gap = first minus second
N_BOOT = 1000
N_BINS = 10


def final_features(features: list[str]) -> list[str]:
    """The current inputs minus the protected attributes, order kept."""
    return [f for f in features if f not in PROTECTED_FEATURES]


def forest_builder(params: dict | None = None):
    return lambda X: build_forest(X, params or FOREST_PARAMS)


# ---------------------------------------------------------------- fit

def fit(train: pd.DataFrame, features: list[str], build) -> dict:
    """Fit a model on all training rows, its isotonic calibrator on out-of-fold scores, and its cutoffs."""
    X, y, groups = train[features], train["at_risk"], train["code_presentation"]
    oof = oof_scores(build(X), X, y, groups)
    calibrator = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0).fit(oof, y)
    oof_cal = calibrator.predict(oof)
    model = build(X).fit(X, y)
    return {"model": model, "calibrator": calibrator, "features": list(features),
            "cutoffs": choose_cutoffs(y.to_numpy(), oof_cal), "oof_raw": oof, "oof_calibrated": oof_cal,
            "oof_y": y.to_numpy()}


def choose_cutoffs(y_oof: np.ndarray, oof_cal: np.ndarray) -> dict:
    """Cutoffs on calibrated out-of-fold scores, loosest first. A student is flagged when score >= cutoff.

    Isotonic scores are a step function, so a cutoff can sit on a plateau and flag a little more
    than its nominal share; the out-of-fold share actually flagged is recorded with each cutoff.
    """
    wide = recall_cutoff(y_oof, oof_cal, RECALL_TARGET)
    cutoffs = {"wide_net": {"score": wide["threshold"], "rule": f"out-of-fold recall >= {RECALL_TARGET:.2f}"}}
    for share in sorted(CUTOFF_SHARES, reverse=True):
        cutoffs[f"top_{round(share * 100)}"] = {"score": float(np.quantile(oof_cal, 1 - share)),
                                                "rule": f"top {share:.0%} of out-of-fold scores"}
    for c in cutoffs.values():
        flagged = oof_cal >= c["score"]
        c["oof_flagged_share"] = float(flagged.mean())
        c["oof_recall"] = float(flagged[y_oof == 1].mean())
        c["oof_precision"] = float(y_oof[flagged].mean())
    return cutoffs


def predict(fitted: dict, X: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """(raw, calibrated) at-risk scores."""
    raw = fitted["model"].predict_proba(X[fitted["features"]])[:, 1]
    return raw, fitted["calibrator"].predict(raw)


# ---------------------------------------------------------------- metrics

def reliability(y: np.ndarray, p: np.ndarray, bins: int = N_BINS) -> dict:
    """Brier score, expected calibration error and an equal-width reliability table."""
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, bins - 1)
    table, ece = [], 0.0
    for j in range(bins):
        m = idx == j
        row = {"bin": f"{edges[j]:.1f}-{edges[j + 1]:.1f}", "count": int(m.sum())}
        if m.any():
            row["mean_predicted"], row["observed_rate"] = float(p[m].mean()), float(y[m].mean())
            ece += m.mean() * abs(row["mean_predicted"] - row["observed_rate"])
        table.append(row)
    return {"brier": float(np.mean((p - y) ** 2)), "ece": float(ece), "reliability": table}


def flag_metrics(y: np.ndarray, flagged: np.ndarray) -> dict:
    tp = int((flagged & (y == 1)).sum())
    precision = tp / flagged.sum() if flagged.any() else float("nan")
    return {"flagged_share": float(flagged.mean()), "precision": float(precision),
            "recall": float(tp / (y == 1).sum()), "lift": float(precision / y.mean()), "flagged": int(flagged.sum())}


def top_share(score: np.ndarray, share: float) -> np.ndarray:
    k = int(round(share * len(score)))
    flagged = np.zeros(len(score), bool)
    flagged[np.argsort(-score, kind="stable")[:k]] = True
    return flagged


def evaluate(fitted: dict, val: pd.DataFrame) -> dict:
    """Validation metrics for one fitted configuration."""
    y = val["at_risk"].to_numpy()
    raw, cal = predict(fitted, val)
    return {
        "n": len(y), "prevalence": float(y.mean()),
        "roc_auc": float(roc_auc_score(y, raw)), "pr_auc": float(average_precision_score(y, raw)),
        "roc_auc_calibrated": float(roc_auc_score(y, cal)), "pr_auc_calibrated": float(average_precision_score(y, cal)),
        "raw": reliability(y, raw), "calibrated": reliability(y, cal),
        "cutoffs": {name: {"score": c["score"], **flag_metrics(y, cal >= c["score"])}
                    for name, c in fitted["cutoffs"].items()},
        # Ranked by the raw score; calibration is monotonic, so this only breaks its ties.
        "capacity": {f"{round(s * 100)}%": flag_metrics(y, top_share(raw, s)) for s in CAPACITY_SHARES},
    }


def bootstrap(fitted: dict, val: pd.DataFrame, n_boot: int = N_BOOT, seed: int = RANDOM_SEED) -> dict:
    """Percentile 95% CIs over resampled validation rows, with the cutoff held fixed."""
    y = val["at_risk"].to_numpy()
    raw, cal = predict(fitted, val)
    flagged = cal >= fitted["cutoffs"][FAIRNESS_CUTOFF]["score"]
    rng = np.random.default_rng(seed)
    draws = {"roc_auc": [], "pr_auc": [], f"precision_{FAIRNESS_CUTOFF}": [], f"recall_{FAIRNESS_CUTOFF}": []}
    for _ in range(n_boot):
        i = rng.integers(0, len(y), len(y))
        yy, ff = y[i], flagged[i]
        draws["roc_auc"].append(roc_auc_score(yy, raw[i]))
        draws["pr_auc"].append(average_precision_score(yy, raw[i]))
        draws[f"precision_{FAIRNESS_CUTOFF}"].append(yy[ff].mean())
        draws[f"recall_{FAIRNESS_CUTOFF}"].append(ff[yy == 1].mean())
    return {"resamples": n_boot, "seed": seed, **{k: _ci(v) for k, v in draws.items()}}


def fairness(fitted: dict, val: pd.DataFrame, n_boot: int = N_BOOT, seed: int = RANDOM_SEED) -> dict:
    """Recall and flag rate by gender and age band at the fairness cutoff, with bootstrap CIs on the gaps."""
    y = val["at_risk"].to_numpy()
    _, cal = predict(fitted, val)
    flagged = cal >= fitted["cutoffs"][FAIRNESS_CUTOFF]["score"]
    cols = {c: val[c].astype(str).to_numpy() for c in FAIRNESS_GROUPS}

    def stats(yy, ff, col, group):
        m = col == group
        return {"n": int(m.sum()), "at_risk_n": int((m & (yy == 1)).sum()),
                "flag_rate": float(ff[m].mean()), "recall": float(ff[m & (yy == 1)].mean())}

    out = {"cutoff": FAIRNESS_CUTOFF, "resamples": n_boot, "seed": seed, "groups": {}, "gaps": {}}
    rng = np.random.default_rng(seed)
    idx = [rng.integers(0, len(y), len(y)) for _ in range(n_boot)]
    for col, (a, b) in FAIRNESS_GROUPS.items():
        out["groups"][col] = {g: stats(y, flagged, cols[col], g) for g in (a, b)}
        for metric in ("recall", "flag_rate"):
            point = out["groups"][col][a][metric] - out["groups"][col][b][metric]
            draws = [stats(y[i], flagged[i], cols[col][i], a)[metric] - stats(y[i], flagged[i], cols[col][i], b)[metric]
                     for i in idx]
            out["gaps"][f"{metric}_gap_{col}_{a}_minus_{b}"] = {"point": float(point), **_ci(draws, rate=False)}
    return out


def _ci(draws, rate: bool = True) -> dict:
    lo, hi = np.percentile(draws, [2.5, 97.5])
    out = {"mean": float(np.mean(draws)), "ci95": [float(lo), float(hi)]}
    if not rate:
        out["ci_excludes_zero"] = bool(lo > 0 or hi < 0)
    return out


def leave_one_presentation_out(matrix: pd.DataFrame, features: list[str], build) -> dict:
    """Refit the whole configuration (model, calibrator, cutoffs) with each presentation held out."""
    rows = {}
    for p in sorted(matrix["code_presentation"].unique()):
        train, held = make_split(matrix, test_presentation=p)
        fitted = fit(train, features, build)
        ev = evaluate(fitted, held)
        rows[p] = {"train_rows": len(train), "n": ev["n"], "prevalence": ev["prevalence"],
                   "roc_auc": ev["roc_auc"], "pr_auc": ev["pr_auc"],
                   "brier_calibrated": ev["calibrated"]["brier"], "ece_calibrated": ev["calibrated"]["ece"],
                   "cutoffs": {k: {m: v[m] for m in ("score", "flagged_share", "precision", "recall", "lift")}
                               for k, v in ev["cutoffs"].items()}}
    return rows


# ---------------------------------------------------------------- run

def run(matrix: pd.DataFrame, features_30: list[str], params: dict | None = None,
        n_boot: int = N_BOOT, seed: int = RANDOM_SEED) -> tuple[dict, dict]:
    """Fit the final configuration and its comparison rows; return (final fit, metrics)."""
    features = final_features(features_30)
    rf = forest_builder(params)
    train, val = make_split(matrix)
    final = fit(train, features, rf)
    comparisons = {
        "forest_30_features": fit(train, features_30, rf),
        "logistic_baseline": fit(train, features_30, build_pipeline),
    }
    metrics = {
        "validation_presentation": sorted(val["code_presentation"].unique()),
        "train_rows": len(train), "validation_rows": len(val),
        "final": {
            "features": len(features),
            "cutoffs_from_oof": final["cutoffs"],
            "oof_training": {"raw": {k: v for k, v in reliability(final["oof_y"], final["oof_raw"]).items()
                                     if k != "reliability"},
                             "calibrated": {k: v for k, v in reliability(final["oof_y"], final["oof_calibrated"]).items()
                                            if k != "reliability"}},
            "validation": evaluate(final, val),
            "bootstrap_validation": bootstrap(final, val, n_boot, seed),
            "fairness_validation": fairness(final, val, n_boot, seed),
            "leave_one_presentation_out": leave_one_presentation_out(matrix, features, rf),
        },
        "comparisons": {name: {"features": len(f["features"]), "cutoffs_from_oof": f["cutoffs"],
                               "validation": evaluate(f, val)}
                        for name, f in comparisons.items()},
    }
    return final, metrics


# ---------------------------------------------------------------- save / load

def save_final(fitted: dict, config: dict, metrics: dict, out_dir: Path | None = None) -> dict[str, Path]:
    """Write model.joblib, calibrator.joblib, config.json (with both hashes) and metrics.json."""
    out = Path(out_dir or FINAL_DIR)
    out.mkdir(parents=True, exist_ok=True)
    paths = {"model": out / "model.joblib", "calibrator": out / "calibrator.joblib",
             "config": out / "config.json", "metrics": out / "metrics.json"}
    joblib.dump(fitted["model"], paths["model"])
    joblib.dump(fitted["calibrator"], paths["calibrator"])
    config = {**config, "features": fitted["features"], "cutoffs": fitted["cutoffs"],
              "model_file": paths["model"].name, "model_sha256": file_sha256(paths["model"]),
              "calibrator_file": paths["calibrator"].name, "calibrator_sha256": file_sha256(paths["calibrator"])}
    paths["config"].write_text(json.dumps(config, indent=2))
    paths["metrics"].write_text(json.dumps(metrics, indent=2))
    return paths


def load_final(out_dir: Path | None = None) -> tuple[dict, dict]:
    """Load the model and calibrator named in config.json, hash-checked (sar.models.forest.load_checked)."""
    out = Path(out_dir or FINAL_DIR)
    config = json.loads((out / "config.json").read_text())
    script = "scripts/train_final.py"
    model = load_checked(out / config["model_file"], config["model_sha256"], "config.json", script)
    calibrator = load_checked(out / config["calibrator_file"], config["calibrator_sha256"], "config.json", script)
    return {"model": model, "calibrator": calibrator, "features": config["features"],
            "cutoffs": config["cutoffs"]}, config
