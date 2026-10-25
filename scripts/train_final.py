"""Train the final Day 30 model: 24-feature Random Forest, isotonic calibration, out-of-fold cutoffs.

Run from the repo root, after scripts/run_baseline.py:  python scripts/train_final.py
Writes models/final/model.joblib, calibrator.joblib, config.json and metrics.json (git-ignored).
2014J is the validation presentation: it is scored for the metrics and used for nothing else.
"""

import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import sklearn  # noqa: E402

from sar.config import CUTOFF_DAY, PROCESSED_DIR, PROJECT_ROOT, PROTECTED_FEATURES, RANDOM_SEED  # noqa: E402
from sar.models.final import N_BOOT, run, save_final  # noqa: E402
from sar.models.forest import FOREST_PARAMS, file_sha256  # noqa: E402

start = time.perf_counter()
matrix_path = PROCESSED_DIR / "day30_feature_matrix.csv"
matrix = pd.read_csv(matrix_path)
features_30 = json.loads((PROJECT_ROOT / "reports" / "metrics_baseline.json").read_text())["features"]

final, metrics = run(matrix, features_30)
runtime = time.perf_counter() - start

config = {
    "model": "Random Forest, 24 features, isotonic calibration",
    "cutoff_day": CUTOFF_DAY,
    "excluded_protected_features": list(PROTECTED_FEATURES),
    "params": FOREST_PARAMS,
    "seed": RANDOM_SEED,
    "bootstrap_resamples": N_BOOT,
    "train_presentations": sorted(set(matrix["code_presentation"]) - set(metrics["validation_presentation"])),
    "validation_presentation": metrics["validation_presentation"],
    "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    "runtime_seconds": round(runtime, 1),
    "data_file": str(matrix_path.relative_to(PROJECT_ROOT)).replace("\\", "/"),
    "data_sha256": file_sha256(matrix_path),
    "versions": {"python": sys.version.split()[0], "scikit-learn": sklearn.__version__,
                 "pandas": pd.__version__, "numpy": np.__version__},
}
paths = save_final(final, config, metrics)

v = metrics["final"]["validation"]
print("saved", *(str(p) for p in paths.values()))
print(f"runtime {runtime:.0f} s | model.joblib {paths['model'].stat().st_size / 1e6:.1f} MB")
for name, c in final["cutoffs"].items():
    print(f"cutoff {name}: score {c['score']:.4f} (out-of-fold flagged {c['oof_flagged_share']:.3f}, "
          f"recall {c['oof_recall']:.3f}) | validation flagged {v['cutoffs'][name]['flagged_share']:.3f}, "
          f"precision {v['cutoffs'][name]['precision']:.3f}, recall {v['cutoffs'][name]['recall']:.3f}")
print(f"validation ROC-AUC {v['roc_auc']:.4f} PR-AUC {v['pr_auc']:.4f} | Brier raw {v['raw']['brier']:.4f} "
      f"calibrated {v['calibrated']['brier']:.4f} | ECE raw {v['raw']['ece']:.4f} calibrated {v['calibrated']['ece']:.4f}")
