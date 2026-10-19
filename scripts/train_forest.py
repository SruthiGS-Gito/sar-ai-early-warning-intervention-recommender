"""SCRUM-187 (support): train the PROVISIONAL Random Forest and its threshold.

Stand-in until the team model (tickets 212 and 217, SCRUM-184 and SCRUM-185) and the decision
threshold (ticket 214, SCRUM-186) are in the repo.
Run from the repo root, after scripts/run_baseline.py:  python scripts/train_forest.py
Writes models/forest_provisional.joblib and models/forest_provisional.json (git-ignored)
and reports/model_forest_provisional.json (aggregate metrics only, committed).
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import sklearn  # noqa: E402

from sar.config import CUTOFF_DAY, PROCESSED_DIR, PROJECT_ROOT  # noqa: E402
from sar.data.split import TEST_PRESENTATION, make_split  # noqa: E402
from sar.explain.factors import BACKGROUND  # noqa: E402
from sar.models.baseline import evaluate  # noqa: E402
from sar.models.forest import (  # noqa: E402
    FOREST_PARAMS,
    RECALL_TARGET,
    oof_threshold,
    save_model,
    train_forest,
)

matrix = pd.read_csv(PROCESSED_DIR / "day30_feature_matrix.csv")
features = json.loads((PROJECT_ROOT / "reports" / "metrics_baseline.json").read_text())["features"]
train, val = make_split(matrix)
X_train, y_train = train[features], train["at_risk"]

model = train_forest(X_train, y_train)
threshold = oof_threshold(X_train, y_train, train["code_presentation"], RECALL_TARGET)
at_threshold = evaluate(model, val[features], val["at_risk"], threshold["threshold"])
at_half = evaluate(model, val[features], val["at_risk"], 0.5)

metadata = {
    "status": ("PROVISIONAL stand-in until the team model (tickets 212 and 217, SCRUM-184 and SCRUM-185) "
               "and the decision threshold (ticket 214, SCRUM-186) are in the repo"),
    "source": "tuned settings (tickets 212 and 217) from notebooks/xgboost-random-forest (2).ipynb, cell 69, commit c4c3ba6",
    "cutoff_day": CUTOFF_DAY,
    "features": features,
    "background_features": list(BACKGROUND),
    "params": FOREST_PARAMS,
    "train_presentations": sorted(train["code_presentation"].unique()),
    "validation_presentation": TEST_PRESENTATION,
    "train_rows": len(train),
    "validation_rows": len(val),
    "threshold": threshold,
    "validation_at_threshold": {**at_threshold, "flagged_share": float(
        (model.predict_proba(val[features])[:, 1] >= threshold["threshold"]).mean())},
    "validation_at_0_5": at_half,
    "versions": {"scikit-learn": sklearn.__version__, "pandas": pd.__version__, "numpy": np.__version__},
}
model_path, local_meta = save_model(model, metadata)
report_path = PROJECT_ROOT / "reports" / "model_forest_provisional.json"
report_path.write_text(json.dumps({"provisional": True, **metadata}, indent=2))

print("saved", model_path, "|", local_meta, "|", report_path)
print(f"threshold {threshold['threshold']:.3f} (out-of-fold recall {threshold['oof_recall']:.3f}, "
      f"precision {threshold['oof_precision']:.3f})")
print(f"{TEST_PRESENTATION} at threshold: recall {at_threshold['recall']:.3f} precision "
      f"{at_threshold['precision']:.3f} | PR-AUC {at_threshold['pr_auc']:.3f} ROC-AUC {at_threshold['roc_auc']:.3f}")
