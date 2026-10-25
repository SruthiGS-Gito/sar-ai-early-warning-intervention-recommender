"""Precompute SHAP explanations of the final model for whole presentations, for the app's Student Lookup.

Run from the repo root, after scripts/train_final.py:
    python scripts/precompute_shap.py                       # the validation presentation (2014J)
    python scripts/precompute_shap.py --presentation 2013J  # one presentation
    python scripts/precompute_shap.py --all                 # every presentation
Writes models/final/shap/<presentation>.parquet and .json (git-ignored; student-level values).
Each file records the model file's hash, so the app ignores it after the model is retrained.
SHAP takes about 0.25 s per student on one core; --jobs spreads the rows over several cores.
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd  # noqa: E402

from sar.config import MODELS_DIR, PROCESSED_DIR  # noqa: E402
from sar.data.split import TEST_PRESENTATION  # noqa: E402
from sar.explain.shap_explain import explain_batch  # noqa: E402
from sar.models.final import load_final  # noqa: E402
from sar.service.data import final_dir, save_shap_cache  # noqa: E402
from sar.service.views import KEY  # noqa: E402

parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
parser.add_argument("--presentation", default=TEST_PRESENTATION)
parser.add_argument("--all", action="store_true", help="every presentation in the matrix")
parser.add_argument("--jobs", type=int, default=-1, help="worker processes (-1: all cores)")
args = parser.parse_args()

fitted, config = load_final(final_dir(MODELS_DIR))
matrix = pd.read_csv(PROCESSED_DIR / "day30_feature_matrix.csv")
presentations = sorted(matrix["code_presentation"].unique()) if args.all else [args.presentation]

for presentation in presentations:
    start = time.perf_counter()
    part = matrix[matrix["code_presentation"] == presentation]
    base, contributions, raw = explain_batch(fitted["model"], part[fitted["features"]], n_jobs=args.jobs)
    path = save_shap_cache(MODELS_DIR, presentation, part[KEY], contributions, raw, base, config)
    print(f"{presentation}: {len(part):,} students in {time.perf_counter() - start:.0f} s -> {path.name}")
