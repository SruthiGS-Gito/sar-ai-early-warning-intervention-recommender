"""Write docs/final_model.md from models/final/config.json and metrics.json.

Run from the repo root, after scripts/train_final.py:  python scripts/write_model_summary.py
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sar.config import PROJECT_ROOT  # noqa: E402
from sar.models.final import FINAL_DIR  # noqa: E402
from sar.models.summary import render_summary  # noqa: E402

config = json.loads((FINAL_DIR / "config.json").read_text())
metrics = json.loads((FINAL_DIR / "metrics.json").read_text())
out = PROJECT_ROOT / "docs" / "final_model.md"
out.write_text(render_summary(config, metrics), encoding="utf-8")
print("saved", out.relative_to(PROJECT_ROOT))
