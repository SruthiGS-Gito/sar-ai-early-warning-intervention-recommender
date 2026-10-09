import numpy as np
import pandas as pd

from sar.models.baseline import evaluate, save_metrics, train_baseline


def _xy(n=400, seed=0):
    rng = np.random.default_rng(seed)
    clicks = rng.poisson(40, n).astype(float)
    y = pd.Series((clicks + rng.normal(0, 10, n) < 35).astype(int))
    X = pd.DataFrame({
        "total_clicks": clicks,
        "region": rng.choice(["A", "B", None], n),
        "mean_score": np.where(rng.random(n) < 0.2, np.nan, rng.normal(60, 15, n)),
    })
    return X, y


def test_train_evaluate_save(tmp_path):
    X, y = _xy()
    model = train_baseline(X.iloc[:300], y.iloc[:300])
    m = evaluate(model, X.iloc[300:], y.iloc[300:])
    assert m["roc_auc"] > 0.7
    assert 0 <= m["recall"] <= 1 and 0 <= m["pr_auc"] <= 1
    assert sum(m["confusion_matrix"].values()) == 100
    assert save_metrics(m, tmp_path / "metrics_baseline.json").exists()


def test_unseen_category_does_not_crash():
    X, y = _xy()
    model = train_baseline(X, y)
    new = X.iloc[:5].copy()
    new["region"] = "Z"
    assert model.predict_proba(new).shape == (5, 2)
