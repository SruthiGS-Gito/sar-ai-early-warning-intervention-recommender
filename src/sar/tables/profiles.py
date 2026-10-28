"""Ticket 210 (SCRUM-183): student profiles from Day 30 engagement and submission behaviour.

KMeans on five Day 30 features: VLE volume (total_clicks, log-transformed before scaling), VLE
regularity (active_days), recency (trailing_inactive_days), and submissions (n_submitted,
n_missed). No protected attribute and no outcome is a clustering input. The scaler and the
clusters are fitted on the training presentations (sar.data.split.make_split); validation
students (2014J) are then assigned to the nearest cluster.

Number of clusters: the k in K_RANGE (3 to 5) with the highest silhouette score on the training
rows (a fixed sample of SILHOUETTE_SAMPLE rows). The chosen k was then checked for
interpretability by inspecting its centroids and at-risk rates (docs/app.md).

Names were written after that inspection of the k = 4 centroids, and are assigned by rules on the
centroids, in this order:
  * Little VLE activity     the cluster with the fewest active days
  * Missed early assessment the remaining cluster with the most missed assessments
  * Highly engaged          the remaining cluster with the most clicks
  * Moderately engaged      any cluster left
Each description is filled in from that cluster's mean values and observed at-risk rate. The
at-risk rate (from final outcomes) is attached after clustering and is not a clustering input.
"""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from sar.config import CUTOFF_DAY, PROTECTED_FEATURES, RANDOM_SEED
from sar.service.views import MIN_CELL

PROFILE_FEATURES = ("total_clicks", "active_days", "trailing_inactive_days", "n_submitted", "n_missed")
LOG_FEATURES = ("total_clicks",)
K_RANGE = (3, 4, 5)
SILHOUETTE_SAMPLE = 10_000
N_INIT = 10
COLUMNS = ("scope", "presentations", "profile", "cluster", "description", "students", "share_of_scope",
           "at_risk_rate", *(f"mean_{f}" for f in PROFILE_FEATURES), "suppressed")

def check_features(features) -> None:
    banned = (set(PROTECTED_FEATURES) | {"at_risk", "final_result"}) & set(features)
    if banned:
        raise ValueError(f"Profile features must not include protected attributes or outcomes: {sorted(banned)}")


def prepare(df: pd.DataFrame, features=PROFILE_FEATURES) -> pd.DataFrame:
    """Clustering inputs before scaling: the features as floats, with LOG_FEATURES log1p-transformed."""
    check_features(features)
    X = df[list(features)].astype(float).fillna(0.0)
    for f in LOG_FEATURES:
        if f in X:
            X[f] = np.log1p(X[f].clip(lower=0))
    return X


def _kmeans(k: int, seed: int) -> KMeans:
    return KMeans(n_clusters=k, n_init=N_INIT, random_state=seed)


def select_k(Xs: np.ndarray, k_range=K_RANGE, seed: int = RANDOM_SEED) -> pd.DataFrame:
    """Silhouette, inertia and smallest cluster share for each candidate k, on scaled training rows."""
    rows = []
    for k in k_range:
        km = _kmeans(k, seed).fit(Xs)
        sil = silhouette_score(Xs, km.labels_, sample_size=min(SILHOUETTE_SAMPLE, len(Xs)), random_state=seed)
        rows.append({"k": k, "silhouette": float(sil), "inertia": float(km.inertia_),
                     "smallest_share": float(np.bincount(km.labels_).min() / len(Xs))})
    return pd.DataFrame(rows)


def name_clusters(centroids: pd.DataFrame) -> dict[int, str]:
    """Profile name per cluster from its centroid (mean feature values in original units)."""
    left = list(centroids.index)
    names = {}
    for name, column, pick in [("Little VLE activity", "active_days", "idxmin"),
                               ("Missed early assessment", "n_missed", "idxmax"),
                               ("Highly engaged", "total_clicks", "idxmax")]:
        if not left:
            break
        c = getattr(centroids.loc[left, column], pick)()
        names[c] = name
        left.remove(c)
    for i, c in enumerate(centroids.loc[left, "total_clicks"].sort_values(ascending=False).index):
        names[c] = "Moderately engaged" if i == 0 else f"Moderately engaged {i + 1}"
    return names


def fit_profiles(train: pd.DataFrame, k: int | None = None, seed: int = RANDOM_SEED) -> dict:
    """Scaler, KMeans and profile names fitted on the training rows. k=None selects it from K_RANGE."""
    scaler = StandardScaler().fit(prepare(train))
    Xs = scaler.transform(prepare(train))
    selection = select_k(Xs, K_RANGE, seed)
    if k is None:
        k = int(selection.loc[selection["silhouette"].idxmax(), "k"])
    km = _kmeans(k, seed).fit(Xs)
    centroids = train[list(PROFILE_FEATURES)].groupby(km.labels_).mean()
    return {"scaler": scaler, "kmeans": km, "k": k, "seed": seed, "selection": selection,
            "names": name_clusters(centroids)}


def assign(fitted: dict, df: pd.DataFrame) -> np.ndarray:
    return fitted["kmeans"].predict(fitted["scaler"].transform(prepare(df)))


def describe(means: pd.Series, at_risk_rate: float) -> str:
    """One sentence from a cluster's mean values and observed at-risk rate."""
    return (f"On average {means['total_clicks']:,.0f} clicks on {means['active_days']:.1f} active days by Day "
            f"{CUTOFF_DAY}, {means['trailing_inactive_days']:.1f} days without activity just before it, "
            f"{means['n_submitted']:.2f} assessments submitted and {means['n_missed']:.2f} missed; "
            f"{at_risk_rate:.0%} went on to fail or withdraw.")


def profile_rows(fitted: dict, df: pd.DataFrame, scope: str, min_cell: int = MIN_CELL) -> list[dict]:
    labels = assign(fitted, df)
    presentations = ", ".join(sorted(df["code_presentation"].unique()))
    rows = []
    for c in range(fitted["k"]):
        g = df[labels == c]
        name = fitted["names"][c]
        hidden = len(g) < min_cell
        means = g[list(PROFILE_FEATURES)].mean()
        rate = g["at_risk"].mean() if len(g) else np.nan
        rows.append({"scope": scope, "presentations": presentations, "profile": name, "cluster": c,
                     "description": "" if hidden else describe(means, rate), "students": len(g),
                     "share_of_scope": len(g) / len(df) if len(df) else np.nan,
                     "at_risk_rate": np.nan if hidden else rate,
                     **{f"mean_{f}": np.nan if hidden else float(means[f]) for f in PROFILE_FEATURES},
                     "suppressed": hidden})
    return rows


def student_profiles(fitted: dict, train: pd.DataFrame, val: pd.DataFrame, min_cell: int = MIN_CELL) -> pd.DataFrame:
    """One row per profile for the validation students (scope "validation") and the training rows."""
    rows = profile_rows(fitted, val, "validation", min_cell) + profile_rows(fitted, train, "training", min_cell)
    return pd.DataFrame(rows, columns=list(COLUMNS))
