"""K-Means customer segmentation on log-scaled RFM features."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

FEATURES = ["recency", "frequency", "monetary"]


def _scaled_matrix(rfm: pd.DataFrame) -> np.ndarray:
    # RFM variables are right-skewed -> log transform before scaling
    return StandardScaler().fit_transform(np.log1p(rfm[FEATURES]))


def evaluate_k(rfm: pd.DataFrame, k_range=range(2, 9), seed: int = 42) -> pd.DataFrame:
    """Inertia (elbow) and silhouette score for each candidate k."""
    X = _scaled_matrix(rfm)
    rows = []
    for k in k_range:
        model = KMeans(n_clusters=k, n_init=10, random_state=seed).fit(X)
        rows.append({"k": k, "inertia": model.inertia_, "silhouette": silhouette_score(X, model.labels_)})
    return pd.DataFrame(rows)


def fit_kmeans(rfm: pd.DataFrame, k: int = 4, seed: int = 42) -> pd.DataFrame:
    """Return a copy of rfm with a `cluster` column (clusters ordered by avg spend, 0 = highest)."""
    X = _scaled_matrix(rfm)
    labels = KMeans(n_clusters=k, n_init=10, random_state=seed).fit_predict(X)
    out = rfm.copy()
    out["cluster"] = labels
    order = out.groupby("cluster")["monetary"].mean().sort_values(ascending=False).index
    out["cluster"] = out["cluster"].map({old: new for new, old in enumerate(order)})
    return out


def profile_clusters(clustered: pd.DataFrame) -> pd.DataFrame:
    """Average RFM values, size and revenue share per cluster."""
    profile = (
        clustered.groupby("cluster")
        .agg(
            customers=("customer_id", "count"),
            avg_recency=("recency", "mean"),
            avg_frequency=("frequency", "mean"),
            avg_monetary=("monetary", "mean"),
            total_revenue=("monetary", "sum"),
        )
        .reset_index()
    )
    profile["revenue_share_%"] = 100 * profile["total_revenue"] / profile["total_revenue"].sum()
    return profile.round(2)
