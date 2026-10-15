"""RFM (Recency, Frequency, Monetary) feature engineering and rule-based segments."""
from __future__ import annotations

import numpy as np
import pandas as pd

REQUIRED_COLUMNS = {"invoice_id", "customer_id", "invoice_date", "quantity", "unit_price"}


def clean_transactions(df: pd.DataFrame) -> pd.DataFrame:
    """Validate columns, parse dates, drop returns / invalid rows, add revenue."""
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    out = df.copy()
    out["invoice_date"] = pd.to_datetime(out["invoice_date"])
    out = out.dropna(subset=["customer_id", "invoice_id", "invoice_date"])
    out["revenue"] = out["quantity"] * out["unit_price"]
    out = out[(out["quantity"] > 0) & (out["unit_price"] > 0)]
    return out.reset_index(drop=True)


def _quintile_score(values: pd.Series, higher_is_better: bool) -> pd.Series:
    """Score 1-5 using rank-based quintiles (rank avoids duplicate-bin errors)."""
    ranks = values.rank(method="first", ascending=higher_is_better)
    bins = min(5, len(values))
    return pd.qcut(ranks, bins, labels=False).astype(int) + 1


def assign_segment(r: pd.Series, f: pd.Series) -> pd.Series:
    """Classic RFM segment names derived from R and F scores."""
    conditions = [
        (r >= 4) & (f >= 4),
        (r >= 3) & (f >= 4),
        (r >= 4) & (f.between(2, 3)),
        (r >= 4) & (f <= 1),
        (r <= 2) & (f >= 3),
        (r <= 2) & (f <= 2),
    ]
    labels = ["Champions", "Loyal Customers", "Potential Loyalists", "New Customers", "At Risk", "Hibernating"]
    return pd.Series(np.select(conditions, labels, default="Need Attention"), index=r.index)


def compute_rfm(df: pd.DataFrame, snapshot_date: str | pd.Timestamp | None = None) -> pd.DataFrame:
    """Return one row per customer with recency, frequency, monetary, scores and segment."""
    data = clean_transactions(df)
    snapshot = pd.Timestamp(snapshot_date) if snapshot_date else data["invoice_date"].max() + pd.Timedelta(days=1)

    rfm = (
        data.groupby("customer_id")
        .agg(
            first_purchase=("invoice_date", "min"),
            last_purchase=("invoice_date", "max"),
            frequency=("invoice_id", "nunique"),
            monetary=("revenue", "sum"),
        )
        .reset_index()
    )
    rfm["recency"] = (snapshot - rfm["last_purchase"]).dt.days
    rfm["R"] = _quintile_score(rfm["recency"], higher_is_better=False)
    rfm["F"] = _quintile_score(rfm["frequency"], higher_is_better=True)
    rfm["M"] = _quintile_score(rfm["monetary"], higher_is_better=True)
    rfm["rfm_score"] = rfm["R"].astype(str) + rfm["F"].astype(str) + rfm["M"].astype(str)
    rfm["segment"] = assign_segment(rfm["R"], rfm["F"])
    return rfm


def segment_summary(rfm: pd.DataFrame) -> pd.DataFrame:
    """Aggregate KPIs per RFM segment."""
    summary = (
        rfm.groupby("segment")
        .agg(
            customers=("customer_id", "count"),
            avg_recency_days=("recency", "mean"),
            avg_orders=("frequency", "mean"),
            avg_spend=("monetary", "mean"),
            total_revenue=("monetary", "sum"),
        )
        .reset_index()
    )
    summary["customer_share_%"] = 100 * summary["customers"] / summary["customers"].sum()
    summary["revenue_share_%"] = 100 * summary["total_revenue"] / summary["total_revenue"].sum()
    return summary.sort_values("total_revenue", ascending=False).round(2).reset_index(drop=True)
