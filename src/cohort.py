"""Monthly cohort retention analysis and customer lifetime value (CLV)."""
from __future__ import annotations

import pandas as pd

from .rfm import clean_transactions


def cohort_retention(df: pd.DataFrame) -> pd.DataFrame:
    """Share of each acquisition cohort (rows) that purchased N months later (columns)."""
    data = clean_transactions(df)
    month_index = data["invoice_date"].dt.year * 12 + data["invoice_date"].dt.month
    first_month_index = month_index.groupby(data["customer_id"]).transform("min")
    data["period"] = month_index - first_month_index
    data["cohort"] = data.groupby("customer_id")["invoice_date"].transform("min").dt.to_period("M").astype(str)

    counts = data.groupby(["cohort", "period"])["customer_id"].nunique().unstack(fill_value=0)
    return counts.div(counts[0], axis=0).round(3)


def segment_clv(rfm: pd.DataFrame, lifespan_years: float = 3.0, margin: float = 0.25) -> pd.DataFrame:
    """Simple historical CLV per RFM segment:

        CLV = average order value x purchases per year x expected lifespan x profit margin
    """
    observed_years = max((rfm["last_purchase"].max() - rfm["first_purchase"].min()).days / 365.0, 1 / 12)
    grouped = rfm.groupby("segment").agg(
        customers=("customer_id", "count"),
        orders=("frequency", "sum"),
        revenue=("monetary", "sum"),
        avg_orders=("frequency", "mean"),
    )
    grouped["avg_order_value"] = grouped["revenue"] / grouped["orders"]
    grouped["orders_per_year"] = grouped["avg_orders"] / observed_years
    grouped["clv"] = grouped["avg_order_value"] * grouped["orders_per_year"] * lifespan_years * margin
    return grouped[["customers", "avg_order_value", "orders_per_year", "clv"]].round(2).sort_values("clv", ascending=False)
