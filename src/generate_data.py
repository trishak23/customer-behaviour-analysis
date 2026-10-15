"""Generate a realistic synthetic e-commerce transactions dataset.

You can swap the output for a real dataset (e.g. UCI "Online Retail II") as long
as it has these columns:

    invoice_id, customer_id, invoice_date, category, quantity, unit_price

Usage:
    python -m src.generate_data --customers 2000 --seed 42
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "data" / "transactions.csv"

CATEGORY_BASE_PRICE = {
    "Electronics": 120.0,
    "Fashion": 45.0,
    "Home & Kitchen": 35.0,
    "Beauty": 22.0,
    "Books": 12.0,
    "Sports": 55.0,
    "Groceries": 8.0,
}
CITIES = ["Mumbai", "Pune", "Delhi", "Bengaluru", "Hyderabad", "Chennai", "Kolkata", "Ahmedabad"]
CHANNELS = ["Web", "Mobile App", "Store"]

# Behavioural archetypes used to simulate customers (hidden ground truth: not
# written to the CSV, so the models have to rediscover the structure).
PROFILES = {
    "champion": dict(share=0.10, orders=18, recency_mean=15, spend=1.4),
    "loyal": dict(share=0.20, orders=10, recency_mean=40, spend=1.1),
    "occasional": dict(share=0.30, orders=4, recency_mean=110, spend=0.9),
    "new": dict(share=0.15, orders=1.5, recency_mean=25, spend=0.9),
    "churned": dict(share=0.25, orders=3, recency_mean=400, spend=0.8),
}

START = pd.Timestamp("2024-01-01")
END = pd.Timestamp("2025-12-31")


def generate(n_customers: int = 2000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    names = list(PROFILES)
    shares = np.array([PROFILES[n]["share"] for n in names])
    categories = list(CATEGORY_BASE_PRICE)
    total_days = (END - START).days

    rows = []
    invoice_id = 10_000
    for cid in range(1, n_customers + 1):
        profile_name = rng.choice(names, p=shares / shares.sum())
        p = PROFILES[profile_name]

        if profile_name == "new":
            join = END - pd.Timedelta(days=int(rng.integers(0, 60)))
        else:
            join = START + pd.Timedelta(days=int(rng.integers(0, int(total_days * 0.75))))

        last_active = END - pd.Timedelta(days=int(rng.exponential(p["recency_mean"])))
        last_active = max(min(last_active, END), join)

        n_orders = max(1, int(rng.poisson(p["orders"])))
        span = (last_active - join).days
        offsets = np.sort(rng.integers(0, span + 1, size=n_orders))
        offsets[-1] = span

        preference = rng.dirichlet(np.ones(len(categories)) * 0.7)
        city = rng.choice(CITIES)
        channel = rng.choice(CHANNELS, p=[0.45, 0.4, 0.15])

        for off in offsets:
            invoice_id += 1
            date = join + pd.Timedelta(days=int(off))
            for _ in range(int(rng.integers(1, 5))):
                cat = rng.choice(categories, p=preference)
                price = CATEGORY_BASE_PRICE[cat] * p["spend"] * rng.lognormal(0, 0.35)
                rows.append(
                    (invoice_id, cid, date, cat, int(rng.integers(1, 4)), round(price, 2), city, channel)
                )

    return pd.DataFrame(
        rows,
        columns=["invoice_id", "customer_id", "invoice_date", "category", "quantity", "unit_price", "city", "channel"],
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--customers", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    df = generate(args.customers, args.seed)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"Saved {len(df):,} rows for {df.customer_id.nunique():,} customers -> {args.out}")


if __name__ == "__main__":
    main()
