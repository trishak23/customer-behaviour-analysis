"""End-to-end pipeline: load data -> RFM -> clustering -> cohorts -> charts + CSV reports.

Usage:
    python -m src.analysis                      # uses data/transactions.csv
    python -m src.analysis --data my_file.csv --k 5
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from .cohort import cohort_retention, segment_clv
from .rfm import clean_transactions, compute_rfm, segment_summary
from .segmentation import evaluate_k, fit_kmeans, profile_clusters

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
FIGURES = REPORTS / "figures"


def _save(fig, name: str) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(FIGURES / name, dpi=150)
    plt.close(fig)


def plot_monthly_revenue(data: pd.DataFrame) -> None:
    monthly = data.set_index("invoice_date")["revenue"].resample("MS").sum()
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(monthly.index, monthly.values, marker="o", color="#2563eb")
    ax.set(title="Monthly revenue", xlabel="Month", ylabel="Revenue")
    ax.grid(alpha=0.3)
    _save(fig, "monthly_revenue.png")


def plot_category_revenue(data: pd.DataFrame) -> None:
    if "category" not in data.columns:
        return
    cat = data.groupby("category")["revenue"].sum().sort_values()
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.barh(cat.index, cat.values, color="#10b981")
    ax.set(title="Revenue by product category", xlabel="Revenue")
    _save(fig, "category_revenue.png")


def plot_segments(summary: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].barh(summary["segment"], summary["customers"], color="#6366f1")
    axes[0].set(title="Customers per RFM segment", xlabel="Customers")
    axes[1].barh(summary["segment"], summary["revenue_share_%"], color="#f59e0b")
    axes[1].set(title="Revenue share per RFM segment (%)", xlabel="% of revenue")
    for ax in axes:
        ax.invert_yaxis()
    _save(fig, "rfm_segments.png")


def plot_k_selection(scores: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(scores["k"], scores["inertia"], marker="o")
    axes[0].set(title="Elbow method", xlabel="k", ylabel="Inertia")
    axes[1].plot(scores["k"], scores["silhouette"], marker="o", color="#ef4444")
    axes[1].set(title="Silhouette score", xlabel="k", ylabel="Score")
    for ax in axes:
        ax.grid(alpha=0.3)
    _save(fig, "k_selection.png")


def plot_clusters(clustered: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    scatter = ax.scatter(
        clustered["frequency"], clustered["monetary"], c=clustered["cluster"], cmap="viridis", alpha=0.6, s=18
    )
    ax.set(title="K-Means clusters (frequency vs monetary)", xlabel="Orders", ylabel="Total spend", xscale="log", yscale="log")
    ax.legend(*scatter.legend_elements(), title="Cluster")
    _save(fig, "clusters.png")


def plot_cohorts(retention: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.imshow(retention.values, cmap="Blues", aspect="auto", vmin=0, vmax=1)
    ax.set_xticks(range(retention.shape[1]), retention.columns)
    ax.set_yticks(range(retention.shape[0]), retention.index)
    ax.set(title="Cohort retention (share of cohort purchasing N months after first order)", xlabel="Months since first purchase", ylabel="Acquisition cohort")
    fig.colorbar(im, ax=ax)
    _save(fig, "cohort_retention.png")


def run(data_path: Path, k: int = 4) -> dict:
    raw = pd.read_csv(data_path)
    data = clean_transactions(raw)
    rfm = compute_rfm(raw)

    summary = segment_summary(rfm)
    k_scores = evaluate_k(rfm)
    clustered = fit_kmeans(rfm, k=k)
    profile = profile_clusters(clustered)
    retention = cohort_retention(raw)
    clv = segment_clv(rfm)

    REPORTS.mkdir(exist_ok=True)
    clustered.to_csv(REPORTS / "customer_segments.csv", index=False)
    summary.to_csv(REPORTS / "rfm_segment_summary.csv", index=False)
    profile.to_csv(REPORTS / "cluster_profile.csv", index=False)
    clv.to_csv(REPORTS / "segment_clv.csv")
    retention.to_csv(REPORTS / "cohort_retention.csv")

    plot_monthly_revenue(data)
    plot_category_revenue(data)
    plot_segments(summary)
    plot_k_selection(k_scores)
    plot_clusters(clustered)
    plot_cohorts(retention)

    # Pareto: how concentrated is revenue?
    spend = rfm["monetary"].sort_values(ascending=False)
    top20_share = 100 * spend.head(max(1, int(len(spend) * 0.2))).sum() / spend.sum()
    best_k = int(k_scores.loc[k_scores["silhouette"].idxmax(), "k"])

    print("=" * 60)
    print(f"Customers: {len(rfm):,} | Orders: {data['invoice_id'].nunique():,} | Revenue: {data['revenue'].sum():,.0f}")
    print(f"Top 20% of customers generate {top20_share:.1f}% of revenue")
    print(f"Best k by silhouette: {best_k} (used k={k})")
    print("\nRFM segment summary:\n", summary.to_string(index=False))
    print("\nCluster profile:\n", profile.to_string(index=False))
    print("\nSegment CLV:\n", clv.to_string())
    print(f"\nFigures saved to {FIGURES}")
    return {"rfm": rfm, "summary": summary, "profile": profile, "retention": retention, "clv": clv}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, default=ROOT / "data" / "transactions.csv")
    parser.add_argument("--k", type=int, default=4, help="Number of K-Means clusters")
    args = parser.parse_args()
    if not args.data.exists():
        raise SystemExit(f"{args.data} not found. Run `python -m src.generate_data` first.")
    run(args.data, args.k)


if __name__ == "__main__":
    main()
