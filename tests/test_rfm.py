import pandas as pd
import pytest

from src.cohort import cohort_retention, segment_clv
from src.generate_data import generate
from src.rfm import clean_transactions, compute_rfm, segment_summary
from src.segmentation import fit_kmeans


@pytest.fixture(scope="module")
def raw():
    return generate(n_customers=150, seed=1)


def test_clean_drops_returns_and_bad_prices():
    df = pd.DataFrame(
        {
            "invoice_id": [1, 2, 3],
            "customer_id": [1, 1, 2],
            "invoice_date": ["2025-01-01", "2025-01-02", "2025-01-03"],
            "quantity": [2, -1, 1],
            "unit_price": [10.0, 10.0, 0.0],
        }
    )
    out = clean_transactions(df)
    assert len(out) == 1
    assert out.loc[0, "revenue"] == 20.0


def test_missing_columns_raise():
    with pytest.raises(ValueError):
        clean_transactions(pd.DataFrame({"customer_id": [1]}))


def test_rfm_values_are_correct():
    df = pd.DataFrame(
        {
            "invoice_id": [1, 2, 3],
            "customer_id": [1, 1, 2],
            "invoice_date": ["2025-01-01", "2025-01-10", "2025-01-05"],
            "quantity": [1, 2, 1],
            "unit_price": [10.0, 5.0, 100.0],
        }
    )
    rfm = compute_rfm(df, snapshot_date="2025-01-11").set_index("customer_id")
    assert rfm.loc[1, "recency"] == 1
    assert rfm.loc[1, "frequency"] == 2
    assert rfm.loc[1, "monetary"] == 20.0
    assert rfm.loc[2, "recency"] == 6


def test_scores_and_segments(raw):
    rfm = compute_rfm(raw)
    assert rfm[["R", "F", "M"]].min().min() == 1
    assert rfm[["R", "F", "M"]].max().max() == 5
    assert rfm["segment"].notna().all()
    assert rfm["customer_id"].is_unique
    summary = segment_summary(rfm)
    assert summary["customers"].sum() == len(rfm)
    assert abs(summary["revenue_share_%"].sum() - 100) < 0.5


def test_clustering_returns_k_clusters(raw):
    clustered = fit_kmeans(compute_rfm(raw), k=3)
    assert clustered["cluster"].nunique() == 3
    means = clustered.groupby("cluster")["monetary"].mean()
    assert means.is_monotonic_decreasing  # cluster 0 = highest spend


def test_cohort_retention_first_period_is_100pct(raw):
    retention = cohort_retention(raw)
    assert (retention[0] == 1.0).all()
    assert retention.max().max() <= 1.0


def test_clv_is_positive(raw):
    clv = segment_clv(compute_rfm(raw))
    assert (clv["clv"] > 0).all()
