"""Streamlit dashboard for customer behaviour analysis.

Run:  streamlit run app.py
"""
from pathlib import Path

import pandas as pd
import streamlit as st

from src.cohort import cohort_retention, segment_clv
from src.rfm import clean_transactions, compute_rfm, segment_summary
from src.segmentation import evaluate_k, fit_kmeans, profile_clusters

DATA_PATH = Path(__file__).parent / "data" / "transactions.csv"

st.set_page_config(page_title="Customer Behaviour Analysis", page_icon="🛒", layout="wide")
st.title("🛒 Customer Behaviour Analysis")
st.caption("RFM segmentation, K-Means clustering, cohort retention and CLV")


@st.cache_data(show_spinner=False)
def load(file) -> pd.DataFrame:
    return pd.read_csv(file)


with st.sidebar:
    st.header("Data")
    uploaded = st.file_uploader("Upload transactions CSV", type="csv")
    st.caption("Required columns: invoice_id, customer_id, invoice_date, quantity, unit_price")
    k = st.slider("Number of clusters (K-Means)", 2, 8, 4)
    margin = st.slider("Profit margin for CLV", 0.05, 0.60, 0.25, 0.05)
    lifespan = st.slider("Expected customer lifespan (years)", 1.0, 10.0, 3.0, 0.5)

if uploaded is not None:
    raw = load(uploaded)
elif DATA_PATH.exists():
    raw = load(DATA_PATH)
else:
    st.warning("No data found. Run `python -m src.generate_data` or upload a CSV.")
    st.stop()

try:
    data = clean_transactions(raw)
    rfm = compute_rfm(raw)
except ValueError as err:
    st.error(str(err))
    st.stop()

c1, c2, c3, c4 = st.columns(4)
c1.metric("Customers", f"{len(rfm):,}")
c2.metric("Orders", f"{data['invoice_id'].nunique():,}")
c3.metric("Revenue", f"{data['revenue'].sum():,.0f}")
c4.metric("Avg order value", f"{data['revenue'].sum() / data['invoice_id'].nunique():,.0f}")

tab_rfm, tab_cluster, tab_cohort, tab_clv, tab_data = st.tabs(
    ["RFM segments", "K-Means clusters", "Cohort retention", "CLV", "Raw data"]
)

with tab_rfm:
    summary = segment_summary(rfm)
    st.dataframe(summary, use_container_width=True, hide_index=True)
    left, right = st.columns(2)
    left.subheader("Customers per segment")
    left.bar_chart(summary.set_index("segment")["customers"])
    right.subheader("Revenue share (%)")
    right.bar_chart(summary.set_index("segment")["revenue_share_%"])
    choice = st.selectbox("Download customers from a segment", summary["segment"])
    st.download_button(
        "Download CSV",
        rfm[rfm["segment"] == choice].to_csv(index=False),
        file_name=f"{choice.lower().replace(' ', '_')}.csv",
    )

with tab_cluster:
    scores = evaluate_k(rfm)
    st.subheader("Choosing k")
    a, b = st.columns(2)
    a.line_chart(scores.set_index("k")["inertia"])
    b.line_chart(scores.set_index("k")["silhouette"])
    clustered = fit_kmeans(rfm, k=k)
    st.subheader(f"Cluster profile (k={k})")
    st.dataframe(profile_clusters(clustered), use_container_width=True, hide_index=True)
    st.scatter_chart(clustered, x="frequency", y="monetary", color="cluster")

with tab_cohort:
    retention = cohort_retention(raw)
    st.dataframe(retention.style.format("{:.0%}").background_gradient(cmap="Blues", vmin=0, vmax=1), use_container_width=True)

with tab_clv:
    st.dataframe(segment_clv(rfm, lifespan, margin), use_container_width=True)
    st.caption("CLV = average order value x orders per year x lifespan x margin")

with tab_data:
    st.dataframe(raw.head(500), use_container_width=True)
