import sys
import os
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

sys.path.append(os.path.join(os.path.dirname(__file__), "src"))
from score import score_leads

st.set_page_config(page_title="Lead Scoring Dashboard", page_icon="🎯", layout="wide")
st.title("Lead Scoring Dashboard")
st.caption("Upload a CSV of leads to get a conversion-likelihood score (0-100) for each, ranked highest first.")

uploaded_file = st.file_uploader("Upload leads CSV", type=["csv"])

if uploaded_file is not None:
    df_raw = pd.read_csv(uploaded_file)
    st.write(f"Loaded {len(df_raw)} leads.")

    if st.button("Score Leads"):
        with st.spinner("Scoring leads..."):
            scored = score_leads(df_raw)

        st.success("Done.")

        col1, col2, col3 = st.columns(3)
        col1.metric("Total Leads", len(scored))
        col2.metric("Avg Score", f"{scored['lead_score'].mean():.1f}")
        col3.metric("High-Priority (score ≥ 70)", int((scored["lead_score"] >= 70).sum()))

        st.subheader("Ranked Leads")
        st.dataframe(scored, use_container_width=True)

        st.subheader("Score Distribution")
        fig, ax = plt.subplots(figsize=(8, 3))
        ax.hist(scored["lead_score"], bins=20, color="#4C72B0")
        ax.set_xlabel("Lead Score")
        ax.set_ylabel("Number of Leads")
        st.pyplot(fig)

        csv_out = scored.to_csv(index=False).encode("utf-8")
        st.download_button("Download scored leads as CSV", csv_out, "scored_leads.csv", "text/csv")
else:
    st.info("Upload a CSV with the same columns as the training data to get started.")
