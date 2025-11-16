#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import json
from datetime import datetime
from typing import Optional

import pandas as pd
import numpy as np
import joblib
import streamlit as st
import plotly.express as px

# ---------------------------
# Page config
# ---------------------------
st.set_page_config(
    page_title="Dataforce: Fraud Detection Dashboard",
    layout="wide",
    page_icon="🧠",
)

# ---------------------------
# Helper functions
# ---------------------------
@st.cache_data(show_spinner=False)
def safe_read_json(path: str) -> dict:
    if os.path.exists(path):
        try:
            with open(path, "r") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}

@st.cache_data(show_spinner=False)
def safe_read_csv(path: str) -> pd.DataFrame:
    if os.path.exists(path):
        try:
            return pd.read_csv(path)
        except Exception:
            return pd.DataFrame()
    return pd.DataFrame()

@st.cache_resource(show_spinner=False)
def load_model(path: str) -> Optional[object]:
    if os.path.exists(path):
        try:
            return joblib.load(path)
        except Exception:
            return None
    return None


def find_probability_column(df: pd.DataFrame) -> Optional[str]:
    candidates = [
        "Fraud_Probability", "Fraud Probability", "Fraud_Prob", "FraudProbability",
        "probability", "score", "prob"
    ]
    for c in candidates:
        if c in df.columns:
            return c
    # fallback: check numeric columns with values in [0,1]
    for c in df.select_dtypes(include=[np.number]).columns:
        col = df[c].dropna()
        if not col.empty and col.between(0, 1).all():
            return c
    return None


# ---------------------------
# Resource loading
# ---------------------------
METRICS_PATH = "model_metrics.json"
FI_PATH = "feature_importance.csv"
PRED_PATH = "predictions.csv"
SHAP_PATH = "shap_summary.csv"
MODEL_GLOB = "final_model_Random Forest_20251106_223359.joblib"

metrics = safe_read_json(METRICS_PATH)
feature_importance = safe_read_csv(FI_PATH)
predictions_df = safe_read_csv(PRED_PATH)
shap_summary = safe_read_csv(SHAP_PATH)

# attempt to auto-load a model file if present
model_obj = None
for f in os.listdir('.'):
    if f.startswith(MODEL_GLOB) and f.endswith('.joblib'):
        model_obj = load_model(f)
        break

# ---------------------------
# Top header
# ---------------------------
col1, col2 = st.columns([0.78, 0.22])
with col1:
    st.title("Dataforce: Healthcare Fraud Detection")
with col2:
    if os.path.exists("Dataforce.png"):
        st.image("Dataforce.png", width=120)

st.divider()

# ---------------------------
# Sidebar navigation
# ---------------------------
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to:", [
    "Welcome",
    "Overview",
    "Model Insights",
    "Predict New Data",
    "About"
])

# small utilities shown in sidebar
st.sidebar.markdown("---")
if model_obj is not None:
    st.sidebar.success("Local model loaded")
else:
    st.sidebar.info("No local model file found (optional)")

# ---------------------------
# PAGE: Welcome
# ---------------------------
if page == "Welcome":
    st.header("Welcome to Dataforce: Healthcare Fraud Detection")
    st.markdown(
        """
        ### Project overview
        This dashboard helps you explore model performance, understand which features influence predictions, 
        and inspect dataset-level fraud statistics. It is designed for analysts, model reviewers and fraud investigators.
        """
    )

    st.subheader("How to navigate dashboard")
    st.markdown(
        """
        1. **Welcome** — You're here. Read this short guide before using the dashboard.
        2. **Overview** — View high-level model KPIs and prediction outcome charts.
        3. **Model Insights** — Inspect feature importance and SHAP summaries to understand why the model makes decisions.
        4. **Predict New Data** — Upload a CSV with a `Fraud` column (0/1) for quick dataset-level analysis and to download results.
        5. **About** — See the project team and contact information.
        """
    )

# ---------------------------
# PAGE: Overview
# ---------------------------
elif page == "Overview":
    st.header("Overview & Key Metrics")

    # top KPI strip
    k1, k2, k3, k4, k5 = st.columns(5)
    acc = metrics.get("Accuracy")
    prec = metrics.get("Precision")
    rec = metrics.get("Recall")
    f1 = metrics.get("F1_Score")
    roc = metrics.get("ROC_AUC")

    k1.metric("Accuracy", f"{acc:.3f}" if acc is not None else "N/A")
    k2.metric("Precision", f"{prec:.3f}" if prec is not None else "N/A")
    k3.metric("Recall", f"{rec:.3f}" if rec is not None else "N/A")
    k4.metric("F1 Score", f"{f1:.3f}" if f1 is not None else "N/A")
    k5.metric("ROC AUC", f"{roc:.3f}" if roc is not None else "N/A")

    st.markdown("---")

    # Model status card
    model_name = metrics.get("Model_Name", "Random Forest")
    model_ver = metrics.get("Version", "v1.0")
    training_date = metrics.get("Training_Date", "2025-11-06")

    st.subheader("Model Summary")
    colA, colB = st.columns([0.65, 0.35])
    with colA:
        st.markdown(f"**{model_name}** — {model_ver}")
        st.write(f"Trained on: {training_date}")
        st.write(metrics.get("Notes", "No additional notes."))
    with colB:
        # derive a rough health score
        health_score = 0.0
        has_acc = acc is not None
        has_roc = roc is not None
        if has_acc:
            try:
                health_score += float(acc)
            except Exception:
                pass
        if has_roc:
            try:
                health_score += float(roc)
            except Exception:
                pass
        denom = 0
        if has_acc:
            denom += 1
        if has_roc:
            denom += 1
        if denom > 0:
            health_score = health_score / denom
            grade = "A" if health_score > 0.85 else "B" if health_score > 0.7 else "C"
            st.metric("Model Health", f"{health_score:.2f} - {grade}")
        else:
            st.info("Health score: Requires metrics file")

    st.markdown("---")

    # Prediction distributions (if available)
    # ensure predictions_df exists and has required columns
    if 'predictions_df' in globals() and not predictions_df.empty and {"Actual", "Predicted"}.issubset(predictions_df.columns):
        st.subheader("Prediction Distribution & Outcomes")
        preds = predictions_df.copy()

        # normalize Actual label for display
        if preds['Actual'].dtype == object:
            preds['ActualLabel'] = preds['Actual']
        else:
            preds['ActualLabel'] = preds['Actual'].map({0: 'Legit', 1: 'Fraud'})

        col1, col2 = st.columns([0.5, 0.5])

        # safe counts dataframe for pie chart
        counts = preds['ActualLabel'].value_counts().rename_axis('label').reset_index(name='count')

        with col1:
            fig = px.pie(
                counts,
                values='count',
                names='label',
                title='Actual label split',
                color_discrete_sequence=["#38B2AC", "#E53E3E"]
            )
            st.plotly_chart(fig, use_container_width=True)

        # compute outcomes and plot
        preds['Outcome'] = preds.apply(
            lambda r: 'TP' if (r['Actual'] == 1 and r['Predicted'] == 1)
            else ('TN' if (r['Actual'] == 0 and r['Predicted'] == 0)
            else ('FP' if (r['Actual'] == 0 and r['Predicted'] == 1)
            else 'FN')) , axis=1
        )

        outcome_counts = preds['Outcome'].value_counts().rename_axis('outcome').reset_index(name='count')

        with col2:
            fig2 = px.bar(
                outcome_counts,
                x='outcome',
                y='count',
                title='Prediction Outcomes',
                color='outcome',
                color_discrete_map={
                    "TP": "#2B6CB0",
                    "TN": "#68D391",
                    "FP": "#F6AD55",
                    "FN": "#E53E3E"
                }
            )
            st.plotly_chart(fig2, use_container_width=True)

        st.markdown("Use *Explore Predictions* to drill down into individual records.")
    else:
        st.info("No predictions.csv with required columns found — charts hidden.")

    st.markdown("---")

    # Business KPIs area
    st.subheader("Business Impact (Estimates)")
    b1, b2, b3 = st.columns(3)
    b1.metric("Estimated Annual Fraud Loss", "R 3,200,000")
    b2.metric("Average Claim Flag Time", "2.3 days")
    b3.metric("Investigation Cases / Month", "84")
    
# ---------------------------
# PAGE: Model Insights
# ---------------------------
elif page == "Model Insights":
    st.header("Model Insights & Explainability")

    # Feature importance
    st.subheader("Feature Importance")
    if not feature_importance.empty:
        topn = st.slider("Show top N features", min_value=5, max_value=50, value=15)
        fig = px.bar(feature_importance.sort_values('Importance', ascending=True).tail(topn), x='Importance', y='Feature', orientation='h', title=f'Top {topn} Features')
        st.plotly_chart(fig, use_container_width=True)

        with st.expander("Show full feature importance table"):
            st.dataframe(feature_importance)
    else:
        st.info("feature_importance.csv not found or empty.")

    st.markdown("---")

    # SHAP or surrogate explanation
    st.subheader("SHAP / Model Explanation")
    if not shap_summary.empty:
        fig_shap = px.bar(shap_summary.sort_values('Mean_Abs_SHAP', ascending=True).tail(20), x='Mean_Abs_SHAP', y='Feature', orientation='h', title='Top SHAP Values')
        st.plotly_chart(fig_shap, use_container_width=True)
        with st.expander("Show full SHAP summary"):
            st.dataframe(shap_summary)
    else:
        st.info("No shap_summary.csv found. Consider generating SHAP summary CSV from model training notebooks.")

    st.markdown("---")

    # Quick textual insights (auto-generated from feature importance)
    st.subheader("Auto Insights")
    if not feature_importance.empty:
        top_feats = feature_importance.sort_values('Importance', ascending=False).head(5)['Feature'].tolist()
        st.write(f"Top features influencing the model: {', '.join(top_feats)}")
        st.write("Suggested checks:")
        st.write("- Check distribution drift for these features over time.")
        st.write("- Review encoding or missingness for top features.")
    else:
        st.write("No feature importance available to generate automatic insights.")

# ---------------------------
# PAGE: Predict New Data
# ---------------------------
elif page == "Predict New Data":
    st.subheader("🧾 Upload Data for Fraud Analysis")
    uploaded = st.file_uploader("Upload a CSV (must contain a 'Fraud' column)", type=['csv'])

    if uploaded:
        df = pd.read_csv(uploaded)

        st.write("Uploaded Data Preview:")
        st.dataframe(df.head())

        # ---- Validate Fraud column ----
        if "Fraud" not in df.columns:
            st.error("❌ The uploaded file must contain a 'Fraud' column (0 = non-fraud, 1 = fraud).")
            st.stop()

        # ---- Calculate fraud metrics ----
        vals = pd.to_numeric(df["Fraud"], errors="coerce").fillna(0).astype(int)
        fraud_total = int((vals == 1).sum())
        total = len(df)
        fraud_pct = 100 * fraud_total / max(total, 1)

        st.metric("Total Rows", total)
        st.metric("Fraudulent Claims", fraud_total)
        st.metric("Fraud %", f"{fraud_pct:.2f}%")

        # ---- Fraud Distribution visualization (reads 'Fraud') ----
        df["Fraud Probability"] = df["Fraud"].astype(float)  # substitute for real probabilities

        fig_pred = px.histogram(
            df,
            x="Fraud Probability",
            nbins=2,
            title=f"Fraud Distribution (Fraud {fraud_pct:.2f}%)",
            color="Fraud",
            color_discrete_map={0: "#4CAF50", 1: "#F44336"}
        )
        st.plotly_chart(fig_pred, use_container_width=True)

        # ---- Allow Download ----
        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Download Data with Fraud Indicators",
            data=csv,
            file_name="fraud_analysis.csv",
            mime="text/csv"
        )

        st.success("Analysis complete!")
# ---------------------------
# PAGE: About
# ---------------------------
elif page == "About":
    st.header("About the Project Team")
    st.markdown(
        """
        Project Team

        - Lorraine Oratile Kodisang — Data Engineer
        - Kamogelo Realeboga Kale — Feature Engineer
        - Thabo Baleni — Modeling Lead
        - Lesego Lengolo — Explainability
        - Dineo Mmope — Documentation
        
        """
    )

# ---------------------------
# Footer
# ---------------------------
st.markdown("---")
st.caption(f"Last updated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%SZ')} — Dataforce Fraud Detection")

