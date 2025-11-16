#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
from typing import List

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import joblib
import json

# --------------------------------------------
# PAGE CONFIG
# --------------------------------------------
st.set_page_config(
    page_title="Healthcare Fraud Detection Dashboard",
    layout="wide",
    page_icon="🧠"
)

# --------------------------------------------
# HEADER & LOGO (top-right)
# --------------------------------------------
col1, col2 = st.columns([0.85, 0.15])
with col1:
    st.title("Healthcare Insurance Fraud Detection Dashboard")
    st.markdown("A dashboard to explore model evaluation, feature importance, and dataset-level fraud metrics.")
with col2:
    # show Dataforce logo in the top-right if available
    if os.path.exists("Dataforce.png"):
        st.image("Dataforce.png", width=100)
    else:
        st.empty()

st.divider()

# --------------------------------------------
# LOAD DATA / RESOURCES
# --------------------------------------------
@st.cache_data
def load_metrics():
    if os.path.exists("model_metrics.json"):
        with open("model_metrics.json") as f:
            return json.load(f)
    return {}

@st.cache_data
def load_feature_importance():
    if os.path.exists("feature_importance.csv"):
        return pd.read_csv("feature_importance.csv")
    return pd.DataFrame(columns=["Feature", "Importance"])

@st.cache_data
def load_predictions():
    if os.path.exists("predictions.csv"):
        return pd.read_csv("predictions.csv")
    return pd.DataFrame()

@st.cache_resource
def load_model():
    # optional local model: keep attempt but don't fail app if missing
    model_path = "final_model_Random Forest_20251106_223359.joblib"
    if os.path.exists(model_path):
        try:
            return joblib.load(model_path)
        except Exception:
            return None
    return None

# --------------------------------------------
# SIDEBAR NAVIGATION (title updated)
# --------------------------------------------
st.sidebar.title("Dataforce: Health Insurance fraud dashboard")
page = st.sidebar.radio("Navigation", [
    "Welcome",
    "Overview",
    "Model Insights",
    "Predict New Data",
    "Explore Predictions",
    "About Us"
])

# --------------------------------------------
# PAGE: WELCOME
# --------------------------------------------
if page == "Welcome":
    st.header("Welcome — Healthcare Fraud Detection Project")
    st.markdown(
        """
        **Project purpose**
        - Detect potentially fraudulent health insurance claims to reduce financial loss and speed up investigative workflows.
        
        **Goals**
        - Build and evaluate machine-learning models that separate fraudulent from non-fraudulent claims.
        - Provide clear visualizations for model performance, feature importance, and dataset-level fraud metrics.
        - Offer an interactive interface for analysts to explore predictions and sample datasets.
        
        **Who should use this dashboard?**
        - Data scientists verifying model behaviour.
        - Fraud investigators looking for trends and suspicious patterns.
        - Stakeholders who want quick KPIs about estimated fraud rates and model performance.
        
        **How to use this dashboard (quick start)**
        1. Click **Overview** to see model evaluation metrics and a breakdown of predicted vs actual fraud (if prediction files are present).
        2. Visit **Model Insights** for feature importance and SHAP summaries (if available).
        3. Use **Predict New Data** to upload a dataset with a `Fraud` column for dataset-level analysis (fraud counts and distribution).
        4. Use **Explore Predictions** to review model predictions, errors, and confidence distributions (requires `predictions.csv`).
        
        If you are a first-time viewer, read each page from top to bottom. Files the app expects (optional/for advanced pages):
        - `model_metrics.json` — model evaluation metrics (Accuracy, Precision, Recall, F1_Score, ROC_AUC)
        - `feature_importance.csv` — feature importance table
        - `predictions.csv` — predictions file with columns `Actual`, `Predicted`, `Fraud_Probability` (or similar)
        
        For support or questions, contact the project team via the **About Us** page.
        """
    )

# --------------------------------------------
# PAGE 1: OVERVIEW
# --------------------------------------------
elif page == "Overview":
    metrics = load_metrics()
    st.subheader("Model Evaluation Summary")

    # safe read of metrics keys
    acc = metrics.get("Accuracy", None)
    prec = metrics.get("Precision", None)
    rec = metrics.get("Recall", None)
    f1 = metrics.get("F1_Score", None)
    roc = metrics.get("ROC_AUC", None)

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Accuracy", f"{acc:.3f}" if acc is not None else "N/A")
    col2.metric("Precision", f"{prec:.3f}" if prec is not None else "N/A")
    col3.metric("Recall", f"{rec:.3f}" if rec is not None else "N/A")
    col4.metric("F1 Score", f"{f1:.3f}" if f1 is not None else "N/A")
    col5.metric("ROC-AUC", f"{roc:.3f}" if roc is not None else "N/A")

    # Load predictions for charting (optional)
    preds = load_predictions()
    if not preds.empty and {"Actual", "Predicted"}.issubset(preds.columns):
        st.markdown("### Fraud vs Non-Fraud Distribution (Actual)")
        fraud_counts = preds["Actual"].map({0: "Legit", 1: "Fraud"}).value_counts()
        fig_pie = px.pie(
            values=fraud_counts.values,
            names=fraud_counts.index,
            title="Actual Fraud Distribution",
            color_discrete_sequence=["#38B2AC", "#E53E3E"]
        )
        st.plotly_chart(fig_pie, use_container_width=True)

        # Prediction outcome confusion plot
        preds["Outcome"] = preds.apply(lambda x: "TP" if x["Actual"] == 1 and x["Predicted"] == 1
                                       else "TN" if x["Actual"] == 0 and x["Predicted"] == 0
                                       else "FP" if x["Actual"] == 0 and x["Predicted"] == 1
                                       else "FN", axis=1)
        outcome_counts = preds["Outcome"].value_counts()
        fig_bar = px.bar(
            outcome_counts,
            x=outcome_counts.index,
            y=outcome_counts.values,
            color=outcome_counts.index,
            color_discrete_map={"TP": "#2B6CB0", "TN": "#68D391", "FP": "#F6AD55", "FN": "#E53E3E"},
            title="Prediction Outcome Breakdown"
        )
        st.plotly_chart(fig_bar, use_container_width=True)
    else:
        st.info("No `predictions.csv` found or it lacks required columns `Actual` and `Predicted`. Skipping prediction charts.")

# --------------------------------------------
# PAGE 2: MODEL INSIGHTS
# --------------------------------------------
elif page == "Model Insights":
    st.subheader("Feature Importance & SHAP Analysis")

    fi = load_feature_importance()
    if not fi.empty:
        fig_imp = px.bar(
            fi.head(15),
            x="Importance", y="Feature",
            orientation="h",
            color="Importance",
            color_continuous_scale="viridis",
            title="Top 15 Most Important Features"
        )
        fig_imp.update_layout(yaxis={'categoryorder': 'total ascending'})
        st.plotly_chart(fig_imp, use_container_width=True)
    else:
        st.info("feature_importance.csv not found or empty.")

    shap_path = "shap_summary.csv"
    if os.path.exists(shap_path):
        shap_data = pd.read_csv(shap_path)
        if not shap_data.empty:
            fig_shap = px.bar(
                shap_data.head(15),
                x="Mean_Abs_SHAP",
                y="Feature",
                orientation="h",
                color="Mean_Abs_SHAP",
                color_continuous_scale="Tealgrn",
                title="Top SHAP Value Impact Features"
            )
            fig_shap.update_layout(yaxis={'categoryorder': 'total ascending'})
            st.plotly_chart(fig_shap, use_container_width=True)
        else:
            st.info("shap_summary.csv found but empty.")
    else:
        st.info("shap_summary.csv not found. Place SHAP summary CSV if available.")

# --------------------------------------------
# PAGE 3: PREDICTION TOOL (dataset-level analysis)
# --------------------------------------------
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

# --------------------------------------------
# PAGE 4: EXPLORE PREDICTIONS
# --------------------------------------------
elif page == "Explore Predictions":
    st.subheader("Explore Model Predictions and Errors")
    df = load_predictions()

    if df.empty:
        st.info("predictions.csv not found or empty.")
    else:
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total Predictions", len(df))
            if "Predicted" in df.columns:
                st.metric("Detected Frauds", int(df["Predicted"].sum()))
            else:
                st.metric("Detected Frauds", "N/A")
        with col2:
            if {"Actual", "Predicted"}.issubset(df.columns):
                st.metric("False Positives", int(((df["Actual"] == 0) & (df["Predicted"] == 1)).sum()))
                st.metric("False Negatives", int(((df["Actual"] == 1) & (df["Predicted"] == 0)).sum()))
            else:
                st.metric("False Positives", "N/A")
                st.metric("False Negatives", "N/A")

        st.markdown("### Prediction Confidence")
        # use possible probability column names safely
        prob_col_candidates = ["Fraud_Probability", "Fraud Probability", "Fraud_Prob", "FraudProbability"]
        prob_col = next((c for c in prob_col_candidates if c in df.columns), None)
        if prob_col:
            fig_hist = px.histogram(
                df,
                x=prob_col,
                color="Actual" if "Actual" in df.columns else None,
                barmode="overlay",
                title="Model Confidence by Actual Label",
                color_discrete_map={0: "#38B2AC", 1: "#E53E3E"}
            )
            st.plotly_chart(fig_hist, use_container_width=True)
        else:
            st.info("No probability column found in predictions CSV. Expected one of: " + ", ".join(prob_col_candidates))

        st.dataframe(df.head(50))

# --------------------------------------------
# PAGE: ABOUT US
# --------------------------------------------
elif page == "About Us":
    st.subheader("About the Project Team")
    st.markdown(
        """
        **Project Members & Roles**
        
        - **Lorraine Oratile Kodisang** — Feature Engineer  
        - **Kamogelo Realeboga Kale** — Feature Engineer  
        - **Thabo Baleni** — Modeling Lead  
        - **Lesego Lengolo** — Explainability  
        - **Dineo Mmope** — Documentation  
        """
    )
    st.markdown("Dataforce:Healthcare Insurance Fraud")

# --------------------------------------------
# FOOTER
# --------------------------------------------
st.markdown("---")
st.caption("Healthcare Fraud Detection Dashboard — Dataforce")

