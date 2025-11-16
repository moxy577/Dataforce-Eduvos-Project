import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import joblib
import json
import numpy as np

# --------------------------------------------
# PAGE CONFIG
# --------------------------------------------
st.set_page_config(
    page_title="Healthcare Fraud Detection Dashboard",
    layout="wide"
)

# --------------------------------------------
# LOAD DATA
# --------------------------------------------
@st.cache_data
def load_metrics():
    with open(r"model_metrics.json") as f:
        return json.load(f)

@st.cache_data
def load_feature_importance():
    return pd.read_csv(r"feature_importance.csv")

@st.cache_data
def load_predictions():
    return pd.read_csv(r"predictions.csv")

@st.cache_resource
def load_model():
    return joblib.load(r"final_model_Random Forest_20251106_223359.joblib")

# --------------------------------------------
# HEADER
# --------------------------------------------
st.title("Healthcare Insurance Fraud Detection Dashboard")

# --------------------------------------------
# SIDEBAR NAVIGATION
# --------------------------------------------
page = st.sidebar.radio("Navigation", [
    "Overview",
    "Model Insights",
    "Predict New Data",
    "Explore Predictions"
])

# --------------------------------------------
# PAGE 1: OVERVIEW
# --------------------------------------------
if page == "Overview":
    metrics = load_metrics()
    st.subheader("Model Evaluation Summary")

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Accuracy", f"{metrics['Accuracy']:.3f}")
    col2.metric("Precision", f"{metrics['Precision']:.3f}")
    col3.metric("Recall", f"{metrics['Recall']:.3f}")
    col4.metric("F1 Score", f"{metrics['F1_Score']:.3f}")
    col5.metric("ROC-AUC", f"{metrics['ROC_AUC']:.3f}")

    # Load predictions for charting
    preds = load_predictions()
    st.markdown("### Fraud vs Non-Fraud Distribution")

    fraud_counts = preds["Actual"].value_counts().rename({0: "Legit", 1: "Fraud"})
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

# --------------------------------------------
# PAGE 2: MODEL INSIGHTS
# --------------------------------------------
elif page == "Model Insights":
    st.subheader("Feature Importance & SHAP Analysis")

    fi = load_feature_importance()
    fig_imp = px.bar(
        fi.head(15),
        x="Importance", y="Feature",
        orientation="h",
        color="Importance",
        color_continuous_scale="viridis",
        title="Top 15 Most Important Features"
    )
    fig_imp.update_layout(yaxis={'categoryorder':'total ascending'})
    st.plotly_chart(fig_imp, use_container_width=True)

    shap_data = pd.read_csv(r"shap_summary.csv")
    fig_shap = px.bar(
        shap_data.head(15),
        x="Mean_Abs_SHAP",
        y="Feature",
        orientation="h",
        color="Mean_Abs_SHAP",
        color_continuous_scale="Tealgrn",
        title="Top SHAP Value Impact Features"
    )
    fig_shap.update_layout(yaxis={'categoryorder':'total ascending'})
    st.plotly_chart(fig_shap, use_container_width=True)

# --------------------------------------------
# PAGE 3: PREDICTION TOOL
# --------------------------------------------
elif page == "Predict New Data":
    st.subheader("🧾 Upload Data for Fraud Analysis")
    uploaded = st.file_uploader("Upload a CSV (must contain a 'fraud' column)", type=['csv'])

    if uploaded:
        df = pd.read_csv(uploaded)

        st.write("Uploaded Data Preview:")
        st.dataframe(df.head())

        # ---- Validate fraud column ----
        if "Fraud" not in df.columns:
            st.error("❌ The uploaded file must contain a 'fraud' column (0 = non-fraud, 1 = fraud).")
            st.stop()

        # ---- Calculate fraud metrics ----
        vals = pd.to_numeric(df["Fraud"], errors="coerce").fillna(0).astype(int)
        fraud_total = int((vals == 1).sum())
        total = len(df)
        fraud_pct = 100 * fraud_total / max(total, 1)

        st.metric("Total Rows", total)
        st.metric("Fraudulent Claims", fraud_total)
        st.metric("Fraud %", f"{fraud_pct:.2f}%")

        # ---- Fraud Probability Distribution Substitute ----
        # No model → use fraud=1 as 100% fraud, fraud=0 as 0% fraud
        df["Fraud Probability"] = df["fraud"].astype(float)

        fig_pred = px.histogram(
            df,
            x="Fraud Probability",
            nbins=2,
            title=f"Fraud Distribution (Fraud {fraud_pct:.2f}%)",
            color="fraud",
            color_discrete_map={0: "#4CAF50", 1: "#F44336"}
        )
        st.plotly_chart(fig_pred, use_container_width=True)

        # ---- Allow Download ----
        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Download Data with Fraud % Indicators",
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

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Total Predictions", len(df))
        st.metric("Detected Frauds", df["Predicted"].sum())
    with col2:
        st.metric("False Positives", ((df["Actual"] == 0) & (df["Predicted"] == 1)).sum())
        st.metric("False Negatives", ((df["Actual"] == 1) & (df["Predicted"] == 0)).sum())

    st.markdown("### Prediction Confidence")
    fig_hist = px.histogram(
        df,
        x="Fraud_Probability",
        color="Actual",
        barmode="overlay",
        title="Model Confidence by Actual Label",
        color_discrete_map={0: "#38B2AC", 1: "#E53E3E"}
    )
    st.plotly_chart(fig_hist, use_container_width=True)

    st.dataframe(df.head(50))

# --------------------------------------------
# FOOTER
# --------------------------------------------
st.markdown("---")
st.caption("Healthcare Fraud Detection Dashboard")