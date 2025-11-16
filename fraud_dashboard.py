import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import joblib
import json
from PIL import Image
import numpy as np

# --------------------------------------------
# PAGE CONFIG
# --------------------------------------------
st.set_page_config(
    page_title="Healthcare Fraud Detection Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --------------------------------------------
# STYLING
# --------------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    background-color: #F9FAFB;
    color: #1A202C;
}
h1, h2, h3 {
    color: #2D3748;
    font-weight: 700;
}
.highlight-heading {
    background: linear-gradient(to right, #319795, #3182CE);
    padding: 20px;
    border-radius: 12px;
    margin-bottom: 30px;
    color: white;
    text-align: center;
}
.footer {
    text-align: center;
    color: #A0AEC0;
    font-size: 13px;
    margin-top: 40px;
}
</style>
""", unsafe_allow_html=True)

# --------------------------------------------
# LOAD FUNCTIONS
# --------------------------------------------
@st.cache_resource
def load_model():
    return joblib.load(r"C:\Users\kamog\Downloads\dataset\final_model_Random Forest_20251106_223359.joblib")

@st.cache_data
def load_metrics():
    with open(r"C:\Users\kamog\Downloads\dataset\dashboard_data\model_metrics.json", "r") as f:
        metrics = json.load(f)
    return metrics

@st.cache_data
def load_feature_importance():
    return pd.read_csv(r"C:\Users\kamog\Downloads\dataset\dashboard_data\feature_importance.csv")

# --------------------------------------------
# MAIN LAYOUT
# --------------------------------------------
st.markdown('<div class="highlight-heading"><h1>Healthcare Insurance Fraud Detection Dashboard</h1></div>', unsafe_allow_html=True)

page = st.sidebar.radio("Navigation", ["Overview", "Model Insights", "Predict New Data", "About Project"])

# --------------------------------------------
# PAGE 1 - OVERVIEW
# --------------------------------------------
if page == "Overview":
    st.header("Model Evaluation Summary")

    metrics = load_metrics()
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Accuracy", f"{metrics['Accuracy']:.3f}")
    col2.metric("Precision", f"{metrics['Precision']:.3f}")
    col3.metric("Recall", f"{metrics['Recall']:.3f}")
    col4.metric("F1 Score", f"{metrics['F1_Score']:.3f}")
    col5.metric("ROC-AUC", f"{metrics['ROC_AUC']:.3f}")

    st.markdown("### Model Performance Visualizations")
    col1, col2, col3 = st.columns(3)
    col1.image(r"C:\Users\kamog\Downloads\dataset\dashboard_data\confusion_matrix.png", caption="Confusion Matrix", use_container_width=True)
    col2.image(r"C:\Users\kamog\Downloads\dataset\dashboard_data\roc_curve.png", caption="ROC Curve", use_container_width=True)
    col3.image(r"C:\Users\kamog\Downloads\dataset\dashboard_data\precision_recall_curve.png", caption="Precision–Recall Curve", use_container_width=True)

# --------------------------------------------
# PAGE 2 - MODEL INSIGHTS
# --------------------------------------------
elif page == "Model Insights":
    st.header("Feature Importance & SHAP Analysis")

    st.markdown("### Feature Importance")
    fi = load_feature_importance()
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.barplot(data=fi.head(15), x="Importance", y="Feature", palette="viridis", ax=ax)
    ax.set_title("Top 15 Most Important Features")
    st.pyplot(fig)

    st.markdown("### SHAP Explainability")
    col1, col2 = st.columns(2)
    col1.image(r"C:\Users\kamog\Downloads\dataset\dashboard_data\shap_bar_plot.png", caption="Mean SHAP Importance", use_container_width=True)
    col2.image(r"C:\Users\kamog\Downloads\dataset\dashboard_data\shap_summary_plot.png", caption="SHAP Summary (Beeswarm)", use_container_width=True)

# --------------------------------------------
# PAGE 3 - PREDICTION TOOL
# --------------------------------------------
elif page == "Predict New Data":
    st.header("Predict Fraudulent Claims")
    st.info("Upload a single-row CSV (same columns as training data, without the target column) to predict fraud probability.")

    uploaded = st.file_uploader("Upload claim data (CSV)", type=["csv"])
    if uploaded:
        input_df = pd.read_csv(uploaded)
        model = load_model()
        prediction = model.predict(input_df)
        proba = model.predict_proba(input_df)[:, 1]
        result = "Fraudulent" if prediction[0] == 1 else "Legitimate"

        st.success(f"Prediction: **{result}**")
        st.metric("Fraud Probability", f"{proba[0]*100:.2f}%")

        st.subheader("Input Data")
        st.dataframe(input_df)

# --------------------------------------------
# PAGE 4 - ABOUT PROJECT
# --------------------------------------------
elif page == "About Project":
    st.header("About This Project")

    st.markdown("""
    ### Project Overview
    This dashboard demonstrates a machine learning model designed to detect **fraudulent healthcare insurance claims** using structured claim data.  
    It integrates model metrics, feature importance, and explainability through SHAP visualizations.

    ### Model
    - **Algorithm:** Random Forest Classifier  
    - **Evaluation Metrics:** Accuracy, Precision, Recall, F1, ROC-AUC  
    - **Explainability:** SHAP value analysis for feature-level impact

    ### Data
    The model was trained on cleaned and preprocessed healthcare claims data, with balanced fraud vs non-fraud classes.
    """)

# --------------------------------------------
# FOOTER
# --------------------------------------------
st.markdown('<div class="footer">Healthcare Fraud Detection Dashboard', unsafe_allow_html=True)