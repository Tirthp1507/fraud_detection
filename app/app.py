# app/app.py

import streamlit as st
import pandas as pd
import plotly.express as px
import os
import joblib
import numpy as np
from datetime import datetime

st.set_page_config(page_title="Fraud Detection Dashboard", layout="wide", page_icon="📊")

# --- STYLING ---
st.markdown("""
    <style>
    .main {background-color: #f9f9f9;}
    .block-container {padding-top: 2rem;}
    h1, h2, h3 {color: #0d3b66;}
    .stMetricValue {font-size: 24px;}
    </style>
""", unsafe_allow_html=True)

# Title
st.markdown("""
# 🔍 Fraud Detection Dashboard
Analyze transactions, prediction results, and model confidence.
---
""")

# Sidebar input method toggle
st.sidebar.header("📂 Data Input Method")
data_mode = st.sidebar.radio("Select data source:", ["Upload CSV", "Use Demo Data", "Manual Entry"])

# Paths
data_path = "data/predictions.csv"
model_path = "models/fraud_model.pkl"
demo_path = "data/demo_fraud_data.csv"

# Load model
@st.cache_resource
def load_model():
    return joblib.load(model_path)

model = load_model()

# Process file function
def process_file(df):
    expected_features = [
        "Time", "V1", "V2", "V3", "V4", "V5", "V6", "V7", "V8", "V9",
        "V10", "V11", "V12", "V13", "V14", "V15", "V16", "V17", "V18", "V19",
        "V20", "V21", "V22", "V23", "V24", "V25", "V26", "V27", "V28", "Amount"
    ]
    aligned_df = df.reindex(columns=expected_features, fill_value=0.0)
    predictions = model.predict(aligned_df)
    probs = model.predict_proba(aligned_df)
    df["Prediction"] = ["Fraud" if p == 1 else "Non-Fraud" for p in predictions]
    df["Confidence"] = (np.max(probs, axis=1) * 100).round(2)
    df["Timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return df

# Handle input method
if data_mode == "Upload CSV":
    uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type="csv")
    if uploaded_file is not None:
        df = process_file(pd.read_csv(uploaded_file))
        st.sidebar.success("✅ Predictions generated successfully!")
    elif os.path.exists(data_path):
        df = pd.read_csv(data_path)
    else:
        st.error("❌ Please upload a file or add default data.")
        st.stop()

elif data_mode == "Use Demo Data":
    if os.path.exists(demo_path):
        df = process_file(pd.read_csv(demo_path))
    else:
        st.error("❌ Demo data not found.")
        st.stop()

elif data_mode == "Manual Entry":
    st.sidebar.markdown("Enter values below to check a single transaction")
    manual_input = {}
    for feature in ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]:
        manual_input[feature] = st.sidebar.number_input(f"{feature}", value=0.0, step=0.01)
    df = pd.DataFrame([manual_input])
    df = process_file(df)
    st.success("✅ Prediction for manual entry complete")

# KPIs
st.subheader("📈 Key Insights")
total_transactions = len(df)
total_frauds = df[df["Prediction"] == "Fraud"].shape[0]
avg_confidence = round(df["Confidence"].mean(), 2)

kpi1, kpi2, kpi3 = st.columns(3)
kpi1.metric("Total Transactions", f"{total_transactions:,}")
kpi2.metric("Fraudulent Transactions", f"{total_frauds:,}", delta=f"{(total_frauds/total_transactions)*100:.2f}%")
kpi3.metric("Average Confidence", f"{avg_confidence}%")

st.divider()

# Donut Chart
st.subheader("📊 Prediction Distribution")
pie_data = df["Prediction"].value_counts().reset_index()
pie_data.columns = ["Prediction", "Count"]
fig_pie = px.pie(pie_data, names="Prediction", values="Count", hole=0.5,
                 color_discrete_sequence=["#e63946", "#2a9d8f"],
                 title="Fraud vs Non-Fraud Transactions")
st.plotly_chart(fig_pie, use_container_width=True)

# Confidence Distribution
st.subheader("🎯 Confidence Score Distribution")
fig_hist = px.histogram(df, x="Confidence", color="Prediction", nbins=50, barmode="overlay",
                        color_discrete_map={"Fraud": "#e63946", "Non-Fraud": "#2a9d8f"})
st.plotly_chart(fig_hist, use_container_width=True)

# Time Trend (if timestamp present)
if "Timestamp" in df.columns:
    st.subheader("⏱️ Fraud Trend Over Time")
    df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors='coerce')
    frauds = df[df["Prediction"] == "Fraud"]
    frauds_by_time = frauds.groupby(df["Timestamp"].dt.date).size().reset_index(name="Fraud Count")
    fig_trend = px.line(frauds_by_time, x="Timestamp", y="Fraud Count",
                        title="Fraudulent Transactions Over Time",
                        markers=True, line_shape="spline")
    st.plotly_chart(fig_trend, use_container_width=True)

# Transaction Table
st.subheader("📋 Transactions Table")
st.dataframe(df[[col for col in ["Amount", "Prediction", "Confidence", "Timestamp"] if col in df.columns]], use_container_width=True, height=350)

# PDF Download Option
pdf_path = os.path.join("data", "fraud_detection_dashboard.pdf")
if os.path.exists(pdf_path):
    st.divider()
    st.subheader("📎 Download Power BI Report")
    with open(pdf_path, "rb") as f:
        st.download_button("⬇ Download fraud_detection_dashboard.pdf", f, file_name="fraud_dashboard.pdf")

# Footer
st.markdown("---")
st.markdown("""
<center>
✨ Built with ❤️ by Tirth | Streamlit + Plotly + ML | 2025 Internship Project<br>
<i>UI enhanced with custom styling, demo/live/manual modes</i>
</center>
""", unsafe_allow_html=True)