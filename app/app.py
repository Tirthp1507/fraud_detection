# app/app.py

import streamlit as st
import pandas as pd
import plotly.express as px
import joblib
import os
import numpy as np
from datetime import datetime

# Set page config
st.set_page_config(page_title="Fraud Detection Dashboard", layout="wide", page_icon="📊")

# Custom CSS for styling
st.markdown("""
    <style>
    .main {background-color: #f4f4f4;}
    .block-container {padding-top: 2rem;}
    h1, h2, h3 {color: #0d3b66;}
    .stMetricValue {font-size: 22px;}
    </style>
""", unsafe_allow_html=True)

# Title
st.markdown("""
# 🔍 Fraud Detection Dashboard
Analyze transactions, prediction results, and model confidence.
---
""")

# Sidebar toggle
st.sidebar.header("📂 Data Input Method")
data_mode = st.sidebar.radio("Select data source:", ["Upload CSV", "Use Demo Data", "Manual Entry"])

# Paths
model_path = "models/fraud_model.pkl"
demo_path = "data/predictions.csv"
pdf_path = "data/fraud_dashboard_v2.pdf"

# Load ML model
@st.cache_resource
def load_model():
    return joblib.load(model_path)

model = load_model()

# Process uploaded/manual data
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

# Handle each data input option
if data_mode == "Upload CSV":
    uploaded_file = st.sidebar.file_uploader("Upload transaction CSV", type="csv")
    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file)
            df = process_file(df)
            st.sidebar.success("✅ Predictions generated successfully!")
        except Exception as e:
            st.error(f"Error: {e}")
            st.stop()
    else:
        st.warning("Please upload a CSV file.")
        st.stop()

elif data_mode == "Use Demo Data":
    if os.path.exists(demo_path):
        df = pd.read_csv(demo_path)
        df = process_file(df)
    else:
        st.error("Demo file not found.")
        st.stop()

elif data_mode == "Manual Entry":
    st.sidebar.markdown("Enter values to simulate a transaction")
    manual_input = {}
    for feature in ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]:
        manual_input[feature] = st.sidebar.number_input(f"{feature}", value=0.0)
    df = pd.DataFrame([manual_input])
    df = process_file(df)
    st.success("✅ Manual entry processed")

# KPIs
st.subheader("📈 Key Insights")
total = len(df)
frauds = df[df["Prediction"] == "Fraud"].shape[0]
confidence = round(df["Confidence"].mean(), 2)

col1, col2, col3 = st.columns(3)
col1.metric("Total Transactions", f"{total:,}")
col2.metric("Fraudulent", f"{frauds:,}", delta=f"{(frauds/total)*100:.2f}%")
col3.metric("Avg. Confidence", f"{confidence}%")

st.divider()

# Pie / Donut Chart
st.subheader("📊 Prediction Distribution")
pie_data = df["Prediction"].value_counts().reset_index()
pie_data.columns = ["Prediction", "Count"]
fig_pie = px.pie(pie_data, names="Prediction", values="Count", hole=0.5,
                 color_discrete_sequence=["#2a9d8f", "#e63946"])
st.plotly_chart(fig_pie, use_container_width=True)

# Confidence Histogram
st.subheader("🎯 Confidence Score Distribution")
fig_hist = px.histogram(df, x="Confidence", color="Prediction", nbins=50,
                        barmode="overlay", color_discrete_map={"Fraud": "#e63946", "Non-Fraud": "#2a9d8f"})
st.plotly_chart(fig_hist, use_container_width=True)

# Line Chart - Time Trend
if "Timestamp" in df.columns:
    st.subheader("⏱️ Fraud Trends Over Time")
    df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors='coerce')
    frauds_df = df[df["Prediction"] == "Fraud"]
    frauds_by_day = frauds_df.groupby(df["Timestamp"].dt.date).size().reset_index(name="Fraud Count")
    if not frauds_by_day.empty:
        fig_trend = px.line(frauds_by_day, x="Timestamp", y="Fraud Count", markers=True, title="Frauds Over Time")
        st.plotly_chart(fig_trend, use_container_width=True)

# Scatter Plot - Confidence vs Amount
if "Amount" in df.columns:
    st.subheader("💸 Scatter: Confidence vs Transaction Amount")
    fig_scatter = px.scatter(df, x="Amount", y="Confidence", color="Prediction",
                             color_discrete_map={"Fraud": "#e63946", "Non-Fraud": "#2a9d8f"},
                             title="Confidence by Transaction Amount")
    st.plotly_chart(fig_scatter, use_container_width=True)

# Transactions Table
st.subheader("📋 Transaction Records")
columns_to_show = [c for c in ["Amount", "Prediction", "Confidence", "Timestamp"] if c in df.columns]
st.dataframe(df[columns_to_show], use_container_width=True, height=350)

# PDF Report
if os.path.exists(pdf_path):
    st.divider()
    st.subheader("📎 Power BI Dashboard")
    with open(pdf_path, "rb") as f:
        st.download_button("⬇ Download PDF Report", f, file_name="fraud_dashboard.pdf")
