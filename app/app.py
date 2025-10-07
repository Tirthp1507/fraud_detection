import streamlit as st
import pandas as pd
import plotly.express as px
import joblib
import os
import numpy as np
from datetime import datetime

# Set page configuration. This must be the first Streamlit command.
st.set_page_config(
    page_title="Fraud Detection Dashboard",
    layout="wide",
    page_icon="🔍"
)

# --- Custom CSS for Styling ---
st.markdown("""
    <style>
        .main {
            background-color: #0E1117; /* Dark background */
            color: #FAFAFA; /* Light text */
        }
        .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
        }
        h1, h2, h3 {
            color: #1E88E5; /* A vibrant blue for headers */
        }
        .stMetric {
            background-color: #262730;
            border-radius: 10px;
            padding: 15px;
        }
        .stMetricValue {
            font-size: 2.2rem;
            color: #FAFAFA;
        }
        .stMetricLabel {
            color: #A0A0A0;
        }
        .stMetricDelta {
            color: #EF5350; /* Red for delta */
        }
    </style>
""", unsafe_allow_html=True)

# --- Page Title and Header ---
st.markdown("<h1 style='text-align: center;'>🔍 Fraud Detection Dashboard</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Analyze transactions, prediction results, and model confidence.</p>", unsafe_allow_html=True)
st.divider()

# --- Sidebar for Data Input ---
st.sidebar.header("📂 Data Input Method")
data_mode = st.sidebar.radio("Select data source:", ["Upload CSV", "Use Demo Data", "Manual Entry"], index=1)

# --- File Paths (using os.path.join for cross-platform compatibility) ---
base_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(base_dir) # Go up one level from app/ to the project root

model_path = os.path.join(parent_dir, "models", "fraud_model.pkl")
demo_path = os.path.join(parent_dir, "data", "predictions.csv")
pdf_path = os.path.join(parent_dir, "data", "fraud_dashboard_v2.pdf")


# --- Load Machine Learning Model ---
# Caching the model loading function for better performance.
@st.cache_resource
def load_model():
    """Load the pre-trained fraud detection model from disk."""
    if not os.path.exists(model_path):
        st.error(f"Model file not found at {model_path}")
        return None
    try:
        model = joblib.load(model_path)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

model = load_model()
if model is None:
    st.stop() # Stop execution if model fails to load

# --- Data Processing Function ---
def process_data(df):
    """Process the input dataframe to generate predictions and confidence scores."""
    # These are the features the model was trained on.
    expected_features = [
        "Time", "V1", "V2", "V3", "V4", "V5", "V6", "V7", "V8", "V9",
        "V10", "V11", "V12", "V13", "V14", "V15", "V16", "V17", "V18", "V19",
        "V20", "V21", "V22", "V23", "V24", "V25", "V26", "V27", "V28", "Amount"
    ]
    
    # Ensure the dataframe has all the expected columns, filling missing ones with 0.
    aligned_df = df.reindex(columns=expected_features, fill_value=0.0)

    # Make predictions
    predictions = model.predict(aligned_df)
    probs = model.predict_proba(aligned_df)

    # Add results to the original dataframe
    df["Prediction"] = ["Fraud" if p == 1 else "Non-Fraud" for p in predictions]
    df["Confidence"] = (np.max(probs, axis=1) * 100).round(2)
    df["Timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return df

# --- Main Logic for Data Handling ---
df = None
if data_mode == "Upload CSV":
    uploaded_file = st.sidebar.file_uploader("Upload your transaction CSV file", type="csv")
    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file)
            df = process_data(df)
            st.sidebar.success("✅ Predictions generated successfully!")
        except Exception as e:
            st.error(f"Error processing CSV file: {e}")
            st.stop()
    else:
        st.info("Awaiting CSV file upload. Please upload a file to begin analysis.")
        st.stop()

elif data_mode == "Use Demo Data":
    if os.path.exists(demo_path):
        df = pd.read_csv(demo_path)
        # The demo data might already have predictions, so we re-process it for consistency
        df = process_data(df)
        st.sidebar.info("📊 Loaded demo data.")
    else:
        st.error(f"Demo file not found at {demo_path}")
        st.stop()

elif data_mode == "Manual Entry":
    st.sidebar.markdown("Enter values to simulate a single transaction:")
    manual_input = {}
    # Create two columns for a more compact layout
    c1, c2 = st.sidebar.columns(2)
    features = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]
    for i, feature in enumerate(features):
        if i % 2 == 0:
            manual_input[feature] = c1.number_input(f"{feature}", value=0.0, format="%.2f")
        else:
            manual_input[feature] = c2.number_input(f"{feature}", value=0.0, format="%.2f")
    
    if st.sidebar.button("Predict Manual Entry"):
        df = pd.DataFrame([manual_input])
        df = process_data(df)
        st.success("✅ Manual entry processed successfully!")
    else:
        st.info("Enter transaction details and click 'Predict' to see the result.")
        st.stop()

# --- Dashboard Display ---
if df is not None and not df.empty:
    st.subheader("📈 Key Performance Indicators")
    total_transactions = len(df)
    fraud_count = df[df["Prediction"] == "Fraud"].shape[0]
    avg_confidence = round(df["Confidence"].mean(), 2) if not df.empty else 0
    fraud_percentage = (fraud_count / total_transactions) * 100 if total_transactions > 0 else 0

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Transactions", f"{total_transactions:,}")
    col2.metric("Detected Frauds", f"{fraud_count:,}", delta=f"{fraud_percentage:.2f}% of total")
    col3.metric("Average Confidence", f"{avg_confidence}%")

    st.divider()

    # --- Visualizations ---
    col_viz1, col_viz2 = st.columns(2)

    with col_viz1:
        st.subheader("📊 Prediction Distribution")
        pie_data = df["Prediction"].value_counts().reset_index()
        pie_data.columns = ["Prediction", "Count"]
        fig_pie = px.pie(
            pie_data, names="Prediction", values="Count", hole=0.5,
            color="Prediction",
            color_discrete_map={"Fraud": "#EF5350", "Non-Fraud": "#4CAF50"}
        )
        fig_pie.update_layout(legend_title_text='Prediction Type')
        st.plotly_chart(fig_pie, use_container_width=True)

    with col_viz2:
        st.subheader("🎯 Confidence Score Distribution")
        fig_hist = px.histogram(
            df, x="Confidence", color="Prediction", nbins=50,
            barmode="overlay",
            color_discrete_map={"Fraud": "#EF5350", "Non-Fraud": "#4CAF50"}
        )
        st.plotly_chart(fig_hist, use_container_width=True)

    # --- Transaction Data Table ---
    st.subheader("📋 Transaction Records")
    # Define columns to show, checking if they exist in the dataframe
    columns_to_show = [c for c in ["Amount", "Prediction", "Confidence", "Timestamp"] if c in df.columns]
    st.dataframe(df[columns_to_show], use_container_width=True, height=350)

    # --- PDF Report Download ---
    if os.path.exists(pdf_path):
        st.divider()
        st.subheader("📎 Download Full Report")
        with open(pdf_path, "rb") as f:
            st.download_button(
                "⬇️ Download PDF Report",
                data=f,
                file_name="fraud_detection_dashboard_report.pdf",
                mime="application/pdf"
            )
else:
    st.info("Dashboard is ready. Please provide data via the sidebar to see the analysis.")
