# scripts/exporter.py

import pandas as pd
import os
from datetime import datetime

EXPORT_FILE = "data/predictions.csv"

def log_prediction(input_data, prediction_label, probability):
    row = dict(zip(input_data.columns, input_data.values[0]))
    row["Prediction"] = "Fraud" if prediction_label == 1 else "Non-Fraud"
    row["Confidence"] = f"{probability * 100:.2f}%"
    row["Timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    df_row = pd.DataFrame([row])

    if os.path.exists(EXPORT_FILE):
        df_row.to_csv(EXPORT_FILE, mode="a", header=False, index=False)
    else:
        df_row.to_csv(EXPORT_FILE, mode="w", header=True, index=False)
