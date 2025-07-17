import pandas as pd
import numpy as np
import os

# Create column names
columns = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]

# Generate 200 rows of random data
data = np.random.normal(loc=0, scale=1, size=(200, 30))

# Add a few high amounts to simulate fraud-like cases
data[:10, -1] = np.random.uniform(5000, 10000, size=10)

df = pd.DataFrame(data, columns=columns)

# Save to data folder
os.makedirs("data", exist_ok=True)
df.to_csv("data/demo_fraud_data.csv", index=False)

print("✅ demo_fraud_data.csv generated at: data/demo_fraud_data.csv")
