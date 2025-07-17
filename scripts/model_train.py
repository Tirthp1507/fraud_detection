import os
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE

# === 1. Load dataset ===
print("📦 Loading dataset...")
df = pd.read_csv('data/creditcard.csv')

# === 2. Separate features and target ===
X = df.drop('Class', axis=1)
y = df['Class']

# === 3. Scale features ===
print("📊 Scaling features...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# === 4. Apply SMOTE ===
print("🔄 Applying SMOTE for class balancing...")
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_scaled, y)

# === 5. Train-test split ===
print("✂️ Splitting into train and test sets...")
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

# === 6. Train models ===
print("🧠 Training Logistic Regression...")
lr = LogisticRegression(max_iter=1000)
lr.fit(X_train, y_train)

print("🌲 Training Random Forest...")
rf = RandomForestClassifier(n_jobs=-1, random_state=42)
rf.fit(X_train, y_train)

# === 7. Evaluate models ===
print("✅ Evaluation Report (Logistic Regression):")
y_pred_lr = lr.predict(X_test)
print(classification_report(y_test, y_pred_lr))

print("✅ Evaluation Report (Random Forest):")
y_pred_rf = rf.predict(X_test)
print(classification_report(y_test, y_pred_rf))

# === 8. Save best model (Random Forest) ===
print("💾 Saving Random Forest model...")
os.makedirs('models', exist_ok=True)
joblib.dump(rf, 'models/fraud_model.pkl')
print("🎉 Model saved to models/fraud_model.pkl")
