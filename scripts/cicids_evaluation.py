import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report


file_path = "datasets/raw/CICIDS2017/MachineLearningCVE/Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv"

print("Loading CICIDS2017 dataset...")

df = pd.read_csv(file_path)

# clean column names
df.columns = df.columns.str.strip()

# remove NaN / infinite values
df = df.replace([np.inf, -np.inf], np.nan)
df = df.dropna()

# label conversion
df["Label"] = df["Label"].apply(lambda x: 0 if x == "BENIGN" else 1)

# select numeric features
X = df.select_dtypes(include=[np.number])
y = df["Label"]

# scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Isolation Forest
model = IsolationForest(n_estimators=200, contamination=0.2, random_state=42)

model.fit(X_scaled)

preds = model.predict(X_scaled)
preds = np.where(preds == -1, 1, 0)

print("\nCICIDS2017 Results")
print(classification_report(y, preds))
