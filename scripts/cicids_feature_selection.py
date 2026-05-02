import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

file_path = "datasets/raw/CICIDS2017/MachineLearningCVE/Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv"

print("Loading CICIDS dataset...")

df = pd.read_csv(file_path)

# clean column names
df.columns = df.columns.str.strip()

# remove infinity values
df = df.replace([np.inf, -np.inf], np.nan)

# drop rows with NaN values
df = df.dropna()

# label conversion
df["Label"] = df["Label"].apply(lambda x: 0 if x == "BENIGN" else 1)

# keep numeric features
X = df.select_dtypes(include=["float64", "int64"])
y = df["Label"]

print("Training Random Forest for feature importance...")

rf = RandomForestClassifier(n_estimators=100)

rf.fit(X, y)

importances = pd.Series(rf.feature_importances_, index=X.columns)

top_features = importances.sort_values(ascending=False).head(20)

print("\nTop 20 CICIDS Features:\n")
print(top_features)
