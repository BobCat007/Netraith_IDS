import pandas as pd
import numpy as np
import glob
import joblib

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from xgboost import XGBClassifier


print("Loading CICIDS dataset...")

# -------------------------------
# Load all CICIDS files
# -------------------------------
files = glob.glob("datasets/raw/CICIDS2017/MachineLearningCVE/*.csv")

dfs = []

for file in files:
    print("Loading:", file)
    df = pd.read_csv(file)
    df.columns = df.columns.str.strip()
    dfs.append(df)

df = pd.concat(dfs, ignore_index=True)

print("Dataset size:", df.shape)

# -------------------------------
# Clean dataset
# -------------------------------
df = df.replace([np.inf, -np.inf], np.nan)
df = df.dropna()

# -------------------------------
# Behavioral Features
# -------------------------------
df["Flow Duration"] = df["Flow Duration"].replace(0, 1)

df["packet_rate"] = df["Total Fwd Packets"] / df["Flow Duration"]
df["byte_rate"] = df["Total Length of Fwd Packets"] / df["Flow Duration"]

df["avg_packet_size"] = (
    df["Total Length of Fwd Packets"] /
    (df["Total Fwd Packets"].replace(0, 1))
)

# -------------------------------
# Keep only attacks
# -------------------------------
attack_df = df[df["Label"] != "BENIGN"]

print("Attack samples:", attack_df.shape)

# -------------------------------
# Feature list (must match IDS)
# -------------------------------
features = [
"Fwd Packet Length Max",
"Avg Fwd Segment Size",
"Total Length of Fwd Packets",
"Fwd IAT Max",
"Fwd IAT Std",
"Fwd Packet Length Mean",
"Fwd IAT Total",
"Init_Win_bytes_forward",
"Subflow Fwd Bytes",
"Fwd IAT Mean",
"Total Fwd Packets",
"Fwd Header Length.1",
"act_data_pkt_fwd",
"Destination Port",
"Fwd Packet Length Std",
"packet_rate",
"byte_rate",
"avg_packet_size"
]

X = attack_df[features]
y = attack_df["Label"]

# -----------------------------
# Feature Scaling
# -----------------------------

scaler = StandardScaler()
X = scaler.fit_transform(X)

# -------------------------------
# Encode attack labels
# -------------------------------
le = LabelEncoder()
y = le.fit_transform(y)

# -------------------------------
# Train-test split
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# -------------------------------
# XGBoost Classifier
# -------------------------------
model = XGBClassifier(
    n_estimators=300,
    max_depth=8,
    learning_rate=0.1,
    tree_method="hist",
    n_jobs=-1
)

print("Training XGBoost attack classifier...")

model.fit(X_train, y_train)

# -------------------------------
# Evaluation
# -------------------------------
preds = model.predict(X_test)

print("\nAttack Classification Results")

print(classification_report(y_test, preds))

# -------------------------------
# Save models
# -------------------------------
joblib.dump(model, "models/attack_classifier_xgb.pkl")
joblib.dump(le, "models/attack_label_encoder.pkl")
joblib.dump(scaler, "models/feature_scaler.pkl")

print("Attack classifier saved.")
