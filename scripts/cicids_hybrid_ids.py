import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import glob
import joblib
import os

from collections import Counter
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report
from sklearn.neighbors import LocalOutlierFactor
from sklearn.cluster import DBSCAN

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Loading CICIDS dataset...")

# -------------------------------
# Load dataset
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

df["Label"] = df["Label"].apply(lambda x: 0 if x == "BENIGN" else 1)

# -------------------------------
# Behavioral features
# -------------------------------

df["Flow Duration"] = df["Flow Duration"].replace(0, 1)

df["packet_rate"] = df["Total Fwd Packets"] / df["Flow Duration"]

df["byte_rate"] = df["Total Length of Fwd Packets"] / df["Flow Duration"]

df["avg_packet_size"] = (
    df["Total Length of Fwd Packets"] /
    (df["Total Fwd Packets"].replace(0, 1))
)

# -------------------------------
# Selected features
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

X = df[features]
y = df["Label"]

# -------------------------------
# Scaling
# -------------------------------

scaler = joblib.load("models/feature_scaler.pkl")

X_scaled = scaler.transform(X)

X_tensor = torch.tensor(X_scaled, dtype=torch.float32).to(device)

input_dim = X_tensor.shape[1]

# -------------------------------
# Autoencoder Model
# -------------------------------

class Autoencoder(nn.Module):

    def __init__(self, input_dim):

        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 8)
        )

        self.decoder = nn.Sequential(
            nn.Linear(8, 16),
            nn.ReLU(),
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, input_dim)
        )

    def forward(self, x):

        encoded = self.encoder(x)
        decoded = self.decoder(encoded)

        return decoded


# -------------------------------
# Deep SVDD Model
# -------------------------------

class DeepSVDD(nn.Module):

    def __init__(self, input_dim):

        super().__init__()

        self.net = nn.Sequential(

            nn.Linear(input_dim, 32),
            nn.ReLU(),

            nn.Linear(32, 16),
            nn.ReLU(),

            nn.Linear(16, 8)

        )

    def forward(self, x):

        return self.net(x)


# -------------------------------
# Load Autoencoder
# -------------------------------

model = Autoencoder(input_dim).to(device)

model.load_state_dict(
    torch.load(
        "models/anomaly_detection/cicids_autoencoder.pth",
        weights_only=True
    )
)

model.eval()

# -------------------------------
# Load Deep SVDD
# -------------------------------

deep_svdd_model = DeepSVDD(input_dim).to(device)

deep_svdd_model.load_state_dict(
    torch.load("models/deep_svdd_model.pth")
)

deep_svdd_model.eval()

# -------------------------------
# Autoencoder anomaly score
# -------------------------------

with torch.no_grad():

    reconstructed = model(X_tensor)

reconstruction_error = torch.mean(
    (X_tensor - reconstructed) ** 2,
    dim=1
).cpu().numpy()

reconstruction_error = (
    (reconstruction_error - reconstruction_error.min()) /
    (reconstruction_error.max() - reconstruction_error.min())
)

# -------------------------------
# Deep SVDD anomaly score
# -------------------------------

with torch.no_grad():

    deep_features = deep_svdd_model(X_tensor)

svdd_scores = torch.mean(
    deep_features ** 2,
    dim=1
).cpu().numpy()

svdd_scores = (
    (svdd_scores - svdd_scores.min()) /
    (svdd_scores.max() - svdd_scores.min())
)

# -------------------------------
# Isolation Forest
# -------------------------------

print("Training Isolation Forest...")

normal_X = X_scaled[y == 0]

if len(normal_X) > 100000:

    idx = np.random.choice(len(normal_X), 100000, replace=False)
    normal_sample = normal_X[idx]

else:

    normal_sample = normal_X

iso = IsolationForest(

    n_estimators=200,
    contamination=0.05,
    random_state=42

)

iso.fit(normal_sample)

iso_scores = iso.decision_function(X_scaled)

iso_scores = (
    (iso_scores - iso_scores.min()) /
    (iso_scores.max() - iso_scores.min())
)

# -------------------------------
# LOF Detector
# -------------------------------

lof_path = "models/lof_detector.pkl"

if os.path.exists(lof_path):

    print("Loading pre-trained LOF detector...")

    lof = joblib.load(lof_path)

else:

    print("Training LOF detector...")

    if len(normal_X) > 50000:

        idx = np.random.choice(len(normal_X), 50000, replace=False)
        lof_train = normal_X[idx]

    else:

        lof_train = normal_X

    lof = LocalOutlierFactor(

        n_neighbors=20,
        contamination=0.05,
        novelty=True

    )

    lof.fit(lof_train)

    joblib.dump(lof, lof_path)

lof_scores = -lof.decision_function(X_scaled)

lof_scores = (
    (lof_scores - lof_scores.min()) /
    (lof_scores.max() - lof_scores.min())
)

# -------------------------------
# Hybrid Ensemble Score
# -------------------------------

hybrid_score = (

    0.4 * reconstruction_error +
    0.25 * (1 - iso_scores) +
    0.2 * lof_scores +
    0.15 * svdd_scores

)

# -------------------------------
# Dynamic Threshold Calibration
# -------------------------------

normal_scores = hybrid_score[y == 0]

mean_score = np.mean(normal_scores)
std_score = np.std(normal_scores)

threshold = mean_score + 3 * std_score

print("\nDynamic Threshold Calibration")

print("Mean anomaly score:", mean_score)
print("Std deviation:", std_score)
print("Detection threshold:", threshold)

final_preds = (hybrid_score > threshold).astype(int)

# -------------------------------
# Attack Classification
# -------------------------------

attack_model = joblib.load("models/attack_classifier_xgb.pkl")
label_encoder = joblib.load("models/attack_label_encoder.pkl")

attack_indices = np.where(final_preds == 1)[0]

decoded_attacks = []

if len(attack_indices) > 0:

    attack_features = X.iloc[attack_indices]

    attack_types = attack_model.predict(attack_features)

    decoded_attacks = label_encoder.inverse_transform(attack_types)

    print("\nSample detected attack types:")

    print(decoded_attacks[:20])

# -------------------------------
# Attack Distribution
# -------------------------------

if len(decoded_attacks) > 0:

    attack_counts = Counter(decoded_attacks)

    print("\nDetected attack distribution:")

    for attack, count in attack_counts.items():

        print(f"{attack}: {count}")

# -------------------------------
# Self Learning IDS
# -------------------------------

print("\nSaving detected attacks for self-learning...")

self_learning_file = "datasets/self_learning/new_attacks.csv"

detected_df = df.iloc[attack_indices]

if not os.path.exists(self_learning_file):

    detected_df.to_csv(self_learning_file, index=False)

else:

    detected_df.to_csv(
        self_learning_file,
        mode="a",
        header=False,
        index=False
    )

print("New attacks stored for future training.")

# -------------------------------
# Zero Day Detection
# -------------------------------

if len(attack_indices) > 0:

    anomaly_features = X.iloc[attack_indices]

# limit DBSCAN size to prevent memory crash
    if len(anomaly_features) > 10000:
        anomaly_features = anomaly_features.sample(10000, random_state=42)

    scaler = StandardScaler()

    anomaly_scaled = scaler.fit_transform(anomaly_features)

    db = DBSCAN(eps=2.5, min_samples=20)

    clusters = db.fit_predict(anomaly_scaled)

    unknown_attacks = np.sum(clusters == -1)

    print("\nPossible zero-day samples:", unknown_attacks)

else:

    print("\nNo anomalies detected for zero-day analysis.")

# -------------------------------
# Final Results
# -------------------------------

print("\nCICIDS Hybrid IDS Results")

print(classification_report(y, final_preds))
