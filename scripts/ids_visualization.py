import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import glob

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor


# -------------------------------
# Load CICIDS dataset
# -------------------------------
print("Loading CICIDS datasets...")

files = glob.glob("datasets/raw/CICIDS2017/MachineLearningCVE/*.csv")

dfs = []

for file in files:
    df = pd.read_csv(file)
    df.columns = df.columns.str.strip()
    dfs.append(df)

df = pd.concat(dfs, ignore_index=True)

df = df.replace([np.inf, -np.inf], np.nan)
df = df.dropna()

df["Label"] = df["Label"].apply(lambda x: 0 if x == "BENIGN" else 1)

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
"Fwd Packet Length Std"
]

X = df[features]
y = df["Label"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# -------------------------------
# Load Autoencoder
# -------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
        return self.decoder(self.encoder(x))


model = Autoencoder(len(features)).to(device)

model.load_state_dict(
    torch.load("models/anomaly_detection/cicids_autoencoder.pth", weights_only=True)
)

model.eval()

X_tensor = torch.tensor(X_scaled, dtype=torch.float32).to(device)

with torch.no_grad():
    reconstructed = model(X_tensor)

reconstruction_error = torch.mean((X_tensor - reconstructed) ** 2, dim=1).cpu().numpy()

reconstruction_error = (
    (reconstruction_error - reconstruction_error.min()) /
    (reconstruction_error.max() - reconstruction_error.min())
)

# -------------------------------
# Isolation Forest
# -------------------------------
normal_X = X_scaled[y == 0]

iso = IsolationForest(n_estimators=300, contamination=0.05, random_state=42)
iso.fit(normal_X)

iso_scores = iso.decision_function(X_scaled)

iso_scores = (
    (iso_scores - iso_scores.min()) /
    (iso_scores.max() - iso_scores.min())
)

# -------------------------------
# LOF
# -------------------------------
lof = LocalOutlierFactor(n_neighbors=20, contamination=0.05, novelty=True)
lof.fit(normal_X)

lof_scores = -lof.decision_function(X_scaled)

lof_scores = (
    (lof_scores - lof_scores.min()) /
    (lof_scores.max() - lof_scores.min())
)

# -------------------------------
# Hybrid Score
# -------------------------------
hybrid_score = (
    0.4 * reconstruction_error +
    0.35 * (1 - iso_scores) +
    0.25 * lof_scores
)

normal_scores = hybrid_score[y == 0]

threshold = np.mean(normal_scores) + 1.5 * np.std(normal_scores)

preds = (hybrid_score > threshold).astype(int)

# -------------------------------
# Confusion Matrix
# -------------------------------
cm = confusion_matrix(y, preds)

plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")

plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.savefig("confusion_matrix.png")
plt.close()

# -------------------------------
# ROC Curve
# -------------------------------
fpr, tpr, _ = roc_curve(y, hybrid_score)
roc_auc = auc(fpr, tpr)

plt.figure()

plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
plt.plot([0,1], [0,1], linestyle="--")

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")

plt.title("ROC Curve")
plt.legend()

plt.savefig("roc_curve.png")
plt.close()

# -------------------------------
# Precision-Recall Curve
# -------------------------------
precision, recall, _ = precision_recall_curve(y, hybrid_score)

plt.figure()

plt.plot(recall, precision)

plt.xlabel("Recall")
plt.ylabel("Precision")

plt.title("Precision-Recall Curve")

plt.savefig("precision_recall_curve.png")
plt.close()

# -------------------------------
# Anomaly Score Distribution
# -------------------------------
plt.figure()

sns.histplot(hybrid_score[y == 0], label="Normal", bins=50, color="green")
sns.histplot(hybrid_score[y == 1], label="Attack", bins=50, color="red")

plt.legend()
plt.title("Anomaly Score Distribution")

plt.savefig("anomaly_distribution.png")
plt.close()

print("\nGraphs generated successfully:")
print("confusion_matrix.png")
print("roc_curve.png")
print("precision_recall_curve.png")
print("anomaly_distribution.png")
