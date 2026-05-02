import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report
from models.anomaly_detection.autoencoder import Autoencoder

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_path = "/mnt/d/ai_ids_project/datasets/raw/NSL_KDD/KDDTrain+.txt"
test_path = "/mnt/d/ai_ids_project/datasets/raw/NSL_KDD/KDDTest+.txt"

columns = [
"duration","protocol_type","service","flag","src_bytes","dst_bytes",
"land","wrong_fragment","urgent","hot","num_failed_logins","logged_in",
"num_compromised","root_shell","su_attempted","num_root","num_file_creations",
"num_shells","num_access_files","num_outbound_cmds","is_host_login",
"is_guest_login","count","srv_count","serror_rate","srv_serror_rate",
"rerror_rate","srv_rerror_rate","same_srv_rate","diff_srv_rate",
"srv_diff_host_rate","dst_host_count","dst_host_srv_count",
"dst_host_same_srv_rate","dst_host_diff_srv_rate",
"dst_host_same_src_port_rate","dst_host_srv_diff_host_rate",
"dst_host_serror_rate","dst_host_srv_serror_rate",
"dst_host_rerror_rate","dst_host_srv_rerror_rate","label","difficulty"
]

test = pd.read_csv(test_path, names=columns)

test["label"] = test["label"].astype(str).str.strip()

test["label"] = test["label"].apply(lambda x: 0 if x == "normal" else 1)

categorical = ["protocol_type", "service", "flag"]

for col in categorical:
    le = LabelEncoder()
    test[col] = le.fit_transform(test[col])

selected_features = [
"src_bytes",
"dst_bytes",
"flag",
"same_srv_rate",
"dst_host_same_srv_rate",
"diff_srv_rate",
"dst_host_srv_count",
"logged_in",
"dst_host_diff_srv_rate",
"protocol_type",
"dst_host_same_src_port_rate",
"count",
"dst_host_srv_serror_rate",
"service",
"dst_host_srv_diff_host_rate"
]

X = test[selected_features]
y = test["label"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_tensor = torch.tensor(X_scaled, dtype=torch.float32).to(device)

input_dim = X_scaled.shape[1]

model = Autoencoder(input_dim).to(device)

model = Autoencoder(input_dim).to(device)
model.load_state_dict(torch.load("models/anomaly_detection/autoencoder.pth", weights_only=True))
model.eval()

with torch.no_grad():
    reconstructed = model(X_tensor)

error = torch.mean((X_tensor - reconstructed) ** 2, dim=1).cpu().numpy()

threshold = np.percentile(error, 85)

print("Detection threshold:", threshold)

predictions = (error > threshold).astype(int)

print(classification_report(y, predictions))
