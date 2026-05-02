import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import glob
from sklearn.preprocessing import StandardScaler

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Training Deep SVDD...")

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

# ------------------------------
# Behavioral Feature Engineering
# ------------------------------

df["Flow Duration"] = df["Flow Duration"].replace(0,1)

df["packet_rate"] = df["Total Fwd Packets"] / df["Flow Duration"]

df["byte_rate"] = df["Total Length of Fwd Packets"] / df["Flow Duration"]

df["avg_packet_size"] = (
    df["Total Length of Fwd Packets"] /
    (df["Total Fwd Packets"].replace(0,1))
)

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

normal_df = df[df["Label"] == 0]

X = normal_df[features]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_tensor = torch.tensor(X_scaled, dtype=torch.float32).to(device)

input_dim = X_tensor.shape[1]


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


model = DeepSVDD(input_dim).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

center = torch.zeros(8).to(device)

for epoch in range(20):

    optimizer.zero_grad()

    outputs = model(X_tensor)

    loss = torch.mean((outputs - center) ** 2)

    loss.backward()

    optimizer.step()

    print(f"Epoch {epoch+1}/20 | Loss {loss.item():.6f}")

torch.save(model.state_dict(), "models/deep_svdd_model.pth")

print("Deep SVDD model saved.")
