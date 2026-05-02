import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import glob
from sklearn.preprocessing import StandardScaler

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Loading CICIDS datasets...")

# load all CICIDS files
files = glob.glob("datasets/raw/CICIDS2017/MachineLearningCVE/*.csv")

dfs = []

for file in files:
    print("Loading:", file)
    df = pd.read_csv(file)
    df.columns = df.columns.str.strip()
    dfs.append(df)

# merge all datasets
df = pd.concat(dfs, ignore_index=True)

print("Total dataset size:", df.shape)

# clean dataset
df = df.replace([np.inf, -np.inf], np.nan)
df = df.dropna()

# convert labels
df["Label"] = df["Label"].apply(lambda x: 0 if x == "BENIGN" else 1)
# behavioral traffic features
df["packet_rate"] = df["Total Fwd Packets"] / (df["Flow Duration"] + 1)
df["byte_rate"] = df["Total Length of Fwd Packets"] / (df["Flow Duration"] + 1)
df["avg_packet_size"] = df["Total Length of Fwd Packets"] / (df["Total Fwd Packets"] + 1)

# selected features
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

# behavioral features
"packet_rate",
"byte_rate",
"avg_packet_size"
]

# train only on normal traffic
normal_df = df[df["Label"] == 0]

print("Normal samples used for training:", normal_df.shape)

X = normal_df[features]

# remove infinities created by division
X = X.replace([np.inf, -np.inf], np.nan)

# drop rows with invalid values
X = X.dropna()

# scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_tensor = torch.tensor(X_scaled, dtype=torch.float32).to(device)

input_dim = X_tensor.shape[1]

# autoencoder model
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
        x = self.encoder(x)
        x = self.decoder(x)
        return x


model = Autoencoder(input_dim).to(device)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

print("Training CICIDS autoencoder...")

for epoch in range(20):

    optimizer.zero_grad()

    outputs = model(X_tensor)

    loss = criterion(outputs, X_tensor)

    loss.backward()

    optimizer.step()

    print(f"Epoch {epoch+1}/20, Loss: {loss.item():.4f}")

# save model
torch.save(model.state_dict(), "models/anomaly_detection/cicids_autoencoder.pth")

print("CICIDS Autoencoder training complete.")
