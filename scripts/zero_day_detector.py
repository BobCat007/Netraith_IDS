import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import pandas as pd
import glob


print("Loading anomaly samples...")

files = glob.glob("datasets/raw/CICIDS2017/MachineLearningCVE/*.csv")

dfs = []

for file in files:
    df = pd.read_csv(file)
    df.columns = df.columns.str.strip()
    dfs.append(df)

df = pd.concat(dfs)

df = df.replace([np.inf, -np.inf], np.nan)
df = df.dropna()

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

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

print("Running DBSCAN clustering...")

db = DBSCAN(eps=2.5, min_samples=20)

clusters = db.fit_predict(X_scaled)

unknown_attacks = np.sum(clusters == -1)

print("Potential zero-day samples:", unknown_attacks)
