import pandas as pd
import glob
import torch
from torch_geometric.data import Data
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import kneighbors_graph

print("Loading CICIDS dataset...")

files = glob.glob("datasets/raw/CICIDS2017/MachineLearningCVE/*.csv")

dfs = []

for f in files:
    df = pd.read_csv(f)
    df.columns = df.columns.str.strip()
    dfs.append(df)

df = pd.concat(dfs, ignore_index=True)

df = df.replace([float("inf"), float("-inf")], float("nan"))
df = df.dropna()

print("Dataset size:", df.shape)

# features used for graph
features = [
"Fwd Packet Length Max",
"Total Fwd Packets",
"Flow Duration",
"Destination Port"
]

X = df[features]

# scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# build k-nearest neighbor graph
print("Building graph...")

A = kneighbors_graph(X_scaled, n_neighbors=5, mode="connectivity", include_self=False)

edge_index = torch.tensor(A.nonzero(), dtype=torch.long)

x = torch.tensor(X_scaled, dtype=torch.float)

data = Data(x=x, edge_index=edge_index)

torch.save(data, "datasets/network_graph.pt")

print("Graph dataset saved successfully.")
print("Nodes:", data.num_nodes)
print("Edges:", data.num_edges)
