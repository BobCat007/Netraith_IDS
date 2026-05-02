import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data

print("Loading graph dataset...")

data = torch.load("datasets/network_graph.pt")

class GNN_IDS(torch.nn.Module):

    def __init__(self, num_features):
        super().__init__()

        self.conv1 = GCNConv(num_features, 32)
        self.conv2 = GCNConv(32, 16)
        self.fc = torch.nn.Linear(16, 2)

    def forward(self, x, edge_index):

        x = self.conv1(x, edge_index)
        x = F.relu(x)

        x = self.conv2(x, edge_index)
        x = F.relu(x)

        x = self.fc(x)

        return F.log_softmax(x, dim=1)


model = GNN_IDS(data.num_features)

optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

for epoch in range(20):

    optimizer.zero_grad()

    out = model(data.x, data.edge_index)

    loss = out.mean()

    loss.backward()

    optimizer.step()

    print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

print("GNN training complete.")
