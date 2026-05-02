import torch
import torch.nn as nn
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Starting GAN attack generator...")

# feature dimension must match IDS
feature_dim = 18
latent_dim = 32

# -------------------------
# Generator Network
# -------------------------
class Generator(nn.Module):

    def __init__(self):
        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, feature_dim)
        )

    def forward(self, z):
        return self.model(z)


generator = Generator().to(device)

generator.eval()

# -------------------------
# Generate Synthetic Attacks
# -------------------------
num_samples = 5000

noise = torch.randn(num_samples, latent_dim).to(device)

with torch.no_grad():
    fake_attacks = generator(noise).cpu().numpy()

print("Generated synthetic attack samples:", fake_attacks.shape)

# -------------------------
# Save Generated Attacks
# -------------------------
np.save("datasets/synthetic_attacks.npy", fake_attacks)

print("Synthetic attacks saved to datasets/synthetic_attacks.npy")
