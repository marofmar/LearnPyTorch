import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class autoencoder(nn.Module):
    def __init__(self):
        super(autoencoder, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(5, 8),
            nn.ReLU(True),
            nn.Linear(8, 16),
            nn.ReLU(True), nn.Linear(16, 12), nn.ReLU(True), nn.Linear(12, 3)
        )

        self.decoder = nn.Sequential(
            nn.Linear(3, 12),
            nn.ReLU(True),
            nn.Linear(12, 16),
            nn.ReLU(True), nn.Linear(16, 8), nn.ReLU(True), nn.Linear(8, 5)
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        print(f"{encoded}->{decoded}")
        return decoded


model = autoencoder()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)

sequence = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5], dtype=torch.float)
n_in = len(sequence)
sequence = sequence.reshape((1, n_in, 1))

for epoch in range(50):
    output = model(sequence)
    loss = criterion(output, sequence)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


