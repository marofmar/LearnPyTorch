import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class Encoder(nn.Module):

    def __init__(self):
        super(Encoder, self).__init__()
        self.fc1 = nn.Linear(5, 8)  # 6*6 img dimension
        self.fc2 = nn.Linear(8, 5)  # input 120 dim, output 84 dim

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x


model = Encoder()

inp = [0.1, 0.2, 0.3, 0.4, 0.5]
input = torch.tensor(inp, dtype=torch.float)
output = torch.tensor(inp, dtype=torch.float)

loss_fn = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

for i in range(10):
    pred = model(input)
    loss = loss_fn(pred, output)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print(pred)
