import torch
from torch import nn

# create models

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("Using {} device".format(device))


# define models
class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()  # flatten the data to feed into the linear layer
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28 * 28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),  # 10 classes for the outcome categories
            nn.ReLU()  # last ReLU layer
        )

    def forward(self, x):
        x = self.flatten(x)  # first flateen
        logits = self.linear_relu_stack(x)  # feed into the linear nn model
        return logits  # get the 10 dim vector (logits)


model = NeuralNetwork().to(device)
print(model)
