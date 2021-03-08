import os
import torch
from torch import nn  # torch.nn namespace provides all the building blocks to build neural networks

# create models

device = 'cuda' if torch.cuda.is_available() else 'cpu'  # hardware accelerator GPU if possible (check availability)
print("Using {} device".format(device))


# define models
class NeuralNetwork(nn.Module):  # every module in PyTorch subclasses the nn.Module (define nn by subclassing nn.Module)
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()  # convert 2D 28*28 into array of 784 pixel values (the minibatch dim standstill)
        self.linear_relu_stack = nn.Sequential(  # Sequential: ordered container of modules
            nn.Linear(28 * 28, 512),  # applies a linear transformation on the input using its stored weights and biases
            nn.ReLU(),  # non-linear activations (introducing nonlinearity)
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),  # 10 classes for the outcome categories
            nn.ReLU()  # last ReLU layer
        )

    def forward(self, x):
        x = self.flatten(x)  # first flateen
        logits = self.linear_relu_stack(x)  # get the 10 dim tensor with raw predicted values for each class [-inf, inf]
        pred_probab = nn.Softmax(dim=1)(logits)  # get prediction probabilities [0, 1]
        y_pred = pred_probab.argmax(1)  # return the class of the highest probability
        return y_pred


model = NeuralNetwork().to(device)  # create an instance of NeuralNetwork, and send it (move it) to device
print("Model structure: ", model, "\n\n")  # let the model show its structure !
for name, param in model.named_parameters():
    print(f"Layer: {name} | Size: {param.size()} | Values: {param[:2]} \n")
