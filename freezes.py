import torch, torchvision
from torch import nn, optim

model = torchvision.models.resnet18(pretrained=True)

# freeze all the params in the network
for param in model.parameters():
    param.requires_grad = False


model.fc = nn.Linear(512, 10)  # replace the last classifying layer to the linear of 10 classes


optimizer = optim.SGD(model.fc.parameters(), lr = 1e-2, momentum=0.9)  # except the model.fc, all layers were frozen

