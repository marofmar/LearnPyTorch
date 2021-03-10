import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        # 1 input img channel, 6 output channels, 3x3 square conv
        self.conv1 = nn.Conv2d(1, 6, 3)
        self.conv2 = nn.Conv2d(6, 16, 3)  # input 6 ch, output 16 ch, 3x3 conv
        # an affine operation: y = Wx+b
        self.fc1 = nn.Linear(16*6*6, 120)  # 6*6 img dimension
        self.fc2 = nn.Linear(120, 84)   # input 120 dim, output 84 dim
        self.fc3 = nn.Linear(84, 10)  # input 94 dim, output 10 dim

    def forward(self, x):
        # Max pooling over a (2,2) window
        x = F.max_pool2d(F.relu(self.conv1(x)), (2,2))
        # if the size is a square, only specifying a number is enough
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):


