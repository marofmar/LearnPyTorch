import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        # 1 input img channel, 6 output channels, 3x3 square conv
        self.conv1 = nn.Conv2d(1, 6, 3)  # nSamples x nChannels x Height x Width
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
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


net = Net()
#print(net)

params = list(net.parameters())
#print(len(params))  # 10
#print(params[0].size())


input = torch.randn(1, 1, 32, 32)
out = net(input)
#print(out)
net.zero_grad()
out.backward(torch.randn(1, 10))

"""
By far, 

Covered >
- Defining NN
- Processing inputs and calling backward 

To be covered >
- Computing the loss 
- Updating the weights of the NN 
"""

output = net(input)
print(f"Output shape: {output.size()}")
target = torch.rand(10)  # a dummy target, for example
print(f"target: {target} \n Length: {len(target)} \n Size: {target.size()}")
target = target.view(1, -1)  # make it the same shape as output
print(f"Fake batched: {target.size()}")
criterion = nn.MSELoss()

loss = criterion(output, target)  # compute between output and target
# print(f"Calculated loss: {loss} \nLoss type: {type(loss)}")
#
# print(f"loss.grad_fn: {loss.grad_fn}")
# print(f"loss.grad_fn.next_functions[0][0]: {loss.grad_fn.next_functions[0][0]}")
# print(f"loss.grad_fn.next_functions[0][0].grad_fn.next_functions[0][0]: {loss.grad_fn.next_functions[0][0].grad_fn.next_functions[0][0]}")

net.zero_grad()  # zeros the gradient buffers of all parameters

print('conv1.bias.grad before backward')
print(net.conv1.bias.grad)

loss.backward()

print('conv2.bias.grad after backward')
print(net.conv1.bias.grad)


"""
Updating the weights of the network 
"""

learning_rate = 0.01
for p in net.parameters():
    p.data.sub_(p.grad.data*learning_rate)

import torch.optim as optim

optimizer = optim.SGD(net.parameters(), lr = 0.01)

optimizer.zero_grad()
output = net(input)
loss = criterion(output, target)
loss.backeward()
optimizer.step()  # does the update

