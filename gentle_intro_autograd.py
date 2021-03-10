"""

Training NN
1_forward propagation:
take in inputs and returns the best guessed outputs

2_backward propagation:
adjusts its params by traversing backwards from outputs to inputs, collecting derivatives of the errors wrt the params
"""

import torch, torchvision

model = torchvision.models.resnet18(pretrained=True)
data = torch.rand(1, 3, 64, 64)  # 3 channels, height and width of 64, single image.
labels = torch.rand(1, 1000)  # random label value assigned

prediction = model(data)  # forward pass: prediction

loss = (prediction - labels).sum()
loss.backward()  # backward propagation is kicked off

optim = torch.optim.SGD(model.parameters(), lr=1e-2, momentum=0.9)  # momentum adds partial prev updates to the current

optim.step()  # calling step() initiates gradient descent
# the optimizer adjust each param by its gradient stored in .grad


"""
Differentiation in Autograd 
- how 'autograd' collects gradients? 

Q = 3*a^3 - b^2
"""

a = torch.tensor([2., 3.], requires_grad=True)
b = torch.tensor([6., 4.], requires_grad=True)

Q = 3*a**3-b**2

external_gard = torch.tensor([1.,1.])
Q.backward(gradient=external_gard)

print(9*a**2 == a.grad)
print(-2*b == b.grad)


"""
Vector Calculus using autograd
"""

