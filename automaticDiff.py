import torch
"""
When training nn, the most frequently used algorithm is back propagation.
Back propagation: parameters (model weights) are adjusted according to the gradient of the loss function wrt the param)
"""
x = torch.ones(5)  # input tensor
y = torch.zeros(3)
w = torch.randn(5, 3, requires_grad=True)  # param weight (optimization target 1)
b = torch.randn(3, requires_grad=True)  # param bias (optimization target 2)
z = torch.matmul(x, w) + b
loss = torch.nn.funcitonal.binary_cross_entropy_with_logits(z, y)