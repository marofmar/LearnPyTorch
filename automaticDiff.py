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
loss = torch.nn.functional.binary_cross_entropy_with_logits(z, y)

print("Gradient function for z =", z.grad_fn)  # grad_fn property of a tensor contains the back prop function
print("Gradient function for loss =", loss.grad_fn)

    # COMPUTING GRADIENTS
loss.backward()  # to compute the derivatives of the loss wrt to w, and b under the condition that x, and y are fixed.
print(w.grad)  # retrieve the values
print(b.grad)  # .grad is only possible for the variables with "requires_grad=True"


    # DISABLING GRADIENT TRACKING
    # method 1 : torch.no_grad()
z = torch.matmul(x, w) + b
print(z.requires_grad)  # true maybe

with torch.no_grad():  # stop tracking gradients computations, and do only FORWARD computations
    z = torch.matmul(x, w) +b
print(z.requires_grad)  # false
    # method 2 : detach()
z = torch.matmul(x, w) + b
z_det = z.detach()
print(z_det.requires_grad)  # false

inp = torch.eye(5, requires_grad=True)  # identity matrix 5*5
out = (inp+1).pow(2)  # add one to every element of inp, pow(2)
out.backward(torch.ones_like(inp), retain_graph=True)
print("First call\n", inp.grad)
out.backward(torch.ones_like(inp), retain_graph=True)  # accumulated gradients !!! INCORRECT!!
print("Second call\n", inp.grad)
inp.grad.zero_()   # thereby, zeroing, zero_grad() is necessary to compute the right ones
out.backward(torch.ones_like(inp), retain_graph=True)
print("\nCall after zeroing gradients\n", inp.grad)
