
import torch
from torch import nn
from model import *
from data import *


# optimizing the model parameters
learning_rate = 1e-3  # how much to update model params at each batch/epoch.
batch_size = 64  # the number of data samples seen by the model in each epoch
epochs = 5  # the number of times to iterate over the dataset EACH ITERATION OF THE OPTIMIZATION IS CALLED EPOCH
            # Each epoch consists of two main parts: The train loop and the validation/test loop

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)  # SGD: Stochastic Gradient Descent
    # All optimization logic is encapsulated in the optimizer object.

def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)  # move X, y tensors to device (hopefully GPU)

        pred = model(X)  # get the predicted output
        loss = loss_fn(pred, y)  # calculate the loss

        optimizer.zero_grad()  # sets the all gradients to zero (RESET THE GRADIENTS)
        loss.backward()  # accumulate gradients for each param (thereby need zero_gard()) (BACKPROPAGATE LOSS)
        optimizer.step()  # parameter update based on the current gradients (ADJUST THE PARAMS BY THE GRADIENTS)

        if batch % 100 == 0:
            loss, current = loss.item(), batch*len(X)  # check how far we have come
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")  # :>7f left aligned decimal 7


def test_loop(dataloader, model):
    size = len(dataloader.dataset)
    model.eval()  # evaluation mode
    test_loss, correct = 0, 0
    with torch.no_grad():  # not updating gradients, only evaluation
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()  # loss between predicted and the true
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= size
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")



