
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets,transforms
from torchvision.transforms import ToTensor, Lambda, Compose
import matplotlib.pyplot as plt

training_data = datasets.FashionMNIST(
    root='data',  # root directory where images are gonna be downloaded
    train=True,  # specifies training or testing
    download=True,  # download if it is not available at 'root'
    transform=ToTensor(),  # feature transformation(PIL or ndarray -> FloatTensor,
                           # scale the img's pixel intensity values in [0., 1.])
    target_transform=Lambda(lambda y: torch.zeros(10, dtype=torch.float).scatter_(0, torch.tensor(y), value=1))
                           # label transformation # lambda transforamtion: user-defined transformation.
                           # trn the integer into a one-hot encoded tensor
                           # scatter_: this function assigns a value of 1 on the given index by the label y
)

test_data = datasets.FashionMNIST(
    root='data',  # root directory where images are gonna be downloaded
    train=False,
    download=True,
    transform=ToTensor(),
)

batch_size = 64
train_dataloader = DataLoader(training_data, batch_size=batch_size)
test_dataloader = DataLoader(test_data, batch_size=batch_size)

for X, y in test_dataloader:
    print("Shape of X [N, C, H, W]: ", X.shape)
    print("Shape of y: ", y.shape, y.dtype)

    break

"""
Iterating and Visualizing the Dataset
"""
labels_map = {
    0: "T-Shirt",
    1: "Trouser",
    2: "Pullover",
    3: "Dress",
    4: "Coat",
    5: "Sandal",
    6: "Shirt",
    7: "Sneaker",
    8: "Bag",
    9: "Ankle Boot",
}
figure = plt.figure(figsize = (8,8))
cols, rows = 3, 3
for i in range(1, cols*rows+1):
    sample_idx = torch.randint(len(training_data), size=(1,)).item()
    img, label = training_data[sample_idx]
    figure.add_subplot(rows, cols, i)
    plt.title(labels_map[label])
    plt.axis("off")
    plt.imshow(img.squeeze(), cmap='gray')
plt.show()


