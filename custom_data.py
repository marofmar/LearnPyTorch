"""

A custom dataset class must implement three functions:
    __init__: this function is run once to instantiate the Dataset object.
    __len__: this function returns the number of samples in the dataset.
    __getitem__: this function loads and returns a sample from the dataset at the given index.

"""
from data import *
import os
import pandas as pd
from torchvision.io import read_image

from torch.utils.data import DataLoader, Dataset

class CustomImageDataset(Dataset):
    """
    suppose we have images under the 'img_dir'
    and corresponding labels in 'annotations_file' which is in the form of csv.
    We can create a custom dataset by following codes.
    """
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx,0])  # label: filename, classname
        image = read_image(img_path)  # converts the img into tensors
        label = self.img_labels.iloc[idx, 1]  # read class name -> label (retrieve the corresponding label)
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        sample = {"image": image, "label": label}
        return sample


train_dataloader = DataLoader(training_data, batch_size=64, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=64, shuffle = True)

train_features, train_labels = next(iter(train_dataloader))
print(f"Feature batch shape: {train_features.size()}")
print(f"Label batch shape: {train_labels.size()}")

img = train_features[0].squeeze()
label = train_labels[0]
plt.imshow(img, cmap='gray')
plt.show()
print(f"Label: {label}")

