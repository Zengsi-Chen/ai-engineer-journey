import torch
import torch.nn as nn

from dataset import create_dataloaders


train_loader, _ = create_dataloaders()

images, labels = next(iter(train_loader))

print("Input:", images.shape)

conv = nn.Conv2d(
    in_channels=1,
    out_channels=16,
    kernel_size=3,
    padding=1
)

relu = nn.ReLU()

pool = nn.MaxPool2d(
    kernel_size=3
)

x = conv(images)

print("After Conv :", x.shape)

x = relu(x)

print("After ReLU :", x.shape)

x = pool(x)

print("After Pool :", x.shape)