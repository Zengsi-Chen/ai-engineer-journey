import torch
import torch.nn as nn


x = torch.randn(64, 1, 28, 28)

conv = nn.Conv2d(
    in_channels=1,
    out_channels=16,
    kernel_size=3,
    padding=1
)

relu = nn.ReLU()

pool = nn.MaxPool2d(
    kernel_size=2
)

x = conv(x)

print("After Conv :", x.shape)

x = relu(x)

print("After ReLU :", x.shape)

x = pool(x)

print("After Pool :", x.shape)