import torch


x = torch.randn(64, 16, 14, 14)

print("Before:", x.shape)

x = torch.flatten(x, start_dim=1)

print("After :", x.shape)