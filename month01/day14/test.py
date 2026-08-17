import torch

checkpoint = torch.load(
    "checkpoint.pth",
    weights_only=False
)

print(checkpoint.keys())