import numpy as np
import torch
import torch.nn as nn


# PyTorch example
x = torch.tensor([
    [1.0, 2.0, 3.0],
    [4.0, 5.0, 6.0]
])

class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()

        self.linear1 = nn.Linear(3, 4)
        self.linear2 = nn.Linear(4, 2)

    def forward(self, x):
        x = self.linear1(x)
        x = torch.relu(x)
        x = self.linear2(x)

        return x

model = NeuralNetwork()
output = model(x)

print("Model output:")
print(output)

loss_fn = nn.MSELoss()

target = torch.tensor([
    [1.0, 0.0],
    [0.0, 1.0]
])

loss = loss_fn(output, target)

print("Loss:")
print(loss)

'''
print("Model parameters:")
for param in model.parameters():
    print(param)
for name, parameter in model.named_parameters():
    print(name, parameter.shape)
'''