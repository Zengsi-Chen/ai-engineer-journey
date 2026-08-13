# Day 4 - PyTorch Fundamentals and Training Loop

## Goal

Learn the basic PyTorch workflow for building and training a simple neural network.

## What I Learned

- PyTorch Tensor
- `requires_grad`
- Autograd
- Gradient
- Backpropagation
- `nn.Module`
- `nn.Linear`
- `nn.Parameter`
- `model.parameters()`
- `MSELoss`
- SGD optimizer
- Training loop
- `optimizer.zero_grad()`
- `loss.backward()`
- `optimizer.step()`
- `model.train()`
- `model.eval()`
- `torch.no_grad()`
- Training history

## Model

The model is a simple linear layer:

```text
Input features: 3
Output features: 2