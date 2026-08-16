import torch
import torch.nn as nn

x = torch.tensor([
    [1.0],
    [2.0],
    [3.0],
    [4.0]
])

y_true = torch.tensor([
    [5.0],
    [8.0],
    [11.0],
    [14.0]
])

w = torch.tensor([[0.0]], requires_grad=True)
b = torch.tensor([0.0], requires_grad=True)

lr = 0.1

for epoch in range(500):

    # Forward
    y_pred = x @ w + b

    # Loss
    loss_fn = nn.MSELoss()
    loss = loss_fn(y_pred, y_true)
    

    # Backward
    loss.backward()

    # Update
    with torch.no_grad():
        w -= lr * w.grad
        b -= lr * b.grad

    # Clear gradients
    w.grad.zero_()
    b.grad.zero_()

    if epoch % 100 == 0:
        print(
            f"Epoch {epoch}, "
            f"Loss: {loss.item():.4f}, "
            f"w: {w.item():.4f}, "
            f"b: {b.item():.4f}"
        )