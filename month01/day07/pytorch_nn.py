import torch
import matplotlib.pyplot as plt


# Data
X = torch.tensor([
    [1.0],
    [2.0],
    [3.0],
    [4.0]
])

Y = torch.tensor([
    [2.0],
    [4.0],
    [6.0],
    [8.0]
])

def train(lr):

    torch.manual_seed(42)

    model = torch.nn.Sequential(
        torch.nn.Linear(1, 2),
        torch.nn.ReLU(),
        torch.nn.Linear(2, 1)
    )

    loss_fn = torch.nn.MSELoss()

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=lr
    )

    losses = []

    for epoch in range(1000):

        optimizer.zero_grad()

        Y_pred = model(X)

        loss = loss_fn(Y_pred, Y)

        loss.backward()

        optimizer.step()

        losses.append(loss.item())

    return losses

loss_001 = train(0.001)
loss_01 = train(0.01)
loss_1 = train(0.1)
loss_10 = train(1.0)



plt.figure()

plt.plot(loss_001, label="lr=0.001")
plt.plot(loss_01, label="lr=0.01")
plt.plot(loss_1, label="lr=0.1")
plt.plot(loss_10, label="lr=1.0")

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.yscale("log")
plt.title("Learning Rate Comparison")
plt.legend()

plt.show()