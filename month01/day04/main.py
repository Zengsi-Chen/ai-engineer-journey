import torch
import torch.nn as nn


X = torch.tensor([
    [1.0, 2.0, 3.0],
    [4.0, 5.0, 6.0]
])

target = torch.tensor([
    [1.0, 1.0],
    [2.0, 2.0]
])


model = nn.Linear(
    in_features=3,
    out_features=2
)

loss_fn = nn.MSELoss()

optimizer = torch.optim.SGD(
    model.parameters(),
    lr=0.01
)


def train(
    model: nn.Module,
    X: torch.Tensor,
    target: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    epochs: int = 10
) -> list[float]:

    model.train()

    history = []

    for epoch in range(epochs):

        optimizer.zero_grad()

        prediction = model(X)

        loss = loss_fn(prediction, target)

        loss.backward()

        optimizer.step()

        current_loss = loss.item()
        history.append(current_loss)

        print(
            f"Epoch {epoch + 1}: "
            f"loss = {current_loss:.4f}"
        )

    return history


def evaluate(
    model: nn.Module,
    X: torch.Tensor,
    target: torch.Tensor,
    loss_fn: nn.Module
) -> float:

    model.eval()

    with torch.no_grad():
        prediction = model(X)
        loss = loss_fn(prediction, target)

    return loss.item()


history = train(
    model,
    X,
    target,
    optimizer,
    loss_fn
)

evaluation_loss = evaluate(
    model,
    X,
    target,
    loss_fn
)

print("Initial loss:", history[0])
print("Final training loss:", history[-1])
print("Evaluation loss:", evaluation_loss)