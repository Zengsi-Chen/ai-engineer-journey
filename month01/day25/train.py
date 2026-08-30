import torch
import torch.nn as nn

from dataset import create_toy_dataloader
from model import ShallowCNN, DeepPlainCNN


def train_model(model, dataloader, epochs=5):
    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=0.01,
    )

    model.train()

    for epoch in range(epochs):

        total_loss = 0.0
        correct = 0
        total = 0

        for x, y in dataloader:

            optimizer.zero_grad()

            logits = model(x)

            loss = criterion(logits, y)

            loss.backward()

            first_layer = next(model.parameters())

            gradient_norm = first_layer.grad.norm().item()

            optimizer.step()

            total_loss += loss.item() * x.size(0)

            predictions = logits.argmax(dim=1)

            correct += (predictions == y).sum().item()

            total += x.size(0)

        avg_loss = total_loss / total
        accuracy = correct / total

        print(
            f"Epoch {epoch + 1}/{epochs} "
            f"- Loss: {avg_loss:.4f} "
            f"- Accuracy: {accuracy:.4f}"
            f"- Grad: {gradient_norm:.6f}"
        )


def main():

    dataloader = create_toy_dataloader()

    print("\n=== Shallow CNN ===")

    shallow_model = ShallowCNN()

    train_model(
        shallow_model,
        dataloader,
        epochs=5,
    )

    print("\n=== Deep Plain CNN ===")

    deep_model = DeepPlainCNN()

    train_model(
        deep_model,
        dataloader,
        epochs=5,
    )


if __name__ == "__main__":
    main()