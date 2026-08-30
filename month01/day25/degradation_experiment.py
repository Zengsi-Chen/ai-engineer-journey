import torch

from dataset import create_toy_dataloader

from model import (
    ShallowCNN,
    DeepPlainCNN,
)

from resnet import DeepResidualCNN


def train_model(
    model,
    dataloader,
    epochs=10,
):
    criterion = torch.nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.001,
    )

    model.train()

    for epoch in range(epochs):

        correct = 0
        total = 0

        for x, y in dataloader:

            optimizer.zero_grad()

            logits = model(x)

            loss = criterion(
                logits,
                y,
            )

            loss.backward()

            optimizer.step()

            predictions = logits.argmax(
                dim=1
            )

            correct += (
                predictions == y
            ).sum().item()

            total += y.size(0)

        accuracy = correct / total

        print(
            f"Epoch {epoch + 1}/{epochs} "
            f"Accuracy: {accuracy:.4f}"
        )


def main():

    torch.manual_seed(42)

    dataloader = create_toy_dataloader(
        n_samples=512,
        batch_size=32,
    )

    models = {

        "Shallow CNN":
        ShallowCNN(),

        "Deep Plain CNN":
        DeepPlainCNN(),

        "Deep Residual CNN":
        DeepResidualCNN(
            num_blocks=4
        ),
    }

    for name, model in models.items():

        print("\n" + "=" * 40)

        print(name)

        print("=" * 40)

        train_model(
            model,
            dataloader,
            epochs=10,
        )


if __name__ == "__main__":
    main()