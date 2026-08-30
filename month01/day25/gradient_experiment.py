import torch
import torch.nn as nn

from model import (
    ShallowCNN,
    DeepPlainCNN,
    VeryDeepPlainCNN,
)


def get_first_layer_gradient(model):

    x = torch.randn(
        16,
        1,
        8,
        8,
    )

    y = torch.randint(
        0,
        2,
        (16,),
    )

    criterion = nn.CrossEntropyLoss()

    model.zero_grad()

    output = model(x)

    loss = criterion(output, y)

    loss.backward()

    first_parameter = next(model.parameters())

    return first_parameter.grad.norm().item()


def main():

    torch.manual_seed(42)

    models = {
        "Shallow CNN": ShallowCNN(),
        "Deep CNN": DeepPlainCNN(),
        "Very Deep CNN": VeryDeepPlainCNN(
            num_layers=20
        ),
    }

    print("\n=== Gradient Flow Experiment ===\n")

    for name, model in models.items():

        gradient_norm = get_first_layer_gradient(
            model
        )

        print(
            f"{name:20} "
            f"First Layer Gradient: "
            f"{gradient_norm:.10f}"
        )


if __name__ == "__main__":
    main()