import torch

from resnet18 import ResNet18


def count_parameters(model):

    return sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )


def main():

    torch.manual_seed(42)

    model = ResNet18(
        num_classes=10
    )

    x = torch.randn(
        4,
        3,
        32,
        32,
    )

    output = model(x)

    print(
        "\n=== ResNet-18 ===\n"
    )

    print(
        f"Input shape:  {x.shape}"
    )

    print(
        f"Output shape: {output.shape}"
    )

    print(
        f"\nParameters: "
        f"{count_parameters(model):,}"
    )


if __name__ == "__main__":
    main()