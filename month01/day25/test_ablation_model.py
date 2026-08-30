import torch

from resnet_ablation import (
    NoSkipResNet18,
    NoBNResNet18,
)

from resnet18 import ResNet18


def main():

    # model = NoSkipResNet18()
    model = NoBNResNet18()

    x = torch.randn(
        4,
        3,
        32,
        32,
    )

    output = model(x)

    print(
        "Input shape:",
        x.shape,
    )

    print(
        "Output shape:",
        output.shape,
    )


def count_parameters(model):

    return sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

standard_model = ResNet18()

# ablation_model = NoSkipResNet18()
ablation_model = NoBNResNet18()

print(
    "Standard ResNet-18:",
    count_parameters(
        standard_model
    ),
)

print(
    "No-BN ResNet-18:",
    count_parameters(
        ablation_model
    ),
)

if __name__ == "__main__":
    main()