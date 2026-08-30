import torch
import torch.nn as nn

from resnet import ResidualBlock


def main():

    torch.manual_seed(42)

    block = ResidualBlock(
        channels=8
    )

    for module in block.modules():

        if isinstance(
            module,
            nn.Conv2d,
        ):
            nn.init.zeros_(
                module.weight
            )

            if module.bias is not None:
                nn.init.zeros_(
                    module.bias
                )

    x = torch.randn(
        1,
        8,
        8,
        8,
    )

    with torch.no_grad():

        output = block(x)

    difference = (
        output - x
    ).abs().mean().item()

    print(
        "\n=== Identity Experiment ===\n"
    )

    print(
        f"Mean difference: "
        f"{difference:.10f}"
    )

    print(
        "\nExpected:"
    )

    print(
        "Residual F(x) ≈ 0"
    )

    print(
        "Output ≈ x"
    )


if __name__ == "__main__":
    main()