import torch
import torch.nn as nn

from identity_residual_block import (
    IdentityResidualBlock,
)


def main():

    torch.manual_seed(42)

    block = IdentityResidualBlock(
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
    ).abs().max().item()

    print(
        "\n=== Identity Mapping Test ===\n"
    )

    print(
        f"Maximum difference: "
        f"{difference:.10f}"
    )

    print(
        "\nExpected:"
    )

    print(
        "F(x) = 0"
    )

    print(
        "Output = x"
    )


if __name__ == "__main__":
    main()