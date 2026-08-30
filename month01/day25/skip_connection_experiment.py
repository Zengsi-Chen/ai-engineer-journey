import torch

from resnet import (
    ProjectionResidualBlock,
)


def main():

    x = torch.randn(
        4,
        8,
        8,
        8,
    )

    print(
        "\n=== Identity Shortcut ===\n"
    )

    identity_block = (
        ProjectionResidualBlock(
            in_channels=8,
            out_channels=8,
            stride=1,
        )
    )

    output = identity_block(x)

    print(f"Input:  {x.shape}")

    print(
        f"Output: {output.shape}"
    )

    print(
        "\n=== Projection Shortcut ===\n"
    )

    projection_block = (
        ProjectionResidualBlock(
            in_channels=8,
            out_channels=16,
            stride=2,
        )
    )

    output = projection_block(x)

    print(f"Input:  {x.shape}")

    print(
        f"Output: {output.shape}"
    )


if __name__ == "__main__":
    main()