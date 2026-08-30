import torch
import torch.nn as nn


def main():

    x = torch.randn(
        4,
        8,
        8,
        8,
    )

    projection = nn.Conv2d(
        in_channels=8,
        out_channels=16,
        kernel_size=1,
        stride=2,
    )

    output = projection(x)

    print("\n=== Projection Shortcut ===\n")

    print(f"Input shape:  {x.shape}")

    print(
        f"Output shape: {output.shape}"
    )


if __name__ == "__main__":
    main()