import torch

from resnet import Bottleneck


def main():

    torch.manual_seed(42)

    x = torch.randn(
        4,
        256,
        8,
        8,
    )

    block = Bottleneck(
        in_channels=256,
        base_channels=128,
        stride=1,
    )

    output = block(x)

    print(
        "\n=== Bottleneck ===\n"
    )

    print(
        f"Input shape:  {x.shape}"
    )

    print(
        f"Output shape: {output.shape}"
    )

    print(
        f"Parameters:   "
        f"{sum(p.numel() for p in block.parameters()):,}"
    )


if __name__ == "__main__":
    main()