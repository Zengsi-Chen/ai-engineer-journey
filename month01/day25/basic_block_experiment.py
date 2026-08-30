import torch

from resnet import BasicBlock


def main():

    torch.manual_seed(42)

    x = torch.randn(
        4,
        8,
        8,
        8,
    )

    print(
        "\n=== BasicBlock: Identity Shortcut ===\n"
    )

    block1 = BasicBlock(
        in_channels=8,
        out_channels=8,
        stride=1,
    )

    output1 = block1(x)

    print(
        f"Input shape:  {x.shape}"
    )

    print(
        f"Output shape: {output1.shape}"
    )

    print(
        "\n=== BasicBlock: Projection Shortcut ===\n"
    )

    block2 = BasicBlock(
        in_channels=8,
        out_channels=16,
        stride=2,
    )

    output2 = block2(x)

    print(
        f"Input shape:  {x.shape}"
    )

    print(
        f"Output shape: {output2.shape}"
    )


if __name__ == "__main__":
    main()