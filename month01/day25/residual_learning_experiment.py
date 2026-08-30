import torch

from resnet import ResidualBlock


def main():

    torch.manual_seed(42)

    block = ResidualBlock(
        channels=8
    )

    x = torch.randn(
        1,
        8,
        8,
        8,
    )

    with torch.no_grad():

        identity = x

        out = block.conv1(x)

        out = block.relu(out)

        residual = block.conv2(out)

        output = residual + identity

        output = block.relu(output)

    print("\n=== Residual Learning Experiment ===\n")

    print(
        f"Input norm:     "
        f"{identity.norm().item():.6f}"
    )

    print(
        f"Residual norm:  "
        f"{residual.norm().item():.6f}"
    )

    print(
        f"Output norm:    "
        f"{output.norm().item():.6f}"
    )

    print("\nRelationship:")

    print(
        "H(x) = F(x) + x"
    )


if __name__ == "__main__":
    main()