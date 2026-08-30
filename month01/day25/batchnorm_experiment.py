import torch
import torch.nn as nn


def main():

    torch.manual_seed(42)

    x = torch.randn(
        16,
        8,
        8,
        8,
    ) * 10 + 5

    batch_norm = nn.BatchNorm2d(
        num_features=8
    )

    batch_norm.train()

    output = batch_norm(x)

    input_mean = (
        x.mean(
            dim=(0, 2, 3)
        )
    )

    output_mean = (
        output.mean(
            dim=(0, 2, 3)
        )
    )

    print(
        "\n=== BatchNorm Experiment ===\n"
    )

    print(
        "Input channel means:"
    )

    print(
        input_mean
    )

    print(
        "\nOutput channel means:"
    )

    print(
        output_mean
    )


if __name__ == "__main__":
    main()