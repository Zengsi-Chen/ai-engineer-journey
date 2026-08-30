from torchvision import datasets
from torchvision import transforms


def calculate_statistics(
    dataset,
    max_samples=1000,
):

    mean = 0.0

    std = 0.0

    total_samples = 0

    for index, (image, _) in enumerate(dataset):

        if index >= max_samples:
            break

        mean += image.mean(
            dim=(1, 2)
        )

        std += image.std(
            dim=(1, 2)
        )

        total_samples += 1

    mean /= total_samples

    std /= total_samples

    return mean, std


def main():

    raw_transform = transforms.Compose(
        [
            transforms.ToTensor(),
        ]
    )

    normalized_transform = transforms.Compose(
        [
            transforms.ToTensor(),

            transforms.Normalize(
                mean=(
                    0.4914,
                    0.4822,
                    0.4465,
                ),

                std=(
                    0.2470,
                    0.2435,
                    0.2616,
                ),
            ),
        ]
    )

    raw_dataset = datasets.CIFAR10(
        root="data",
        train=True,
        download=True,
        transform=raw_transform,
    )

    normalized_dataset = datasets.CIFAR10(
        root="data",
        train=True,
        download=True,
        transform=normalized_transform,
    )

    raw_mean, raw_std = (
        calculate_statistics(
            raw_dataset
        )
    )

    normalized_mean, normalized_std = (
        calculate_statistics(
            normalized_dataset
        )
    )

    print(
        "\n=== Before Normalization ==="
    )

    print(
        f"Mean: {raw_mean}"
    )

    print(
        f"Std:  {raw_std}"
    )

    print(
        "\n=== After Normalization ==="
    )

    print(
        f"Mean: {normalized_mean}"
    )

    print(
        f"Std:  {normalized_std}"
    )


if __name__ == "__main__":
    main()