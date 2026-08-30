from cifar10_data import (
    create_cifar10_loaders,
)


def main():

    train_loader, validation_loader, _ = (
        create_cifar10_loaders(
            batch_size=4,
            train_samples=100,
            validation_samples=20,
        )
    )

    train_images, train_labels = (
        next(
            iter(train_loader)
        )
    )

    validation_images, validation_labels = (
        next(
            iter(validation_loader)
        )
    )

    print(
        "\n=== Training Batch ==="
    )

    print(
        train_images.shape
    )

    print(
        "\n=== Validation Batch ==="
    )

    print(
        validation_images.shape
    )


if __name__ == "__main__":
    main()