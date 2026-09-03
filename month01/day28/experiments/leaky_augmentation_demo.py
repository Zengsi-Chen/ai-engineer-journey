import numpy as np

from src.augmentation import random_horizontal_flip


def main():
    image = np.arange(
        16,
        dtype=np.float32,
    ).reshape(1, 4, 4)

    rng = np.random.default_rng(42)

    augmented_image = random_horizontal_flip(
        image,
        rng=rng,
        probability=1.0,
    )

    # WRONG:
    # Original and augmented versions are treated
    # as independent samples.

    train_images = np.stack(
        [image]
    )

    val_images = np.stack(
        [augmented_image]
    )

    print("=== Leaky Augmentation Demo ===")

    print("\nOriginal image:")
    print(image)

    print("\nAugmented image:")
    print(augmented_image)

    print("\nTrain contains original image.")
    print("Validation contains augmented version.")

    print(
        "\nWARNING: "
        "Same source image crossed train/validation!"
    )


if __name__ == "__main__":
    main()