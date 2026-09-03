import numpy as np

from src.normalization import TrainOnlyNormalizer


def main():
    rng = np.random.default_rng(42)

    train_images = rng.normal(
        loc=100.0,
        scale=20.0,
        size=(80, 1, 8, 8),
    ).astype(np.float32)

    val_images = rng.normal(
        loc=200.0,
        scale=20.0,
        size=(20, 1, 8, 8),
    ).astype(np.float32)

    # --------------------------------------------------
    # WRONG:
    # Fit normalization on train + validation
    # --------------------------------------------------

    all_images = np.concatenate(
        [train_images, val_images],
        axis=0,
    )

    normalizer = TrainOnlyNormalizer()

    normalizer.fit(all_images)

    train_images = normalizer.transform(train_images)
    val_images = normalizer.transform(val_images)

    print("=== Leaky Preprocessing Demo ===")

    print(
        "Normalizer mean:",
        normalizer.mean.flatten(),
    )

    print(
        "Train mean after normalization:",
        float(train_images.mean()),
    )

    print(
        "Validation mean after normalization:",
        float(val_images.mean()),
    )

    print("\nWARNING:")
    print(
        "Normalization was fitted using "
        "validation data!"
    )


if __name__ == "__main__":
    main()