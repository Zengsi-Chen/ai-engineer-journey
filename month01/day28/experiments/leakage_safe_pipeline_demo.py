import numpy as np

from src.leakage_safe_pipeline import (
    LeakageSafeImagePipeline,
)


def main():
    rng = np.random.default_rng(42)

    # Simulated image dataset
    images = rng.normal(
        loc=100.0,
        scale=20.0,
        size=(100, 1, 8, 8),
    ).astype(np.float32)

    labels = np.array(
        [0] * 50 + [1] * 50
    )

    pipeline = LeakageSafeImagePipeline(
        validation_size=0.2,
        test_size=0.2,
        random_state=42,
        augmentation_seed=42,
    )

    result = pipeline.prepare(
        images,
        labels,
    )

    print("=== Leakage-Safe Image Pipeline ===")

    for split_name in [
        "train",
        "validation",
        "test",
    ]:
        split = result[split_name]

        print(f"\n{split_name.upper()}")

        print(
            "Images shape:",
            split["images"].shape,
        )

        print(
            "Labels shape:",
            split["labels"].shape,
        )

        print(
            "Mean:",
            float(split["images"].mean()),
        )

        print(
            "Std:",
            float(split["images"].std()),
        )


if __name__ == "__main__":
    main()