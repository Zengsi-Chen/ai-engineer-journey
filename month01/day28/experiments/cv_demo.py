import numpy as np

from src.cross_validation import KFoldValidator


def main():
    X = np.arange(20).reshape(
        10,
        2,
    )

    validator = KFoldValidator(
        n_splits=5,
        shuffle=True,
        random_state=42,
    )

    print("=== 5-Fold Cross Validation ===")

    for fold_number, (
        train_indices,
        val_indices,
    ) in enumerate(
        validator.split(X),
        start=1,
    ):
        print(f"\nFold {fold_number}")

        print("Train indices:")
        print(train_indices)

        print("Validation indices:")
        print(val_indices)

    fold_scores = [
        0.80,
        0.90,
        0.85,
        0.95,
        0.90,
    ]

    result = validator.aggregate(
        fold_scores
    )

    print("\n=== CV Result ===")

    print(
        "Fold scores:",
        result.fold_scores,
    )

    print(
        f"Mean score: "
        f"{result.mean_score:.4f}"
    )

    print(
        f"Std score: "
        f"{result.std_score:.4f}"
    )

    print(
        f"\nCV Summary: "
        f"{result.mean_score:.4f} "
        f"± {result.std_score:.4f}"
    )


if __name__ == "__main__":
    main()