import numpy as np

from src.cross_validation import StratifiedKFoldValidator


def main():
    # Simulate an imbalanced binary classification dataset
    X = np.arange(100).reshape(50, 2)

    # 40 samples of class 0
    # 10 samples of class 1
    y = np.array([0] * 40 + [1] * 10)

    validator = StratifiedKFoldValidator(
        n_splits=5,
        shuffle=True,
        random_state=42,
    )

    print("=== Stratified 5-Fold Cross Validation ===")

    for fold_number, (train_indices, val_indices) in enumerate(
        validator.split(X, y),
        start=1,
    ):
        train_labels = y[train_indices]
        val_labels = y[val_indices]

        print(f"\nFold {fold_number}")

        print(
            "Train class distribution:",
            {
                0: int((train_labels == 0).sum()),
                1: int((train_labels == 1).sum()),
            },
        )

        print(
            "Validation class distribution:",
            {
                0: int((val_labels == 0).sum()),
                1: int((val_labels == 1).sum()),
            },
        )


if __name__ == "__main__":
    main()