import numpy as np

from src.cross_validation import GroupKFoldValidator


def main():
    # 12 images
    X = np.arange(24).reshape(12, 2)

    # 4 images per patient
    groups = np.array([
        "patient_A",
        "patient_A",
        "patient_A",

        "patient_B",
        "patient_B",
        "patient_B",

        "patient_C",
        "patient_C",
        "patient_C",

        "patient_D",
        "patient_D",
        "patient_D",
    ])

    validator = GroupKFoldValidator(
        n_splits=4
    )

    print("=== Group 4-Fold Cross Validation ===")

    for fold_number, (train_indices, val_indices) in enumerate(
        validator.split(X, groups),
        start=1,
    ):
        train_groups = set(groups[train_indices])
        val_groups = set(groups[val_indices])

        print(f"\nFold {fold_number}")

        print(
            "Train groups:",
            sorted(train_groups),
        )

        print(
            "Validation groups:",
            sorted(val_groups),
        )

        print(
            "Group overlap:",
            train_groups & val_groups,
        )


if __name__ == "__main__":
    main()