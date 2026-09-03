import numpy as np
from sklearn.model_selection import train_test_split


def main():
    groups = np.array(
        [
            "patient_A",
            "patient_A",
            "patient_A",
            "patient_A",
            "patient_A",

            "patient_B",
            "patient_B",
            "patient_B",
            "patient_B",
            "patient_B",

            "patient_C",
            "patient_C",
            "patient_C",
            "patient_C",
            "patient_C",
        ]
    )

    image_ids = np.arange(len(groups))

    # --------------------------------------------------
    # WRONG:
    # Split individual images instead of patients
    # --------------------------------------------------

    train_ids, val_ids = train_test_split(
        image_ids,
        test_size=0.3,
        random_state=42,
    )

    train_groups = set(groups[train_ids])
    val_groups = set(groups[val_ids])

    overlap = train_groups & val_groups

    print("=== Leaky Group Split Demo ===")

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
        sorted(overlap),
    )

    if overlap:
        print(
            "\nWARNING: Group leakage detected!"
        )


if __name__ == "__main__":
    main()