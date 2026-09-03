import numpy as np
from sklearn.model_selection import train_test_split


def main():
    # Simulate 5 images from each patient
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

    train_ids, val_ids = train_test_split(
        image_ids,
        test_size=0.3,
        random_state=42,
    )

    train_groups = set(groups[train_ids])
    val_groups = set(groups[val_ids])

    overlap = train_groups & val_groups

    print("=== Incorrect Image Split ===")

    print("Train image IDs:", train_ids)
    print("Validation image IDs:", val_ids)

    print("Train groups:", sorted(train_groups))
    print("Validation groups:", sorted(val_groups))

    print("Group overlap:", sorted(overlap))

    if overlap:
        print("\nWARNING: Group leakage detected!")
    else:
        print("\nNo group leakage detected.")


if __name__ == "__main__":
    main()