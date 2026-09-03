from src.leakage_checks import (
    check_split_overlap,
    has_leakage,
)


def main():
    train_ids = {
        "patient_A",
        "patient_B",
        "patient_C",
    }

    val_ids = {
        "patient_C",
        "patient_D",
    }

    test_ids = {
        "patient_E",
        "patient_F",
    }

    overlaps = check_split_overlap(
        train_ids,
        val_ids,
        test_ids,
    )

    print("=== Leakage Detector ===")

    print(
        "Train / Validation overlap:",
        overlaps["train_val"],
    )

    print(
        "Train / Test overlap:",
        overlaps["train_test"],
    )

    print(
        "Validation / Test overlap:",
        overlaps["val_test"],
    )

    print(
        "\nLeakage detected:",
        has_leakage(overlaps),
    )


if __name__ == "__main__":
    main()