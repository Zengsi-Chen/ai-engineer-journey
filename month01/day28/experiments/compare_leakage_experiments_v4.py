import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.model_selection import train_test_split


def create_dataset(
    n_groups=100,
    samples_per_group=5,
    random_state=42,
):
    rng = np.random.default_rng(random_state)

    n_samples = n_groups * samples_per_group

    groups = np.repeat(
        np.arange(n_groups),
        samples_per_group,
    )

    # ----------------------------------------
    # True signal
    # ----------------------------------------

    true_signal = rng.normal(
        size=n_samples
    )

    # Weak relationship between signal and target
    y = (
        true_signal
        + rng.normal(
            scale=2.0,
            size=n_samples,
        )
        > 0
    ).astype(int)

    # ----------------------------------------
    # Patient-specific fingerprint
    # ----------------------------------------

    patient_fingerprint = rng.normal(
        size=n_groups
    )

    fingerprint = np.repeat(
        patient_fingerprint,
        samples_per_group,
    )

    # ----------------------------------------
    # IMPORTANT:
    #
    # In the observed dataset, fingerprint
    # correlates with the patient's labels.
    # ----------------------------------------

    patient_bias = (
        patient_fingerprint > 0
    ).astype(int)

    patient_bias = np.repeat(
        patient_bias,
        samples_per_group,
    )

    # Shortcut is much stronger than true signal.
    shortcut = (
        4.0 * patient_bias
        + 0.1 * fingerprint
    )

    X = np.column_stack(
        [
            true_signal,
            shortcut,
        ]
    )

    return X, y, groups


def evaluate_split(
    X,
    y,
    groups,
    train_idx,
    val_idx,
):
    model = LogisticRegression(
        max_iter=1000
    )

    model.fit(
        X[train_idx],
        y[train_idx],
    )

    predictions = model.predict(
        X[val_idx]
    )

    accuracy = accuracy_score(
        y[val_idx],
        predictions,
    )

    train_groups = set(
        groups[train_idx]
    )

    val_groups = set(
        groups[val_idx]
    )

    overlap = train_groups & val_groups

    return (
        accuracy,
        overlap,
        model,
    )


def run_random_split(
    X,
    y,
    groups,
):
    indices = np.arange(
        len(X)
    )

    train_idx, val_idx = train_test_split(
        indices,
        test_size=0.3,
        random_state=42,
        stratify=y,
    )

    return evaluate_split(
        X,
        y,
        groups,
        train_idx,
        val_idx,
    )


def run_group_split(
    X,
    y,
    groups,
):
    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=0.3,
        random_state=42,
    )

    train_idx, val_idx = next(
        splitter.split(
            X,
            y,
            groups,
        )
    )

    return evaluate_split(
        X,
        y,
        groups,
        train_idx,
        val_idx,
    )


def main():
    X, y, groups = create_dataset()

    (
        random_accuracy,
        random_overlap,
        random_model,
    ) = run_random_split(
        X,
        y,
        groups,
    )

    (
        group_accuracy,
        group_overlap,
        group_model,
    ) = run_group_split(
        X,
        y,
        groups,
    )

    print(
        "=== Shortcut Leakage Experiment v4 ==="
    )

    print("\nDataset")
    print("--------------------")

    print(
        "Samples:",
        len(X),
    )

    print(
        "Groups:",
        len(np.unique(groups)),
    )

    print(
        "Features:",
        X.shape[1],
    )

    print(
        "Feature 0: True Signal"
    )

    print(
        "Feature 1: Patient Shortcut"
    )

    print("\nRandom Split")
    print("--------------------")

    print(
        "Validation Accuracy:",
        f"{random_accuracy:.4f}",
    )

    print(
        "Group Overlap:",
        len(random_overlap),
    )

    print(
        "Model Coefficients:",
        random_model.coef_.round(4),
    )

    print("\nGroup Split")
    print("--------------------")

    print(
        "Validation Accuracy:",
        f"{group_accuracy:.4f}",
    )

    print(
        "Group Overlap:",
        len(group_overlap),
    )

    print(
        "Model Coefficients:",
        group_model.coef_.round(4),
    )

    print("\nAccuracy Difference")
    print("--------------------")

    print(
        "Random - Group:",
        f"{random_accuracy - group_accuracy:.4f}",
    )

    print("\nLeakage Checks")
    print("--------------------")

    print(
        "Random split leakage:",
        len(random_overlap) > 0,
    )

    print(
        "Group split leakage:",
        len(group_overlap) > 0,
    )


if __name__ == "__main__":
    main()