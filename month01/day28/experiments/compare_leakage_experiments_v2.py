import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.model_selection import train_test_split


def create_dataset(
    n_groups: int = 100,
    samples_per_group: int = 5,
    random_state: int = 42,
):
    rng = np.random.default_rng(random_state)

    n_samples = n_groups * samples_per_group

    # Each patient/group has its own fingerprint.
    group_fingerprints = rng.normal(
        loc=0.0,
        scale=3.0,
        size=n_groups,
    )

    groups = np.repeat(
        np.arange(n_groups),
        samples_per_group,
    )

    # Each image inherits the fingerprint of its patient.
    X = np.repeat(
        group_fingerprints,
        samples_per_group,
    )

    # Small image-level noise.
    X = X + rng.normal(
        loc=0.0,
        scale=0.2,
        size=n_samples,
    )

    # Binary target.
    y = (group_fingerprints > 0).astype(int)

    y = np.repeat(
        y,
        samples_per_group,
    )

    X = X.reshape(-1, 1)

    return X, y, groups


def run_random_split(
    X,
    y,
    groups,
):
    indices = np.arange(len(X))

    train_idx, val_idx = train_test_split(
        indices,
        test_size=0.3,
        random_state=42,
        stratify=y,
    )

    model = LogisticRegression()

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

    return accuracy, overlap


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

    model = LogisticRegression()

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

    return accuracy, overlap


def main():
    X, y, groups = create_dataset()

    random_accuracy, random_overlap = (
        run_random_split(
            X,
            y,
            groups,
        )
    )

    group_accuracy, group_overlap = (
        run_group_split(
            X,
            y,
            groups,
        )
    )

    print(
        "=== Controlled Leakage Experiment v2 ==="
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
        "Samples per group:",
        len(X) // len(np.unique(groups)),
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

    print("\nAccuracy Difference")
    print("--------------------")
    print(
        "Random - Group:",
        f"{random_accuracy - group_accuracy:.4f}",
    )

    if random_accuracy > group_accuracy:
        print(
            "\nWARNING:"
            " Random split produced a higher"
            " validation score."
        )

    if len(random_overlap) > 0:
        print(
            "Group leakage exists in random split."
        )

    if len(group_overlap) == 0:
        print(
            "Group split is leakage-safe."
        )


if __name__ == "__main__":
    main()