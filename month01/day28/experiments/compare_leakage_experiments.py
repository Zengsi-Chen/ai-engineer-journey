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

    groups = np.repeat(
        np.arange(n_groups),
        samples_per_group,
    )

    group_signal = rng.normal(
        size=(n_groups, 1)
    )

    X = np.repeat(
        group_signal,
        samples_per_group,
        axis=0,
    )

    X += rng.normal(
        scale=0.1,
        size=(n_samples, 1),
    )

    y = (group_signal[:, 0] > 0).astype(int)

    y = np.repeat(
        y,
        samples_per_group,
    )

    return X, y, groups


def run_random_split(
    X,
    y,
    groups,
):
    train_idx, val_idx = train_test_split(
        np.arange(len(X)),
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

    train_groups = set(groups[train_idx])
    val_groups = set(groups[val_idx])

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

    train_groups = set(groups[train_idx])
    val_groups = set(groups[val_idx])

    overlap = train_groups & val_groups

    return accuracy, overlap


def main():
    X, y, groups = create_dataset()

    random_accuracy, random_overlap = run_random_split(
        X,
        y,
        groups,
    )

    group_accuracy, group_overlap = run_group_split(
        X,
        y,
        groups,
    )

    print("=== Leakage Experiment Comparison ===")

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

    print("\nAccuracy Difference:")
    print(
        f"{random_accuracy - group_accuracy:.4f}"
    )


if __name__ == "__main__":
    main()