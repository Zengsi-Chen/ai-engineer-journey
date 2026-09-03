import numpy as np

from src.data_split import GroupDataSplitter


def test_groups_never_cross_splits():
    X = np.arange(
        60
    ).reshape(
        30,
        2,
    )

    y = np.array(
        [0, 1] * 15
    )

    groups = np.repeat(
        np.arange(10),
        3,
    )

    splitter = GroupDataSplitter(
        validation_size=0.2,
        test_size=0.2,
        random_state=42,
    )

    (
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test,
        train_groups,
        val_groups,
        test_groups,
    ) = splitter.split(
        X,
        y,
        groups,
    )

    train_groups = set(
        train_groups
    )

    val_groups = set(
        val_groups
    )

    test_groups = set(
        test_groups
    )

    assert train_groups.isdisjoint(
        val_groups
    )

    assert train_groups.isdisjoint(
        test_groups
    )

    assert val_groups.isdisjoint(
        test_groups
    )


def test_group_split_preserves_all_samples():
    X = np.arange(
        60
    ).reshape(
        30,
        2,
    )

    y = np.array(
        [0, 1] * 15
    )

    groups = np.repeat(
        np.arange(10),
        3,
    )

    splitter = GroupDataSplitter(
        validation_size=0.2,
        test_size=0.2,
        random_state=42,
    )

    (
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test,
        train_groups,
        val_groups,
        test_groups,
    ) = splitter.split(
        X,
        y,
        groups,
    )

    total_samples = (
        len(X_train)
        + len(X_val)
        + len(X_test)
    )

    assert total_samples == len(X)

    assert (
        len(y_train)
        + len(y_val)
        + len(y_test)
        == len(y)
    )