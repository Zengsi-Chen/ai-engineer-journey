import numpy as np
from sklearn.model_selection import (
    train_test_split,
    GroupShuffleSplit,
)


class DataSplitter:
    """
    Split data into train, validation, and test sets.

    The split happens before learned preprocessing
    and random augmentation.
    """

    def __init__(
        self,
        validation_size: float = 0.1,
        test_size: float = 0.1,
        random_state: int = 42,
    ):
        if validation_size <= 0:
            raise ValueError(
                "validation_size must be greater than 0."
            )

        if test_size <= 0:
            raise ValueError(
                "test_size must be greater than 0."
            )

        if validation_size + test_size >= 1:
            raise ValueError(
                "validation_size + test_size must be less than 1."
            )

        self.validation_size = validation_size
        self.test_size = test_size
        self.random_state = random_state

    def split(
        self,
        X: np.ndarray,
        y: np.ndarray,
    ):
        """
        Split X and y into train, validation, and test sets.

        Returns:
            X_train, X_val, X_test,
            y_train, y_val, y_test
        """

        # Step 1: hold out test data
        X_train_val, X_test, y_train_val, y_test = (
            train_test_split(
                X,
                y,
                test_size=self.test_size,
                random_state=self.random_state,
                stratify=y,
            )
        )

        # Step 2: calculate validation fraction
        validation_fraction = (
            self.validation_size
            / (1.0 - self.test_size)
        )

        # Step 3: split remaining data
        X_train, X_val, y_train, y_val = (
            train_test_split(
                X_train_val,
                y_train_val,
                test_size=validation_fraction,
                random_state=self.random_state,
                stratify=y_train_val,
            )
        )

        return (
            X_train,
            X_val,
            X_test,
            y_train,
            y_val,
            y_test,
        )


class GroupDataSplitter:
    def __init__(
        self,
        validation_size: float = 0.2,
        test_size: float = 0.2,
        random_state: int = 42,
    ):
        if validation_size <= 0:
            raise ValueError(
                "validation_size must be greater than 0."
            )

        if test_size <= 0:
            raise ValueError(
                "test_size must be greater than 0."
            )

        if validation_size + test_size >= 1:
            raise ValueError(
                "validation_size + test_size "
                "must be less than 1."
            )

        self.validation_size = validation_size
        self.test_size = test_size
        self.random_state = random_state

    def split(
        self,
        X: np.ndarray,
        y: np.ndarray,
        groups: np.ndarray,
    ):
        # -------------------------------
        # First: Test split
        # -------------------------------
        test_splitter = GroupShuffleSplit(
            n_splits=1,
            test_size=self.test_size,
            random_state=self.random_state,
        )

        train_val_indices, test_indices = next(
            test_splitter.split(
                X,
                y,
                groups,
            )
        )

        # -------------------------------
        # Second: Validation split
        # -------------------------------
        remaining_groups = groups[
            train_val_indices
        ]

        validation_fraction = (
            self.validation_size
            / (1.0 - self.test_size)
        )

        val_splitter = GroupShuffleSplit(
            n_splits=1,
            test_size=validation_fraction,
            random_state=self.random_state,
        )

        train_relative, val_relative = next(
            val_splitter.split(
                X[train_val_indices],
                y[train_val_indices],
                remaining_groups,
            )
        )

        train_indices = train_val_indices[
            train_relative
        ]

        val_indices = train_val_indices[
            val_relative
        ]

        return (
            X[train_indices],
            X[val_indices],
            X[test_indices],
            y[train_indices],
            y[val_indices],
            y[test_indices],
            groups[train_indices],
            groups[val_indices],
            groups[test_indices],
        )