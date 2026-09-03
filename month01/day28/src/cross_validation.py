from dataclasses import dataclass

import numpy as np
from sklearn.model_selection import (
    KFold,
    StratifiedKFold,
    GroupKFold,
    )


@dataclass
class CrossValidationResult:
    """
    Stores cross-validation metrics.
    """

    fold_scores: list[float]
    mean_score: float
    std_score: float


class KFoldValidator:
    """
    Generate reproducible K-Fold splits and aggregate scores.
    """

    def __init__(
        self,
        n_splits: int = 5,
        shuffle: bool = True,
        random_state: int = 42,
    ):
        if n_splits < 2:
            raise ValueError(
                "n_splits must be at least 2."
            )

        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state

    def split(self, X: np.ndarray):
        """
        Generate train/validation indices for each fold.
        """
        random_state = (
            self.random_state
            if self.shuffle
            else None
        )

        kfold = KFold(
            n_splits=self.n_splits,
            shuffle=self.shuffle,
            random_state=random_state,
        )

        return kfold.split(X)

    def aggregate(
        self,
        fold_scores: list[float],
    ) -> CrossValidationResult:
        """
        Calculate mean and standard deviation across folds.
        """
        if len(fold_scores) != self.n_splits:
            raise ValueError(
                "Number of fold scores must equal n_splits."
            )

        scores = np.asarray(
            fold_scores,
            dtype=float,
        )

        return CrossValidationResult(
            fold_scores=scores.tolist(),
            mean_score=float(scores.mean()),
            std_score=float(scores.std()),
        )


class StratifiedKFoldValidator:
    def __init__(
        self,
        n_splits: int = 5,
        shuffle: bool = True,
        random_state: int = 42,
    ):
        if n_splits < 2:
            raise ValueError("n_splits must be at least 2.")

        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state

    def split(self, X: np.ndarray, y: np.ndarray):
        random_state = self.random_state if self.shuffle else None

        stratified_kfold = StratifiedKFold(
            n_splits=self.n_splits,
            shuffle=self.shuffle,
            random_state=random_state,
        )

        return stratified_kfold.split(X, y)

    def aggregate(
        self,
        fold_scores: list[float],
    ) -> CrossValidationResult:

        if len(fold_scores) != self.n_splits:
            raise ValueError(
                "Number of fold scores must equal n_splits."
            )

        scores = np.asarray(fold_scores, dtype=float)

        return CrossValidationResult(
            fold_scores=scores.tolist(),
            mean_score=float(scores.mean()),
            std_score=float(scores.std()),
        )


class GroupKFoldValidator:
    def __init__(self, n_splits: int = 5):
        if n_splits < 2:
            raise ValueError(
                "n_splits must be at least 2."
            )

        self.n_splits = n_splits

    def split(
        self,
        X: np.ndarray,
        groups: np.ndarray,
    ):
        group_kfold = GroupKFold(
            n_splits=self.n_splits
        )

        return group_kfold.split(
            X,
            groups=groups,
        )