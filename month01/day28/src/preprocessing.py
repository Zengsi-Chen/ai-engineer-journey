import numpy as np
from sklearn.preprocessing import StandardScaler


class SafePreprocessor:
    """
    A preprocessing wrapper that prevents accidental use
    before fitting on training data.
    """

    def __init__(self):
        self.scaler = StandardScaler()
        self.is_fitted = False

    def fit(self, X: np.ndarray):
        """
        Fit preprocessing statistics.

        IMPORTANT:
        This method must only receive training data.
        """
        self.scaler.fit(X)
        self.is_fitted = True

        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Transform data using training-derived statistics.
        """
        if not self.is_fitted:
            raise RuntimeError(
                "Preprocessor must be fitted on training data "
                "before transform()."
            )

        return self.scaler.transform(X)

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """
        Convenience method for training data only.

        Equivalent to:
            fit(X)
            transform(X)
        """
        self.fit(X)

        return self.transform(X)