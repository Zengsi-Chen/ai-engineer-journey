import numpy as np


class TrainOnlyNormalizer:
    """
    Calculate normalization statistics using training data only.
    """

    def __init__(self, eps: float = 1e-8):
        self.mean = None
        self.std = None
        self.eps = eps
        self.is_fitted = False

    def fit(self, images: np.ndarray):
        """
        Calculate channel-wise mean and std from training images.

        Expected shape:
            (N, C, H, W)
        """
        if images.ndim != 4:
            raise ValueError(
                "Expected images with shape (N, C, H, W)."
            )

        self.mean = images.mean(
            axis=(0, 2, 3),
            keepdims=True,
        )

        self.std = images.std(
            axis=(0, 2, 3),
            keepdims=True,
        )

        self.is_fitted = True

        return self

    def transform(self, images: np.ndarray) -> np.ndarray:
        """
        Normalize images using training-derived statistics.
        """
        if not self.is_fitted:
            raise RuntimeError(
                "Normalizer must be fitted on training data "
                "before transform()."
            )

        return (images - self.mean) / (self.std + self.eps)

    def fit_transform(self, images: np.ndarray) -> np.ndarray:
        """
        Fit and transform training images.
        """
        self.fit(images)

        return self.transform(images)