import numpy as np


def random_horizontal_flip(
    image: np.ndarray,
    rng: np.random.Generator,
    probability: float = 0.5,
) -> np.ndarray:
    """
    Randomly flip an image horizontally.

    Expected image shape:
        (C, H, W)
    """
    if not 0.0 <= probability <= 1.0:
        raise ValueError(
            "probability must be between 0 and 1."
        )

    if rng.random() < probability:
        return np.flip(
            image,
            axis=-1,
        ).copy()

    return image.copy()


class TrainOnlyAugmenter:
    """
    Apply random augmentation to training data only.

    The augmenter requires an explicit split name so that
    validation and test data cannot accidentally receive
    training augmentation.
    """

    TRAIN_SPLIT = "train"
    VALIDATION_SPLIT = "validation"
    TEST_SPLIT = "test"

    def __init__(
        self,
        seed: int = 42,
        probability: float = 0.5,
    ):
        self.rng = np.random.default_rng(seed)
        self.probability = probability

    def transform(
        self,
        images: np.ndarray,
        split: str,
    ) -> np.ndarray:
        """
        Apply augmentation only to training images.

        Validation and test images are returned unchanged.
        """
        if split == self.TRAIN_SPLIT:
            return np.stack([
                random_horizontal_flip(
                    image,
                    rng=self.rng,
                    probability=self.probability,
                )
                for image in images
            ])

        if split in {
            self.VALIDATION_SPLIT,
            self.TEST_SPLIT,
        }:
            return images.copy()

        raise ValueError(
            f"Unknown split: {split}. "
            "Expected 'train', 'validation', or 'test'."
        )