import numpy as np

from src.augmentation import TrainOnlyAugmenter
from src.data_split import (
    DataSplitter,
    GroupDataSplitter,
)
from src.leakage_checks import (
    check_split_overlap,
    has_leakage,
)
from src.normalization import TrainOnlyNormalizer


class LeakageSafeImagePipeline:
    def __init__(
        self,
        validation_size: float = 0.2,
        test_size: float = 0.2,
        random_state: int = 42,
        augmentation_seed: int = 42,
        split_unit: str = "sample",
    ):
        if split_unit not in {
            "sample",
            "group",
        }:
            raise ValueError(
                "split_unit must be 'sample' or 'group'."
            )

        self.split_unit = split_unit

        self.validation_size = validation_size
        self.test_size = test_size
        self.random_state = random_state

        if split_unit == "sample":
            self.splitter = DataSplitter(
                validation_size=validation_size,
                test_size=test_size,
                random_state=random_state,
            )
        else:
            self.splitter = GroupDataSplitter(
                validation_size=validation_size,
                test_size=test_size,
                random_state=random_state,
            )

        self.normalizer = TrainOnlyNormalizer()

        self.augmenter = TrainOnlyAugmenter(
            seed=augmentation_seed
        )

    def prepare(
        self,
        images: np.ndarray,
        labels: np.ndarray,
        groups: np.ndarray | None = None,
    ):
        if self.split_unit == "group":
            if groups is None:
                raise ValueError(
                    "groups are required when "
                    "split_unit='group'."
                )

            (
                train_images,
                val_images,
                test_images,
                train_labels,
                val_labels,
                test_labels,
                train_groups,
                val_groups,
                test_groups,
            ) = self.splitter.split(
                images,
                labels,
                groups,
            )

            group_overlaps = check_split_overlap(
                train_groups,
                val_groups,
                test_groups,
            )

            if has_leakage(group_overlaps):
                raise RuntimeError(
                    "Group leakage detected after splitting."
                )

        else:
            (
                train_images,
                val_images,
                test_images,
                train_labels,
                val_labels,
                test_labels,
            ) = self.splitter.split(
                images,
                labels,
            )

            train_groups = None
            val_groups = None
            test_groups = None

        # ----------------------------------------
        # Fit preprocessing ONLY on training data
        # ----------------------------------------

        self.normalizer.fit(
            train_images
        )

        train_images = self.normalizer.transform(
            train_images
        )

        val_images = self.normalizer.transform(
            val_images
        )

        test_images = self.normalizer.transform(
            test_images
        )

        # ----------------------------------------
        # Augmentation ONLY on training data
        # ----------------------------------------

        train_images = self.augmenter.transform(
            train_images,
            split="train",
        )

        val_images = self.augmenter.transform(
            val_images,
            split="validation",
        )

        test_images = self.augmenter.transform(
            test_images,
            split="test",
        )

        result = {
            "train": {
                "images": train_images,
                "labels": train_labels,
            },
            "validation": {
                "images": val_images,
                "labels": val_labels,
            },
            "test": {
                "images": test_images,
                "labels": test_labels,
            },
        }

        if self.split_unit == "group":
            result["train"]["groups"] = train_groups
            result["validation"]["groups"] = val_groups
            result["test"]["groups"] = test_groups

        return result