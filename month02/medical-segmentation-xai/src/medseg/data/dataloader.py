from pathlib import Path

from torch.utils.data import DataLoader

from medseg.data.augmentation import SegmentationAugmentation
from medseg.data.dataset import SegmentationDataset


def create_segmentation_dataloader(
    manifest_path: Path,
    split: str,
    image_size: int = 256,
    batch_size: int = 4,
    shuffle: bool = False,
    num_workers: int = 0,
    augmentation: str = "none",
) -> DataLoader:

    transform = None

    if split == "train":

        if augmentation == "none":
            transform = None

        elif augmentation == "hflip":
            transform = SegmentationAugmentation(
                horizontal_flip_prob=0.5,
                vertical_flip_prob=0.0,
                rotation_prob=0.0,
                rotation_degrees=0.0,
            )

        elif augmentation == "full":
            transform = SegmentationAugmentation(
                horizontal_flip_prob=0.5,
                vertical_flip_prob=0.5,
                rotation_prob=0.5,
                rotation_degrees=15.0,
            )

        else:
            raise ValueError(
                f"Unknown augmentation mode: {augmentation}. "
                f"Expected one of: none, hflip, full."
            )

    dataset = SegmentationDataset(
        manifest_path=manifest_path,
        split=split,
        image_size=image_size,
        transform=transform,
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=False,
    )