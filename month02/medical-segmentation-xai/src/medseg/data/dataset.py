from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

from medseg.data.augmentation import SegmentationAugmentation
from medseg.data.preprocessing import (
    resize_image,
    resize_mask,
    image_to_tensor,
    mask_to_tensor,
)


class SegmentationDataset(Dataset):
    """
    Dataset for binary medical image segmentation.

    Augmentation is applied only to the training split.
    Validation and test splits remain deterministic.
    """

    def __init__(
        self,
        manifest_path: Path,
        split: str,
        image_size: int = 256,
        transform: SegmentationAugmentation | None = None,
    ) -> None:

        self.manifest_path = Path(manifest_path)
        self.split = split
        self.image_size = image_size
        self.transform = transform

        manifest = pd.read_csv(self.manifest_path)

        self.samples = manifest[
            manifest["split"] == split
        ].reset_index(drop=True)

        if self.samples.empty:
            raise ValueError(
                f"No samples found for split='{split}'"
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(
        self,
        index: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:

        row = self.samples.iloc[index]

        image_path = Path(row["image_path"])
        mask_path = Path(row["mask_path"])

        image = Image.open(image_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        size = (
            self.image_size,
            self.image_size,
        )

        image = resize_image(image, size)
        mask = resize_mask(mask, size)

        # Apply paired augmentation before converting to tensors.
        if self.transform is not None:
            image, mask = self.transform(
                image,
                mask,
            )

        image_tensor = image_to_tensor(image)
        mask_tensor = mask_to_tensor(mask)

        return image_tensor, mask_tensor