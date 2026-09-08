from pathlib import Path

from torch.utils.data import DataLoader

from medseg.data.dataset import SegmentationDataset


def create_segmentation_dataloader(
    manifest_path: Path,
    split: str,
    image_size: int = 256,
    batch_size: int = 4,
    shuffle: bool = False,
    num_workers: int = 0,
) -> DataLoader:

    dataset = SegmentationDataset(
        manifest_path=manifest_path,
        split=split,
        image_size=image_size,
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=False,
    )