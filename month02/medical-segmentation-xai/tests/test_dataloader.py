from pathlib import Path

import torch

from medseg.data.dataloader import (
    create_segmentation_dataloader,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]

MANIFEST_PATH = (
    PROJECT_ROOT
    / "artifacts"
    / "splits"
    / "split_manifest.csv"
)


def test_train_dataloader_batch_shape():

    loader = create_segmentation_dataloader(
        manifest_path=MANIFEST_PATH,
        split="train",
        image_size=256,
        batch_size=4,
        shuffle=False,
        num_workers=0,
    )

    images, masks = next(iter(loader))

    assert images.shape == (4, 3, 256, 256)
    assert masks.shape == (4, 1, 256, 256)


def test_train_dataloader_dtype():

    loader = create_segmentation_dataloader(
        manifest_path=MANIFEST_PATH,
        split="train",
        image_size=256,
        batch_size=4,
        shuffle=False,
        num_workers=0,
    )

    images, masks = next(iter(loader))

    assert images.dtype == torch.float32
    assert masks.dtype == torch.float32


def test_train_dataloader_image_range():

    loader = create_segmentation_dataloader(
        manifest_path=MANIFEST_PATH,
        split="train",
        image_size=256,
        batch_size=4,
        shuffle=False,
        num_workers=0,
    )

    images, _ = next(iter(loader))

    assert images.min() >= 0.0
    assert images.max() <= 1.0


def test_train_dataloader_mask_binary():

    loader = create_segmentation_dataloader(
        manifest_path=MANIFEST_PATH,
        split="train",
        image_size=256,
        batch_size=4,
        shuffle=False,
        num_workers=0,
    )

    _, masks = next(iter(loader))

    unique_values = torch.unique(masks)

    assert set(unique_values.tolist()).issubset(
        {0.0, 1.0}
    )


def test_val_dataloader_batch_shape():

    loader = create_segmentation_dataloader(
        manifest_path=MANIFEST_PATH,
        split="val",
        image_size=256,
        batch_size=4,
        shuffle=False,
        num_workers=0,
    )

    images, masks = next(iter(loader))

    assert images.shape[1:] == (3, 256, 256)
    assert masks.shape[1:] == (1, 256, 256)