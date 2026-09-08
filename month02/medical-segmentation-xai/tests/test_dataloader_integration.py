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


def validate_full_split(
    split: str,
    expected_samples: int,
) -> None:

    loader = create_segmentation_dataloader(
        manifest_path=MANIFEST_PATH,
        split=split,
        image_size=256,
        batch_size=4,
        shuffle=False,
        num_workers=0,
    )

    total_samples = 0

    for images, masks in loader:

        # Shape
        assert images.ndim == 4
        assert masks.ndim == 4

        assert images.shape[1:] == (
            3,
            256,
            256,
        )

        assert masks.shape[1:] == (
            1,
            256,
            256,
        )

        # Batch alignment
        assert images.shape[0] == masks.shape[0]

        # Dtype
        assert images.dtype == torch.float32
        assert masks.dtype == torch.float32

        # Image validity
        assert torch.isfinite(images).all()
        assert images.min() >= 0.0
        assert images.max() <= 1.0

        # Mask validity
        assert torch.isfinite(masks).all()

        unique_values = torch.unique(masks)

        assert set(unique_values.tolist()).issubset(
            {0.0, 1.0}
        )

        total_samples += images.shape[0]

    assert total_samples == expected_samples


def test_full_train_split():
    validate_full_split(
        split="train",
        expected_samples=700,
    )


def test_full_val_split():
    validate_full_split(
        split="val",
        expected_samples=150,
    )


def test_full_test_split():
    validate_full_split(
        split="test",
        expected_samples=150,
    )