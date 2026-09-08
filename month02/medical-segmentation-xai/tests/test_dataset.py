from pathlib import Path

import torch

from medseg.data.dataset import SegmentationDataset


PROJECT_ROOT = Path(__file__).resolve().parents[1]

MANIFEST_PATH = (
    PROJECT_ROOT
    / "artifacts"
    / "splits"
    / "split_manifest.csv"
)


def test_train_dataset_count():
    dataset = SegmentationDataset(
        MANIFEST_PATH,
        split="train",
        image_size=256,
    )

    assert len(dataset) == 700


def test_val_dataset_count():
    dataset = SegmentationDataset(
        MANIFEST_PATH,
        split="val",
        image_size=256,
    )

    assert len(dataset) == 150


def test_test_dataset_count():
    dataset = SegmentationDataset(
        MANIFEST_PATH,
        split="test",
        image_size=256,
    )

    assert len(dataset) == 150


def test_dataset_sample_shape():
    dataset = SegmentationDataset(
        MANIFEST_PATH,
        split="train",
        image_size=256,
    )

    image, mask = dataset[0]

    assert image.shape == (3, 256, 256)
    assert mask.shape == (1, 256, 256)


def test_dataset_dtype():
    dataset = SegmentationDataset(
        MANIFEST_PATH,
        split="train",
        image_size=256,
    )

    image, mask = dataset[0]

    assert image.dtype == torch.float32
    assert mask.dtype == torch.float32


def test_image_range():
    dataset = SegmentationDataset(
        MANIFEST_PATH,
        split="train",
        image_size=256,
    )

    image, _ = dataset[0]

    assert torch.min(image) >= 0.0
    assert torch.max(image) <= 1.0


def test_mask_is_binary():
    dataset = SegmentationDataset(
        MANIFEST_PATH,
        split="train",
        image_size=256,
    )

    _, mask = dataset[0]

    unique_values = torch.unique(mask)

    assert set(unique_values.tolist()).issubset(
        {0.0, 1.0}
    )