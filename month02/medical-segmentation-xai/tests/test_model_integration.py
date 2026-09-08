from pathlib import Path

import torch

from medseg.data.dataloader import (
    create_segmentation_dataloader,
)
from medseg.models.factory import create_model


PROJECT_ROOT = Path(__file__).resolve().parents[1]

MANIFEST_PATH = (
    PROJECT_ROOT
    / "artifacts"
    / "splits"
    / "split_manifest.csv"
)


def test_dataloader_to_unet():

    loader = create_segmentation_dataloader(
        manifest_path=MANIFEST_PATH,
        split="train",
        image_size=256,
        batch_size=2,
        shuffle=False,
        num_workers=0,
    )

    images, masks = next(iter(loader))

    model = create_model(
        name="unet",
        in_channels=3,
        out_channels=1,
    )

    outputs = model(images)

    assert outputs.shape == masks.shape

    assert outputs.shape == (
        2,
        1,
        256,
        256,
    )

    assert torch.isfinite(outputs).all()