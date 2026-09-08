from pathlib import Path

import torch

from medseg.data.dataloader import create_segmentation_dataloader
from medseg.models.factory import create_model
from medseg.training.losses import BCEDiceLoss
from medseg.training.optimizer import create_optimizer
from medseg.training.train_step import train_one_batch


PROJECT_ROOT = Path(__file__).resolve().parents[1]

MANIFEST_PATH = (
    PROJECT_ROOT
    / "artifacts"
    / "splits"
    / "split_manifest.csv"
)


def test_train_one_batch():
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
        features=(16, 32, 64, 128),
    )

    criterion = BCEDiceLoss()

    optimizer = create_optimizer(
        model=model,
        name="adamw",
        learning_rate=1e-3,
        weight_decay=1e-4,
    )

    first_parameter = next(model.parameters())

    before = first_parameter.detach().clone()

    result = train_one_batch(
        model=model,
        images=images,
        masks=masks,
        criterion=criterion,
        optimizer=optimizer,
    )

    after = first_parameter.detach()

    assert "loss" in result

    assert result["loss"] >= 0.0

    assert torch.isfinite(
        torch.tensor(result["loss"])
    )

    assert not torch.equal(before, after)