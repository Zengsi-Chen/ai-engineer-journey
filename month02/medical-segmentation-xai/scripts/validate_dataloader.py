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


def validate_batch(
    images: torch.Tensor,
    masks: torch.Tensor,
) -> None:

    assert images.ndim == 4
    assert masks.ndim == 4

    assert images.shape[0] == masks.shape[0]

    assert images.shape[1] == 3
    assert masks.shape[1] == 1

    assert images.shape[2:] == (256, 256)
    assert masks.shape[2:] == (256, 256)

    assert images.dtype == torch.float32
    assert masks.dtype == torch.float32

    assert torch.min(images) >= 0.0
    assert torch.max(images) <= 1.0

    unique_mask_values = torch.unique(masks)

    assert set(unique_mask_values.tolist()).issubset(
        {0.0, 1.0}
    )


def main() -> None:

    loader = create_segmentation_dataloader(
        manifest_path=MANIFEST_PATH,
        split="train",
        image_size=256,
        batch_size=4,
        shuffle=False,
        num_workers=0,
    )

    images, masks = next(iter(loader))

    validate_batch(images, masks)

    print("DataLoader Batch Validation: PASS")
    print(f"Images shape: {tuple(images.shape)}")
    print(f"Masks shape:  {tuple(masks.shape)}")
    print(f"Images dtype: {images.dtype}")
    print(f"Masks dtype:  {masks.dtype}")
    print(
        f"Image range: "
        f"{images.min().item():.4f} - "
        f"{images.max().item():.4f}"
    )
    print(
        f"Mask values: "
        f"{torch.unique(masks).tolist()}"
    )


if __name__ == "__main__":
    main()