from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

from medseg.data.preprocessing import (
    image_to_tensor,
    resize_image,
)
from medseg.models.factory import create_model
from medseg.xai.saliency import compute_input_saliency


IMAGE_PATH = Path(
    "data/raw/kvasir-seg/images/"
    "cju35k2fr3vc50988c85qkrwg.jpg"
)

CHECKPOINT_PATH = Path(
    "artifacts/day37/baseline/best_model.pt"
)

OUTPUT_PATH = Path(
    "artifacts/day38_saliency_complete_miss.png"
)

MASK_PATH = Path(
    "data/raw/kvasir-seg/masks/"
    "cju35k2fr3vc50988c85qkrwg.jpg"
)

IMAGE_SIZE = 256
THRESHOLD = 0.55


def load_model():
    model = create_model(
        name="unet",
        in_channels=3,
        out_channels=1,
        features=(16, 32, 64, 128),
    )

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location="cpu",
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.eval()

    return model


def main():
    print("=" * 80)
    print("Day 38 — Saliency Map")
    print("=" * 80)

    print(f"Image:      {IMAGE_PATH}")
    print(f"Checkpoint: {CHECKPOINT_PATH}")

    image = Image.open(IMAGE_PATH).convert("RGB")

    resized = resize_image(
        image,
        (IMAGE_SIZE, IMAGE_SIZE),
    )

    image_tensor = image_to_tensor(
        resized
    ).unsqueeze(0)

    model = load_model()

    saliency = compute_input_saliency(
        model=model,
        image=image_tensor,
        threshold=THRESHOLD,
        target_mode="soft_foreground",
    )

    mask = Image.open(MASK_PATH).convert("L")

    mask = mask.resize(
        (IMAGE_SIZE, IMAGE_SIZE),
        resample=Image.Resampling.NEAREST,
    )

    mask_tensor = torch.from_numpy(
        np.array(mask, copy=True)
    )

    gt_mask = mask_tensor > 0

    saliency_inside_gt = saliency[gt_mask]

    saliency_outside_gt = saliency[~gt_mask]

    inside_mean = saliency_inside_gt.mean().item()
    outside_mean = saliency_outside_gt.mean().item()

    if outside_mean > 0:
        concentration_ratio = (
            inside_mean / outside_mean
        )
    else:
        concentration_ratio = float("inf")

    print(
        f"GT pixels:             {gt_mask.sum().item()}"
    )

    print(
        f"Saliency mean inside GT:  "
        f"{inside_mean:.6f}"
    )

    print(
        f"Saliency mean outside GT: "
        f"{outside_mean:.6f}"
    )

    print(
        f"Saliency concentration:    "
        f"{concentration_ratio:.3f}x"
    )

    print(
        f"Saliency shape: {tuple(saliency.shape)}"
    )

    print(
        f"Saliency min: {saliency.min().item():.6f}"
    )

    print(
        f"Saliency max: {saliency.max().item():.6f}"
    )

    print(
        f"Saliency mean: {saliency.mean().item():.6f}"
    )

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    plt.figure(figsize=(12, 4))

    plt.subplot(1, 3, 1)
    plt.imshow(resized)
    plt.title("Input")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(saliency.numpy(), cmap="hot")
    plt.title("Soft Foreground Saliency")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.imshow(resized)
    plt.imshow(
        saliency.numpy(),
        cmap="hot",
        alpha=0.5,
    )
    plt.title("Saliency Overlay")
    plt.axis("off")

    plt.tight_layout()

    plt.savefig(
        OUTPUT_PATH,
        dpi=150,
        bbox_inches="tight",
    )

    plt.close()

    print(f"Saved: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()