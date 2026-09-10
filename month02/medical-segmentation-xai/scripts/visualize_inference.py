from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

from medseg.data.preprocessing import resize_mask
from medseg.inference.segmenter import Segmenter
from medseg.models.unet import UNet


PROJECT_ROOT = Path(__file__).resolve().parents[1]

CHECKPOINT_PATH = (
    PROJECT_ROOT
    / "artifacts"
    / "checkpoints"
    / "best_model.pt"
)

IMAGE_DIR = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "kvasir-seg"
    / "images"
)

MASK_DIR = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "kvasir-seg"
    / "masks"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "artifacts"
    / "figures"
)

IMAGE_SIZE = (256, 256)
THRESHOLD = 0.5


def load_model(
    checkpoint_path: Path,
) -> torch.nn.Module:

    model = UNet(
        in_channels=3,
        out_channels=1,
        features=(16, 32, 64, 128),
    )

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )

    if "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    elif "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict)

    model.eval()

    return model


def load_ground_truth(
    mask_path: Path,
) -> np.ndarray:

    mask = Image.open(
        mask_path
    ).convert("L")

    mask = resize_mask(
        mask,
        IMAGE_SIZE,
    )

    array = np.asarray(
        mask,
        dtype=np.uint8,
    )

    return (array > 0).astype(np.uint8)


def find_matching_mask(
    image_path: Path,
) -> Path:

    mask_path = MASK_DIR / image_path.name

    if not mask_path.exists():
        raise FileNotFoundError(
            f"Matching mask not found: {mask_path}"
        )

    return mask_path


def main() -> None:

    model = load_model(
        CHECKPOINT_PATH,
    )

    segmenter = Segmenter(
        model=model,
        device="cpu",
        threshold=THRESHOLD,
        image_size=IMAGE_SIZE,
    )

    images = sorted(
        IMAGE_DIR.glob("*")
    )

    if not images:
        raise FileNotFoundError(
            f"No images found in {IMAGE_DIR}"
        )

    image_path = images[0]

    mask_path = find_matching_mask(
        image_path
    )

    image = Image.open(
        image_path
    ).convert("RGB")

    ground_truth = load_ground_truth(
        mask_path
    )

    probability = segmenter.predict_proba(
        image
    )

    prediction = segmenter.predict(
        image
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    figure, axes = plt.subplots(
        2,
        2,
        figsize=(10, 10),
    )

    axes[0, 0].imshow(
        image.resize(IMAGE_SIZE)
    )
    axes[0, 0].set_title(
        "Original Image"
    )
    axes[0, 0].axis("off")

    axes[0, 1].imshow(
        ground_truth,
        cmap="gray",
    )
    axes[0, 1].set_title(
        "Ground Truth"
    )
    axes[0, 1].axis("off")

    axes[1, 0].imshow(
        probability,
        cmap="gray",
        vmin=0.0,
        vmax=1.0,
    )
    axes[1, 0].set_title(
        "Probability Map"
    )
    axes[1, 0].axis("off")

    axes[1, 1].imshow(
        prediction,
        cmap="gray",
    )
    axes[1, 1].set_title(
        f"Prediction (threshold={THRESHOLD})"
    )
    axes[1, 1].axis("off")

    figure.suptitle(
        f"Kvasir-SEG Inference: {image_path.name}",
        fontsize=14,
    )

    figure.tight_layout()

    output_path = (
        OUTPUT_DIR
        / "day35_inference_visualization.png"
    )

    figure.savefig(
        output_path,
        dpi=150,
        bbox_inches="tight",
    )

    plt.close(figure)

    print("=" * 60)
    print("Day 35 Inference Visualization")
    print("=" * 60)
    print(f"Image:       {image_path.name}")
    print(f"Ground truth:{mask_path.name}")
    print(f"Threshold:   {THRESHOLD}")
    print(f"Output:      {output_path}")
    print()
    print(
        f"GT foreground ratio: "
        f"{ground_truth.mean():.4%}"
    )
    print(
        f"Prediction foreground ratio: "
        f"{prediction.mean():.4%}"
    )
    print(
        f"Mean probability: "
        f"{probability.mean():.4f}"
    )


if __name__ == "__main__":
    main()