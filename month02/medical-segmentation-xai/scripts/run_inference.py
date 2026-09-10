from pathlib import Path

import torch
from PIL import Image

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

OUTPUT_DIR = (
    PROJECT_ROOT
    / "artifacts"
    / "predictions"
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

    if isinstance(checkpoint, dict):

        if "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]

        elif "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]

        else:
            state_dict = checkpoint

    else:
        raise TypeError(
            "Unsupported checkpoint format."
        )

    model.load_state_dict(state_dict)

    model.eval()

    return model


def main() -> None:

    print("=" * 60)
    print("Medical Segmentation Inference")
    print("=" * 60)

    print(f"Checkpoint: {CHECKPOINT_PATH}")

    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {CHECKPOINT_PATH}"
        )

    model = load_model(
        CHECKPOINT_PATH,
    )

    print("Model loaded successfully.")

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

    print(f"Input image: {image_path.name}")

    image = Image.open(
        image_path
    ).convert("RGB")

    probability_map = segmenter.predict_proba(
        image
    )

    binary_mask = segmenter.predict(
        image
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    probability_path = (
        OUTPUT_DIR
        / f"{image_path.stem}_probability.png"
    )

    mask_path = (
        OUTPUT_DIR
        / f"{image_path.stem}_mask.png"
    )

    probability_image = (
        probability_map * 255
    ).clip(0, 255).astype("uint8")

    mask_image = (
        binary_mask * 255
    ).astype("uint8")

    Image.fromarray(
        probability_image
    ).save(probability_path)

    Image.fromarray(
        mask_image
    ).save(mask_path)

    print()
    print("Inference completed.")
    print(f"Probability map: {probability_path}")
    print(f"Binary mask:     {mask_path}")
    print()
    print(
        f"Probability range: "
        f"{probability_map.min():.4f} - "
        f"{probability_map.max():.4f}"
    )

    print(
        f"Foreground ratio: "
        f"{binary_mask.mean():.4%}"
    )


if __name__ == "__main__":
    main()