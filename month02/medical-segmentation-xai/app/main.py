import base64
from contextlib import asynccontextmanager
from io import BytesIO
from pathlib import Path

import numpy as np
import torch
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel

from medseg.inference.segmenter import Segmenter
from medseg.models.unet import UNet


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT_PATH = (
    PROJECT_ROOT / "artifacts" / "checkpoints" / "best_model.pt"
)

IMAGE_SIZE = (256, 256)
THRESHOLD = 0.5
MAX_UPLOAD_BYTES = 10 * 1024 * 1024  # 10 MiB


def encode_grayscale_png(image_array: np.ndarray) -> str:
    """Encode a uint8 grayscale image as a Base64 PNG string."""
    image = Image.fromarray(image_array.astype(np.uint8), mode="L")

    buffer = BytesIO()
    image.save(buffer, format="PNG")

    return base64.b64encode(buffer.getvalue()).decode("ascii")

def load_segmenter(checkpoint_path: Path) -> Segmenter:
    """Load the trained model once and wrap it for inference."""
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"Model checkpoint not found: {checkpoint_path}"
        )

    model = UNet(
        in_channels=3,
        out_channels=1,
        features=(16, 32, 64, 128),
    )

    # This checkpoint is a locally produced, trusted project artifact.
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )

    if not isinstance(checkpoint, dict):
        raise TypeError("Unsupported checkpoint format.")

    if "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    elif "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict)
    model.eval()

    return Segmenter(
        model=model,
        device="cpu",
        threshold=THRESHOLD,
        image_size=IMAGE_SIZE,
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize the model at startup, not on every request."""
    app.state.segmenter = load_segmenter(CHECKPOINT_PATH)
    yield
    app.state.segmenter = None


app = FastAPI(
    title="Medical Image Segmentation API",
    version="0.1.0",
    description=(
        "A research demonstration API for image segmentation. "
        "It is not intended for clinical diagnosis."
    ),
    lifespan=lifespan,
)


class ModelInfo(BaseModel):
    model: str
    device: str
    image_size: list[int]
    threshold: float


@app.get("/health")
def health(request: Request) -> dict:
    """Report whether the application has loaded its model."""
    segmenter = getattr(request.app.state, "segmenter", None)

    if segmenter is None:
        raise HTTPException(
            status_code=503,
            detail="Model is not loaded.",
        )

    return {"status": "ok", "model_loaded": True}


@app.get("/model-info", response_model=ModelInfo)
def model_info(request: Request) -> ModelInfo:
    """Return the inference configuration."""
    segmenter = getattr(request.app.state, "segmenter", None)

    if segmenter is None:
        raise HTTPException(
            status_code=503,
            detail="Model is not loaded.",
        )

    return ModelInfo(
        model="UNet",
        device=str(segmenter.device),
        image_size=list(segmenter.image_size),
        threshold=segmenter.threshold,
    )


@app.post("/predict")
async def predict(
    request: Request,
    file: UploadFile = File(...),
) -> dict:
    """Return a binary mask and probability-map summary."""
    if file.content_type and not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=415,
            detail="Unsupported media type. Upload an image.",
        )

    content = await file.read(MAX_UPLOAD_BYTES + 1)

    if not content:
        raise HTTPException(
            status_code=400,
            detail="Uploaded file is empty.",
        )

    if len(content) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail="Image exceeds the 10 MiB upload limit.",
        )

    try:
        with Image.open(BytesIO(content)) as uploaded_image:
            uploaded_image.load()
            image = uploaded_image.convert("RGB")
    except (UnidentifiedImageError, OSError, ValueError):
        raise HTTPException(
            status_code=400,
            detail="Invalid or unreadable image file.",
        ) from None

    segmenter = getattr(request.app.state, "segmenter", None)

    if segmenter is None:
        raise HTTPException(
            status_code=503,
            detail="Model is not loaded.",
        )

    try:
        probability_map = segmenter.predict_proba(image)
        binary_mask = (
            probability_map >= segmenter.threshold
        ).astype(np.uint8)
    except Exception:
        # Keep internal model errors out of the public response.
        raise HTTPException(
            status_code=500,
            detail="Inference failed.",
        ) from None

    probability_uint8 = np.clip(
        probability_map * 255.0, 0, 255
    ).astype(np.uint8)

    mask_uint8 = binary_mask * 255

    probability_png = encode_grayscale_png(probability_uint8)
    mask_png = encode_grayscale_png(mask_uint8)

    return {
        "filename": file.filename,
        "model": "UNet",
        "input_size": {
            "width": image.width,
            "height": image.height,
        },
        "output_size": {
            "width": int(binary_mask.shape[1]),
            "height": int(binary_mask.shape[0]),
        },
        "threshold": segmenter.threshold,
        "probability_summary": {
            "min": float(probability_map.min()),
            "max": float(probability_map.max()),
            "mean": float(probability_map.mean()),
        },
        "foreground_ratio": float(binary_mask.mean()),
        "probability_map": {
            "format": "png",
            "encoding": "base64",
            "data": probability_png,
        },
        "binary_mask": {
            "format": "png",
            "encoding": "base64",
            "data": mask_png,
        },
    }
