import base64
import io

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

import app.main as api


class FakeSegmenter:
    """A lightweight deterministic segmenter for API tests."""

    device = "cpu"
    image_size = (256, 256)
    threshold = 0.5

    def predict_proba(self, image):
        probability_map = np.zeros((256, 256), dtype=np.float32)
        probability_map[64:192, 64:192] = 0.9
        probability_map[32:64, 32:64] = 0.5
        return probability_map


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(
        api,
        "load_segmenter",
        lambda checkpoint_path: FakeSegmenter(),
    )

    with TestClient(api.app) as test_client:
        yield test_client


def make_test_image() -> bytes:
    image = Image.new("RGB", (100, 80), color=(120, 150, 180))
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def test_health(client):
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "model_loaded": True,
    }


def test_model_info(client):
    response = client.get("/model-info")

    assert response.status_code == 200
    data = response.json()

    assert data["model"] == "UNet"
    assert data["device"] == "cpu"
    assert data["image_size"] == [256, 256]
    assert data["threshold"] == 0.5


def test_predict_valid_image(client):
    response = client.post(
        "/predict",
        files={
            "file": (
                "test.png",
                make_test_image(),
                "image/png",
            )
        },
    )

    assert response.status_code == 200
    data = response.json()

    assert data["filename"] == "test.png"
    assert data["input_size"] == {"width": 100, "height": 80}
    assert data["output_size"] == {"width": 256, "height": 256}
    assert data["threshold"] == 0.5
    assert data["foreground_ratio"] == pytest.approx(
        (128 * 128 + 32 * 32) / (256 * 256)
    )

    summary = data["probability_summary"]
    assert summary["min"] == pytest.approx(0.0)
    assert summary["max"] == pytest.approx(0.9)
    assert summary["mean"] == pytest.approx(
        (128 * 128 * 0.9 + 32 * 32 * 0.5) / (256 * 256)
    )

    probability_data = data["probability_map"]
    mask_data = data["binary_mask"]

    assert probability_data["format"] == "png"
    assert probability_data["encoding"] == "base64"
    assert mask_data["format"] == "png"
    assert mask_data["encoding"] == "base64"

    probability_bytes = base64.b64decode(
        probability_data["data"],
        validate=True,
    )
    mask_bytes = base64.b64decode(
        mask_data["data"],
        validate=True,
    )

    with Image.open(io.BytesIO(probability_bytes)) as probability_image:
        probability_image.load()
        assert probability_image.format == "PNG"
        assert probability_image.mode == "L"
        assert probability_image.size == (256, 256)
        probability_array = np.asarray(probability_image)
        assert probability_array.min() == 0
        assert probability_array.max() == 229
        assert len(np.unique(probability_array)) > 2

    with Image.open(io.BytesIO(mask_bytes)) as mask_image:
        mask_image.load()
        assert mask_image.format == "PNG"
        assert mask_image.mode == "L"
        assert mask_image.size == (256, 256)

        mask_array = np.asarray(mask_image)
        assert set(np.unique(mask_array).tolist()) == {0, 255}
        assert int(np.count_nonzero(mask_array)) == 128 * 128 + 32 * 32


def test_predict_rejects_non_image_media_type(client):
    response = client.post(
        "/predict",
        files={
            "file": (
                "notes.txt",
                b"this is not an image",
                "text/plain",
            )
        },
    )

    assert response.status_code == 415


def test_predict_rejects_corrupt_image(client):
    response = client.post(
        "/predict",
        files={
            "file": (
                "broken.png",
                b"not a real image",
                "image/png",
            )
        },
    )

    assert response.status_code == 400


def test_predict_rejects_oversized_upload(client):
    oversized_content = b"x" * (10 * 1024 * 1024 + 1)

    response = client.post(
        "/predict",
        files={
            "file": (
                "large.png",
                oversized_content,
                "image/png",
            )
        },
    )

    assert response.status_code == 413
    assert response.json()["detail"] == (
        "Image exceeds the 10 MiB upload limit."
    )