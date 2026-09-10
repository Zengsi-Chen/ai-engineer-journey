import numpy as np
import torch
from PIL import Image

from medseg.inference.segmenter import Segmenter
from medseg.models.unet import UNet


def create_test_image() -> Image.Image:
    """
    Create a deterministic synthetic RGB image.
    """
    array = np.zeros(
        (256, 256, 3),
        dtype=np.uint8,
    )

    array[:, :, 0] = 100
    array[:, :, 1] = 150
    array[:, :, 2] = 200

    return Image.fromarray(array)


def test_segmenter_predict_proba_shape():
    model = UNet(
        features=(16, 32, 64, 128),
    )

    segmenter = Segmenter(
        model=model,
        device="cpu",
    )

    image = create_test_image()

    probabilities = segmenter.predict_proba(image)

    assert probabilities.shape == (256, 256)


def test_segmenter_predict_proba_range():
    model = UNet(
        features=(16, 32, 64, 128),
    )

    segmenter = Segmenter(
        model=model,
        device="cpu",
    )

    image = create_test_image()

    probabilities = segmenter.predict_proba(image)

    assert probabilities.min() >= 0.0
    assert probabilities.max() <= 1.0


def test_segmenter_predict_returns_binary_mask():
    model = UNet(
        features=(16, 32, 64, 128),
    )

    segmenter = Segmenter(
        model=model,
        device="cpu",
        threshold=0.5,
    )

    image = create_test_image()

    mask = segmenter.predict(image)

    assert mask.shape == (256, 256)

    unique_values = np.unique(mask)

    assert set(unique_values).issubset({0, 1})


def test_segmenter_sets_model_to_eval_mode():
    model = UNet(
        features=(16, 32, 64, 128),
    )

    model.train()

    Segmenter(
        model=model,
        device="cpu",
    )

    assert model.training is False


def test_segmenter_threshold_validation():
    model = UNet(
        features=(16, 32, 64, 128),
    )

    try:
        Segmenter(
            model=model,
            threshold=1.5,
        )
        assert False
    except ValueError:
        pass