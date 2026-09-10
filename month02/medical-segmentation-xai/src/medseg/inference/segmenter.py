from __future__ import annotations

from typing import Union

import numpy as np
import torch
from PIL import Image

from medseg.data.preprocessing import (
    image_to_tensor,
    resize_image,
)


class Segmenter:
    """
    Production-style inference wrapper for semantic segmentation.

    Pipeline:

        PIL Image
            ↓
        resize
            ↓
        image_to_tensor
            ↓
        model.eval()
            ↓
        torch.no_grad()
            ↓
        logits
            ↓
        sigmoid
            ↓
        probability map
            ↓
        threshold
            ↓
        binary mask
    """

    def __init__(
        self,
        model: torch.nn.Module,
        device: str = "cpu",
        threshold: float = 0.5,
        image_size: tuple[int, int] = (256, 256),
    ) -> None:

        if not 0.0 <= threshold <= 1.0:
            raise ValueError(
                "threshold must be between 0.0 and 1.0"
            )

        self.model = model
        self.device = torch.device(device)
        self.threshold = threshold
        self.image_size = image_size

        self.model.to(self.device)
        self.model.eval()

    def _preprocess(
        self,
        image: Image.Image,
    ) -> torch.Tensor:
        """
        Apply the same preprocessing used during training.
        """

        image = image.convert("RGB")

        image = resize_image(
            image,
            self.image_size,
        )

        tensor = image_to_tensor(image)

        tensor = tensor.unsqueeze(0)

        tensor = tensor.to(self.device)

        return tensor

    def predict_proba(
        self,
        image: Image.Image,
    ) -> np.ndarray:
        """
        Predict a pixel-level probability map.

        Returns
        -------
        np.ndarray
            Float32 array with shape (H, W),
            containing values in [0, 1].
        """

        tensor = self._preprocess(image)

        self.model.eval()

        with torch.no_grad():

            logits = self.model(tensor)

            probabilities = torch.sigmoid(logits)

        probabilities = probabilities.squeeze(0).squeeze(0)

        return probabilities.cpu().numpy()

    def predict(
        self,
        image: Image.Image,
    ) -> np.ndarray:
        """
        Predict a binary segmentation mask.

        Returns
        -------
        np.ndarray
            UInt8 array with shape (H, W),
            containing only 0 and 1.
        """

        probabilities = self.predict_proba(image)

        mask = (
            probabilities >= self.threshold
        ).astype(np.uint8)

        return mask