import random

from PIL import Image


class SegmentationAugmentation:
    """
    Paired geometric augmentation for image segmentation.

    The same random transformation is applied to both
    the image and its corresponding mask.
    """

    def __init__(
        self,
        horizontal_flip_prob: float = 0.5,
        vertical_flip_prob: float = 0.5,
        rotation_prob: float = 0.5,
        rotation_degrees: float = 15.0,
    ) -> None:

        if not 0.0 <= horizontal_flip_prob <= 1.0:
            raise ValueError(
                "horizontal_flip_prob must be between 0 and 1"
            )

        if not 0.0 <= vertical_flip_prob <= 1.0:
            raise ValueError(
                "vertical_flip_prob must be between 0 and 1"
            )

        if not 0.0 <= rotation_prob <= 1.0:
            raise ValueError(
                "rotation_prob must be between 0 and 1"
            )

        if rotation_degrees < 0:
            raise ValueError(
                "rotation_degrees must be non-negative"
            )

        self.horizontal_flip_prob = horizontal_flip_prob
        self.vertical_flip_prob = vertical_flip_prob
        self.rotation_prob = rotation_prob
        self.rotation_degrees = rotation_degrees

    def __call__(
        self,
        image: Image.Image,
        mask: Image.Image,
    ) -> tuple[Image.Image, Image.Image]:

        if random.random() < self.horizontal_flip_prob:
            image = image.transpose(
                Image.Transpose.FLIP_LEFT_RIGHT
            )
            mask = mask.transpose(
                Image.Transpose.FLIP_LEFT_RIGHT
            )

        if random.random() < self.vertical_flip_prob:
            image = image.transpose(
                Image.Transpose.FLIP_TOP_BOTTOM
            )
            mask = mask.transpose(
                Image.Transpose.FLIP_TOP_BOTTOM
            )

        if random.random() < self.rotation_prob:
            angle = random.uniform(
                -self.rotation_degrees,
                self.rotation_degrees,
            )

            image = image.rotate(
                angle,
                resample=Image.Resampling.BILINEAR,
                fillcolor=0,
            )

            mask = mask.rotate(
                angle,
                resample=Image.Resampling.NEAREST,
                fillcolor=0,
            )

        return image, mask