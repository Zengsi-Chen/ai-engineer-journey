from PIL import Image
import numpy as np
import torch
import torch.nn.functional as F


def resize_image(
    image: Image.Image,
    size: tuple[int, int],
) -> Image.Image:
    """
    Resize an image using bilinear interpolation.
    """
    return image.resize(
        size,
        resample=Image.Resampling.BILINEAR,
    )


def resize_mask(
    mask: Image.Image,
    size: tuple[int, int],
) -> Image.Image:
    """
    Resize a segmentation mask using nearest-neighbor interpolation.
    """
    return mask.resize(
        size,
        resample=Image.Resampling.NEAREST,
    )


def image_to_tensor(image: Image.Image) -> torch.Tensor:
    """
    Convert RGB image to float tensor in CHW format.
    """
    array = np.asarray(image, dtype=np.float32)

    tensor = torch.from_numpy(array)

    tensor = tensor.permute(2, 0, 1)

    tensor = tensor / 255.0

    return tensor


def mask_to_tensor(mask: Image.Image) -> torch.Tensor:
    """
    Convert mask to binary float tensor in 1xHxW format.
    """
    mask = mask.convert("L")

    array = np.asarray(mask, dtype=np.uint8)

    binary = (array > 0).astype(np.float32)

    tensor = torch.from_numpy(binary)

    tensor = tensor.unsqueeze(0)

    return tensor