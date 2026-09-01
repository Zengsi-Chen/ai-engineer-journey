from dataclasses import dataclass

import torch
import torchvision.transforms as transforms


@dataclass
class DomainShiftConfig:
    brightness: float = 0.0
    contrast: float = 0.0
    grayscale: bool = False
    noise_std: float = 0.0


class GaussianNoise:

    def __init__(self, std: float):
        self.std = std

    def __call__(
        self,
        tensor: torch.Tensor,
    ) -> torch.Tensor:

        if self.std <= 0:
            return tensor

        noise = torch.randn_like(tensor) * self.std

        tensor = tensor + noise

        return tensor.clamp(0.0, 1.0)


def create_domain_shift_transform(
    config: DomainShiftConfig,
    normalize_mean=None,
    normalize_std=None,
):
    transform_list = []

    if (
        config.brightness > 0
        or config.contrast > 0
    ):
        transform_list.append(
            transforms.ColorJitter(
                brightness=config.brightness,
                contrast=config.contrast,
            )
        )

    if config.grayscale:
        transform_list.append(
            transforms.Grayscale(
                num_output_channels=3
            )
        )

    transform_list.append(
        transforms.ToTensor()
    )

    if config.noise_std > 0:
        transform_list.append(
            GaussianNoise(
                std=config.noise_std
            )
        )

    if (
        normalize_mean is not None
        and normalize_std is not None
    ):
        transform_list.append(
            transforms.Normalize(
                mean=normalize_mean,
                std=normalize_std,
            )
        )

    return transforms.Compose(
        transform_list
    )

class DeterministicDomainShift:

    def __init__(
        self,
        brightness=0.0,
        contrast=0.0,
        grayscale=False,
    ):
        self.brightness = brightness
        self.contrast = contrast
        self.grayscale = grayscale

    def __call__(self, tensor):

        if self.brightness != 0.0:
            tensor = tensor + self.brightness

        if self.contrast != 0.0:
            mean = tensor.mean(
                dim=(-2, -1),
                keepdim=True,
            )

            tensor = (
                (tensor - mean)
                * (1.0 + self.contrast)
                + mean
            )

        if self.grayscale:
            gray = (
                0.299 * tensor[:, 0:1]
                + 0.587 * tensor[:, 1:2]
                + 0.114 * tensor[:, 2:3]
            )
                
            

            tensor = gray.repeat(
                1,
                3,
                1,
                1,
            )

        return tensor.clamp(
            0.0,
            1.0,
        )

def create_deterministic_domain_shift_transform(
    config: DomainShiftConfig,
    normalize_mean=None,
    normalize_std=None,
    ):

    transform_list = [
        transforms.ToTensor(),
        DeterministicDomainShift(
            brightness=config.brightness,
            contrast=config.contrast,
            grayscale=config.grayscale,
        ),
    ]

    if (
        normalize_mean is not None
        and normalize_std is not None
    ):
        transform_list.append(
            transforms.Normalize(
                mean=normalize_mean,
                std=normalize_std,
            )
        )

    return transforms.Compose(
        transform_list
    )
    
