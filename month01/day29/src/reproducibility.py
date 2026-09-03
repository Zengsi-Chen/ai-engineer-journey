import platform
import random

import numpy as np
import torch


def set_seed(seed):
    if (
        not isinstance(seed, int)
        or isinstance(seed, bool)
    ):
        raise TypeError(
            "Seed must be an integer."
        )

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def get_environment_metadata():
    metadata = {
        "python_version": platform.python_version(),
        "pytorch_version": torch.__version__,
        "platform": platform.platform(),
        "device": (
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        ),
        "cuda_version": torch.version.cuda,
        "cudnn_version": (
            torch.backends.cudnn.version()
            if torch.cuda.is_available()
            else None
        ),
    }

    return metadata


def get_reproducibility_metadata(seed):
    return {
        "seed": seed,
        "environment": (
            get_environment_metadata()
        ),
    }