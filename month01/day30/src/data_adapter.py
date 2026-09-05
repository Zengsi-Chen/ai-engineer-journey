import importlib
import sys
from pathlib import Path


def _load_day25_data_module():
    project_root = (
        Path(__file__).resolve().parents[3]
    )

    day25_path = (
        project_root
        / "month01"
        / "day25"
    )

    day25_path_str = str(day25_path)

    if day25_path_str not in sys.path:
        sys.path.insert(
            0,
            day25_path_str,
        )

    return importlib.import_module(
        "cifar10_data"
    )


def create_data_loaders(config):
    """
    Create train, validation, and test
    dataloaders from the Day 25 CIFAR-10
    implementation.

    The complete PipelineConfig is accepted
    so that dataset parameters and experiment
    seed remain available in one interface.
    """

    if (
        config.data.dataset_name
        != "CIFAR10"
    ):
        raise ValueError(
            "Only CIFAR10 is currently supported."
        )

    data_module = (
        _load_day25_data_module()
    )

    train_loader, validation_loader, test_loader = (
        data_module.create_cifar10_loaders(
            batch_size=config.data.batch_size,
            train_samples=(
                config.data.max_train_samples
            ),
            validation_samples=(
                config.data.max_val_samples
            ),
            seed=config.experiment.seed,
            num_workers=config.data.num_workers,
        )
    )

    return (
        train_loader,
        validation_loader,
        test_loader,
    )
