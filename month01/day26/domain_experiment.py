import torch

from dataclasses import dataclass

from torchvision import datasets
from torch.utils.data import DataLoader

from domain_shift import (
    create_domain_shift_transform,
    DomainShiftConfig,
)


CIFAR10_MEAN = (
    0.4914,
    0.4822,
    0.4465,
)

CIFAR10_STD = (
    0.2470,
    0.2435,
    0.2616,
)


def create_shifted_dataset(
    root: str,
    shift_config,
    download: bool = True,
):
    transform = create_domain_shift_transform(
        config=shift_config,
        normalize_mean=CIFAR10_MEAN,
        normalize_std=CIFAR10_STD,
    )

    return datasets.CIFAR10(
        root=root,
        train=False,
        download=download,
        transform=transform,
    )

def create_experiment_dataloader(
    experiment,
    root: str,
    batch_size: int,
):
    dataset = create_shifted_dataset(
        root=root,
        shift_config=experiment.shift_config,
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
    )

@torch.no_grad()
def evaluate_model(
    model,
    dataloader,
    device,
):
    model.eval()

    correct = 0
    total = 0

    for images, labels in dataloader:

        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)

        predictions = outputs.argmax(
            dim=1
        )

        correct += (
            predictions == labels
        ).sum().item()

        total += labels.size(0)

    accuracy = correct / total

    return {
        "accuracy": accuracy,
        "correct": correct,
        "total": total,
    }

@dataclass
class DomainExperiment:
    name: str
    shift_config: DomainShiftConfig


def create_domain_shift_experiments():

    return [
        DomainExperiment(
            name="no_shift",
            shift_config=DomainShiftConfig(),
        ),

        DomainExperiment(
            name="moderate_shift",
            shift_config=DomainShiftConfig(
                brightness=0.3,
                contrast=0.3,
                noise_std=0.03,
            ),
        ),

        DomainExperiment(
            name="severe_shift",
            shift_config=DomainShiftConfig(
                brightness=0.6,
                contrast=0.6,
                grayscale=True,
                noise_std=0.08,
            ),
        ),
    ]


@dataclass
class DomainExperimentResult:
    name: str
    accuracy: float
    correct: int
    total: int

def run_domain_shift_experiments(
    model,
    root,
    batch_size,
    device,
):
    experiments = (
        create_domain_shift_experiments()
    )

    results = []

    for experiment in experiments:

        dataloader = (
            create_experiment_dataloader(
                experiment=experiment,
                root=root,
                batch_size=batch_size,
            )
        )

        metrics = evaluate_model(
            model=model,
            dataloader=dataloader,
            device=device,
        )

        results.append(
            DomainExperimentResult(
                name=experiment.name,
                accuracy=metrics["accuracy"],
                correct=metrics["correct"],
                total=metrics["total"],
            )
        )

    return results