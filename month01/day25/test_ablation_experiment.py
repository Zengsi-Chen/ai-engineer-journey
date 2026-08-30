import torch

from torch.utils.data import (
    DataLoader,
    TensorDataset,
)

from experiment_config import (
    CPUExperimentConfig,
)

from resnet18 import ResNet18

from ablation_experiment import (
    run_experiment,
)


def test_run_experiment():

    images = torch.randn(
        8,
        3,
        32,
        32,
    )

    labels = torch.randint(
        0,
        10,
        (8,),
    )

    dataset = TensorDataset(
        images,
        labels,
    )

    train_loader = DataLoader(
        dataset,
        batch_size=4,
    )

    validation_loader = DataLoader(
        dataset,
        batch_size=4,
    )

    config = CPUExperimentConfig(
        epochs=1,
        learning_rate=0.001,
    )

    model = ResNet18()

    result = run_experiment(
        model=model,
        train_loader=train_loader,
        validation_loader=validation_loader,
        config=config,
        name="Test ResNet",
    )

    assert result.name == (
        "Test ResNet"
    )

    assert len(
        result.epoch_results
    ) == 1

    assert (
        result.epoch_results[0].epoch
        == 1
    )