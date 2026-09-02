import pytest
import torch
import torch.nn as nn

from finetuning_config import (
FineTuningConfig,
)

from optimizer_factory import (
    create_layer_wise_learning_rates,
    get_trainable_layer_names,
    create_discriminative_parameter_groups,
    create_optimizer,
)

class SimpleModel(nn.Module):

    def __init__(self):
        super().__init__()

        self.layer1 = nn.Linear(
            10,
            10,
        )

        self.layer2 = nn.Linear(
            10,
            10,
        )

        self.classifier = nn.Linear(
            10,
            2,
        )


def test_layer_wise_learning_rate_decay():


    learning_rates = (
        create_layer_wise_learning_rates(
            layer_names=[
                "layer1",
                "layer2",
                "classifier",
            ],
            base_learning_rate=1e-3,
            layer_wise_decay=0.3,
        )
    )

    assert (
        learning_rates["layer1"]
        == pytest.approx(9e-5)
    )

    assert (
        learning_rates["layer2"]
        == pytest.approx(3e-4)
    )

    assert (
        learning_rates["classifier"]
        == pytest.approx(1e-3)
    )

def test_layer_wise_decay_one_produces_uniform_lr():


    learning_rates = (
        create_layer_wise_learning_rates(
            layer_names=[
                "layer1",
                "layer2",
                "classifier",
            ],
            base_learning_rate=1e-3,
            layer_wise_decay=1.0,
        )
    )

    for learning_rate in (
        learning_rates.values()
    ):

        assert (
            learning_rate
            == pytest.approx(1e-3)
        )
    

def test_get_trainable_layer_names():


    model = SimpleModel()

    for parameter in (
        model.parameters()
    ):
        parameter.requires_grad = False

    for parameter in (
        model.layer2.parameters()
    ):
        parameter.requires_grad = True

    for parameter in (
        model.classifier.parameters()
    ):
        parameter.requires_grad = True

    layer_names = (
        get_trainable_layer_names(
            model
        )
    )

    assert layer_names == [
        "layer2",
        "classifier",
    ]
    

def test_create_discriminative_parameter_groups():

    model = SimpleModel()

    learning_rates = {
        "layer1": 1e-4,
        "layer2": 1e-3,
    }

    parameter_groups = (
        create_discriminative_parameter_groups(
            model=model,
            learning_rates=learning_rates,
        )
    )

    assert len(
        parameter_groups
    ) == 2

    assert (
        parameter_groups[0]["lr"]
        == pytest.approx(1e-4)
    )

    assert (
        parameter_groups[1]["lr"]
        == pytest.approx(1e-3)
    )


def test_optimizer_uses_uniform_lr():


    model = SimpleModel()

    config = FineTuningConfig(
        base_learning_rate=1e-3,
        layer_wise_decay=1.0,
    )

    optimizer = create_optimizer(
        model=model,
        config=config,
    )

    learning_rates = [
        group["lr"]
        for group in (
            optimizer.param_groups
        )
    ]

    assert learning_rates == [
        pytest.approx(1e-3)
    ]


def test_optimizer_uses_manual_discriminative_lr():

    model = SimpleModel()

    config = FineTuningConfig(
        discriminative_learning_rates={
            "layer1": 1e-4,
            "layer2": 3e-4,
            "classifier": 1e-3,
        },
    )

    optimizer = create_optimizer(
        model=model,
        config=config,
    )

    learning_rates = [
        group["lr"]
        for group in (
            optimizer.param_groups
        )
    ]

    assert learning_rates == pytest.approx(
        [
            1e-4,
            3e-4,
            1e-3,
        ]
    )


def test_optimizer_uses_automatic_layer_wise_decay():

    model = SimpleModel()

    config = FineTuningConfig(
        base_learning_rate=1e-3,
        layer_wise_decay=0.1,
    )

    optimizer = create_optimizer(
        model=model,
        config=config,
    )

    learning_rates = [
        group["lr"]
        for group in (
            optimizer.param_groups
        )
    ]

    assert learning_rates == pytest.approx(
        [
            1e-5,
            1e-4,
            1e-3,
        ]
    )


def test_manual_lr_has_highest_priority():

    model = SimpleModel()

    config = FineTuningConfig(
        base_learning_rate=1e-3,
        layer_wise_decay=0.1,
        discriminative_learning_rates={
            "layer1": 1e-4,
            "layer2": 3e-4,
            "classifier": 1e-3,
        },
    )

    optimizer = create_optimizer(
        model=model,
        config=config,
    )

    learning_rates = [
        group["lr"]
        for group in (
            optimizer.param_groups
        )
    ]

    assert learning_rates == pytest.approx(
        [
            1e-4,
            3e-4,
            1e-3,
        ]
    )
    
