import pytest

from experiment_registry import (
    create_experiment
)


def test_create_binary_mlp():

    config = create_experiment(
        "binary_mlp"
    )

    assert config.data.source == "binary"

    assert config.model.name == "mlp"

    assert (
        config.task.task_type
        == "binary_classification"
    )


def test_create_mnist_cnn():

    config = create_experiment(
        "mnist_cnn"
    )

    assert config.data.source == "mnist"

    assert config.model.name == "cnn"

    assert (
        config.task.task_type
        == "multiclass_classification"
    )


def test_unknown_experiment():

    with pytest.raises(ValueError):

        create_experiment(
            "unknown_experiment"
        )
        