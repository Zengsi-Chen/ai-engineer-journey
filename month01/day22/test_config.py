import pytest

from config import validate_config

from experiment_configs import (
    create_binary_mlp_config,
    create_mnist_cnn_config,
    create_invalid_mnist_binary_config,
    create_invalid_binary_cnn_config,
)


@pytest.mark.parametrize(
    "config_factory, expected_valid",
    [
        (
            create_binary_mlp_config,
            True
        ),

        (
            create_mnist_cnn_config,
            True
        ),

        (
            create_invalid_mnist_binary_config,
            False
        ),

        (
            create_invalid_binary_cnn_config,
            False
        ),
    ]
)
def test_config_matrix(
    config_factory,
    expected_valid
):

    config = config_factory()

    if expected_valid:

        validate_config(config)

    else:

        with pytest.raises(ValueError):

            validate_config(config)