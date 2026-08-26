from experiment_configs import (
    create_binary_mlp_config,
    create_mnist_cnn_config,
)


EXPERIMENTS = {

    "binary_mlp": create_binary_mlp_config,

    "mnist_cnn": create_mnist_cnn_config,

}


def create_experiment(name):

    name = name.lower()

    if name not in EXPERIMENTS:

        available = ", ".join(
            EXPERIMENTS.keys()
        )

        raise ValueError(
            f"Unknown experiment: {name}. "
            f"Available experiments: {available}"
        )

    return EXPERIMENTS[name]()