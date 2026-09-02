import torch

def get_trainable_parameters(
    layer,
    ):
    """
    Return only trainable parameters
    from a model layer.
    """

    return [
        parameter
        for parameter in layer.parameters()
        if parameter.requires_grad
    ]


def get_trainable_layer_names(
    model,
    ):
    """
    Return the model layers that contain
    trainable parameters.
    
    The order is important because it defines
    the learning-rate decay order.
    """

    layer_names = [
        "stem",
        "layer1",
        "layer2",
        "layer3",
        "layer4",
        "classifier",
    ]

    trainable_layer_names = []

    for layer_name in layer_names:

        if not hasattr(
            model,
            layer_name,
        ):
            continue

        layer = getattr(
            model,
            layer_name,
        )

        parameters = (
            get_trainable_parameters(
                layer
            )
        )

        if parameters:

            trainable_layer_names.append(
                layer_name
            )

    return trainable_layer_names


def create_layer_wise_learning_rates(
    layer_names,
    base_learning_rate,
    layer_wise_decay,
    ):
    """
    Automatically create learning rates
    from shallow layers to deep layers.

    ```
    The deepest layer receives the base
    learning rate.

    Earlier layers receive progressively
    smaller learning rates.
    """

    learning_rates = {}

    total_layers = len(
        layer_names
    )

    for index, layer_name in enumerate(
        layer_names
    ):

        distance_from_last = (
            total_layers
            - index
            - 1
        )

        learning_rate = (
            base_learning_rate
            * (
                layer_wise_decay
                ** distance_from_last
            )
        )

        learning_rates[
            layer_name
        ] = learning_rate

    return learning_rates


def create_discriminative_parameter_groups(
    model,
    learning_rates,
    ):
    """
    Create parameter groups with
    different learning rates.
    """

    parameter_groups = []

    for layer_name, learning_rate in (
        learning_rates.items()
    ):

        layer = getattr(
            model,
            layer_name,
        )

        parameters = (
            get_trainable_parameters(
                layer
            )
        )

        if not parameters:
            continue

        parameter_groups.append(
            {
                "params": parameters,
                "lr": learning_rate,
            }
        )

    return parameter_groups


def create_optimizer(
    model,
    config,
    ):
    """
    Create an AdamW optimizer.

    Priority:

    1. Manual discriminative learning rates
    2. Automatic layer-wise learning-rate decay
    3. Uniform learning rate
    """

    # --------------------------------------------------
    # Manual Discriminative Learning Rates
    # --------------------------------------------------

    if (
        config.discriminative_learning_rates
        is not None
    ):

        parameter_groups = (
            create_discriminative_parameter_groups(
                model=model,
                learning_rates=(
                    config
                    .discriminative_learning_rates
                ),
            )
        )

        return torch.optim.AdamW(
            parameter_groups,
            weight_decay=config.weight_decay,
        )

    # --------------------------------------------------
    # Automatic Layer-wise Learning-rate Decay
    # --------------------------------------------------

    if config.layer_wise_decay < 1.0:

        trainable_layer_names = (
            get_trainable_layer_names(
                model
            )
        )

        learning_rates = (
            create_layer_wise_learning_rates(
                layer_names=(
                    trainable_layer_names
                ),
                base_learning_rate=(
                    config
                    .base_learning_rate
                ),
                layer_wise_decay=(
                    config
                    .layer_wise_decay
                ),
            )
        )

        parameter_groups = (
            create_discriminative_parameter_groups(
                model=model,
                learning_rates=learning_rates,
            )
        )

        return torch.optim.AdamW(
            parameter_groups,
            weight_decay=config.weight_decay,
        )

    # --------------------------------------------------
    # Uniform Learning Rate
    # --------------------------------------------------

    trainable_parameters = [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad
    ]

    return torch.optim.AdamW(
        trainable_parameters,
        lr=config.base_learning_rate,
        weight_decay=config.weight_decay,
    )

