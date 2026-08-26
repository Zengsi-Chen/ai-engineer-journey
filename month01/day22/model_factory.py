from model import MLP, SimpleCNN


def create_model(config):

    model_name = config.name.lower()

    if model_name == "mlp":

        return MLP(
            input_dim=config.input_dim,
            hidden_dim=config.hidden_dim,
            output_dim=config.output_dim
        )

    if model_name == "cnn":

        return SimpleCNN(
            input_channels=config.input_channels,
            num_classes=config.num_classes
        )

    raise ValueError(
        f"Unknown model: {config.name}"
    )