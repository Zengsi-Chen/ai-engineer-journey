import torch
import torch.nn as nn


class MLP(nn.Module):

    def __init__(
        self,
        input_dim=1,
        hidden_dim=16,
        output_dim=1
    ):

        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):

        return self.network(x)


def create_model(config):

    model_name = config.name.lower()

    if model_name == "mlp":

        return MLP(
            **config.params
        )

    raise ValueError(
        f"Unknown model: {config.name}"
    )

if __name__ == "__main__":

    from config import Config

    config = Config()

    config.model.name = "mlp"

    config.model.params = {
        "input_dim": 1,
        "hidden_dim": 16,
        "output_dim": 1
    }

    model = create_model(
        config.model
    )

    print(model)

    x = torch.randn(4, 1)

    output = model(x)

    print("Input shape:", x.shape)
    print("Output shape:", output.shape)