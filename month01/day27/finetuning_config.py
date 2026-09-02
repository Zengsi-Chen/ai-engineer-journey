from dataclasses import dataclass


@dataclass
class FineTuningConfig:

    base_learning_rate: float = 1e-3

    layer_wise_decay: float = 1.0

    weight_decay: float = 1e-4

    discriminative_learning_rates: dict = None