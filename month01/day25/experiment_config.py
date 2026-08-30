from dataclasses import dataclass


@dataclass
class CPUExperimentConfig:

    batch_size: int = 128

    train_samples: int = 2000

    validation_samples: int = 500

    epochs: int = 3

    learning_rate: float = 0.001

    num_workers: int = 0

    num_threads: int = 4

    seed: int = 42