from dataclasses import dataclass


@dataclass
class TransferLearningConfig:
    seed: int = 42

    batch_size: int = 1
    epochs: int = 1

    classifier_lr: float = 1e-3
    backbone_lr: float = 1e-4

    num_classes: int = 10