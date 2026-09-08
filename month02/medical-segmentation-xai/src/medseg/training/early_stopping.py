class EarlyStopping:
    def __init__(
        self,
        monitor: str = "val_dice",
        mode: str = "max",
        patience: int = 3,
        min_delta: float = 0.0,
    ) -> None:
        self.monitor = monitor
        self.mode = mode
        self.patience = patience
        self.min_delta = min_delta

        if mode not in {"min", "max"}:
            raise ValueError(
                "mode must be either 'min' or 'max'"
            )

        if patience < 0:
            raise ValueError(
                "patience must be non-negative"
            )

        if min_delta < 0:
            raise ValueError(
                "min_delta must be non-negative"
            )

        if mode == "max":
            self.best_value = float("-inf")
        else:
            self.best_value = float("inf")

        self.num_bad_epochs = 0
        self.best_epoch = None

    def _is_improvement(self, value: float) -> bool:
        if self.mode == "max":
            return value > self.best_value + self.min_delta

        return value < self.best_value - self.min_delta

    def step(
        self,
        value: float,
        epoch: int | None = None,
    ) -> bool:
        if self._is_improvement(value):
            self.best_value = value
            self.best_epoch = epoch
            self.num_bad_epochs = 0
            return False

        self.num_bad_epochs += 1

        return self.num_bad_epochs > self.patience