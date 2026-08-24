class EarlyStopping:

    def __init__(
        self,
        patience=20,
        min_delta=0.001,
        mode="min"
    ):

        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode

        self.counter = 0
        self.best_value = None


    def step(
        self,
        current_value
    ):

        if self.best_value is None:

            self.best_value = current_value
            return False

        if self.mode == "min":

            improved = (
                current_value
                < self.best_value - self.min_delta
            )

        elif self.mode == "max":

            improved = (
                current_value
                > self.best_value + self.min_delta
            )

        else:

            raise ValueError(
                f"Unknown mode: {self.mode}"
            )

        if improved:

            self.best_value = current_value
            self.counter = 0

        else:

            self.counter += 1

        return (
            self.counter >= self.patience
        )