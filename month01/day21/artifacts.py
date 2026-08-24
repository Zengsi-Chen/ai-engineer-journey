import os


class ArtifactsManager:

    def __init__(
        self,
        root="artifacts",
        experiment_id=None
    ):

        self.root = root

        if experiment_id is None:

            raise ValueError(
                "experiment_id is required"
            )

        self.root = root
        self.experiment_id = experiment_id

        self.experiment_dir = os.path.join(
            self.root,
            "experiments",
            self.experiment_id
        )

        self.checkpoints = os.path.join(
            self.experiment_dir,
            "checkpoints"
        )

        self.plots = os.path.join(
            self.experiment_dir,
            "plots"
        )

        self.logs = os.path.join(
            self.experiment_dir,
            "logs"
        )

        self.history = os.path.join(
            self.experiment_dir,
            "history"
        )

        os.makedirs(
            self.checkpoints,
            exist_ok=True
        )

        os.makedirs(
            self.plots,
            exist_ok=True
        )

        os.makedirs(
            self.logs,
            exist_ok=True
        )

        os.makedirs(
            self.history,
            exist_ok=True
        )