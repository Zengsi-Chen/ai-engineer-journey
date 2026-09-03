import json
from pathlib import Path


class ExperimentComparator:
    def __init__(self, experiments_dir):
        self.experiments_dir = Path(experiments_dir)
        self.experiments = []


    def load_all(self):
        if not self.experiments_dir.exists():
            raise FileNotFoundError(
                f"Experiments directory not found: "
                f"{self.experiments_dir}"
            )

        self.experiments = []

        experiment_files = sorted(
            self.experiments_dir.glob("*.json")
        )

        for experiment_file in experiment_files:
            with open(
                experiment_file,
                "r",
                encoding="utf-8",
            ) as file:
                experiment = json.load(file)

            self.experiments.append(experiment)

        return self.experiments
    

    def get_experiment_count(self):
        return len(self.experiments)


    def get_best_experiment(self):
        if not self.experiments:
            raise RuntimeError(
                "No experiments loaded. "
                "Call load_all() first."
            )

        experiments_with_metric = [
            experiment
            for experiment in self.experiments
            if experiment.get("best_metric") is not None
        ]

        if not experiments_with_metric:
            raise RuntimeError(
                "No experiments contain a best metric."
            )

        return max(
            experiments_with_metric,
            key=lambda experiment: (
                experiment["best_metric"]["value"]
            ),
        )


    def _get_best_checkpoint(self, experiment):
        checkpoint = experiment.get("best_checkpoint")

        if checkpoint is None:
            raise RuntimeError(
                "Best experiment does not have a checkpoint."
            )

        required_fields = {
            "path",
            "epoch",
            "metric_name",
            "metric_value",
        }

        missing_fields = (
            required_fields - checkpoint.keys()
        )

        if missing_fields:
            raise ValueError(
                "Best checkpoint is missing required fields: "
                f"{sorted(missing_fields)}"
            )

        best_metric = experiment.get("best_metric")

        if best_metric is None:
            raise ValueError(
                "Best experiment does not have best metric metadata."
            )

        if (
            checkpoint["metric_name"]
            != best_metric["name"]
        ):
            raise ValueError(
                "Checkpoint metric name does not match "
                "experiment best metric."
            )

        if (
            checkpoint["metric_value"]
            != best_metric["value"]
        ):
            raise ValueError(
                "Checkpoint metric value does not match "
                "experiment best metric."
            )

        return checkpoint


    def _validate_checkpoint_file(self, checkpoint):
        checkpoint_path = Path(
            checkpoint["path"]
        )

        if not checkpoint_path.exists():
            raise FileNotFoundError(
                f"Checkpoint file not found: "
                f"{checkpoint_path}"
            )

        if not checkpoint_path.is_file():
            raise ValueError(
                f"Checkpoint path is not a file: "
                f"{checkpoint_path}"
            )

        return checkpoint_path
    

    def select_best_model(self):
        best_experiment = (
            self.get_best_experiment()
        )

        checkpoint = self._get_best_checkpoint(
            best_experiment
        )

        checkpoint_path = (
            self._validate_checkpoint_file(
                checkpoint
            )
        )

        return {
            "experiment_id": (
                best_experiment["experiment_id"]
            ),
            "checkpoint_path": str(checkpoint_path),
            "epoch": checkpoint["epoch"],
            "metric_name": checkpoint["metric_name"],
            "metric_value": checkpoint["metric_value"],
        }


    