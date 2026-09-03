import json
import platform
from datetime import datetime
from pathlib import Path
from uuid import uuid4

import torch


class ExperimentTracker:
    REQUIRED_METRICS = {
        "epoch",
        "train_loss",
        "val_loss",
        "train_accuracy",
        "val_accuracy",
    }

    def __init__(self, experiment_id=None):
        if experiment_id is None:
            experiment_id = self._generate_experiment_id()

        self.record = {
            "experiment_id": experiment_id,
            "metadata": self._create_metadata(),
            "hyperparameters": {},
            "training_history": [],
            "best_epoch": None,
            "best_metric": None,
            "best_checkpoint": None,
            "reproducibility": None,
        }

    @staticmethod
    def _generate_experiment_id():
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        unique_suffix = uuid4().hex[:8]

        return f"exp_{timestamp}_{unique_suffix}"

    @staticmethod
    def _create_metadata():
        return {
            "created_at": datetime.now().isoformat(),
            "python_version": platform.python_version(),
            "torch_version": torch.__version__,
        }

    def log_hyperparameters(self, params):
        self._validate_hyperparameters(params)

        self.record["hyperparameters"].update(params)


    @staticmethod
    def _validate_hyperparameters(params):
        if "learning_rate" in params:
            if params["learning_rate"] <= 0:
                raise ValueError(
                    "learning_rate must be greater than 0."
                )

        if "batch_size" in params:
            if (
                not isinstance(params["batch_size"], int)
                or isinstance(params["batch_size"], bool)
                or params["batch_size"] <= 0
            ):
                raise ValueError(
                    "batch_size must be a positive integer."
                )

        if "epochs" in params:
            if (
                not isinstance(params["epochs"], int)
                or isinstance(params["epochs"], bool)
                or params["epochs"] <= 0
            ):
                raise ValueError(
                    "epochs must be a positive integer."
                )

    def log_metadata(self, metadata):
        self.record["metadata"].update(metadata)

    def log_epoch(self, metrics):
        self._validate_epoch_metrics(metrics)

        self.record["training_history"].append(metrics)

        self._update_best_metric(metrics)

    def _update_best_metric(self, metrics):
        current_value = metrics["val_accuracy"]
        best_metric = self.record["best_metric"]

        if best_metric is None:
            self.record["best_epoch"] = metrics["epoch"]
            self.record["best_metric"] = {
                "name": "val_accuracy",
                "value": current_value,
            }
            return

        if current_value > best_metric["value"]:
            self.record["best_epoch"] = metrics["epoch"]
            self.record["best_metric"] = {
                "name": "val_accuracy",
                "value": current_value,
            }

    def _validate_epoch_metrics(self, metrics):
        missing_metrics = (
            self.REQUIRED_METRICS - metrics.keys()
        )

        if missing_metrics:
            raise ValueError(
                f"Missing required metrics: {sorted(missing_metrics)}"
            )

        epoch = metrics["epoch"]

        if (
            not isinstance(epoch, int)
            or isinstance(epoch, bool)
            or epoch <= 0
        ):
            raise ValueError(
                "epoch must be a positive integer."
            )

        self._validate_epoch_order(epoch)

    def _validate_epoch_order(self, epoch):
        history = self.record["training_history"]

        if not history:
            expected_epoch = 1
        else:
            last_epoch = history[-1]["epoch"]
            expected_epoch = last_epoch + 1

        if epoch != expected_epoch:
            raise ValueError(
                f"Expected epoch {expected_epoch}, "
                f"but received epoch {epoch}."
            )

    def set_best_checkpoint(self, path):
        if not isinstance(path, str) or not path.strip():
            raise ValueError(
                "Checkpoint path must be a non-empty string."
            )

        if self.record["best_epoch"] is None:
            raise RuntimeError(
                "Cannot set best checkpoint before logging "
                "a best epoch."
            )

        best_metric = self.record["best_metric"]

        self.record["best_checkpoint"] = {
            "path": path,
            "epoch": self.record["best_epoch"],
            "metric_name": best_metric["name"],
            "metric_value": best_metric["value"],
        }

    def get_record(self):
        return self.record


    def save(self, output_dir="artifacts/experiments"):
        output_dir = Path(output_dir)

        output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        experiment_id = self.record["experiment_id"]

        output_path = (
            output_dir / f"{experiment_id}.json"
        )

        with open(
            output_path,
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(
                self.record,
                file,
                indent=4,
            )

        return output_path


    @staticmethod
    def load(path):
        path = Path(path)

        with open(
            path,
            "r",
            encoding="utf-8",
        ) as file:
            return json.load(file)


    def set_reproducibility(self, seed):
        from src.reproducibility import (
            get_reproducibility_metadata,
            set_seed,
        )

        set_seed(seed)

        self.record["reproducibility"] = (
            get_reproducibility_metadata(seed)
        )