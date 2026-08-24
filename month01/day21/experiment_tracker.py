import json
import os

from dataclasses import asdict
from datetime import datetime


class ExperimentTracker:

    def __init__(
        self,
        experiment_id,
        output_dir
    ):

        self.experiment_id = experiment_id
        self.output_dir = output_dir

        os.makedirs(
            self.output_dir,
            exist_ok=True
        )


    def create_experiment_id(self):

        return datetime.now().strftime(
            "%Y%m%d_%H%M%S"
        )


    def save(
        self,
        result,
        config
    ):

        experiment_id = result.experiment_id

        experiment = {

            "experiment_id":
                experiment_id,

            "timestamp":
                datetime.now().isoformat(),

            "config":
                asdict(config),

            "best_epoch":
                result.best_epoch,

            "best_threshold":
                result.best_threshold,

            "best_validation_score":
                result.best_validation_score,

            "best_model_path":
                result.best_model_path,

            "test_results":
                result.test_results,

            "history":
                result.history.to_dict()
                if result.history is not None
                else None
        }

        path = os.path.join(
            self.output_dir,
            "experiment.json"
        )

        with open(
            path,
            "w",
            encoding="utf-8"
        ) as f:

            json.dump(
                experiment,
                f,
                indent=4
            )

        return path