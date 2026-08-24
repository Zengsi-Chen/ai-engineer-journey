from dataclasses import dataclass, field
from typing import Optional


@dataclass
class PipelineResult:

    best_epoch: Optional[int] = None

    best_threshold: Optional[float] = None

    best_validation_score: Optional[float] = None

    best_model_path: Optional[str] = None

    test_results: dict = field(
        default_factory=dict
    )

    experiment_id: Optional[str] = None

    history: object = None

    def summary(self):

        print()
        print("========================")
        print("Pipeline Result")
        print("========================")

        print(
            f"Best Epoch: "
            f"{self.best_epoch}"
        )

        print(
            f"Best Threshold: "
            f"{self.best_threshold}"
        )

        print(
            f"Best Validation Score: "
            f"{self.best_validation_score:.4f}"
            if self.best_validation_score is not None
            else "Best Validation Score: None"
        )

        print(
            f"Best Model: "
            f"{self.best_model_path}"
        )

        print()
        print("Test Results")

        for name, value in self.test_results.items():

            print(
                f"{name}: {value:.4f}"
            )

        print(
            f"Experiment ID: "
            f"{self.experiment_id}"
        )