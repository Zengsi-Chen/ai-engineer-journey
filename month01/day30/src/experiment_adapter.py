from configs.pipeline_config import PipelineConfig


class ExperimentAdapter:
    """
    Connects Day 30 PipelineConfig with the
    Day 29 ExperimentTracker interface.
    """

    def __init__(
        self,
        config: PipelineConfig,
        tracker,
    ):
        self.config = config
        self.tracker = tracker

    def initialize(self):
        self.tracker.set_reproducibility(
            self.config.experiment.seed
        )

        self.tracker.log_metadata(
            {
                "experiment_name":
                    self.config.experiment.experiment_name,
            }
        )

        self._log_hyperparameters()

        return self.tracker

    def _log_hyperparameters(self):
        hyperparameters = {
            "dataset_name":
                self.config.data.dataset_name,
            "batch_size":
                self.config.data.batch_size,
            "max_train_samples":
                self.config.data.max_train_samples,
            "max_val_samples":
                self.config.data.max_val_samples,
            "model_name":
                self.config.model.model_name,
            "num_classes":
                self.config.model.num_classes,
            "pretrained":
                self.config.model.pretrained,
            "epochs":
                self.config.training.epochs,
            "learning_rate":
                self.config.training.learning_rate,
            "device":
                self.config.training.device,
        }

        self.tracker.log_hyperparameters(
            hyperparameters
        )