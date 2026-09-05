from configs.pipeline_config import PipelineConfig

from src.data_adapter import (
    create_data_loaders,
)

from src.experiment_adapter import (
    ExperimentAdapter,
)

from src.model_adapter import (
    create_model,
)

from src.training_adapter import (
    train_model,
)


def run_pipeline(
    config: PipelineConfig,
    tracker,
):
    """
    Orchestrate the complete Day 30
    experiment pipeline.

    The pipeline is intentionally responsible
    only for coordinating adapters.

    It does not implement:

    - data loading logic
    - model construction logic
    - training logic
    - evaluation logic
    - checkpoint logic
    - experiment tracking logic
    """

    # -------------------------------------------------
    # 1. Initialize experiment
    # -------------------------------------------------

    experiment_adapter = ExperimentAdapter(
        config=config,
        tracker=tracker,
    )

    tracker = (
        experiment_adapter.initialize()
    )

    # -------------------------------------------------
    # 2. Create data loaders
    # -------------------------------------------------

    (
        train_loader,
        validation_loader,
        test_loader,
    ) = create_data_loaders(
        config
    )

    # -------------------------------------------------
    # 3. Create model
    # -------------------------------------------------

    model = create_model(
        config
    )

    # -------------------------------------------------
    # 4. Train model
    #
    # Training adapter handles:
    # - training
    # - validation
    # - best model selection
    # - checkpoint saving
    # - checkpoint loading
    # - final test evaluation
    # -------------------------------------------------

    training_result = train_model(
        model=model,
        train_loader=train_loader,
        validation_loader=validation_loader,
        test_loader=test_loader,
        config=config,
        tracker=tracker,
    )

    # -------------------------------------------------
    # 5. Save experiment record
    # -------------------------------------------------

    experiment_path = tracker.save()

    # -------------------------------------------------
    # 6. Return pipeline result
    # -------------------------------------------------

    return {
        "experiment_id": (
            tracker.get_record()[
                "experiment_id"
            ]
        ),
        "best_epoch": (
            training_result[
                "best_epoch"
            ]
        ),
        "best_metric": (
            training_result[
                "best_metric"
            ]
        ),
        "best_checkpoint": (
            training_result[
                "best_checkpoint"
            ]
        ),
        "test_metrics": (
            training_result[
                "test_metrics"
            ]
        ),
        "experiment_path": str(
            experiment_path
        ),
    }
