import os
from datetime import datetime

import torch

from config import Config
from dataset import create_data_pipeline
from model import create_model
from loss import create_loss

from optimizer import (
    create_optimizer,
    create_scheduler
)

from checkpoint import CheckpointManager
from early_stopping import EarlyStopping
from trainer import Trainer

from metrics import create_metrics
from output_adapter import create_output_adapter
from evaluation import Evaluator
from threshold import ThresholdOptimizer

from pipeline_result import PipelineResult
from experiment_tracker import ExperimentTracker
from artifacts import ArtifactsManager

from plots import plot_training_history




def main():

    # =========================
    # 1. Config
    # =========================

    config = Config()

    config.task.task_type = "classification"
    config.task.threshold = 0.5
    # config.training.epochs = 520

    # =========================
    # 2. Experiment
    # =========================
    config.training.resume = True
    config.training.resume_experiment_id = "20260825_010839"

    if (
        config.training.resume
        and config.training.resume_experiment_id
    ):

        experiment_id = (
            config.training.resume_experiment_id
        )

    else:

        experiment_id = datetime.now().strftime(
            "%Y%m%d_%H%M%S"
        )


    artifacts = ArtifactsManager(
        root=config.artifacts.root,
        experiment_id=experiment_id
    )

    tracker = ExperimentTracker(
        experiment_id=experiment_id,
        output_dir=artifacts.experiment_dir
    )

    

    print(
        f"Experiment ID: {experiment_id}"
    )

    config.model.name = "mlp"

    config.model.params = {
        "input_dim": 1,
        "hidden_dim": 16,
        "output_dim": 1
    }

    # =========================
    # 2. Device
    # =========================

    if config.general.device == "auto":

        device = (
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

    else:

        device = config.general.device

    print(
        f"Device: {device}"
    )


    # =========================
    # 3. Data
    # =========================

    (
        train_loader,
        val_loader,
        test_loader
    ) = create_data_pipeline(
        config=config
    )


    # =========================
    # 4. Model
    # =========================

    model = create_model(
        config.model
    )

    

    # =========================
    # 5. Loss
    # =========================

    loss_fn = create_loss(
        config
    )


    # =========================
    # 6. Optimizer
    # =========================

    optimizer = create_optimizer(
        model,
        config
    )


    # =========================
    # 7. Scheduler
    # =========================

    scheduler = create_scheduler(
        optimizer,
        config
    )


    # =========================
    # 8. Checkpoint
    # =========================

    checkpoint = None

    if config.checkpoint.enabled:

        checkpoint = CheckpointManager(
            path=artifacts.checkpoints,
            monitor=config.checkpoint.monitor,
            mode=config.checkpoint.mode
        )

        print(
            f"Checkpoint directory: "
            f"{artifacts.checkpoints}"
        )


    # =========================
    # 9. Early Stopping
    # =========================

    early_stopping = None

    if config.early_stopping.enabled:

        early_stopping = EarlyStopping(
            patience=config.early_stopping.patience,
            min_delta=config.early_stopping.min_delta,
            mode=config.early_stopping.mode
        )


    # =========================
    # 10. Resume
    # =========================

    start_epoch = 0

    if (
        checkpoint is not None
        and config.training.resume
    ):

        start_epoch = checkpoint.load_checkpoint(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            early_stopping=early_stopping
        )

        print(
            f"Resume from epoch: {start_epoch}"
        )

    else:

        print(
            "Starting a new experiment."
        )


    # =========================
    # 11. Metrics
    # =========================

    metrics = create_metrics(
        config
    )


    # =========================
    # 12. Output Adapter
    # =========================

    output_adapter = create_output_adapter(
        config
    )


    # =========================
    # 13. Trainer
    # =========================

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        loss_fn=loss_fn,
        scheduler=scheduler,
        checkpoint=checkpoint,
        early_stopping=early_stopping,
        device=device
    )


    # =========================
    # 14. Training
    # =========================

    history = trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=config.training.epochs,
        start_epoch=start_epoch
    )

    plot_training_history(
        history=history,
        save_path=os.path.join(
            artifacts.plots,
            "loss_curve.png"
        )
    )

    # =========================
    # 15. Load Best Model
    # =========================

    if checkpoint is not None:

        checkpoint.load_best_model(
            model=model,
            device=device
        )


    # =========================
    # 16. Evaluator
    # =========================

    evaluator = Evaluator(
        model=model,
        metrics=metrics,
        output_adapter=output_adapter,
        device=device
    )


    # =========================
    # 17. Validation Prediction
    # =========================

    probabilities, targets = (
        evaluator.predict(
            val_loader
        )
    )


    # =========================
    # 18. Threshold Optimization
    # =========================

    threshold_optimizer = (
        ThresholdOptimizer(
            thresholds=torch.arange(
                0.1,
                1.0,
                0.1
            ),
            metric="f1"
        )
    )


    threshold_result = (
        threshold_optimizer.optimize(
            probabilities,
            targets
        )
    )


    best_threshold = (
        threshold_result[
            "best_threshold"
        ]
    )

    best_validation_score = (
        threshold_result[
            "best_score"
        ]
    )





    # =========================
    # 19. Apply Best Threshold
    # =========================

    output_adapter.threshold = (
        best_threshold
    )


    # =========================
    # 20. Final Test
    # =========================

    test_results = evaluator.evaluate(
        test_loader
    )


    # =========================
    # 21. Pipeline Result
    # =========================


    result = PipelineResult(

        best_epoch=(
            checkpoint.best_epoch
            if checkpoint is not None
            else None
        ),

        best_threshold=(
            best_threshold
        ),

        best_validation_score=(
            best_validation_score
        ),

        best_model_path = (
            checkpoint.best_model_path
            if checkpoint is not None
            else None
        ),

        test_results=(
            test_results
        ),

        experiment_id=(
            experiment_id
        ),

        history=(
            history
        )
    )


    # =========================
    # 22. Save Experiment
    # =========================

    experiment_path = tracker.save(
        result=result,
        config=config
    )


    # =========================
    # Print Result
    # =========================

    result.summary()

    print(
        f"Experiment saved: "
        f"{experiment_path}"
    )
        
   

if __name__ == "__main__":

    main()