from config import validate_config

from experiment_registry import create_experiment

from data_factory import create_data_loaders

from model_factory import create_model

from task_factory import create_task

from metrics_factory import create_metrics

from optimizer_factory import create_optimizer

from pipeline import MLPipeline


# =========================
# 1. Create Experiment
# =========================

experiment_name = "mnist_cnn"

config = create_experiment(
    experiment_name
)

# =========================
# 2. Validate Configuration
# =========================

validate_config(config)

# =========================
# Build components
# =========================

model = create_model(
    config.model
)

task = create_task(
    config.task
)

metrics = create_metrics(
    config.task.task_type
)

optimizer = create_optimizer(
    model,
    config.training
)

train_loader, val_loader = create_data_loaders(
    config.data
)


# =========================
# Build pipeline
# =========================

pipeline = MLPipeline(
    model=model,
    task=task,
    optimizer=optimizer,
    train_loader=train_loader,
    val_loader=val_loader,
    metrics=metrics
)


# =========================
# Training
# =========================

for epoch in range(
    config.training.epochs
):

    train_loss = pipeline.train_epoch()

    results = pipeline.evaluate()

    print(
        f"Epoch [{epoch + 1}/"
        f"{config.training.epochs}] "
        f"Train Loss: {train_loss:.4f} "
        f"Val Loss: {results['loss']:.4f} "
        f"Accuracy: {results['accuracy']:.4f}"
    )