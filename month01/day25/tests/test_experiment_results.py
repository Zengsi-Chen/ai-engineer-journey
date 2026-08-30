from experiment_results import (
    EpochResult,
    ExperimentResult,
)


def test_epoch_result():

    result = EpochResult(
        epoch=1,
        train_loss=1.0,
        train_accuracy=0.5,
        validation_loss=1.2,
        validation_accuracy=0.4,
    )

    assert result.epoch == 1

    assert result.train_loss == 1.0

    assert result.train_accuracy == 0.5

    assert result.validation_loss == 1.2

    assert result.validation_accuracy == 0.4


def test_experiment_result():

    epoch_result = EpochResult(
        epoch=1,
        train_loss=1.0,
        train_accuracy=0.5,
        validation_loss=1.2,
        validation_accuracy=0.4,
    )

    experiment_result = ExperimentResult(
        name="Test Model",
        epoch_results=[
            epoch_result,
        ],
        best_validation_accuracy=0.4,
        training_time=10.0,
    )

    assert (
        experiment_result.name
        == "Test Model"
    )

    assert len(
        experiment_result.epoch_results
    ) == 1

    assert (
        experiment_result.best_validation_accuracy
        == 0.4
    )

    assert (
        experiment_result.training_time
        == 10.0
    )