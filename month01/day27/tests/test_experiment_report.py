import pytest

from dataclasses import dataclass

from experiment_report import (
    calculate_tradeoff,
    get_best_experiment,
    generate_experiment_report,
)

@dataclass
class FakeResult:

    best_epoch: int

    best_accuracy: float

    final_accuracy: float

    training_time: float


def create_results():

    return {
        "uniform_lr": {
            "result": FakeResult(
                best_epoch=3,
                best_accuracy=0.70,
                final_accuracy=0.68,
                training_time=100.0,
            ),
        },

        "manual_discriminative_lr": {
            "result": FakeResult(
                best_epoch=4,
                best_accuracy=0.75,
                final_accuracy=0.74,
                training_time=120.0,
            ),
        },

        "automatic_layer_wise_decay": {
            "result": FakeResult(
                best_epoch=5,
                best_accuracy=0.80,
                final_accuracy=0.79,
                training_time=110.0,
            ),
        },
    }


def test_get_best_experiment():

    results = create_results()

    name, data = (
        get_best_experiment(
            results
        )
    )

    assert (
        name
        == "automatic_layer_wise_decay"
    )

    assert (
        data["result"].best_accuracy
        == 0.80
    )


def test_calculate_tradeoff():

    baseline = FakeResult(
        best_epoch=1,
        best_accuracy=0.70,
        final_accuracy=0.70,
        training_time=100.0,
    )

    comparison = FakeResult(
        best_epoch=2,
        best_accuracy=0.75,
        final_accuracy=0.75,
        training_time=120.0,
    )

    (
        accuracy_difference,
        time_difference,
    ) = calculate_tradeoff(
        baseline_result=baseline,
        comparison_result=comparison,
    )

    assert accuracy_difference == pytest.approx(0.05)

    assert ( time_difference == pytest.approx(20.0) )


def test_generate_experiment_report(
    tmp_path,
    ):

    results = create_results()

    report_path = (
        tmp_path
        / "experiment_report.md"
    )

    generate_experiment_report(
        results=results,
        report_path=report_path,
    )

    assert (
        report_path.exists()
    )

    report_text = (
        report_path.read_text(
            encoding="utf-8"
        )
    )

    assert (
        "# Day 27 Experiment Report"
        in report_text
    )

    assert (
        "automatic_layer_wise_decay"
        in report_text
    )

    assert (
        "Best Accuracy"
        in report_text
    )

    assert (
        "Automatic layer-wise"
        in report_text
    )