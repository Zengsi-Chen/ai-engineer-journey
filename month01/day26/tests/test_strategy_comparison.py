from strategy_comparison import (
    StrategyResult,
    calculate_accuracy_gain,
    compare_strategies,
    find_best_strategy,
    calculate_accuracy_per_second,
    find_most_efficient_strategy,
)


def test_calculate_accuracy_gain():

    gain = calculate_accuracy_gain(
        baseline_accuracy=0.3205,
        fine_tuned_accuracy=0.4005,
    )

    assert gain == 0.08


def test_compare_strategies():

    baseline_accuracy = 0.3205

    results = [

        StrategyResult(
            name="feature_extraction",
            best_accuracy=0.35,
            best_epoch=3,
            training_time=10.0,
        ),

        StrategyResult(
            name="partial_fine_tuning",
            best_accuracy=0.40,
            best_epoch=4,
            training_time=20.0,
        ),

    ]

    comparison = compare_strategies(
        baseline_accuracy=baseline_accuracy,
        results=results,
    )

    assert len(comparison) == 2

    assert (
        comparison[0]["accuracy_gain"]
        == 0.0295
    )


def test_find_best_strategy():

    results = [

        StrategyResult(
            name="feature_extraction",
            best_accuracy=0.35,
            best_epoch=3,
            training_time=10.0,
        ),

        StrategyResult(
            name="partial_fine_tuning",
            best_accuracy=0.40,
            best_epoch=4,
            training_time=20.0,
        ),

        StrategyResult(
            name="full_fine_tuning",
            best_accuracy=0.38,
            best_epoch=5,
            training_time=30.0,
        ),

    ]

    best_strategy = find_best_strategy(
        results
    )

    assert (
        best_strategy.name
        == "partial_fine_tuning"
    )

def test_calculate_accuracy_per_second():

    value = calculate_accuracy_per_second(
        accuracy_gain=0.06,
        training_time=120.0,
    )

    assert value == 0.0005

def test_find_most_efficient_strategy():

    comparison = [

        {
            "strategy": "feature_extraction",
            "accuracy_per_second": 0.0005,
        },

        {
            "strategy": "partial_fine_tuning",
            "accuracy_per_second": 0.0003,
        },

        {
            "strategy": "full_fine_tuning",
            "accuracy_per_second": 0.0002,
        },

    ]

    best = find_most_efficient_strategy(
        comparison
    )

    assert (
        best["strategy"]
        == "feature_extraction"
    )