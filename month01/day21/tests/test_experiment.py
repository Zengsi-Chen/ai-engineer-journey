from pathlib import Path


def test_experiment_directory_exists():

    experiment_root = Path(
        "artifacts/experiments"
    )

    assert experiment_root.exists()


def test_experiment_has_required_files():

    experiment_root = Path(
        "artifacts/experiments"
    )

    experiments = [
        path
        for path in experiment_root.iterdir()
        if path.is_dir()
    ]

    assert len(experiments) > 0

    latest_experiment = max(
        experiments,
        key=lambda path: path.stat().st_mtime
    )

    assert (
        latest_experiment
        / "experiment.json"
    ).exists()

    assert (
        latest_experiment
        / "checkpoints"
        / "checkpoint.pth"
    ).exists()

    assert (
        latest_experiment
        / "checkpoints"
        / "best_model.pth"
    ).exists()