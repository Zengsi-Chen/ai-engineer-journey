from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_required_directories_exist():
    required_directories = [
        "configs",
        "data/raw",
        "data/processed",
        "src/medseg/data",
        "src/medseg/models",
        "src/medseg/training",
        "src/medseg/evaluation",
        "src/medseg/xai",
        "src/medseg/inference",
        "tests",
        "artifacts/reports",
        "artifacts/figures",
    ]

    for directory in required_directories:
        path = PROJECT_ROOT / directory
        assert path.exists(), f"Missing directory: {directory}"