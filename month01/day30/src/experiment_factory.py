import importlib.util
import sys
from pathlib import Path


def create_experiment_tracker():
    project_root = (
        Path(__file__).resolve().parents[3]
    )

    day29_root = (
        project_root
        / "month01"
        / "day29"
    )

    day29_src = day29_root / "src"

    # Day 29 ExperimentTracker 
    # "from src.reproducibility import ..."
    #
    #  Python should find Day 29
    #  src package。
    day29_root_str = str(day29_root)

    if day29_root_str in sys.path:
        sys.path.remove(day29_root_str)

    sys.path.insert(
        0,
        day29_root_str,
    )

    tracker_path = (
        day29_src
        / "experiment_tracker.py"
    )

    spec = importlib.util.spec_from_file_location(
        "day29_experiment_tracker",
        tracker_path,
    )

    if spec is None or spec.loader is None:
        raise ImportError(
            "Unable to load Day 29 ExperimentTracker."
        )

    module = importlib.util.module_from_spec(
        spec
    )

    spec.loader.exec_module(module)

    return module.ExperimentTracker()

