import json
from pathlib import Path

import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[1]

RESULT_PATH = (
    PROJECT_ROOT
    / "artifacts"
    / "experiments"
    / "day34_trainer_experiment.json"
)

FIGURE_DIR = (
    PROJECT_ROOT
    / "artifacts"
    / "figures"
)


def load_history() -> list[dict]:
    with RESULT_PATH.open(
        "r",
        encoding="utf-8",
    ) as file:
        data = json.load(file)

    return data["history"]


def plot_losses(history: list[dict]) -> None:
    epochs = [item["epoch"] for item in history]
    train_loss = [item["train_loss"] for item in history]
    val_loss = [item["val_loss"] for item in history]

    plt.figure(figsize=(8, 5))

    plt.plot(
        epochs,
        train_loss,
        label="Train Loss",
    )

    plt.plot(
        epochs,
        val_loss,
        label="Validation Loss",
    )

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")

    plt.legend()
    plt.grid(True)

    output_path = FIGURE_DIR / "day34_loss_curve.png"

    plt.savefig(
        output_path,
        dpi=150,
        bbox_inches="tight",
    )

    plt.close()

    print(f"Saved: {output_path}")


def plot_segmentation_metrics(history: list[dict]) -> None:
    epochs = [item["epoch"] for item in history]
    val_dice = [item["val_dice"] for item in history]
    val_iou = [item["val_iou"] for item in history]

    plt.figure(figsize=(8, 5))

    plt.plot(
        epochs,
        val_dice,
        label="Validation Dice",
    )

    plt.plot(
        epochs,
        val_iou,
        label="Validation IoU",
    )

    plt.xlabel("Epoch")
    plt.ylabel("Score")
    plt.title("Validation Segmentation Metrics")

    plt.ylim(0, 1)

    plt.legend()
    plt.grid(True)

    output_path = (
        FIGURE_DIR
        / "day34_segmentation_metrics.png"
    )

    plt.savefig(
        output_path,
        dpi=150,
        bbox_inches="tight",
    )

    plt.close()

    print(f"Saved: {output_path}")


def plot_learning_rate(history: list[dict]) -> None:
    epochs = [item["epoch"] for item in history]
    learning_rates = [
        item["learning_rate"]
        for item in history
    ]

    plt.figure(figsize=(8, 5))

    plt.plot(
        epochs,
        learning_rates,
        marker="o",
    )

    plt.xlabel("Epoch")
    plt.ylabel("Learning Rate")
    plt.title("Learning Rate Schedule")

    plt.yscale("log")

    plt.grid(True)

    output_path = (
        FIGURE_DIR
        / "day34_learning_rate.png"
    )

    plt.savefig(
        output_path,
        dpi=150,
        bbox_inches="tight",
    )

    plt.close()

    print(f"Saved: {output_path}")


def main() -> None:
    FIGURE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    history = load_history()

    if not history:
        raise ValueError(
            "Training history is empty"
        )

    plot_losses(history)
    plot_segmentation_metrics(history)
    plot_learning_rate(history)

    best_epoch = max(
        history,
        key=lambda item: item["val_dice"],
    )

    print("=" * 60)
    print("Training Curve Analysis")
    print("=" * 60)

    print(
        f"Best Epoch: "
        f"{best_epoch['epoch']}"
    )

    print(
        f"Best Val Dice: "
        f"{best_epoch['val_dice']:.6f}"
    )

    print(
        f"Best Val IoU: "
        f"{best_epoch['val_iou']:.6f}"
    )

    print(
        f"Val Loss at Best Epoch: "
        f"{best_epoch['val_loss']:.6f}"
    )

    print(
        f"Learning Rate: "
        f"{best_epoch['learning_rate']:.6g}"
    )

    print("=" * 60)


if __name__ == "__main__":
    main()