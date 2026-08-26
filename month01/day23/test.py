import torch

from checkpoint import CheckpointManager
from dataset import create_dataloaders
from evaluation import (
    CIFAR10_CLASSES,
    confusion_matrix,
    evaluate_test,
    per_class_accuracy,
    most_confused_pairs,
)
from model import CIFAR10CNN
from plots import plot_confusion_matrix


def main():
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    _, _, test_loader = create_dataloaders(
        batch_size=64
    )

    model = CIFAR10CNN(
        num_classes=10
    ).to(device)

    checkpoint_manager = CheckpointManager(
        directory="artifacts"
    )

    checkpoint = checkpoint_manager.load_best(
        model
    )

    model.to(device)

    print(
        "Best epoch:",
        checkpoint["epoch"],
    )

    print(
        "Best validation accuracy:",
        f"{checkpoint['best_val_accuracy']:.4f}",
    )

    test_accuracy, predictions, labels = (
        evaluate_test(
            model,
            test_loader,
            device,
        )
    )

    print()
    print(
        "Test Accuracy:",
        f"{test_accuracy:.4f}",
    )

    matrix = confusion_matrix(
        predictions,
        labels,
        num_classes=10,
    )

    print()
    print("Confusion Matrix:")
    print(matrix)

    print()
    print("Classes:")

    for index, name in enumerate(
        CIFAR10_CLASSES
    ):
        print(
            f"{index}: {name}"
        )

    class_accuracies = per_class_accuracy(
        matrix
    )

    print()
    print("Per-class accuracy:")

    for index, accuracy in (
        class_accuracies.items()
    ):
        print(
            f"{CIFAR10_CLASSES[index]:12s}"
            f": {accuracy:.4f}"
        )

    pairs = most_confused_pairs(
        matrix,
        top_k=10,
    )

    print()
    print("Most confused pairs:")

    for (
        count,
        true_class,
        predicted_class,
    ) in pairs:

        print(
            f"{CIFAR10_CLASSES[true_class]}"
            f" → "
            f"{CIFAR10_CLASSES[predicted_class]}"
            f": {count}"
        )

    plot_confusion_matrix(
        matrix,
        CIFAR10_CLASSES,
    )

if __name__ == "__main__":
    main()