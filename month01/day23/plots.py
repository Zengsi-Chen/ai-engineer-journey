import matplotlib.pyplot as plt


def plot_training_history(
    history,
    save_path="artifacts/training_history.png",
):
    epochs = range(
        1,
        len(history["train_loss"]) + 1,
    )

    plt.figure()

    plt.plot(
        epochs,
        history["train_loss"],
        label="Train Loss",
    )

    plt.plot(
        epochs,
        history["val_loss"],
        label="Validation Loss",
    )

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    plt.tight_layout()

    plt.savefig(save_path)

    plt.close()

    print(f"Saved: {save_path}")


def plot_accuracy_history(
    history,
    save_path="artifacts/accuracy_history.png",
):
    epochs = range(
        1,
        len(history["train_accuracy"]) + 1,
    )

    plt.figure()

    plt.plot(
        epochs,
        history["train_accuracy"],
        label="Train Accuracy",
    )

    plt.plot(
        epochs,
        history["val_accuracy"],
        label="Validation Accuracy",
    )

    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()

    plt.tight_layout()

    plt.savefig(save_path)

    plt.close()

    print(f"Saved: {save_path}")


def plot_confusion_matrix(
    matrix,
    class_names,
    save_path="artifacts/confusion_matrix.png",
):
    plt.figure(figsize=(10, 8))

    plt.imshow(matrix.numpy())

    plt.colorbar()

    plt.xticks(
        range(len(class_names)),
        class_names,
        rotation=45,
        ha="right",
    )

    plt.yticks(
        range(len(class_names)),
        class_names,
    )

    plt.xlabel("Predicted")
    plt.ylabel("True")

    plt.title(
        "CIFAR-10 Confusion Matrix"
    )

    plt.tight_layout()

    plt.savefig(save_path)

    plt.close()

    print(f"Saved: {save_path}")