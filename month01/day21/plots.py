import os

import matplotlib.pyplot as plt


def plot_training_history(
    history,
    save_path=None
):

    train_loss = history.train_loss
    val_loss = history.val_loss
    learning_rate = history.learning_rate

    epochs = range(
        1,
        history.epochs + 1
    )


    plt.figure()

    plt.plot(
        epochs,
        train_loss,
        label="Train Loss"
    )

    plt.plot(
        epochs,
        val_loss,
        label="Val Loss"
    )

   
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training History")
    plt.legend()
    plt.grid()

    if save_path is not None:

        directory = os.path.dirname(
            save_path
        )

        if directory:

            os.makedirs(
                directory,
                exist_ok=True
            )

        plt.savefig(
            save_path,
            bbox_inches="tight"
        )

    plt.close()