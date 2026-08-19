import os


import torch

from config import (
    MODEL_PATH,
    BATCH_SIZE,
    THRESHOLD,
    THRESHOLDS,
    MIN_RECALL,
    RESULTS_DIR,
    METRICS_PATH,
    FINAL_METRICS_PATH,
    CONFUSION_MATRIX_PATH,
    ROC_CURVE_PATH,
    THRESHOLD_METRICS_PATH,
    BEST_THRESHOLD_PATH
)

from model import BinaryClassifier

from dataset import (
    create_dataset,
    split_dataset,
    create_dataloader
)

from evaluation import (
    run_evaluation,
    print_evaluation_summary
)

from artifacts import save_results


def main():

    # =========================
    # 1. Create results directory
    # =========================

    os.makedirs(
        RESULTS_DIR,
        exist_ok=True
    )


    # =========================
    # 2. Create Dataset
    # =========================

    dataset = create_dataset()

    print(
        "Dataset size:",
        len(dataset)
    )


    # =========================
    # 3. Split Dataset
    # =========================

    train_dataset, val_dataset = split_dataset(
        dataset
    )

    print(
        "Train samples:",
        len(train_dataset)
    )

    print(
        "Validation samples:",
        len(val_dataset)
    )


    # =========================
    # 4. Create Validation DataLoader
    # =========================

    val_loader = create_dataloader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False
    )


    # =========================
    # 5. Create Model
    # =========================

    model = BinaryClassifier()


    # =========================
    # 6. Load Best Model
    # =========================

    checkpoint = torch.load(
        MODEL_PATH,
        map_location="cpu"
    )

    model.load_state_dict(
        checkpoint
    )

    model.eval()

    print(
        "Best model loaded."
    )


    # =========================
    # 7. Evaluate
    # =========================

    evaluation_results = run_evaluation(
        model=model,
        val_loader=val_loader,
        threshold=THRESHOLD,
        thresholds=THRESHOLDS,
        min_recall=MIN_RECALL
    )

    print_evaluation_summary(
        evaluation_results,
        min_recall=MIN_RECALL
    )

    save_results(
        evaluation_results=evaluation_results,
        metrics_path=METRICS_PATH,
        final_metrics_path=FINAL_METRICS_PATH,
        threshold_metrics_path=THRESHOLD_METRICS_PATH,
        best_threshold_path=BEST_THRESHOLD_PATH,
        confusion_matrix_path=CONFUSION_MATRIX_PATH,
        roc_curve_path=ROC_CURVE_PATH
    )




    # =========================
    # 12. Print Artifact Paths
    # =========================

    print()
    print("Evaluation artifacts saved:")

    print(
        f"- {METRICS_PATH}"
    )

    print(
        f"- {FINAL_METRICS_PATH}"
    )

    print(
        f"- {CONFUSION_MATRIX_PATH}"
    )

    print(
        f"- {ROC_CURVE_PATH}"
    )

    print(
        f"- {THRESHOLD_METRICS_PATH}"
    )

    print(
        f"- {BEST_THRESHOLD_PATH}"
    )

if __name__ == "__main__":
    main()