MODEL_PATH = "../day18/best_model.pth"

BATCH_SIZE = 32

THRESHOLD = 0.5

THRESHOLDS = [
    0.1,
    0.2,
    0.3,
    0.4,
    0.5,
    0.6,
    0.7,
    0.8,
    0.9
]

MIN_RECALL = 0.95

RESULTS_DIR = "results"

METRICS_PATH = f"{RESULTS_DIR}/metrics.json"

CONFUSION_MATRIX_PATH = (
    f"{RESULTS_DIR}/confusion_matrix.png"
)

ROC_CURVE_PATH = (
    f"{RESULTS_DIR}/roc_curve.png"
)

THRESHOLD_METRICS_PATH = (
    f"{RESULTS_DIR}/threshold_metrics.csv"
)

BEST_THRESHOLD_PATH = (
    f"{RESULTS_DIR}/best_threshold.json"
)

FINAL_METRICS_PATH = (
    f"{RESULTS_DIR}/final_metrics.json"
)