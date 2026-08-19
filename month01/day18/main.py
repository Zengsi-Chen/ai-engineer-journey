import torch
from torch.utils.data import DataLoader, random_split

from dataset import ClassificationDataset
from model import BinaryClassifier
from threshold import (
    generate_thresholds,
    threshold_experiment,
    find_best_threshold,
    find_threshold_by_metric
)


# =========================
# 1. Data
# =========================

x = torch.arange(
    1,
    101,
    dtype=torch.float32
).reshape(-1, 1)

x = x / 100.0

y = (x >= 0.5).float()


# =========================
# 2. Dataset
# =========================

dataset = ClassificationDataset(
    x,
    y
)

train_size = int(
    0.8 * len(dataset)
)

val_size = (
    len(dataset) - train_size
)

generator = torch.Generator().manual_seed(42)

train_dataset, val_dataset = random_split(
    dataset,
    [train_size, val_size],
    generator=generator
)


# =========================
# 3. Validation DataLoader
# =========================

val_loader = DataLoader(
    val_dataset,
    batch_size=4,
    shuffle=False
)


# =========================
# 4. Model
# =========================

model = BinaryClassifier()


# =========================
# 5. Load Best Model
# =========================

best_model = torch.load(
    "best_model.pth",
    weights_only=True
)

model.load_state_dict(
    best_model
)

model.eval()

print(
    "Best model loaded."
)


# =========================
# 6. Generate Probabilities
# =========================

all_logits = []
all_targets = []

with torch.no_grad():

    for batch_x, batch_y in val_loader:

        logits = model(batch_x)

        all_logits.append(logits)
        all_targets.append(batch_y)


all_logits = torch.cat(
    all_logits
)

all_targets = torch.cat(
    all_targets
)

probabilities = torch.sigmoid(
    all_logits
)

# =========================
# 8. Threshold Search
# =========================

thresholds = generate_thresholds()

results = threshold_experiment(
    probabilities,
    all_targets,
    thresholds
)


# =========================
# 9. Find Best Threshold
# =========================

best_threshold, best_f1 = (
    find_best_threshold(
        results
    )
)

best_recall_threshold, best_recall = (
    find_threshold_by_metric(
        results,
        "recall"
    )
)

best_precision_threshold, best_precision = (
    find_threshold_by_metric(
        results,
        "precision"
    )
)


print(
    "\nThreshold Selection"
)

print(
    f"Best F1 Threshold        : "
    f"{best_threshold}"
)

print(
    f"Best F1                  : "
    f"{best_f1:.4f}"
)

print(
    f"Best Recall Threshold    : "
    f"{best_recall_threshold}"
)

print(
    f"Best Recall              : "
    f"{best_recall:.4f}"
)

print(
    f"Best Precision Threshold : "
    f"{best_precision_threshold}"
)

print(
    f"Best Precision           : "
    f"{best_precision:.4f}"
)


# =========================
# 10. Compare Thresholds
# =========================

default_threshold = 0.5

default_metrics = results[
    default_threshold
]

best_metrics = results[
    best_threshold
]


print(
    "\nThreshold Comparison"
)

print(
    f"\nDefault Threshold = "
    f"{default_threshold}"
)

print(
    f"Accuracy  : "
    f"{default_metrics['accuracy']:.4f}"
)

print(
    f"Precision : "
    f"{default_metrics['precision']:.4f}"
)

print(
    f"Recall    : "
    f"{default_metrics['recall']:.4f}"
)

print(
    f"F1        : "
    f"{default_metrics['f1']:.4f}"
)


print(
    f"\nBest Threshold = "
    f"{best_threshold}"
)

print(
    f"Accuracy  : "
    f"{best_metrics['accuracy']:.4f}"
)

print(
    f"Precision : "
    f"{best_metrics['precision']:.4f}"
)

print(
    f"Recall    : "
    f"{best_metrics['recall']:.4f}"
)

print(
    f"F1        : "
    f"{best_metrics['f1']:.4f}"
)