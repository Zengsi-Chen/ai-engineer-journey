import os
import torch
import torch.nn as nn


from torch.utils.data import DataLoader, random_split

from dataset import ClassificationDataset
from model import BinaryClassifier
from metrics import classification_metrics
from predictions_from_logits import predictions_from_logits
from early_stopping import EarlyStopping
from evaluation import evaluate_classification
from confusion_matrix import confusion_matrix
from roc import roc_curve
from auc import auc_score



# -------------------------
# 1. Data
# -------------------------

x = torch.arange(
    1,
    101,
    dtype=torch.float32
).reshape(-1, 1)

x = x / 100.0

y = (x >= 0.5).float()

# -------------------------
# 2. Dataset / DataLoader
# -------------------------

dataset = ClassificationDataset(x, y)

train_size = int(0.8 * len(dataset))

val_size = len(dataset) - train_size


generator = torch.Generator().manual_seed(42)

train_dataset, val_dataset = random_split(
    dataset,
    [train_size, val_size],
    generator=generator
)

train_loader = DataLoader(
    train_dataset,
    batch_size=4,
    shuffle=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=4,
    shuffle=False
)



# -------------------------
# 3. Model
# -------------------------

model = BinaryClassifier()


# -------------------------
# 4. Loss
# -------------------------

criterion = nn.BCEWithLogitsLoss()


# -------------------------
# 5. Optimizer
# -------------------------

optimizer = torch.optim.SGD(
    model.parameters(),
    lr=0.01
)

start_epoch = 0

# -------------------------
# EarlyStopping
# -------------------------

early_stopping = EarlyStopping(
    patience=10,
    min_delta=0.001,
    checkpoint_path="checkpoint.pth",
    best_model_path="best_model.pth"
)

if os.path.exists("checkpoint.pth"):

    checkpoint = torch.load(
        "checkpoint.pth",
        weights_only=False
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    optimizer.load_state_dict(
        checkpoint["optimizer_state_dict"]
    )

    start_epoch = checkpoint["epoch"]

    print(
        f"Resuming from epoch {start_epoch}"
    )

num_epochs = 100

for epoch in range(start_epoch, num_epochs):

    # =========================
    # Training
    # =========================

    model.train()

    total_train_loss = 0

    for batch_x, batch_y in train_loader:

        optimizer.zero_grad()

        logits = model(batch_x)

        loss = criterion(logits, batch_y)

        loss.backward()

        optimizer.step()

        total_train_loss += loss.item()

    avg_train_loss = total_train_loss / len(train_loader)

    # =========================
    # Validation
    # =========================

    model.eval()

    all_logits = []
    all_targets = []

    total_val_loss = 0

    with torch.no_grad():

        for batch_x, batch_y in val_loader:

            logits = model(batch_x)

            loss = criterion(
                logits,
                batch_y
            )

            total_val_loss += loss.item()

            all_logits.append(logits)
            all_targets.append(batch_y)

    avg_val_loss = (
        total_val_loss
        / len(val_loader)
    )

    # =========================
    # Metrics
    # =========================   

    all_logits = torch.cat(all_logits)

    all_targets = torch.cat(all_targets) 

    # -------------------------
    # Default threshold = 0.5
    # -------------------------

    probabilities = torch.sigmoid(all_logits)

    predictions = (
        probabilities >= 0.5
    ).float()

    metrics = classification_metrics(
        all_targets,
        predictions
    )

    # =========================
    # Print
    # =========================

    if (epoch + 1) % 10 == 0:
        print(
            f"Epoch [{epoch + 1}/{num_epochs}] "
            f"Train Loss: {avg_train_loss:.4f} "
            f"Val Loss: {avg_val_loss:.4f} "
            f"Val Acc: {metrics['accuracy']:.4f} "
            f"Val Precision: {metrics['precision']:.4f} "
            f"Val Recall: {metrics['recall']:.4f} "
            f"Val F1: {metrics['f1']:.4f}"
        )


    # =========================
    # Early Stopping
    # =========================

    should_stop = early_stopping(
        avg_val_loss,
        model,
        optimizer,
        epoch
    )

    
    # =========================
    # Stop
    # =========================
    
    if should_stop:

        print(
            f"Early stopping at epoch {epoch + 1}"
        )

        break
    # =================================
    # Simulating interruption
    # =================================
    
    '''
    if epoch == 29:

        print("Simulating interruption...")

        break
    '''

# =================================
# Load Best Model
# =================================

best_model = torch.load(
    "best_model.pth",
    weights_only=True
)

model.load_state_dict(
    best_model
)

print(
    "Best model loaded for final evaluation."
)

# -------------------------
# Final Evaluation
# -------------------------

results = evaluate_classification(
    model,
    val_loader,
    thresholds=[0.3, 0.5, 0.7],
    plot_roc=True,
    roc_save_path="roc_curve.png"
)

print(
    results["auc"]
)

result_05 = results["thresholds"][0.5]

print(
    result_05["accuracy"]
)

print(
    result_05["precision"]
)

print(
    result_05["recall"]
)

print(
    result_05["f1"]
)

print("\nThreshold Experiment")

for threshold, result in results["thresholds"].items():

    print("\n==============================")

    print(
        f"Threshold = {threshold}"
    )

    print(
        "Confusion Matrix"
    )

    print(
        result["confusion_matrix"]
    )

    print(
        f"Accuracy  : "
        f"{result['accuracy']:.4f}"
    )

    print(
        f"Precision : "
        f"{result['precision']:.4f}"
    )

    print(
        f"Recall    : "
        f"{result['recall']:.4f}"
    )

    print(
        f"F1        : "
        f"{result['f1']:.4f}"
    )



