import torch

from model import NonlinearModel
from evaluate import evaluate
from data import create_datasets


BEST_MODEL_PATH = "../day15/best_model.pth"

train_dataset, val_dataset, test_dataset = create_datasets()

# ============================================================
# 1. Create fixed dataset split
# ============================================================

train_dataset, val_dataset, test_dataset = create_datasets(
    n_samples=30,
    train_ratio=0.7,
    val_ratio=0.15,
    seed=42
)


print("Dataset")
print("-------------------")
print(f"Train      : {len(train_dataset)}")
print(f"Validation : {len(val_dataset)}")
print(f"Test       : {len(test_dataset)}")


# ============================================================
# 2. Model
# ============================================================

model = NonlinearModel()


# ============================================================
# 3. Load best model
# ============================================================

state_dict = torch.load(
    BEST_MODEL_PATH,
    map_location="cpu"
)

model.load_state_dict(state_dict)

print("Loaded best model.")




# ============================================================
# 4. Prepare Test Set
# ============================================================

x_test = torch.stack([
    test_dataset[i][0]
    for i in range(len(test_dataset))
])

y_test = torch.stack([
    test_dataset[i][1]
    for i in range(len(test_dataset))
])


# ============================================================
# 5. Evaluation
# ============================================================

results, y_pred = evaluate(
    model,
    x_test,
    y_test
)


# ============================================================
# 6. Evaluation Report
# ============================================================

print()
print("=" * 40)
print("       MODEL EVALUATION")
print("=" * 40)



print()
print("Dataset")
print("-" * 40)
print(f"Train samples    : {len(train_dataset)}")
print(f"Validation       : {len(val_dataset)}")
print(f"Test samples     : {len(test_dataset)}")

print()
print("Metrics")
print("-" * 40)
print(f"MAE              : {results['mae']:.4f}")
print(f"MSE              : {results['mse']:.4f}")
print(f"RMSE             : {results['rmse']:.4f}")
print(f"R2               : {results['r2']:.4f}")

print()
print("Predictions")
print("-" * 40)

for x, true, pred in zip(
    x_test,
    y_test,
    y_pred
):

    error = abs(
        true.item() - pred.item()
    )

    print(
        f"x={x.item():5.2f} "
        f"True={true.item():6.2f} "
        f"Pred={pred.item():6.2f} "
        f"Error={error:5.2f}"
    )

print()
print("=" * 40)