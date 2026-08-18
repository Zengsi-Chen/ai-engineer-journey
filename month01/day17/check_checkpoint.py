import torch


# =========================
# Check checkpoint.pth
# =========================

checkpoint = torch.load(
    "checkpoint.pth",
    weights_only=False
)

print("========== checkpoint.pth ==========")

print(
    "Keys:",
    checkpoint.keys()
)

print(
    "Epoch:",
    checkpoint["epoch"]
)

print(
    "Best Val Loss:",
    checkpoint["best_val_loss"]
)

print(
    "Early Stopping Counter:",
    checkpoint["early_stopping_counter"]
)


# =========================
# Check best_model.pth
# =========================

best_model = torch.load(
    "best_model.pth",
    weights_only=True
)

print("\n========== best_model.pth ==========")

print(
    "Type:",
    type(best_model)
)

print(
    "Number of parameters:",
    len(best_model)
)