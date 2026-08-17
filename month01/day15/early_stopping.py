import torch


class EarlyStopping:
   
    def __init__(
        self,
        patience=20,
        min_delta=0.001,
        path="best_model.pth"
    ):
        self.patience = patience
        self.min_delta = min_delta
        self.path = path

        self.best_val_loss = float("inf")
        self.counter = 0
        self.early_stop = False
        
    def __call__(self, val_loss, model):

        if val_loss < self.best_val_loss - self.min_delta:

            self.best_val_loss = val_loss
            self.counter = 0

            torch.save(
                model.state_dict(),
                self.path
            )

        else:

            self.counter += 1

            if self.counter >= self.patience:
                self.early_stop = True