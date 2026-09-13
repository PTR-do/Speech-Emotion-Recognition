import torch


class EarlyStopping:
    def __init__(self, patience=10, min_delta=0.0001):
        self.patience = patience
        self.min_delta = min_delta
        self.min_validation_loss = torch.inf
        self.counter = 0
        self.early_stop = False

    def __call__(self, validation_loss):
        if (validation_loss + self.min_delta) >= self.min_validation_loss:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.min_validation_loss = validation_loss
            self.counter = 0
