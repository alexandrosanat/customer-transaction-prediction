import pandas as pd
import numpy as np
import torch


def get_predictions(loader, model, device):
    # Model evaluation mode
    model.eval()
    saved_predictions = list()
    true_labels = list()

    # No compute is required here
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            scores = model(x)
            saved_predictions += scores.tolist()
            true_labels += y.tolist()

    # Switch to train mode
    model.train()

    return saved_predictions, true_labels
