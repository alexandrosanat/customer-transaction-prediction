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


def get_submission(model, loader, test_ids, device):
    # Model evaluation mode
    model.eval()
    # Store all predictions in list
    all_predictions = list()

    # No compute is required here
    with torch.no_grad():
        for x in loader:
            x = x[0].to(device)
            score = model(x)
            prediction = score.float()
            all_predictions += prediction.tolist()

    # Switch to train mode
    model.train()

    df = pd.DataFrame({
        "ID_code": test_ids.values,
        "target": np.array(all_predictions)
    })

    df.to_csv("./data/submission.csv", index=False)
