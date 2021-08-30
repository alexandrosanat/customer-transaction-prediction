import pandas as pd
import torch
from torch.utils.data import TensorDataset
from torch.utils.data.dataset import random_split
from math import ceil


def get_data():

    # Load data
    train_data = pd.read_csv("./data/train.csv")
    test_data = pd.read_csv("./data/test.csv")

    # Split features and labels
    # Training data
    y_train = train_data["target"]
    X_train = train_data.drop(["ID_code", "target"], axis=1)
    # Test data
    test_ids = test_data["ID_code"]
    X_test = test_data.drop(["ID_code"], axis=1)

    # Convert data to tensors
    X_tensor_train = torch.tensor(X_train.values, dtype=torch.float32)
    y_tensor_train = torch.tensor(y_train.values, dtype=torch.float32)
    X_tensor_test = torch.tensor(X_test.values, dtype=torch.float32)

    # Convert data to pytorch dataset
    train_ds = TensorDataset(X_tensor_train, y_tensor_train)
    test_ds = TensorDataset(X_tensor_test)

    # Split data for training
    train_ds, val_ds = random_split(train_ds, [int(0.8*len(train_ds)), ceil(0.2*len(train_ds))])

    return train_ds, val_ds, test_ds, test_ids
