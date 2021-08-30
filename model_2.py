import torch
import torch.nn as nn
import torch.optim as optim
from utils import get_predictions
from dataset import get_data
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score
import torch.nn.functional as F
import time


activation = {}


# Function that creates a hook for a layer of the NN
def get_activation(name):
    def hook(model, input, output):
        activation[name] = output.detach()
    return hook


class NN(nn.Module):
    def __init__(self, input_size):
        super(NN, self).__init__()
        self.bn = nn.BatchNorm1d(input_size)
        self.fc1 = nn.Linear(input_size, 50)  # 50 hidden layers
        self.fc2 = nn.Linear(50, 1)

    def forward(self, x):
        x = self.bn(x)  # (BATCH_SIZE, 200)
        x = F.relu(self.fc1(x))
        return torch.sigmoid(self.fc2(x)).view(-1)


# Define gpu if available
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Define model
model = NN(input_size=200).to(DEVICE)
# Define optimiser
optimizer = optim.Adam(model.parameters(), lr=3e-4, weight_decay=1e-4)
# Define loss function
loss_fn = nn.BCELoss()  # (Binary Cross Entropy assuming output has already gone through sigmoid)

# Load the data
train_ds, val_ds, test_ds, test_ids = get_data()
train_loader = DataLoader(train_ds, batch_size=1024, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=1024)
test_loader = DataLoader(test_ds, batch_size=1024)

start = time.time()
# Loop through all epochs
for epoch in range(2):

    # Once per epoch evaluate the model
    probabilities, true_labels = get_predictions(val_loader, model, device=DEVICE)
    print(f"Epoch {epoch} - Validation ROC: {round(roc_auc_score(true_labels, probabilities), 4)}")

    # Loop through all batches
    for batch_idx, (data, targets) in enumerate(train_loader):
        # Test model with single batch first
        # data, targets = next(iter(train_loader))
        # Send data to GPU
        data = data.to(DEVICE)
        targets = targets.to(DEVICE)

        # Forward pass
        scores = model(data)
        model.bn.register_forward_hook(get_activation('bn'))
        model.fc1.register_forward_hook(get_activation('fc1'))
        model.fc2.register_forward_hook(get_activation('fc2'))

        # Define loss
        loss = loss_fn(scores, targets)
        # print(loss)
        # For every mini-batch during training we need to explicitly set the gradients to zero
        # before backpropagation because PyTorch accumulates gradients on subsequent backward passes
        optimizer.zero_grad()
        # Backpropagation
        loss.backward()
        # Perform parameter update based on the current gradient
        optimizer.step()

print(f"{round(time.time() - start, 2)} seconds")