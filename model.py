import torch
import torch.nn as nn
import torch.optim as optim
from utils import get_predictions
from dataset import get_data
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score
from tqdm import tqdm


class NN(nn.Module):
    def __init__(self, input_size):
        super(NN, self).__init__()
        self.net = nn.Sequential(
            nn.BatchNorm1d(input_size),
            nn.Linear(input_size, 50),  # 50 hidden layers
            nn.ReLU(inplace=True),
            nn.Linear(50, 1)
        )

    def forward(self, x):
        return torch.sigmoid(self.net(x)).view(-1)  # Use BCE with logits if you just output net(x)


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

# Loop through all epochs
for epoch in range(20):

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
