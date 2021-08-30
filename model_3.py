import torch
import torch.nn as nn
import torch.optim as optim
from utils import get_predictions, get_submission
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
    """
    Since we established that the features are not correlated,
    we will try to use that information to train a more effective NN.
    We will instead convert each feature to each own single example and we will
    do this for every example. This mean that the input of size (BATCH_SIZE, no_of_features),
    will be converted into a vector of size (BATCH_SIZE * no_of_features, 1) and
    then go through a hidden layer of input size [1] with hidden_layers=16 hidden layers.
    The outputs will then be converted back to size (BATCH_SIZE, no_of_features * hidden_layers)
    no_of_features = input size

    We have also created 200 new features that tell us if the feature values is unique or not.
    After creating the new features we want to pass each feature as its own example together
    with the value indicating whether the feature is unique or not
    """

    def __init__(self, input_size, hidden_dim):
        super(NN, self).__init__()
        self.bn = nn.BatchNorm1d(input_size//2)
        self.fc1 = nn.Linear(2, hidden_dim)
        self.fc2 = nn.Linear(input_size//2 * hidden_dim, 1)

    def forward(self, x):
        BATCH_SIZE = x.shape[0]

        # Split the features (unsqeeze convert its feature into a separate example)
        original_features = x[:, :200].unsqueeze(2)  # (BATCH_SIZE, 200, 1)
        # Send all 200 original features through batch norm
        original_features = self.bn(original_features)  # (BATCH_SIZE, 400)
        new_features = x[:, 200:].unsqueeze(2)  # (BATCH_SIZE, 200, 1)

        # Concatenate the features
        x = torch.cat([original_features, new_features], dim=2)  # (BATCH_SIZE, 200, 2)

        # We need to reshape the output of the first layer from (BATCH_SIZE, 200, hidden_dim)
        # Once packed together the inputs are (200, 2) not (400, 1)
        x = F.relu(self.fc1(x)).reshape(BATCH_SIZE, -1)  # (BATCH_SIZE, input_size//2 * hidden_dim)
        return torch.sigmoid(self.fc2(x)).view(-1)


# Define gpu if available
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Define model
model = NN(input_size=400, hidden_dim=100).to(DEVICE)
# Define optimiser
optimizer = optim.Adam(model.parameters(), lr=2e-3, weight_decay=1e-4)
# Define loss function
loss_fn = nn.BCELoss()  # (Binary Cross Entropy assuming output has already gone through sigmoid)

# Load the data
train_ds, val_ds, test_ds, test_ids = get_data()
train_loader = DataLoader(train_ds, batch_size=1024, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=1024)
test_loader = DataLoader(test_ds, batch_size=1024)

start = time.time()
plot_loss = dict()  # Keep track of loss
# Loop through all epochs
for epoch in range(35):

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
        # Assign hooks to the layers to get their outputs
        model.bn.register_forward_hook(get_activation('bn'))
        model.fc1.register_forward_hook(get_activation('fc1'))
        model.fc2.register_forward_hook(get_activation('fc2'))

        # Define loss
        loss = loss_fn(scores, targets)
        plot_loss[epoch] = loss
        # print(loss)
        # For every mini-batch during training we need to explicitly set the gradients to zero
        # before backpropagation because PyTorch accumulates gradients on subsequent backward passes
        optimizer.zero_grad()
        # Backpropagation
        loss.backward()
        # Perform parameter update based on the current gradient
        optimizer.step()

print(f"{round(time.time() - start, 2)} seconds")

# Export data for submission
get_submission(model, test_loader, test_ids, DEVICE)