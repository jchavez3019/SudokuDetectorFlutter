import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

class NumberModel(nn.Module):
    def __init__(self, lrate, loss_fn, in_size, out_size):
        super(NumberModel, self).__init__()

        self.model = nn.Sequential(
            nn.Conv2d(in_channels=in_size, out_channels=96, kernel_size=(11, 11), stride=4, padding="valid"),
            nn.MaxPool2d(kernel_size=(3, 3), stride=2),
            nn.Conv2d(96, 256, kernel_size=(5, 5), stride=1, padding=2),
            nn.MaxPool2d(kernel_size=(3, 3), stride=2),
            nn.Conv2d(256, 384, kernel_size=(3, 3), stride=1, padding=1),
            nn.Conv2d(384, 384, kernel_size=(3, 3), stride=1, padding=1),
            nn.Conv2d(384, 256, kernel_size=(3, 3), stride=1, padding=1),
            nn.MaxPool2d(kernel_size=(3, 3), stride=2),
            nn.Flatten(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.Linear(4096, 4096),
            nn.Linear(4096, 1000),
            nn.Linear(1000, out_size),
            nn.ReLU(),
        )

        self.loss_fn = loss_fn
        self.optimizer = optim.SGD(self.parameters(), lr=lrate, weight_decay=1e-5)

    def forward(self, x):
        return self.model(x)
    
    def step(self, x, y):
        self.optimizer.zero_grad()

        # pass the batch through the model
        y_hat = self.model(x)

        # compute the loss
        loss = self.loss_fn(y_hat, y)

        # update model
        loss.backward()
        self.optimizer.step()

        return loss.item()
    
class SudokuDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, X,y):
        """
        Args:
            X [np.array]: features vector
            y [np.array]: labels vector          
        """
        self.data = X
        self.labels = y

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        features = self.data[idx,:]
        label = self.labels[idx]
        sample = {'features': features,'labels': label}

        return sample
    
def fit(train_dataset, test_dataset, image_dim, epochs, batch_size=50):
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    NetObject = NumberModel(0.01, nn.CrossEntropyLoss(), image_dim[0], 10)

    losses = []
    for epoch in range(epochs):
        print(f"\rEpoch {epoch}", end="")
        epoch_loss = 0.0
        for batch in train_loader:
            batch_x, batch_y = batch
            epoch_loss += NetObject.step(batch[batch_x], batch[batch_y])
        epoch_loss /= len(train_loader)

    # Set the model to evaluation mode
    NetObject.eval()

    # Define a variable to store the total number of correct predictions
    total_correct = 0
    # Define a variable to store the total number of examples
    total_examples = 0

    # Iterate through the test DataLoader
    for images, labels in test_loader:
        # Move images and labels to the device the model is on (e.g., GPU)
        # images = images.to(device)
        # labels = labels.to(device)

        # Forward pass
        with torch.no_grad():  # No need to compute gradients during inference
            outputs = NetObject(images)

        # Get predicted labels
        _, predicted = torch.max(outputs, 1)

        # Update the total number of examples
        total_examples += labels.size(0)
        # Update the total number of correct predictions
        total_correct += (predicted == labels).sum().item()

        # Calculate the accuracy
        accuracy = total_correct / total_examples

    print('Accuracy:', accuracy)

    return losses, NetObject