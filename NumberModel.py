import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out
class NumberModel(nn.Module):
    def __init__(self, lrate, loss_fn, in_size, out_size):
        """
        Initializes the Neural Network model
        @param lrate:
        @param loss_fn:
        @param in_size:
        @param out_size:
        """
        super(NumberModel, self).__init__()

        # self.conv_layers = nn.Sequential(
        #     nn.Conv2d(in_channels=in_size, out_channels=96, kernel_size=(11, 11), stride=4, padding="valid"),
        #     nn.MaxPool2d(kernel_size=(3, 3), stride=2),
        #     nn.Conv2d(96, 256, kernel_size=(5, 5), stride=1, padding=2),
        #     nn.MaxPool2d(kernel_size=(3, 3), stride=2),
        #     nn.Conv2d(256, 384, kernel_size=(3, 3), stride=1, padding=1),
        #     nn.Conv2d(384, 384, kernel_size=(3, 3), stride=1, padding=1),
        #     nn.Conv2d(384, 256, kernel_size=(3, 3), stride=1, padding=1),
        #     nn.MaxPool2d(kernel_size=(3, 3), stride=2),
        # )
        # self.linear_layers = nn.Sequential(
        #     nn.Flatten(),
        #     nn.Linear(256 * 6 * 6, 4096),
        #     nn.ReLU(),
        #     nn.Linear(4096, 4096),
        #     nn.ReLU(),
        #     nn.Linear(4096, 1000),
        #     nn.ReLU(),
        #     nn.Linear(1000, out_size),
        #     nn.ReLU()
        # )
        block = BasicBlock
        layers = [2,2,2,2]
        self.in_channels = 64
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, out_size)

        self.loss_fn = loss_fn
        self.optimizer = optim.SGD(self.parameters(), lr=lrate, weight_decay=1e-5)

    def _make_layer(self, block, out_channels, blocks, stride=1):
        layers = []
        layers.append(block(self.in_channels, out_channels, stride))
        self.in_channels = out_channels
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the model
        @param x:
        @return:
        """
        # y = self.conv_layers(x)
        # return self.linear_layers(y)
        # return self.model(x)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x
    
    def step(self, x: torch.Tensor, y):
        """
        Returns the loss of a single forward pass
        @param x:
        @param y:
        @return:
        """
        self.optimizer.zero_grad()

        # pass the batch through the model
        y_hat = self.forward(x)

        # compute the loss
        loss = self.loss_fn(y_hat, y.to(dtype=torch.long))

        # update model
        loss.backward()
        self.optimizer.step()

        return loss.item()
    
class SudokuDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, X: np.ndarray,y: np.ndarray):
        """
        Initializes the dataset
        @param X: features vector
        @param y: labels vector
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

    NetObject = NumberModel(0.01, nn.CrossEntropyLoss(), image_dim[0], 10).to(device)

    losses = []
    epoch_progress_bar = tqdm(range(epochs), desc="Epochs", leave=True)
    for epoch in range(epochs):
        epoch_loss = 0.0
        for batch in train_loader:
            batch_x, batch_y = batch
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            epoch_loss += NetObject.step(batch_x, batch_y)
        epoch_loss /= len(train_loader)
        epoch_progress_bar.update(1)
        epoch_progress_bar.set_postfix({'Epoch Loss': epoch_loss})
    print("\n")

    # Set the model to evaluation mode
    NetObject.eval()

    # Define a variable to store the total number of correct predictions
    total_correct = 0
    # Define a variable to store the total number of examples
    total_examples = 0

    # Iterate through the test DataLoader
    for images, labels in test_loader:
        # Move images and labels to the device the model is on (e.g., GPU)
        images = images.to(device)
        labels = labels.to(device)

        # Forward pass
        with torch.no_grad():  # No need to compute gradients during inference
            outputs = NetObject(images)

        # Get predicted labels (and ignore the values themselves)
        _, predicted = torch.max(outputs, 1)

        # Update the total number of examples
        total_examples += labels.size(0)
        # Update the total number of correct predictions
        total_correct += (predicted == labels).sum().item()

        # Calculate the accuracy
        accuracy = total_correct / total_examples

    print('Accuracy:', accuracy)

    return losses, NetObject


if __name__ == "__main__":
    from python_helper_functions import get_sudoku_dataset
    train_data, test_data, single_image_dimension = get_sudoku_dataset()
    losses, NetObject = fit(train_data, test_data, single_image_dimension, 20, 50)