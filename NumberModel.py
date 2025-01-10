import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Subset
from torch import Tensor, Size
from tqdm import tqdm
from typing import Union, Tuple
from torch.utils.mobile_optimizer import optimize_for_mobile

from python_helper_functions import get_sudoku_dataset

save_model = False
model_path = "./models/"
model_name = "NumberModel_v0"
device = torch.device('cuda')


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
    def __init__(
            self,
            lrate: float,
            loss_fn,
            in_size,
            out_size):
        """
        Initializes the Neural Network model
        @param lrate: Learning rate for the model
        @param loss_fn: Loss function for the model
        @param in_size: Input size for a single input
        @param out_size: Output size for a single output
        """
        super(NumberModel, self).__init__()

        block = BasicBlock
        layers = [2, 2, 2, 2]  # number of blocks in each residual layer
        self.in_channels = 64

        self.model = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False),  # conv1
            nn.BatchNorm2d(64),  # bn1
            nn.ReLU(inplace=True),  # relu
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),  # maxpool
            self._make_layer(block, 64, layers[0]),  # layer1
            self._make_layer(block, 128, layers[1], stride=2),  # layer 2
            self._make_layer(block, 256, layers[2], stride=2),  # layer 3
            self._make_layer(block, 512, layers[3], stride=2),  # layer 4
            nn.AdaptiveAvgPool2d((1, 1)),  # avgpool
            nn.Flatten(),  # flatten
            nn.Linear(512, out_size),  # fully connected
        )

        self.loss_fn = loss_fn
        self.optimizer = optim.SGD(self.parameters(), lr=lrate, weight_decay=1e-5)

    def _make_layer(self, block, out_channels, blocks, stride=1):
        """
        Creates a resnet layer
        @param block:
        @param out_channels:
        @param blocks:
        @param stride:
        @return:
        """
        layers = [block(self.in_channels, out_channels, stride)]
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
        return self.model(x)

    def step(self, x: torch.Tensor, y: torch.Tensor):
        """
        Returns the loss of a single forward pass
        @param x:
        @param y:
        @return:
        """
        # zero the gradients
        self.optimizer.zero_grad()

        # pass the batch through the model
        y_hat = self.forward(x)

        # compute the loss
        loss = self.loss_fn(y_hat, y.to(dtype=torch.long))

        # update model
        loss.backward()
        self.optimizer.step()

        return loss.item()


def fit(
        train_dataset: Subset,
        test_dataset: Subset,
        image_dim: Size,
        epochs: int,
        lrate: float = 0.01,
        loss_fn = nn.CrossEntropyLoss(),
        batch_size: int = 50,
        save: bool = False
) -> Tuple[list[float], NumberModel]:
    """
    Trains the model and returns the losses as well as the model.
    @param train_dataset:
    @param test_dataset:
    @param image_dim:       Input dimensions of an image
    @param epochs:          Number of epochs
    @param batch_size:      Number of batches to use per epoch
    @param save:            True if the model parameters should be saved
    @return:
    """
    # get data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    # initialize net and send to device
    NetObject = NumberModel(lrate, loss_fn, image_dim[0], 10).to(device)

    losses = []
    epoch_progress_bar = tqdm(range(epochs), desc="Epochs", leave=True)
    for epoch in range(epochs):
        epoch_loss = 0.0
        for batch in train_loader:
            batch_x, batch_y = batch
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            curr_epoch_loss = NetObject.step(batch_x, batch_y)
            epoch_loss += curr_epoch_loss
            losses.append(curr_epoch_loss)
        epoch_loss /= len(train_loader)
        epoch_progress_bar.update(1)
        epoch_progress_bar.set_postfix({'Epoch Loss': epoch_loss})

    # Set the model to evaluation mode
    NetObject.eval()

    # Define a variable to store the total number of correct predictions
    total_correct = 0
    # Define a variable to store the total number of examples
    total_examples = 0

    # Iterate through the test DataLoader to evaluate misclassification rate
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

    if save:
        pth_dict = {
            "state_dict": NetObject.state_dict(),
            "lrate": lrate,
            "loss_fn": loss_fn,
            "image_dim": image_dim[0],
            "out_size": 10
        }
        torch.save(pth_dict, model_path + model_name + ".pth")
        NetObject.to(device='cpu')
        samples = torch.rand(10, 1, 50, 50)
        output = NetObject(samples)
        print(f"output.dtype: {output.dtype}")
        traced_script_module = torch.jit.trace(NetObject, samples)
        optimized_traced_model = optimize_for_mobile(traced_script_module)
        optimized_traced_model._save_for_lite_interpreter(model_path + model_name + "_mobile.pt")

    return losses, NetObject


if __name__ == "__main__":
    """
    Trains and tests the accuracy of the network
    """
    train_data, test_data, single_image_dimension = get_sudoku_dataset(verbose=True)
    params = {
        "train_dataset": train_data,
        "test_dataset": test_data,
        "image_dim": single_image_dimension,
        "epochs": 20,
        "lrate": 0.01,
        "loss_fn": nn.CrossEntropyLoss(),
        "batch_size": 50,
        "save": save_model
    }
    losses, NetObject = fit(**params)
