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
model_path = "models/"
model_name = "NumberModel_ReLU_NN_v0"
device = torch.device('cuda')


def hook(module, grad_input, grad_output):
    print(f'grad_input: {grad_input[0].shape}')
    print(f'grad_output{grad_output[0].shape}')

class NumberModelReLUNN(nn.Module):
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
        super(NumberModelReLUNN, self).__init__()

        self.model = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False),  # conv1
            nn.BatchNorm2d(64),  # bn1
            nn.ReLU(),  # relu
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),  # maxpool
            nn.Flatten(),
            nn.Linear(10816, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, out_size),  # fully connected
        )

        # register hook to print gradients
        # self.model[5].register_full_backward_hook(hook)

        self.loss_fn = loss_fn
        self.optimizer = optim.SGD(self.parameters(), lr=lrate, weight_decay=1e-5)


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
) -> Tuple[list[float], NumberModelReLUNN]:
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
    NetObject = NumberModelReLUNN(lrate, loss_fn, image_dim[0], 10).to(device)

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
        "epochs": 1,
        "lrate": 0.01,
        "loss_fn": nn.CrossEntropyLoss(),
        "batch_size": 50,
        "save": save_model
    }
    losses, NetObject = fit(**params)
