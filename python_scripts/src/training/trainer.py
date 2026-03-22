"""
Core training logic and loop utilities.
"""
import torch
import torch.nn as nn
import logging
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm
from typing import Tuple, List

log = logging.getLogger(__name__)

class ModelTrainer:
    """
    Handles the training and evaluation loop of a Neural Network.
    """
    def __init__(self, model: nn.Module, optimizer, loss_fn, scheduler=None):
        """
        Initializes the ModelTrainer.

        :param model:       The PyTorch neural network module.
        :param optimizer:   The optimizer (e.g., SGD, Adam).
        :param loss_fn:     The loss function to evaluate performance.
        :param scheduler:   Optional learning rate scheduler.
        """
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.scheduler = scheduler

    def train_step(self, x: torch.Tensor, y: torch.Tensor) -> float:
        """
        Performs a single batch forward and backward pass.

        :param x:   Batch of input samples.
        :param y:   Batch of ground-truth labels.
        :return:    The loss value over this batch.
        """
        self.optimizer.zero_grad() # zero gradients
        y_hat = self.model(x) # pass batch of inputs through the model
        loss = self.loss_fn(y_hat, y.to(dtype=torch.long)) # calculate the loss
        loss.backward() # calculate gradient of loss w.r.t. the inputs
        self.optimizer.step() # update the optimizable parameters
        return loss.item()

    def fit(self, epochs: int, dataloader: torch.utils.data.DataLoader, device: str = "cpu"
        ) -> Tuple[List[float], List[float]]:
        """
        Trains the model for a specified number of epochs.

        :param epochs:      Number of epochs to run.
        :param dataloader:  The training data loader.
        :param device:      Hardware device to process on.
        :return:            A tuple containing a list of epoch losses and a list of epoch learning rates.
        """
        losses = []
        learning_rates = []

        with logging_redirect_tqdm():
            epoch_progress_bar = tqdm(range(epochs), desc="Epochs", leave=True)

            for epoch in epoch_progress_bar:
                self.model.train()
                epoch_loss = 0.0

                for batch_x, batch_y in dataloader:
                    batch_x = batch_x.to(device)
                    batch_y = batch_y.to(device)
                    curr_batch_loss = self.train_step(batch_x, batch_y)
                    epoch_loss += curr_batch_loss

                # Average the loss over the number of batches
                epoch_loss /= len(dataloader)

                # Step the scheduler and extract the new learning rate
                if self.scheduler:
                    self.scheduler.step()
                    epoch_lrate = self.scheduler.get_last_lr()[0]
                else:
                    # No LR scheduler is used. LR is fixed, and we must fall back to reading the LR directly from
                    # the optimizer.
                    epoch_lrate = self.optimizer.param_groups[0]['lr']

                # Update the progress bar output
                epoch_progress_bar.set_postfix({
                    'Epoch Loss': f"{epoch_loss:.4f}",
                    'Epoch lrate': f"{epoch_lrate:.6f}"
                })

                losses.append(epoch_loss)
                learning_rates.append(epoch_lrate)

        return losses, learning_rates