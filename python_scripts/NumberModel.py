"""
Entry point for training and evaluating the NumberModel pipeline.
"""
import logging
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import hydra
from hydra.core.config_store import ConfigStore
from hydra.utils import instantiate
from omegaconf import OmegaConf

# Import typed configurations
from src.facades.hydra_types import HydraSettings

# Import modular training classes and functions
from src.training.trainer import ModelTrainer
from src.training.exporter import save_model
from src.training.training_utils import get_sudoku_dataset, get_lr_scheduler

log = logging.getLogger(__name__)

# Register the ConfigStore for strict typing and help generation
cs = ConfigStore.instance()
cs.store(name="config_schema", node=HydraSettings)

# Set up matplotlib styling
try:
    plt.style.use("seaborn-v0_8-white")
    plt.rcParams['text.usetex'] = True
except Exception:
    log.warning("Seaborn style or LaTeX not available. Falling back to default matplotlib settings.")


@hydra.main(version_base=None, config_path="config", config_name="original_config")
def main(cfg: HydraSettings):
    """
    Main execution pipeline.

    :param cfg: The strongly typed Hydra configuration object.
    """
    try:
        log.info("Initializing NumberModel Pipeline...")
        log.info(f"Configuration:\n{OmegaConf.to_yaml(cfg)}")

        # 1. Instantiate the architecture dynamically via Hydra's _target_
        model = instantiate(cfg.architecture)
        model = model.to(cfg.torch.device)

        # 2. Setup Data
        log.info(f"Loading dataset from {cfg.training.dataset_path}...")
        dataset, train_data, test_data, _ = get_sudoku_dataset(
            cfg.training.dataset_path,
            cfg.training.train_test_split
        )

        train_loader = torch.utils.data.DataLoader(
            train_data,
            batch_size=cfg.training.batch_size,
            shuffle=True
        )

        # 3. Setup Optimizer, Loss, and Scheduler
        optimizer = optim.SGD(
            model.parameters(),
            lr=cfg.training.lrate,
            weight_decay=1e-5
        )

        # Incorporate dataset weights into the loss function to handle class imbalances
        loss_fn = nn.CrossEntropyLoss(weight=dataset.weights.to(cfg.torch.device))

        scheduler = None
        if cfg.training.lrate_scheduler.enable:
            scheduler = get_lr_scheduler(
                cfg.training.lrate_scheduler.type,
                optimizer,
                cfg.training.lrate_scheduler.parameters
            )

        # Initialize the custom trainer object
        trainer = ModelTrainer(model, optimizer, loss_fn, scheduler)

        # 4. Train Loop
        log.info(f"Starting training for {cfg.training.epochs} epochs on {cfg.torch.device}...")
        losses, learning_rates = trainer.fit(
            epochs=cfg.training.epochs,
            dataloader=train_loader,
            device=cfg.torch.device
        )

        # 5. Graphing
        if cfg.misc.display_loss:
            fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(18, 14))
            ax1, ax2 = np.ravel(axs)

            ax1.plot(losses)
            ax1.set_xlabel("Epoch")
            ax1.set_ylabel("Loss")
            ax1.set_title("Training Loss")
            ax1.grid()

            ax2.plot(learning_rates)
            ax2.set_xlabel("Epoch")
            ax2.set_ylabel("Learning Rate")
            ax2.set_title("Training Learning Rates")
            ax2.grid()

            plt.show()

        # 6. Save Model and Export to TFLite
        if cfg.save_parameters.save_model:
            log.info("Saving model and exporting to TFLite...")
            save_model(
                net=model,
                architecture=cfg.architecture,
                training=cfg.training,
                save_parameters=cfg.save_parameters,
                device=cfg.torch.device
            )

        log.info("Pipeline execution completed successfully.")

    except Exception:
        log.exception("Training pipeline ended with an unhandled exception.")
        raise


if __name__ == "__main__":
    main()