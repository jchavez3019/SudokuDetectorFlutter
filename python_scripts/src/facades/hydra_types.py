from __future__ import annotations
"""
Hydra Datatypes to validate the imported .yaml file.
"""
from dataclasses import dataclass, field
from typing import Optional, Any, Dict
from pathlib import Path
import torch
import logging

log = logging.getLogger(__name__)

@dataclass
class HydraSettings:
    architecture: Any = field(metadata={"help": "Model architecture definition. Parsed via Hydra's _target_."})
    save_parameters: SaveParametersSettings = field(default_factory=lambda:SaveParametersSettings)
    training: TrainingSettings = field(default_factory=lambda:TrainingSettings)
    torch: TorchSettings = field(default_factory=lambda:TorchSettings)
    misc: MiscSettings = field(default_factory=lambda:MiscSettings)

@dataclass
class TrainingSettings:
    dataset_path: str = field(
        metadata={"help": "Absolute or relative path to the root directory of the dataset."}
    )
    train_test_split: float = field(
        default=0.8,
        metadata={"help": "Fraction of the dataset to be used for training (0.0 to 1.0)."}
    )
    epochs: int = field(
        default=10,
        metadata={"help": "Number of complete passes through the training dataset."}
    )
    lrate: float = field(
        default=0.015,
        metadata={"help": "Initial learning rate for the optimizer."}
    )
    batch_size: int = field(
        default=128,
        metadata={"help": "Number of samples processed before the model is updated."}
    )
    lrate_scheduler: LrateSettings = field(default_factory=lambda:LrateSettings)

    def __post_init__(self):
        if self.epochs < 0:
            raise ValueError(f"training.epochs is {self.epochs}; must be nonnegative.")
        if self.batch_size < 0:
            raise ValueError(f"training.batch_size is {self.batch_size}; must be nonnegative.")
        if not (0 < self.lrate < 1):
            raise ValueError(f"training.lrate is {self.lrate}; must be in the range (0, 1).")
        if not (0 < self.train_test_split < 1):
            raise ValueError(f"training.train_test_split is {self.train_test_split}; must be in the range (0, 1).")
        if not Path(self.dataset_path).is_dir():
            raise ValueError(f"training.dataset_path is {self.dataset_path} which does not exist.")

@dataclass
class LrateSettings:
    enable: bool = field(default=False, metadata={"help": "Enable learning rate scheduling."})
    type: str = field(default="cosine_annealing", metadata={"help": "Type of scheduler to use."})
    parameters: Dict[str, Any] = field(default_factory=dict, metadata={"help": "Kwargs for the scheduler."})

@dataclass
class SaveParametersSettings:
    model_path: str = field(metadata={"help": "Directory where the model artifacts will be saved."})
    model_name: str = field(metadata={"help": "Base filename for the saved model (without extension)."})
    save_model: bool = field(default=True, metadata={"help": "Whether to save the model after training."})
    load_model: Optional[str] = field(default=None, metadata={"help": "Path to a .pth file to load for evaluation."})
    trace_sample: bool = field(default=False, metadata={"help": "Trace and save layer outputs for a sample."})

@dataclass
class TorchSettings:
    device: str = field(default="cpu", metadata={"help": "Hardware device to use: 'cpu', 'cuda', or 'mps'."})
    backends_cudnn_deterministic: bool = field(default=True, metadata={"help": "Enforce deterministic CuDNN."})
    use_deterministic_algorithms: bool = field(default=True, metadata={"help": "Enforce deterministic PyTorch algorithms."})

    def __post_init__(self):
        if self.device not in ['cpu', 'cuda', 'mps']:
            raise ValueError("torch.device must be in ['cpu', 'cuda', 'mps']")
        if self.device == 'cuda' and not torch.cuda.is_available():
            raise ValueError("torch.device='cuda' but cuda is not available on this device.")
        if self.device == 'mps' and not torch.backends.mps.is_available():
            raise ValueError("torch.device='mps' but mps is not available on this device.")

@dataclass
class MiscSettings:
    display_loss: bool = field(default=False, metadata={"help": "Plot and display loss/lrate graphs after training."})