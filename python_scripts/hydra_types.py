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
    architecture: Any # TODO: This should be cleaned up and maybe even removed
    training: TrainingSettings = field(default_factory=lambda: TrainingSettings)
    save_parameters: SaveParametersSettings = field(default_factory=lambda: SaveParametersSettings)
    torch: TorchSettings = field(default_factory=lambda: TorchSettings)
    misc: MiscSettings = field(default_factory=lambda: MiscSettings)

@dataclass
class TrainingSettings:
    dataset_path: str
    train_test_split: float = 0.8
    epochs: int = 10
    lrate: float = 0.015
    batch_size: int = 128
    lrate_scheduler: LrateSettings = field(default_factory=lambda: LrateSettings)

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
            raise ValueError(f"training.dataset_path is {self.dataset_path} which does not exist; "
                             f"must be a valid path.")

@dataclass
class LrateSettings:
    enable: bool = False
    type: str = "cosine_annealing"
    parameters: Dict[str, Any] = field(default_factory=lambda: {})

@dataclass
class SaveParametersSettings:
    model_path: str
    model_name: str
    save_model: bool = True
    load_model: Optional[str] = None
    trace_sample: bool = False

@dataclass
class TorchSettings:
    device: str = "cpu"
    backends_cudnn_deterministic: bool = True
    use_deterministic_algorithms: bool = True

    def __post_init__(self):
        if self.device not in ['cpu', 'cuda', 'mps']:
            raise ValueError("torch.device must be in ['cpu', 'cuda', 'mps']")
        if self.device == 'cuda' and not torch.cuda.is_available():
            raise ValueError(f"torch.device={'cuda'} but cuda is not available on this device.")
        if self.device == 'mps' and not torch.backends.mps.is_available():
            raise ValueError(f"torch.device={'mps'} but mps is not available on this device.")

@dataclass
class MiscSettings:
    display_loss: bool = False