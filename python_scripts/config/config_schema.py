from dataclasses import dataclass, field
from typing import *

@dataclass
class SaveParameters:
    model_path: str = field(default="./models", metadata={"help": "Directory to save the model."})
    model_name: str = field(default="TestNumberModelForTFLite", metadata={"help": "Name of the model."})
    save_model: bool = field(default=True, metadata={"help": "Set flag if the model should be saved."})
    load_model: Optional[str] = field(default=None, metadata={"help": "Instead of saving a new model, load an existing model."})

@dataclass
class Training:
    epochs: int = field(default=80, metadata={"help": "Number of training epochs."})
    lrate: float = field(default=0.001, metadata={"help": "Learning rate."})
    batch_size: int = field(default=64, metadata={"help": "Batch size per epcch."})

@dataclass
class Hardware:
    device: str = field(default="cuda", metadata={"help": "Device to use (e.g., 'cpu' or 'cuda'; default is 'cuda')."})

@dataclass
class Misc:
    verbose_architecture: bool = field(default=False, metadata={"help": "Set flag if the model architecture should be printed. This helps get a high-level view "
    #                          "of the modules that comprise our custom residual network."})

@dataclass
class Layer:
    type: str # The type of the layer
    params: Dict[str, Any] = field(default_factory=dict, metadata={"help": "Stores additional params in a dict (e.g., kernel_size, stride, etc.)"})

@dataclass
class Architecture:
    layers: List[Layer] = field(default_factory=list, metadata={"help": "List of sequential layers of the residual network."})

@dataclass
class Config:
    save_parameters: SaveParameters = field(default_factory=SaveParameters, metadata={"help": "Save parameters"})
    training: Training = field(default_factory=Training, metadata={"help": "Training configuration"})
    hardware: Hardware = field(default_factory=Hardware, metadata={"help": "Hardware configuration"})
    misc: Misc = field(default_factory=Misc, metadata={"help": "Misc configuration"})
    architecture: Architecture = field(default_factory=Architecture, metadata={"help": "Architecture configuration"})
