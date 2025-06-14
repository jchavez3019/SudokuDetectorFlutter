from dataclasses import dataclass, field
from typing import *

@dataclass
class SaveParameters:
    model_path: str = field(default="./models", metadata={"help": "Path to the saved model."})
    model_name: str = field(default="TestNumberModelForTFLite", metadata={"help": "Name of the model."})
    save_model: bool = field(default=True, metadata={"help": "True if save"})
    load_model: Optional[str] = field(default=None, metadata={"help": "Load the saved model."})

@dataclass
class Training:
    epochs: int = field(default=80, metadata={"help": "Number of epochs"})
    lrate: float = field(default=0.001, metadata={"help": "Learning rate"})
    batch_size: int = field(default=64, metadata={"help": "Batch size"})

@dataclass
class Hardware:
    device: str = field(default="cuda", metadata={"help": "Device"})

@dataclass
class Misc:
    verbose_architecture: bool = field(default=False, metadata={"help": "Verbose architecture"})

@dataclass
class Layer:
    type: str # The type of the layer
    params: Dict[str, Any] = field(default_factory=dict, metadata={"help": "Stores additional params in a dict (e.g., kernel_size, stride, etc.)"})

@dataclass
class Architecture:
    layers: List[Layer] = field(default_factory=list, metadata={"help": "List of layers"})

@dataclass
class Config:
    save_parameters: SaveParameters = field(default_factory=SaveParameters, metadata={"help": "Save parameters"})
    training: Training = field(default_factory=Training, metadata={"help": "Training configuration"})
    hardware: Hardware = field(default_factory=Hardware, metadata={"help": "Hardware configuration"})
    misc: Misc = field(default_factory=Misc, metadata={"help": "Misc configuration"})
    architecture: Architecture = field(default_factory=Architecture, metadata={"help": "Architecture configuration"})
