"""
This is a custom implementation of a Residual network. This script can also train the network.
"""
import torch
import torch.nn as nn
import torch.optim as optim
from hydra.core.config_store import ConfigStore
from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Subset
from torch import Tensor
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm # prevents the standard logger from breaking the progress bar visually
from typing import *
from collections import OrderedDict
import numpy as np
import hydra
from pathlib import Path
import matplotlib.pyplot as plt
import logging

from hydra_types import (
    HydraSettings,
    SaveParametersSettings,
)

# from config.config_schema import Config, Architecture, LRateScheduler
from python_helper_functions import get_sudoku_dataset, get_lr_scheduler
from python_scripts.hydra_types import TrainingSettings

log = logging.getLogger(__name__)

# Register the ConfigStore for strict typing and help generation
cs = ConfigStore.instance()
cs.store(name="config_schema", node=HydraSettings)

print(f"Available matplotlib styles: {plt.style.available}")
plt.rcParams['text.usetex'] = True
plt.style.use("seaborn-v0_8-white")

to_numpy = lambda x : x.detach().cpu().numpy() if isinstance(x, Tensor) else x

# these are the explicit types of the layers that are used for the residual network in this script
ModuleTypes = Union[nn.modules.conv.Conv2d, nn.modules.batchnorm.BatchNorm2d, nn.modules.activation.ReLU,
    nn.modules.flatten.Flatten, nn.modules.linear.Linear, nn.modules.pooling.MaxPool2d, nn.modules.pooling.AvgPool2d,]

class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        """
        Create a basic residual block that composes a residual layer.
        :param in_channels:     Number of incoming channels.
        :param out_channels:    Number of outgoing channels.
        :param stride:          The stride to use at the first convolutional layer.
        """
        super(BasicBlock, self).__init__()

        # sublayers that compose of our feed-forward net in our block
        layers = [
            ("00_Conv2d", nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)),
            ("01_BatchNorm2d", nn.BatchNorm2d(out_channels)),
            ("02_ReLU", nn.ReLU(inplace=True)),
            ("03_Conv2d", nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)),
            ("04_BatchNorm2d", nn.BatchNorm2d(out_channels)),
        ]
        self.ffn = nn.Sequential(OrderedDict(layers))

        # If the stride is greater than 1, then the output will have a smaller image shape than the input,
        # so the residual connection requires the input to be modified (so we just downsample it).
        # When the number of output channels is different from the input channels, we also have an issue where
        # we cannot directly add the output of the block back to the input, so we need to transform the input
        # to have the same number of channels.
        # Both of these issues can be resolved by using a convolutional layer.
        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(OrderedDict([
                ("00_Conv2d", nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False)),
                ("01_BatchNorm2d", nn.BatchNorm2d(out_channels))
            ]))

    def forward(self, x):
        identity = x
        out = self.ffn(x)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = nn.functional.relu(out)

        return out


class NumberModel(nn.Module):
    def __init__(self, lrate: float, architecture_cfg, loss_fn: Optional[Callable] = None,
                 scheduler_type: Optional[str] = None, scheduler_params: Optional[dict] = None):
        """
        Initializes the Neural Network model.
        :param lrate:              Learning rate for the model.
        :param architecture_cfg:   Architecture configuration.
        :param loss_fn:            Loss function for the model. If None, the model cannot be stepped.
        :param scheduler_type:     The learning rate scheduler to use.
        """
        super(NumberModel, self).__init__()

        self.architecture_cfg = architecture_cfg
        self.image_dimensions = architecture_cfg.image_dimensions
        self.in_channels = None
        model_layers = []
        self.residual_args = {}

        for i, layer_cfg in enumerate(architecture_cfg.layers):
            layer_type = layer_cfg.type
            layer_params = layer_cfg.params
            layer_name = f"{i:02d}_{layer_type}"

            if layer_type == "Conv2d":
                self.in_channels = layer_params["out_channels"]
                layer = nn.Conv2d(**layer_params)

            elif layer_type == "BatchNorm2d":
                layer = nn.BatchNorm2d(**layer_params)

            elif layer_type == "ReLU":
                layer = nn.ReLU(inplace=layer_params.get("inplace", True))

            elif layer_type == "MaxPool2d":
                layer = nn.MaxPool2d(**layer_params)

            elif layer_type == "ResidualLayer":
                layer = self._make_layer(
                    BasicBlock,
                    out_channels=layer_params["out_channels"],
                    blocks=layer_params["blocks"],
                    stride=layer_params["stride"]
                )
                self.residual_args[layer_name] = {
                    "out_channels": layer_params["out_channels"],
                    "blocks": layer_params["blocks"],
                    "stride": layer_params["stride"]
                }

            elif layer_type == "AvgPool2d":
                layer = nn.AdaptiveAvgPool2d(tuple(layer_params["output_size"]))

            elif layer_type == "Flatten":
                layer = nn.Flatten()

            elif layer_type == "Linear":
                layer = nn.Linear(layer_params["in_features"], layer_params["out_features"])

            else:
                error_msg = f"Unknown layer type: {layer_type}"
                raise ValueError(error_msg)

            model_layers.append((layer_name, layer))

        self.model = nn.Sequential(OrderedDict(model_layers))

        self.loss_fn = loss_fn
        self.lrate = lrate
        self.optimizer = optim.SGD(self.parameters(), lr=lrate, weight_decay=1e-5)

        self.scheduler = None
        self.scheduler_type = scheduler_type
        if scheduler_type is not None:
            self.scheduler = get_lr_scheduler(scheduler_type, self.optimizer, scheduler_params)


    def _make_layer(self, block, out_channels: int, blocks: int, stride: int = 1):
        """
        Creates a resnet layer
        :param block:
        :param out_channels:
        :param blocks:
        :param stride:
        :return:
        """
        layers = [("00_ResidualBasicBlock", block(self.in_channels, out_channels, stride))]
        self.in_channels = out_channels # update in_channels for the next blocks
        for i in range(1, blocks):
            layers.append((f"{i:02d}_ResidualBasicBlock", block(self.in_channels, out_channels)))
        return nn.Sequential(OrderedDict(layers))

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass of the model.
        :param x:   Batch of samples.
        :return:    Batch of inferences.
        """
        return self.model(x)        
        
    def forward_trace(self, x: Tensor):
        """
        Forward pass of the model where each layer's output is recorded.
        :param x:   Batch of samples.
        :return:    Batch of inferences and trace of each layer's results.
        """
        trace = {}
        module_dict = nn.ModuleDict(self.model._modules)
        for layer_name, layer_fn in module_dict.items():
            x = layer_fn(x)
            trace[layer_name] = to_numpy(x)
        return x, trace

    def step(self, x: Tensor, y: Tensor):
        """
        Returns the loss of a single forward pass.
        :param x:   Batch of input samples.
        :param y:   Batch of ground-truth labels.
        :return:    Returns the loss over this batch step.
        """
        if self.loss_fn is None:
            error_msg = "Network object was not initialized with a loss function, the model cannot be stepped."
            raise Exception(error_msg)
        
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

    def scheduler_step(self) -> float:
        """If used, steps the learning rate scheduler and returns the learning rate used prior to the step."""
        if self.scheduler is not None:
            self.scheduler.step()
            return self.scheduler.get_last_lr()[0]
        else:
            return self.lrate

class NetParser:
    """
    Currently, it is challenging to convert a PyTorch model -> Keras model -> TFLite model. To circumvent this, we
    explicitly parse the PyTorch model into a dictionary model that can be used to create the same model
    programmatically with Keras. Afterward, it is simple to convert a Keras model -> TFLite model.

    This class defines what parameters are necessary for model reconstruction in Keras.
    """
    def __init__(self):
        self.parsers = {
            "Conv2d": self._parse_conv2d,
            "Linear": self._parse_linear,
            "ReLU": self._parse_relu,
            "BatchNorm2d": self._parse_batchnorm2d,
            "MaxPool2d": self._parse_maxpool2d,
            "AvgPool2d": self._parse_avgpool2d,
            "Flatten": self._parse_flatten,
            "ResidualLayer": self._parse_residual_layer,
        }

    def parse_net(self, net: NumberModel) -> dict:
        """
        Parses a network model and creates a custom dictionary that may be later used to
        create an equivalent TensorFlow model.
        :param net:         The network model to parse.
        :return:            Returns a dictionary describing the model's parameters.
        """
        model_dict = {}
        for name, layer in net.model.named_children():
            # each layer starts with a digit followed by its type
            # which is in {ResidualBasicBlock, Conv2d, BatchNorm2d, ReLU, MaxPool2d, AvgPool2d, Flatten, and Linear}
            layer_digit, layer_type = name.split("_", 1)
            log.debug(f"Layer name: {name}")
            log.debug(f"Layer items: {layer}")

            layer_params = self.parse_layer(name, layer_type, layer, net.residual_args)
            model_dict[name] = layer_params

        return model_dict

    def parse_layer(self, layer_name: str, layer_type: str, layer: ModuleTypes,
                    residual_args: Optional[dict] = None) -> dict:
        """
        Parses an individual layer and returns a custom dictionary describing the layer's submodules and parameters.
        :param layer_name:      The custom name of the layer.
        :param layer_type:      The type of layer, e.g., linear, convolutional, relu, etc.
        :param layer:           The PyTorch layer module.
        :param residual_args:   Necessary arguments for a residual layer.
        :return:
        """
        if layer_type not in self.parsers:
            raise ValueError(f"Unknown layer type: {layer_type}")
        return self.parsers[layer_type](layer_name, layer, residual_args)

    @staticmethod
    def _parse_conv2d(name, layer: nn.Conv2d, *args):
        """Parse the parameters of a 2D convolutional layer"""
        return {
            "weight": to_numpy(layer.weight), "bias": to_numpy(layer.bias),
            "stride": layer.stride, "padding": layer.padding, "dilation": layer.dilation,
            "groups": layer.groups, "kernel_size": layer.kernel_size,
            "in_channels": layer.in_channels, "out_channels": layer.out_channels
        }

    @staticmethod
    def _parse_linear(name, layer: nn.Linear, *args):
        """Parse the parameters of a linear layer"""
        return {"weight": to_numpy(layer.weight), "bias": to_numpy(layer.bias)}

    @staticmethod
    def _parse_relu(name, layer: nn.ReLU, *args):
        """Nothing to parse for a relu layer, return an empty dictionary"""
        return {}

    @staticmethod
    def _parse_batchnorm2d(name, layer: nn.BatchNorm2d, *args):
        """Parse the parameters of a 2D batch normalization layer"""
        return {
            "weight": to_numpy(layer.weight), "bias": to_numpy(layer.bias),
            "running_mean": to_numpy(layer.running_mean),
            "running_var": to_numpy(layer.running_var),
            "eps": layer.eps, "momentum": layer.momentum
        }

    @staticmethod
    def _parse_maxpool2d(name, layer: nn.MaxPool2d, *args):
        """Parse the parameters of a 2D max pool layer"""
        return {
            "kernel_size": layer.kernel_size,
            "stride": layer.stride,
            "padding": layer.padding,
        }

    @staticmethod
    def _parse_avgpool2d(name, layer: nn.AdaptiveAvgPool2d, *args):
        """Parse the parameters of a 2D average pool layer"""
        return {"output_size": layer.output_size}

    @staticmethod
    def _parse_flatten(name, layer: nn.Flatten, *args):
        """Nothing to parse for a flattening/vectorizing layer, return an empty dictionary"""
        return {}

    def _parse_residual_layer(self, name, layer: nn.Sequential, residual_args: dict):
        """Parse the parameters of our custom residual layer"""

        if residual_args is None:
            raise ValueError("Residual arguments are required to parse residual layers.")

        # returns the number of immediate children/layers of a module
        _get_num_layers = lambda x: sum([1 for _ in x.named_children()])

        num_blocks = _get_num_layers(layer)  # the number of residual blocks in the layer
        # parse the parameters of the residual layer
        residual_params = {
            "NumberBlocks": residual_args[name]["blocks"],  # the number of residual blocks in the layer
            "OutChannels": residual_args[name]["out_channels"],  # the residual layer's number of output channels
            "Stride": residual_args[name]["stride"],
            # the stride of the residual layer's first convolution in the first block
            "ResidualLayerModel": {}
        }
        # this will hold the parameters of each block in the residual layer
        res_layer_model = residual_params["ResidualLayerModel"]

        def _get_parameters(module: nn.modules.container.Sequential) -> dict:
            ret_params = {}
            for sub_layer_name, sub_layer in module.named_children():
                layer_digit, layer_type = sub_layer_name.split("_", 1)
                layer_params = self.parse_layer(sub_layer_name, layer_type, sub_layer)
                if len(layer_params) > 0:
                    ret_params[sub_layer_name] = layer_params

            return ret_params

        for i, (basic_block_name, basic_block) in zip(range(1, 1 + num_blocks), layer.named_children()):

            # the digit is the sequential order of the block in the layer; the name should always be "ResidualBasicBlock"
            block_digit, block_name = basic_block_name.split("_", 1)
            assert block_name == "ResidualBasicBlock", "The residual layer is not being parsed correctly."

            # get the attributes of the BasicBlock in the residual layer that has optimizable parameters
            ffn_attr = getattr(basic_block, "ffn")
            downsample_attr = getattr(basic_block, "downsample", None)  # the downsample attribute is optional

            # iterate through the layers of the sub feedforward network
            curr_ffn = _get_parameters(ffn_attr)

            # if the residual layer performs downsampling, get its parameters
            downsample = None
            if downsample_attr is not None:
                downsample = _get_parameters(downsample_attr)

            res_layer_model[basic_block_name] = {
                "ffn": curr_ffn,
                "downsample": downsample,
            }

        return residual_params

def get_explicit_model(net: NumberModel) -> dict:
    """
    Creates a custom dictionary which explicitly describes the structure of the model and its parameters.
    :param net:         Network model to parse.
    :return:            A dictionary describing the structure of the model.
    """
    net_parser = NetParser()
    torch_model_dict = net_parser.parse_net(net)
    return torch_model_dict


def train_model(net: NumberModel, epochs: int, train_loader: DataLoader) -> Tuple[List[float], List[float]]:
    """
    Trains a model for a specified number of epochs given a training dataset.

    :param net: Network model to train.
    :param epochs: Number of epochs to run.
    :param train_loader: The training data loader.
    :return: A tuple containing a list of epoch losses and a list of epoch learning rates.
    """
    losses = []
    learning_rates = []

    with logging_redirect_tqdm():
        epoch_progress_bar = tqdm(range(epochs), desc="Epochs", leave=True)
        for epoch in epoch_progress_bar:
            epoch_loss = 0.0
            for batch_x, batch_y in train_loader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)
                curr_epoch_loss = net.step(batch_x, batch_y)
                epoch_loss += curr_epoch_loss

            epoch_loss /= len(train_loader)
            epoch_lrate = net.scheduler_step()

            epoch_progress_bar.set_postfix({'Epoch Loss': f"{epoch_loss:.4f}", 'Epoch lrate': f"{epoch_lrate:.6f}"})

            losses.append(epoch_loss)
            learning_rates.append(epoch_lrate)

    return losses, learning_rates

def evaluate_model(net: NumberModel, test_loader: DataLoader) -> float:
    """
    Evaluates the accuracy of the model on the test-holdout set.
    :param net: Network model to evaluate.
    :param test_loader: Dataloader for the test set.
    :return: Numerical accuracy on the test set.
    :rtype: float
    """
    total_correct = 0
    total_examples = 0

    # Iterate through the test DataLoader to evaluate misclassification rate
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)

        with torch.no_grad():  # No need to compute gradients during inference
            outputs = net(images)

        # Get predicted labels (and ignore the values themselves)
        _, predicted = torch.max(outputs, 1)

        # Update the total number of examples
        total_examples += labels.size(0)
        # Update the total number of correct predictions
        total_correct += (predicted == labels).sum().item()

    accuracy = total_correct / total_examples
    return accuracy

def save_model(net: NumberModel, architecture: Any, training: TrainingSettings,
               save_parameters: SaveParametersSettings) -> None:
    """
    Saves the PyTorch model state dict and attempts to convert it to a TFLite edge model.

    :param net: Network model to save.
    :param architecture: Architecture configuration object containing image dimensions.
    :param training_settings: Configuration object containing training hyperparameters.
    :param save_parameters: Settings dictating where and how to save the model.
    """
    if log.isEnabledFor(logging.DEBUG):
        for name, layer in net.model.named_children():
            # each layer starts with a digit followed by its type
            # which is in {ResidualBasicBlock, Conv2d, BatchNorm2d, ReLU, MaxPool2d, AvgPool2d, Flatten, and Linear}
            layer_digit, layer_type = name.split("_", 1)
            log.debug(f"Layer name: {name}")
            log.debug(f"Layer items: {layer}")

    # We format our own state dictionary with more verbose parameters and attributes
    # instead of the default network state dictionary. This makes it easier to use this .pth
    # file to initialize a Keras model.
    custom_state_dict = get_explicit_model(net)
    torch_test_samples = torch.rand(1, *architecture.image_dimensions).to(dtype=torch.float32, device=device)
    pth_dict = {
        "custom_state_dict": custom_state_dict, # typically you would use NetObject.state_dict()
        "lrate": training.lrate,
        "loss_fn": None,
        "image_dim": architecture.image_dimensions,
        "out_size": 10,
        "test_samples": to_numpy(torch_test_samples),
    }
    if save_parameters.trace_sample:
        # pass our batch of randomized samples and trace the network's output throughout each layer
        pth_dict["trace"] = net.forward_trace(torch_test_samples)[1]
    net = net.to(device=device)
    net.eval()
    output = to_numpy(net(torch_test_samples))
    pth_dict["test_sample_outputs"] = output
    pth_dict["state_dict"] = net.state_dict()

    save_path = Path(save_parameters.model_path) / f"{save_parameters.model_name}.pth"
    torch.save(pth_dict, save_path)
    log.info(f"Torch model saved to path: {save_path}")

    try:
        import litert_torch

        log.info("Starting TFLite conversion...")

        # DIAGNOSTIC: Print what we're converting with
        log.debug(f"architecture.image_dimensions: {architecture.image_dimensions}")
        log.debug(f"torch_test_samples.shape: {torch_test_samples.shape}")

        net_cpu = net.cpu()
        test_samples_cpu = torch_test_samples.cpu()

        # DIAGNOSTIC: Verify input shape before conversion
        log.debug(f"test_samples_cpu.shape before conversion: {test_samples_cpu.shape}")

        edge_model = litert_torch.convert(
            net_cpu.eval(),
            (test_samples_cpu,)
        )

        # DIAGNOSTIC: Test with the same input used for conversion
        tf_output = edge_model(test_samples_cpu)
        log.debug(f"✓ Forward inference with conversion samples works")
        log.debug(f"  Output shape: {tf_output.shape}")

        # DIAGNOSTIC: Now test with 50x50 input (what Flutter provides)
        test_50x50 = torch.rand(1, 1, 50, 50).cpu()
        log.debug(f"Testing with 50×50 input (Flutter size): {test_50x50.shape}")
        try:
            output_50x50 = edge_model(test_50x50)
            log.debug(f"✓ Model accepts 50×50 input!")
            log.debug(f"  Output shape: {output_50x50.shape}")
        except Exception as e:
            log.debug(f"✗ Model REJECTS 50×50 input: {e}")
            log.debug(f"  This confirms the dimension mismatch!")

        tflite_path = str(save_path).replace(".pth", ".tflite")
        edge_model.export(tflite_path)
        log.info(f"✓ TFLite model saved to: {tflite_path}")

        # DIAGNOSTIC: Verify the exported .tflite file's input shape
        from ai_edge_litert.interpreter import Interpreter
        interpreter = Interpreter(model_path=tflite_path)
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        log.info(f"=== EXPORTED TFLITE MODEL INFO ===")
        log.info(f"Input shape: {input_details[0]['shape']}")
        log.info(f"Input dtype: {input_details[0]['dtype']}")
        log.info(f"Output shape: {output_details[0]['shape']}")

        # Calculate expected number of elements
        input_shape = input_details[0]['shape']
        expected_elements = np.prod(input_shape)
        log.info(f"Expected input elements: {expected_elements}")
        log.info(f"Flutter provides: {1 * 50 * 50 * 1} = 2500 elements")

        if expected_elements != 2500:
            log.warning(f"⚠️  MISMATCH CONFIRMED! ⚠️")
            log.warning(f"The TFLite model expects {expected_elements} elements")
            log.warning(f"But Flutter provides 2500 elements (50×50)")
            log.warning(f"You need to change architecture.image_dimensions to [1, 50, 50]")
        else:
            log.info(f"✓ Input dimensions match! Model should work in Flutter.")
    except Exception as e:
        log.error(f"✗ TFLite conversion failed: {e}")
        import traceback
        traceback.print_exc()

    log.info(f"Forward inference in PyTorch with test samples: \n{output}")
    log.info(f"output.dtype: {output.dtype}")

def fit(
        train_dataset: Subset,
        test_dataset: Subset,
        epochs: int,
        architecture: Any,
        lrate: float = 0.01,
        loss_fn = nn.CrossEntropyLoss(),
        batch_size: int = 50,
        scheduler_type: Optional[str] = None,
        scheduler_params: Optional[Dict[str, Any]] = None,
) -> Tuple[List[float], List[float], NumberModel]:
    """
    Initializes and trains the model, returning the loss history and the model itself.

    :param train_dataset: Subset of the dataset used for training.
    :param test_dataset: Subset of the dataset used for evaluating.
    :param epochs: Number of epochs to train.
    :param architecture: Architecture configuration object.
    :param lrate: Learning rate for the optimizer, defaults to 0.01.
    :param loss_fn: Loss function, defaults to CrossEntropyLoss.
    :param batch_size: Number of samples per batch, defaults to 50.
    :param scheduler_type: String identifier for the learning rate scheduler.
    :param scheduler_params: Kwargs for the scheduler.
    :return: A tuple containing lists of losses, learning rates, and the trained model.
    """
    # pass the dataset subsets to dataloaders to pick out shuffled batches of data per epoch
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    # initialize net and send to a device
    net = NumberModel(lrate, architecture, loss_fn, scheduler_type=scheduler_type, scheduler_params=scheduler_params).to(device)

    # train the model
    losses, learning_rates = train_model(net, epochs, train_loader)

    # Set the model to evaluation mode
    net.eval()
    for param in net.parameters():
        param.requires_grad = False

    # evaluate the model's accuracy on the holdout set
    accuracy = evaluate_model(net, test_loader)
    log.info(f'Accuracy: {accuracy}')

    return losses, learning_rates, net

@hydra.main(config_path="./config", version_base=None)
def main(cfg: HydraSettings):
    """
    Main entry point for the training pipeline handled by Hydra.

    :param cfg: Strongly typed configuration object populated by Hydra.
    :type cfg: HydraSettings
    """
    error_file_handler = logging.FileHandler(Path(HydraConfig.get().runtime.output_dir) / 'error.log')
    error_file_handler.setLevel(logging.ERROR)
    log.addHandler(error_file_handler)
    try:
        log.info("Start of training.")
        log.info(f"Configuration:\n{OmegaConf.to_yaml(cfg)}")

        # use deterministic algorithms for the training algorithm
        torch.backends.cudnn.deterministic = cfg.torch.backends_cudnn_deterministic
        torch.use_deterministic_algorithms(cfg.torch.use_deterministic_algorithms)

        global device
        device = cfg.torch.device
        lrate = cfg.training.lrate
        if cfg.save_parameters.load_model is not None:
            pth_path = cfg.save_parameters.load_model
            pth_dict = torch.load(pth_path, map_location=device)
            _, _, _, single_image_dimension = get_sudoku_dataset(
                dataset_path=cfg.training.dataset_path, split=cfg.training.train_test_split
            )
            net = NumberModel(lrate, cfg.architecture, None,
                              scheduler_type=cfg.training.lrate_scheduler.type,
                              scheduler_params=cfg.training.lrate_scheduler.parameters,
                              ).to(device)
            net.load_state_dict(pth_dict["state_dict"])
            net.eval()
            test_samples = torch.from_numpy(pth_dict["test_samples"]).to(dtype=torch.float32, device=device)
            output_samples = to_numpy(net(test_samples))
            gt_output_samples = pth_dict["test_sample_outputs"]
            # for the outputs to match, the device the model is run on should also be the same
            outputs_match = np.allclose(output_samples, gt_output_samples)
            log.info(f"Does this evaluation match the ground truth outputs? {outputs_match}")
        else:

            # load in the sudoku dataset
            sudoku_dataset, train_data, test_data, single_image_dimension = get_sudoku_dataset(
                dataset_path=cfg.training.dataset_path, split=cfg.training.train_test_split
            )
            assert single_image_dimension == torch.Size(cfg.architecture.image_dimensions), \
                ("The dimensions of an image from the Sudoku dataset does not match the dimensions specified in the "
                 "configuration file.")

            # fit the neural network
            params = {
                "train_dataset": train_data,
                "test_dataset": test_data,
                "epochs": cfg.training.epochs,
                "architecture": cfg.architecture,
                "lrate": lrate,
                "loss_fn": nn.CrossEntropyLoss(weight=sudoku_dataset.weights),
                "batch_size": cfg.training.batch_size,
                "scheduler_type": cfg.training.lrate_scheduler.type,
                "scheduler_params": cfg.training.lrate_scheduler.parameters,
            }
            losses, learning_rates, net = fit(**params)

            if cfg.misc.display_loss:
                # if true, show the resulting training loss
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

            if cfg.save_parameters.save_model:
                # save the model after training
                save_model(net, cfg.architecture, cfg.training, cfg.save_parameters)
            log.info("Done training.")
    except Exception:
        log.exception("Training ended with an unhandled exception.")
        raise

if __name__ == "__main__":
    main()
