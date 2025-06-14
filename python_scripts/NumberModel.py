"""
This is a custom implementation of a Residual network. This script can also train the network.
"""
import torch
import torch.nn as nn
import torch.optim as optim
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Subset
from torch import Tensor, Size
from tqdm import tqdm
from typing import *
from collections import OrderedDict
import numpy as np
import hydra

from config.config_schema import Config, Architecture

from python_helper_functions import get_sudoku_dataset

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
            ("1_Conv2d", nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)),
            ("2_BatchNorm2d", nn.BatchNorm2d(out_channels)),
            ("3_ReLU", nn.ReLU(inplace=True)),
            ("4_Conv2d", nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)),
            ("5_BatchNorm2d", nn.BatchNorm2d(out_channels)),
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
                ("0_Conv2d", nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False)),
                ("1_BatchNorm2d", nn.BatchNorm2d(out_channels))
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
    def __init__(self, lrate: float, cfg: Architecture, loss_fn: Optional[Callable] = None):
        """
        Initializes the Neural Network model
        :param lrate:   Learning rate for the model.
        :param cfg:     Architecture configuration.
        :param loss_fn: Loss function for the model. If None, the model cannot be stepped.
        """
        super(NumberModel, self).__init__()

        self.cfg = cfg
        self.image_dimensions = cfg.image_dimensions
        self.in_channels = None
        model_layers = []
        self.residual_args = {}

        for i, layer_cfg in enumerate(cfg.layers):
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
        self.optimizer = optim.SGD(self.parameters(), lr=lrate, weight_decay=1e-5)

    def _make_layer(self, block, out_channels: int, blocks: int, stride: int = 1):
        """
        Creates a resnet layer
        :param block:
        :param out_channels:
        :param blocks:
        :param stride:
        :return:
        """
        layers = [("1_ResidualBasicBlock", block(self.in_channels, out_channels, stride))]
        self.in_channels = out_channels # update in_channels for the next blocks
        for i in range(1, blocks):
            layers.append((f"{i+1}_ResidualBasicBlock", block(self.in_channels, out_channels)))
        return nn.Sequential(OrderedDict(layers))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the model.
        :param x:   Batch of samples.
        :return:    Batch of inferences.
        """
        return self.model(x)
        # TODO: Remove this but can be useful for comparing layer outputs between torch and tf
        # module_dict = nn.ModuleDict(self.model._modules)
        # for layer_name, layer_fn in module_dict.items():
        #     x = layer_fn(x)
        # return x

    def step(self, x: torch.Tensor, y: torch.Tensor):
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

    def parse_net(self, net: NumberModel, verbose: bool = False) -> dict:
        """
        Parses a network model and creates a custom dictionary that may be later used to
        create an equivalent TensorFlow model.
        :param net:         The network model to parse.
        :param verbose:     If true, print the name of each layer and their immediate submodules.
        :return:            Returns a dictionary describing the model's parameters.
        """
        model_dict = {}
        for name, layer in net.model.named_children():
            # each layer starts with a digit followed by its type
            # which is in {ResidualBasicBlock, Conv2d, BatchNorm2d, ReLU, MaxPool2d, AvgPool2d, Flatten, and Linear}
            layer_digit, layer_type = name.split("_", 1)
            if verbose:
                print(f"Layer name: {name}")
                print(f"Layer items: {layer}")

            layer_params = self.parse_layer(name, layer_type, layer, net.residual_args)
            if len(layer_params) > 0:
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
            "NumberBlocks": residual_args.get("blocks"),  # the number of residual blocks in the layer
            "OutChannels": residual_args.get("out_channels"),  # the residual layer's number of output channels
            "Stride": residual_args.get("stride"),
            # the stride of the residual layer's first convolution in the first block
            "ResidualLayerModel": {}
        }
        # this will hold the parameters of each block in the residual layer
        res_layer_model = residual_params.get("ResidualLayerModel")

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

def get_explicit_model(net: NumberModel, verbose: bool = False) -> dict:
    """
    Creates a custom dictionary which explicitly describes the structure of the model and its parameters.
    :param net:         Network model to parse.
    :param verbose:     If true, print the name of each layer and its immediate submodules.
    :return:            A dictionary describing the structure of the model.
    """
    net_parser = NetParser()
    torch_model_dict = net_parser.parse_net(net, verbose=verbose)
    return torch_model_dict


def train_model(net: NumberModel, epochs: int, train_loader: DataLoader) -> List[float]:
    """
    Trains a model for a number of epochs given a training set.
    :param net:             Network model to train.
    :param epochs:          Number of epochs to run.
    :param train_loader:    The training data loader.
    :return:                A list of loss values over epochs.
    """
    losses = []
    epoch_progress_bar = tqdm(range(epochs), desc="Epochs", leave=True)
    for epoch in range(epochs):
        epoch_loss = 0.0
        for batch in train_loader:
            batch_x, batch_y = batch
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            curr_epoch_loss = net.step(batch_x, batch_y)
            epoch_loss += curr_epoch_loss
            losses.append(curr_epoch_loss)
        epoch_loss /= len(train_loader)
        epoch_progress_bar.update(1)
        epoch_progress_bar.set_postfix({'Epoch Loss': epoch_loss})

    return losses

def evaluate_model(net: NumberModel, test_loader: DataLoader) -> float:
    """
    Evaluates the accuracy of the model on the test-holdout set.
    :param net:             Network model to evaluate.
    :param test_loader:     Dataloader for the test set.
    :return:                Numerical accuracy on the test set.
    """
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
            outputs = net(images)

        # Get predicted labels (and ignore the values themselves)
        _, predicted = torch.max(outputs, 1)

        # Update the total number of examples
        total_examples += labels.size(0)
        # Update the total number of correct predictions
        total_correct += (predicted == labels).sum().item()

    # Calculate the accuracy
    accuracy = total_correct / total_examples

    return accuracy

def save_model(net: NumberModel, cfg: Config, verbose_architecture: bool = False):
    """
    Saves the model to the desired path.
    :param net:                     Network model to save.
    :param cfg:                     Global configuration dataclass specifying how to save the model.
    :param verbose_architecture:    If true, print the network's architecture.
    :return: 
    """
    if verbose_architecture:
        for name, layer in net.model.named_children():
            # each layer starts with a digit followed by its type
            # which is in {ResidualBasicBlock, Conv2d, BatchNorm2d, ReLU, MaxPool2d, AvgPool2d, Flatten, and Linear}
            layer_digit, layer_type = name.split("_", 1)
            print(f"Layer name: {name}")
            print(f"Layer items: {layer}")

    # We format our own state dictionary with more verbose parameters and attributes
    # instead of the default network state dictionary. This makes it easier to use this .pth
    # file to initialize a Keras model.
    custom_state_dict = get_explicit_model(net, verbose=True)
    pth_dict = {
        "custom_state_dict": custom_state_dict, # typically you would use NetObject.state_dict()
        "lrate": cfg.training.lrate,
        "loss_fn": None,
        "image_dim": cfg.architecture.image_dimensions,
        "out_size": 10,
        "test_samples": to_numpy(torch.rand(10, *cfg.architecture.image_dimensions)),
    }
    net = net.to(device=device)
    net.eval()
    samples = torch.from_numpy(pth_dict["test_samples"]).to(dtype=torch.float32, device=device)
    output = to_numpy(net(samples))
    pth_dict["test_sample_outputs"] = output
    pth_dict["state_dict"] = net.state_dict()
    torch.save(pth_dict, cfg.save_parameters.model_path + cfg.save_parameters.model_name + ".pth")

    print(f"Forward inference with test samples: \n{output}")
    print(f"output.dtype: {output.dtype}")

def fit(
        train_dataset: Subset,
        test_dataset: Subset,
        epochs: int,
        cfg: Architecture,
        lrate: float = 0.01,
        loss_fn = nn.CrossEntropyLoss(),
        batch_size: int = 50,
) -> Tuple[list[float], NumberModel]:
    """
    Trains the model and returns the losses as well as the model.
    :param train_dataset:   A randomized subset of the dataset to be used for training the model.
    :param test_dataset:    A randomized subset of the dataset to be used for evaluating the model.
    :param epochs:          Number of epochs.
    :param cfg:             Architectural configuration dataclass specifying how to build the model.
    :param lrate:           Learning rate.
    :param loss_fn:         Loss function.
    :param batch_size:      Number of batches to use per epoch.
    :return:                Returns the losses as well as the trained model.
    """
    # pass the dataset subsets to dataloaders to pick out shuffled batches of data per epoch
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    # initialize net and send to a device
    net = NumberModel(lrate, cfg, loss_fn).to(device)

    # train the model
    losses = train_model(net, epochs, train_loader)

    # Set the model to evaluation mode
    net.eval()
    for param in net.parameters():
        param.requires_grad = False

    # evaluate the model's accuracy on the holdout set
    accuracy = evaluate_model(net, test_loader)
    print('Accuracy:', accuracy)

    return losses, net

@hydra.main(config_path="./config", version_base=None)
def main(cfg: Config):
    # print the .yaml configuration settings
    print("Configuration:")
    print(OmegaConf.to_yaml(cfg))
    print("Architecture:")
    print(OmegaConf.to_yaml(cfg.architecture))

    global device
    device = cfg.hardware.device
    lrate = cfg.training.lrate
    loss_fn = nn.CrossEntropyLoss()
    if cfg.save_parameters.load_model is not None:
        pth_path = cfg.save_parameters.load_model
        pth_dict = torch.load(pth_path, map_location=device)
        _, _, single_image_dimension = get_sudoku_dataset(verbose=False)
        net = NumberModel(lrate, cfg.architecture, loss_fn).to(device)
        net.load_state_dict(pth_dict["state_dict"])
        net.eval()
        test_samples = torch.from_numpy(pth_dict["test_samples"]).to(dtype=torch.float32, device=device)
        output_samples = to_numpy(net(test_samples))
        gt_output_samples = pth_dict["test_sample_outputs"]
        # for the outputs to match, the device the model is run on should also be the same
        outputs_match = np.allclose(output_samples, gt_output_samples)
        print(f"Does this evaluation match the ground truth outputs? {outputs_match}")
    else:

        # load in the sudoku dataset
        train_data, test_data, single_image_dimension = get_sudoku_dataset(verbose=True)
        assert single_image_dimension == torch.Size(cfg.architecture.image_dimensions), \
            ("The dimensions of an image from the Sudoku dataset does not match the dimensions specified in the "
             "configuration file.")

        # fit the neural network
        params = {
            "train_dataset": train_data,
            "test_dataset": test_data,
            "epochs": cfg.training.epochs,
            "cfg": cfg.architecture,
            "lrate": lrate,
            "loss_fn": loss_fn,
            "batch_size": cfg.training.batch_size,
        }
        losses, net = fit(**params)
        
        if cfg.save_parameters.save_model:
            # save the model after training
            save_model(net, cfg, cfg.misc.verbose_architecture)

if __name__ == "__main__":
    """
    Trains and tests the accuracy of the network
    """
    # use deterministic algorithms for the training algorithm
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)

    # run the main program
    main()
