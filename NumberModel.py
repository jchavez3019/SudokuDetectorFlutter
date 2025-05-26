import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Subset
from torch import Tensor, Size
from tqdm import tqdm
from typing import *
from torch.utils.mobile_optimizer import optimize_for_mobile
import argparse
from collections import OrderedDict
import numpy as np

from python_helper_functions import get_sudoku_dataset

torch.backends.cudnn.deterministic = True
torch.use_deterministic_algorithms(True)

to_numpy = lambda x : x.detach().cpu().numpy() if isinstance(x, Tensor) else x

# these are the explicit types of the layers that are used for the residual network in this script
ModuleTypes = Union[nn.modules.conv.Conv2d, nn.modules.batchnorm.BatchNorm2d, nn.modules.activation.ReLU,
    nn.modules.flatten.Flatten, nn.modules.linear.Linear, nn.modules.pooling.MaxPool2d, nn.modules.pooling.AvgPool2d,]

def parse_layer(layer_name: str, layer_type: str, layer: ModuleTypes, residual_args: Optional[dict] = None) -> dict:
    """

    :param layer_type:
    :param layer:
    :return:
    """
    if layer_type == "Conv2d":
        assert isinstance(layer, nn.modules.conv.Conv2d)
        return {"weight": to_numpy(layer.weight), "bias": to_numpy(layer.bias), "stride": layer.stride,
                "padding": layer.padding, "dilation": layer.dilation, "groups": layer.groups,
                "kernel_size": layer.kernel_size, "in_channels": layer.in_channels, "out_channels": layer.out_channels}
    elif layer_type == "Linear":
        assert isinstance(layer, nn.Linear)
        return {"weight": to_numpy(layer.weight), "bias": to_numpy(layer.bias)}
    elif layer_type == "ReLU":
        assert isinstance(layer, nn.modules.activation.ReLU)
        return {}
    elif layer_type == "BatchNorm2d":
        assert isinstance(layer, nn.modules.batchnorm.BatchNorm2d)
        return {"weight": to_numpy(layer.weight), "bias": to_numpy(layer.bias),
                "running_mean": to_numpy(layer.running_mean), "running_var": to_numpy(layer.running_var),
                "eps": layer.eps, "momentum": layer.momentum}
    elif layer_type == "MaxPool2d":
        assert isinstance(layer, nn.modules.pooling.MaxPool2d)
        return {"kernel_size": layer.kernel_size, "stride": layer.stride, "padding": layer.padding, }
    elif layer_type == "AvgPool2d":
        # in particular, we use adaptive 2D Average Pooling where adaptive refers to the fact that
        # the layer is adaptive to any input size
        assert isinstance(layer, nn.modules.pooling.AdaptiveAvgPool2d)
        return {"output_size": layer.output_size}
    elif layer_type == "Flatten":
        assert isinstance(layer, nn.modules.flatten.Flatten)
        return {}
    elif layer_type == "ResidualLayer":
        assert isinstance(layer, nn.modules.container.Sequential)
        assert residual_args is not None
        return parse_residual_layer(layer, residual_args[layer_name])
    else:
        raise ValueError(f"Unknown layer type: {layer_type}")

def parse_residual_layer(layer: nn.modules.container.Sequential, residual_args: dict):
    """

    :param layer:
    :return:
    """
    num_blocks = len([None for _, _ in layer.named_children()])
    residual_params = {
        "NumberBlocks": residual_args.get("blocks"),
        "OutChannels": residual_args.get("out_channels"),
        "Stride": residual_args.get("stride"),
        "ResidualLayerModel": {}
    }
    res_layer_model = residual_params.get("ResidualLayerModel")

    for i, (basic_block_name, basic_block) in zip(range(1, 1+num_blocks), layer.named_children()):
        block_digit, block_name = basic_block_name.split("_", 1)
        assert block_name == "ResidualBasicBlock", "The residual layer is not being parsed correctly."

        # get the attributes of the BasicBlock in the residual layer that has optimizable parameters
        ffn_attr = getattr(basic_block, "ffn")
        downsample_attr = getattr(basic_block, "downsample", None)  # the downsample attribute is optional

        # iterate through the layers of the sub feedforward network
        curr_ffn = {}
        downsample = None
        num_ffn_children = len([None for _, _ in ffn_attr.named_children()])
        for j, (ffn_layer_name, ffn_layer) in zip(range(1, 1+num_ffn_children), ffn_attr.named_children()):
            ffn_layer_digit, ffn_layer_type = ffn_layer_name.split("_", 1)
            ffn_layer_params = parse_layer(ffn_layer_name, ffn_layer_type, ffn_layer)
            if len(ffn_layer_params) > 0:
                curr_ffn[ffn_layer_name] = ffn_layer_params

        # if the residual layer performs downsampling, get its parameters
        if downsample_attr is not None:
            downsample = {}
            num_downsample_children = len([None for _, _ in downsample_attr.named_children()])
            for j, (downsample_layer_name, downsample_layer) in zip(range(1, 1+num_downsample_children), downsample_attr.named_children()):
                downsample_layer_digit, downsample_layer_type = downsample_layer_name.split("_", 1)
                downsample_layer_params = parse_layer(downsample_layer_name, downsample_layer_type, downsample_layer)
                if len(downsample_layer_params) > 0:
                    downsample[downsample_layer_name] = downsample_layer_params
        res_layer_model[basic_block_name] = {
            "ffn": curr_ffn,
            "downsample": downsample,
        }

    return residual_params

def save_torch_model(net):
    torch_model_dict = {}
    for name, layer in net.model.named_children():
        # each layer starts with a digit followed by its type
        # which is in {ResidualBasicBlock, Conv2d, BatchNorm2d, ReLU, MaxPool2d, AvgPool2d, Flatten, and Linear}
        layer_digit, layer_type = name.split("_", 1)
        print(f"Layer name: {name}")
        print(f"Layer items: {layer}")

        layer_params = parse_layer(name, layer_type, layer, net.residual_args)
        if len(layer_params) > 0:
            torch_model_dict[name] = layer_params

    return torch_model_dict

class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        """

        :param in_channels:
        :param out_channels:
        :param stride:
        """
        super(BasicBlock, self).__init__()

        layers = [
            ("1_Conv2d", nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)),
            ("2_BatchNorm2d", nn.BatchNorm2d(out_channels)),
            ("3_ReLU", nn.ReLU(inplace=True)),
            ("4_Conv2d", nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)),
            ("5_BatchNorm2d", nn.BatchNorm2d(out_channels)),
        ]
        self.ffn = nn.Sequential(OrderedDict(layers))

        # if the stride is greater than 1, then the output will have a smaller image shape than the input,
        # so the residual connection requires the input to be modified (e.g., just downsample it)
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

        self.in_channels = 64

        # the arguments for initialize the residual layers, useful to save for later
        self.residual_args = {
            "5_ResidualLayer": {"out_channels": 64, "blocks": 2, "stride": 1},
            "6_ResidualLayer": {"out_channels": 128, "blocks": 2, "stride": 2},
            "7_ResidualLayer": {"out_channels": 256, "blocks": 2, "stride": 2},
            "8_ResidualLayer": {"out_channels": 512, "blocks": 2, "stride": 2},
        }
        model_layers = []
        model_layers.extend([
            ("1_Conv2d", nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)),
            ("2_BatchNorm2d", nn.BatchNorm2d(64)), # element-wise normalization along each channel
            ("3_ReLU", nn.ReLU(inplace=True)),
            ("4_MaxPool2d", nn.MaxPool2d(kernel_size=3, stride=2, padding=1, ceil_mode=False)),
            ("5_ResidualLayer", self._make_layer(BasicBlock, **self.residual_args["5_ResidualLayer"])),
            ("6_ResidualLayer", self._make_layer(BasicBlock, **self.residual_args["6_ResidualLayer"])),
            ("7_ResidualLayer", self._make_layer(BasicBlock, **self.residual_args["7_ResidualLayer"])),
            ("8_ResidualLayer", self._make_layer(BasicBlock, **self.residual_args["8_ResidualLayer"])),
            ("9_AvgPool2d", nn.AdaptiveAvgPool2d((1, 1))), # average each convolution image to be a single element
            ("10_Flatten", nn.Flatten()),
            ("11_Linear", nn.Linear(512, out_size)),
        ])
        self.model = nn.Sequential(OrderedDict(model_layers))

        self.loss_fn = loss_fn
        self.optimizer = optim.SGD(self.parameters(), lr=lrate, weight_decay=1e-5)

    def _make_layer(self, block, out_channels: int, blocks: int, stride: int = 1):
        """
        Creates a resnet layer
        @param block:
        @param out_channels:
        @param blocks:
        @param stride:
        @return:
        """
        layers = [("1_ResidualBasicBlock", block(self.in_channels, out_channels, stride))]
        self.in_channels = out_channels # update in_channels for the next blocks
        for i in range(1, blocks):
            layers.append((f"{i+1}_ResidualBasicBlock", block(self.in_channels, out_channels)))
        return nn.Sequential(OrderedDict(layers))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the model
        @param x:
        @return:
        """
        # return self.model(x)
        module_dict = nn.ModuleDict(self.model._modules)
        for layer_name, layer_fn in module_dict.items():
            x = layer_fn(x)
        return x

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
        model_path: str,
        model_name: str,
        image_dim: Size,
        epochs: int,
        lrate: float = 0.01,
        loss_fn = nn.CrossEntropyLoss(),
        batch_size: int = 50,
        save: bool = False,
        verbose_architecture: bool = False,
) -> Tuple[list[float], NumberModel]:
    """
    Trains the model and returns the losses as well as the model.
    @param train_dataset:
    @param test_dataset:
    @param model_path:      Relative directory to save the model
    @param model_name:      Name of the model to save as
    @param image_dim:       Input dimensions of an image
    @param epochs:          Number of epochs
    @param batch_size:      Number of batches to use per epoch
    @param save:            True if the model should be saved
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
    for param in NetObject.parameters():
        param.requires_grad = False

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

    if verbose_architecture:
        for name, layer in NetObject.model.named_children():
            # each layer starts with a digit followed by its type
            # which is in {ResidualBasicBlock, Conv2d, BatchNorm2d, ReLU, MaxPool2d, AvgPool2d, Flatten, and Linear}
            layer_digit, layer_type = name.split("_", 1)
            print(f"Layer name: {name}")
            print(f"Layer items: {layer}")

    if save:
        # We format our own state dictionary with more verbose parameters and attributes
        # instead of the default network state dictionary. This makes it easier to use this .pth
        # file to initialize a Keras model.
        custom_state_dict = save_torch_model(NetObject)
        pth_dict = {
            "custom_state_dict": custom_state_dict, # typically you would use NetObject.state_dict()
            "lrate": lrate,
            "loss_fn": loss_fn,
            "image_dim": image_dim[0],
            "out_size": 10,
            "test_samples": to_numpy(torch.rand(10, 1, 50, 50)),
        }
        NetObject = NetObject.to(device=device)
        NetObject.eval()
        samples = torch.from_numpy(pth_dict["test_samples"]).to(dtype=torch.float32, device=device)
        output = to_numpy(NetObject(samples))
        pth_dict["test_sample_outputs"] = output
        pth_dict["state_dict"] = NetObject.state_dict()
        torch.save(pth_dict, model_path + model_name + ".pth")

        print(f"Forward inference with test samples: \n{output}")
        print(f"output.dtype: {output.dtype}")
        # traced_script_module = torch.jit.trace(NetObject, samples)
        # optimized_traced_model = optimize_for_mobile(traced_script_module)
        # optimized_traced_model._save_for_lite_interpreter(model_path + model_name + "_mobile.pt")

    return losses, NetObject

def main():
    device = args_dict.get("device")
    lrate = args_dict.get("lrate")
    loss_fn = nn.CrossEntropyLoss()
    if args_dict.get("load_model") is not None:
        pth_path = args_dict.get("load_model")
        pth_dict = torch.load(pth_path, map_location=device)
        _, _, single_image_dimension = get_sudoku_dataset(verbose=False)
        NetObject = NumberModel(lrate, loss_fn, single_image_dimension[0], 10).to(device)
        NetObject.load_state_dict(pth_dict["state_dict"])
        NetObject.eval()
        test_samples = torch.from_numpy(pth_dict["test_samples"]).to(dtype=torch.float32, device=device)
        output_samples = to_numpy(NetObject(test_samples))
        gt_output_samples = pth_dict["test_sample_outputs"]
        # for the outputs to match, the device the model is run on should also be the same
        outputs_match = np.allclose(output_samples, gt_output_samples)
        print(f"Does this evaluation match the ground truth outputs? {outputs_match}")
    else:
        train_data, test_data, single_image_dimension = get_sudoku_dataset(verbose=True)
        params = {
            "train_dataset": train_data,
            "test_dataset": test_data,
            "model_path": args_dict.get("model_path"),
            "model_name": args_dict.get("model_name"),
            "image_dim": single_image_dimension,
            "epochs": args_dict.get("epochs"),
            "lrate": lrate,
            "loss_fn": loss_fn,
            "batch_size": args_dict.get("batch_size"),
            "save": args_dict.get("save_model"),
            "verbose_architecture": args_dict.get("verbose_architecture"),
        }
        losses, NetObject = fit(**params)

if __name__ == "__main__":
    """
    Trains and tests the accuracy of the network
    """
    ## BEGIN PROGRAM ARGUMENTS ##
    parser = argparse.ArgumentParser()
    # Build arguments
    parser.add_argument("--model_path", type=str,
                        default="./models/",
                        help="Directory to save the model")
    parser.add_argument("--model_name", type=str,
                        default="NumberModel_v0",
                        help="Name of the model.")
    parser.add_argument("--save_model", action="store_true",
                        help="Set flag if the model should be saved.")
    parser.add_argument("--load_model", type=str,
                        help="Instead of saving a new model, load an existing model.")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lrate", type=float, default=0.01)
    parser.add_argument("--batch_size", type=int, default=50)
    parser.add_argument("--device", type=torch.device, default=torch.device('cuda'),
                        help="Device to use (e.g., 'cpu' or 'cuda'; default is 'cuda').")
    parser.add_argument("--verbose_architecture", action="store_true",
                        help="Set flag if the model architecture should be printed. This helps get a high-level view "
                             "of the modules that comprise our custom residual network.")
    # Parse arguments
    args = parser.parse_args()
    args_dict = vars(args)
    ## END PROGRAM ARGUMENTS

    global device
    device = args_dict.get("device")

    ## RUN MAIN PROGRAM
    main()
