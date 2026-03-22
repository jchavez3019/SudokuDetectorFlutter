"""
Utilities for parsing a PyTorch neural network into an explicit dictionary format
suitable for custom Keras/TFLite reconstruction.
"""
import logging
import torch
import torch.nn as nn
from typing import Optional

log = logging.getLogger(__name__)


# Helper to convert tensors to numpy arrays safely
def to_numpy(x):
    return x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else x


class NetParser:
    """
    Explicitly parses a PyTorch model into a dictionary model that can be used to
    create the same model programmatically with Keras.
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

    def parse_net(self, net: nn.Module) -> dict:
        """
        Parses a network model and creates a custom dictionary.

        :param net:     The network model to parse.
        :return:        A dictionary describing the model's parameters.
        """
        model_dict = {}
        for name, layer in net.model.named_children():
            # each layer starts with a digit followed by its type
            layer_digit, layer_type = name.split("_", 1)

            # Note: Custom explicitly defined architectures must have residual_args
            residual_args = getattr(net, "residual_args", None)
            layer_params = self.parse_layer(name, layer_type, layer, residual_args)
            model_dict[name] = layer_params

        return model_dict

    def parse_layer(self, layer_name: str, layer_type: str, layer: nn.Module,
                    residual_args: Optional[dict] = None) -> dict:
        """
        Parses an individual layer and returns a custom dictionary.

        :param layer_name:      The custom name of the layer.
        :param layer_type:      The type of layer.
        :param layer:           The PyTorch layer module.
        :param residual_args:   Necessary arguments for a residual layer.
        :return:                Dictionary describing the layer's submodules and parameters.
        """
        if layer_type not in self.parsers:
            raise ValueError(f"Unknown layer type: {layer_type}")
        return self.parsers[layer_type](layer_name, layer, residual_args)

    @staticmethod
    def _parse_conv2d(name, layer: nn.Conv2d, *args):
        return {
            "weight": to_numpy(layer.weight), "bias": to_numpy(layer.bias),
            "stride": layer.stride, "padding": layer.padding, "dilation": layer.dilation,
            "groups": layer.groups, "kernel_size": layer.kernel_size,
            "in_channels": layer.in_channels, "out_channels": layer.out_channels
        }

    @staticmethod
    def _parse_linear(name, layer: nn.Linear, *args):
        return {"weight": to_numpy(layer.weight), "bias": to_numpy(layer.bias)}

    @staticmethod
    def _parse_relu(name, layer: nn.ReLU, *args):
        return {}

    @staticmethod
    def _parse_batchnorm2d(name, layer: nn.BatchNorm2d, *args):
        return {
            "weight": to_numpy(layer.weight), "bias": to_numpy(layer.bias),
            "running_mean": to_numpy(layer.running_mean),
            "running_var": to_numpy(layer.running_var),
            "eps": layer.eps, "momentum": layer.momentum
        }

    @staticmethod
    def _parse_maxpool2d(name, layer: nn.MaxPool2d, *args):
        return {
            "kernel_size": layer.kernel_size,
            "stride": layer.stride,
            "padding": layer.padding,
        }

    @staticmethod
    def _parse_avgpool2d(name, layer: nn.AdaptiveAvgPool2d, *args):
        return {"output_size": layer.output_size}

    @staticmethod
    def _parse_flatten(name, layer: nn.Flatten, *args):
        return {}

    def _parse_residual_layer(self, name, layer: nn.Sequential, residual_args: dict):
        if residual_args is None or name not in residual_args:
            raise ValueError("Residual arguments are required to parse residual layers.")

        _get_num_layers = lambda x: sum([1 for _ in x.named_children()])
        num_blocks = _get_num_layers(layer)

        residual_params = {
            "NumberBlocks": residual_args[name]["blocks"],
            "OutChannels": residual_args[name]["out_channels"],
            "Stride": residual_args[name]["stride"],
            "ResidualLayerModel": {}
        }
        res_layer_model = residual_params["ResidualLayerModel"]

        def _get_parameters(module: nn.Sequential) -> dict:
            ret_params = {}
            for sub_layer_name, sub_layer in module.named_children():
                _, layer_type = sub_layer_name.split("_", 1)
                layer_params = self.parse_layer(sub_layer_name, layer_type, sub_layer)
                if len(layer_params) > 0:
                    ret_params[sub_layer_name] = layer_params
            return ret_params

        for i, (basic_block_name, basic_block) in zip(range(1, 1 + num_blocks), layer.named_children()):
            _, block_name = basic_block_name.split("_", 1)
            assert block_name == "ResidualBasicBlock", "Residual layer parsed incorrectly."

            ffn_attr = getattr(basic_block, "ffn")
            downsample_attr = getattr(basic_block, "downsample", None)

            res_layer_model[basic_block_name] = {
                "ffn": _get_parameters(ffn_attr),
                "downsample": _get_parameters(downsample_attr) if downsample_attr is not None else None,
            }

        return residual_params


def get_explicit_model(net: nn.Module) -> dict:
    """
    Creates a custom dictionary which explicitly describes the structure of the model.

    :param net: Network model to parse.
    :return: A dictionary describing the structure of the model.
    """
    net_parser = NetParser()
    return net_parser.parse_net(net)