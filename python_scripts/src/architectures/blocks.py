"""
Defines reusable building blocks for Neural Network architectures.
"""
import torch.nn as nn
from collections import OrderedDict


class BasicBlock(nn.Module):
    """
    A basic residual block that composes a residual layer.
    """

    def __init__(self, in_channels, out_channels, stride=1):
        """
        Create a basic residual block that composes a residual layer.

        :param in_channels:     Number of incoming channels.
        :param out_channels:    Number of outgoing channels.
        :param stride:          The stride to use at the first convolutional layer.
        """
        super(BasicBlock, self).__init__()

        layers = [
            ("00_Conv2d", nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)),
            ("01_BatchNorm2d", nn.BatchNorm2d(out_channels)),
            ("02_ReLU", nn.ReLU(inplace=True)),
            ("03_Conv2d", nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)),
            ("04_BatchNorm2d", nn.BatchNorm2d(out_channels)),
        ]
        self.ffn = nn.Sequential(OrderedDict(layers))

        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(OrderedDict([
                ("00_Conv2d", nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False)),
                ("01_BatchNorm2d", nn.BatchNorm2d(out_channels))
            ]))

    def forward(self, x):
        """
        Forward pass of the basic block.

        :param x:   Input tensor.
        :return:    Output tensor.
        """
        identity = x
        out = self.ffn(x)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = nn.functional.relu(out)
        return out