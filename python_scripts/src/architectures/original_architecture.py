"""
Defines the original NumberModel architecture.
"""
import torch.nn as nn
from collections import OrderedDict
from src.architectures.blocks import BasicBlock


class OriginalArchitecture(nn.Module):
    """
    The original Neural Network architecture.
    """

    def __init__(self):
        """
        Initializes the Original Architecture.
        """
        super(OriginalArchitecture, self).__init__()
        self.image_dimensions = [1, 50, 50]
        self.in_channels = 64
        self.residual_args = {}

        # Features Extractor
        self.model = nn.Sequential(OrderedDict([
            ("00_Conv2d", nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)),
            ("01_BatchNorm2d", nn.BatchNorm2d(64)),
            ("02_ReLU", nn.ReLU(inplace=True)),
            ("03_MaxPool2d", nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),
            ("04_ResidualLayer",
             self._make_layer(BasicBlock, out_channels=64, blocks=2, stride=1, name="04_ResidualLayer")),
            ("05_ResidualLayer",
             self._make_layer(BasicBlock, out_channels=128, blocks=2, stride=2, name="05_ResidualLayer")),
            ("06_ResidualLayer",
             self._make_layer(BasicBlock, out_channels=256, blocks=2, stride=2, name="06_ResidualLayer")),
            ("07_ResidualLayer",
             self._make_layer(BasicBlock, out_channels=512, blocks=2, stride=2, name="07_ResidualLayer")),
            ("08_AvgPool2d", nn.AdaptiveAvgPool2d((1, 1))),
            ("09_Flatten", nn.Flatten()),
            ("10_Linear", nn.Linear(512, 10))
        ]))

    def _make_layer(self, block, out_channels, blocks, stride, name):
        """
        Creates a sequential resnet layer.

        :param block:           The block class to use.
        :param out_channels:    Output channels for the block.
        :param blocks:          Number of blocks.
        :param stride:          Stride for the first block.
        :param name:            Name of the layer for argument tracking.
        :return:                A Sequential layer of blocks.
        """
        self.residual_args[name] = {"out_channels": out_channels, "blocks": blocks, "stride": stride}
        layers = [("00_ResidualBasicBlock", block(self.in_channels, out_channels, stride))]
        self.in_channels = out_channels
        for i in range(1, blocks):
            layers.append((f"{i:02d}_ResidualBasicBlock", block(self.in_channels, out_channels)))
        return nn.Sequential(OrderedDict(layers))

    def forward(self, x):
        """
        Forward pass of the model.

        :param x:   Batch of input samples.
        :return:    Batch of inferences.
        """
        return self.model(x)