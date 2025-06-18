"""
This is a TFLite model that initializes a custom residual network. It is not meant to be trained. Instead,
the model parameters should come from the equivalent PyTorch model whose training script is given in 'NumberModel.py'.

This script is meant to generate a TFLite model that can be used on Android.
"""
# tensorflow imports
from keras.models import Sequential
from keras.layers import Input, Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization, ReLU, GlobalAveragePooling2D, ZeroPadding2D
import tensorflow as tf
from ai_edge_litert.interpreter import Interpreter

# other imports
import numpy as np
import os, argparse, torch
from torch import nn
from typing import *
import hydra
from pathlib import Path
from config.config_schema import Config, Architecture

print("TensorFlow version:", tf.__version__)
print("GPU available:", tf.config.list_physical_devices('GPU'))

to_numpy = lambda x : x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else x

class SequentialModel(tf.keras.Model):
    def __init__(self, layers: Sequential):
        super(SequentialModel, self).__init__()
        self.model = layers

    def call(self, x: tf.Tensor, training: bool=False)->tf.Tensor:
        """
        Forward pass of the model.
        :param x:           Batch of samples.
        :param training:    Flag denoting if the network is being trained.
        :return:            Batch of inferences.
        """
        return self.model(x, training=training)

    def call_trace(self, x, training=False)->Tuple[tf.Tensor, Dict[str, np.ndarray]]:
        """
        Forward pass of the model that also traces the output of each layer.
        :param x:           Batch of samples.
        :param training:    Flag denoting if the network is being trained.
        :return:            Batch of inferences.
        """
        trace = {}
        current_output = current_input = x
        for i, layer in enumerate(self.model.layers):
            current_output = layer(current_input, training=training) \
                if hasattr(layer, "training") or callable(getattr(layer, "call", None)) \
                else layer(current_input)
            # trace[f"{i:02d}_{layer.__class__.__name__}"] = current_output.numpy()
            trace[layer.name] = current_output.numpy()
            current_input = current_output

        return current_output, trace

class KerasModelBuilder:
    """
    This builder class allows a Keras model to build from an explicitly formatted dictionary
    that can be created from the 'NumberModel.py' script.
    """
    def __init__(self, input_shape:tuple=(None, 50, 50, 1), num_classes: int = 10):
        """
        Initializes the builder with a specified input and output shape.
        :param input_shape:     Input shape of the model.
        :param num_classes:     Number of output logits.
        """
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.x_shape = input_shape

        self.builders = {
            "Conv2d": self._build_conv2d,
            "Linear": self._build_linear,
            "ReLU": self._build_relu,
            "BatchNorm2d": self._build_batchnorm2d,
            "MaxPool2d": self._build_maxpool2d,
            "AvgPool2d": self._build_avgpool2d,
            "Flatten": self._build_flatten,
            "ResidualLayer": self._build_residual_layer,
        }

    def build(self, layers_dict: dict) -> tf.keras.Model:
        """
        Given a dictionary which explicitly describes the model's architecture and parameters,
        create an instance of the model as a Keras model.
        :param layers_dict: The dictionary describing the structure and parameters.
        :return:            The Keras model from the dictionary.
        """
        # The first level keys of the dictionary are named as "x_name" where x is an integer and
        # name is the type of the layer. We want to sort the keys with respect to x.
        layer_keys = sorted(layers_dict, key=lambda x: int(x.split("_", 1)[0]))
        input_shape = self.input_shape
        model = Sequential([Input(shape=self.input_shape[1:])])
        for i, key in enumerate(layer_keys):
            layer_type = key.split("_", 1)[1]
            input_shape = self.builders[layer_type](model, input_shape, layers_dict[key], i)

        seq_model = SequentialModel(model)
        seq_model.build(input_shape=(None, 50, 50, 1))  # (batch, height, width, channels)
        return seq_model

    @staticmethod
    def _build_conv2d(parent_layer: Sequential, input_shape: tuple, torch_params: dict, i: int) -> tuple:
        """Builds a 2D convolution layer."""
        conv2d_sequence = Sequential([Input(shape=input_shape[1:])], name=f"{i:02d}_Conv2d")

        conv2d_padding = ZeroPadding2D(padding=torch_params["padding"])
        conv2d_padding.build(input_shape)
        conv2d_input_shape = conv2d_padding.compute_output_shape(input_shape)
        conv2d = Conv2D(filters=torch_params["out_channels"],
                        kernel_size=torch_params["kernel_size"],
                        strides=torch_params["stride"],
                        padding="valid",
                        use_bias=True if torch_params.get("bias") else False
                        )
        conv2d.build(conv2d_input_shape)
        conv2d.set_weights([to_numpy(t) for t in (
            torch_params["weight"].transpose(2, 3, 1, 0),
            # shape [Cout, Cin, h, w] in torch, [h, w, Cin, Cout] in keras
        )])
        conv2d_output_shape = conv2d.compute_output_shape(conv2d_input_shape)
        conv2d_sequence.add(conv2d_padding)
        conv2d_sequence.add(conv2d)
        parent_layer.add(conv2d_sequence)

        return conv2d_output_shape

    @staticmethod
    def _build_linear(parent_layer: Sequential, input_shape: tuple, torch_params: dict, i: int) -> tuple:
        """Builds a linear/dense layer."""
        # In PyTorch, weight has shape [output_shape, input_shape]. In Keras, this is transposed
        # to be [input_shape, output_shape].
        weights = to_numpy(torch_params["weight"]).transpose()
        bias = to_numpy(torch_params["bias"])
        units = weights.shape[1]
        dense = Dense(units, name=f"{i:02d}_Linear")
        dense.build(input_shape)
        dense.set_weights([weights, bias])
        parent_layer.add(dense)
        return dense.compute_output_shape(input_shape)

    @staticmethod
    def _build_relu(parent_layer: Sequential, input_shape: tuple, torch_params: dict, i: int) -> tuple:
        """Builds a ReLU layer."""
        parent_layer.add(ReLU(name=f"{i:02d}_ReLU"))
        return input_shape

    @staticmethod
    def _build_flatten(parent_layer: Sequential, input_shape: tuple, torch_params: dict, i: int) -> tuple:
        """Keras does not have an explicit flatten module."""
        return input_shape

    @staticmethod
    def _build_batchnorm2d(parent_layer: Sequential, input_shape: tuple, torch_params: dict, i: int) -> tuple:
        """Builds a 2D batch normalization layer."""
        batchnorm2d = BatchNormalization(epsilon=torch_params["eps"], momentum=torch_params["momentum"],
                                         name=f"{i:02d}_BatchNorm2d")
        batchnorm2d.build(input_shape)
        batchnorm2d.set_weights([to_numpy(t) for t in (
            torch_params["weight"],
            torch_params["bias"],
            torch_params["running_mean"],
            torch_params["running_var"]
        )])
        batchnorm2d_output_shape = batchnorm2d.compute_output_shape(input_shape)
        parent_layer.add(batchnorm2d)

        return batchnorm2d_output_shape

    @staticmethod
    def _build_maxpool2d(parent_layer: Sequential, input_shape: tuple, torch_params: dict, i: int) -> tuple:
        """Builds a 2d max pooling layer."""
        maxpool_sequential = Sequential([Input(shape=input_shape[1:])], name=f"{i:02d}_MaxPool2d")
        maxpool_padding = ZeroPadding2D(padding=torch_params["padding"])
        maxpool_padding.build(input_shape)
        maxpool_input_shape = maxpool_padding.compute_output_shape(input_shape)
        maxpool = MaxPooling2D(pool_size=torch_params["kernel_size"],
                                    strides=torch_params["stride"],
                               padding="valid")
        maxpool.build(maxpool_input_shape)

        maxpool_output_shape = maxpool.compute_output_shape(maxpool_input_shape)
        maxpool_sequential.add(maxpool_padding)
        maxpool_sequential.add(maxpool)
        parent_layer.add(maxpool_sequential)

        return maxpool_output_shape

    @staticmethod
    def _build_avgpool2d(parent_layer: Sequential, input_shape: tuple, torch_params: dict, i: int) -> tuple:
        """Builds a 2D average pooling layer."""
        global_pool = GlobalAveragePooling2D(name=f"{i:02d}_AvgPool2d")
        global_pool.build(input_shape)

        global_pool_output_shape = global_pool.compute_output_shape(input_shape)
        parent_layer.add(global_pool)

        return global_pool_output_shape

    @staticmethod
    def _build_residual_layer(parent_layer: Sequential, input_shape: tuple, torch_params: dict, i: int) -> tuple:
        """Builds a residual layer"""
        assert len(input_shape) == 4, "Input shape must be (N, H, W, C)"
        residual_in_channels = input_shape[3]

        # build the residual layers and initialize the first block
        num_blocks = torch_params["NumberBlocks"]
        out_channels = torch_params["OutChannels"]
        init_block_params = torch_params["ResidualLayerModel"]["00_ResidualBasicBlock"]
        init_block = BasicBlock(input_shape=input_shape, in_channels=residual_in_channels, out_channels=out_channels,
                       block_params=init_block_params)
        residual_layer = Sequential([Input(shape=input_shape[1:])], name=f"{i:02d}_ResidualLayer")
        residual_layer.add(init_block)
        output_shape = init_block.output_shape  # in case only one block is used

        # initialize the remaining blocks of the layer
        for i in range(1, num_blocks):
            res_block_params = torch_params["ResidualLayerModel"][f"{i:02d}_ResidualBasicBlock"]
            block = BasicBlock(input_shape=output_shape, in_channels=residual_in_channels,
                               out_channels=out_channels, block_params=res_block_params)
            output_shape = block.output_shape
            residual_layer.add(block)

        parent_layer.add(residual_layer)

        return output_shape

class BasicBlock(tf.keras.Model):
    def __init__(self, input_shape: Tuple[int,...], in_channels: int, out_channels: int, block_params: dict):
        """
        Builds a single residual block which a residual layer is composed of.
        :param input_shape:     The input shape to the block.
        :param in_channels:     Number of input channels.
        :param out_channels:    Number of output channels.
        :param block_params:    A dictionary describing the parameters of the block.
        """
        super().__init__()

        # unpack the feed forward network and the downsample that may optionally occur for the
        # residual connection
        ffn_layers = block_params["ffn"]
        downsample_layers = block_params.get("downsample", None) # this attribute is optional

        torch_conv1_params = ffn_layers["00_Conv2d"]
        self.conv1_padding = ZeroPadding2D(padding=torch_conv1_params["padding"])
        conv1_input_shape = self.conv1_padding.compute_output_shape(input_shape)
        self.conv1 = Conv2D(filters=torch_conv1_params["out_channels"],
                            kernel_size=torch_conv1_params["kernel_size"], strides=torch_conv1_params["stride"],
                            padding="valid", use_bias=True if torch_conv1_params.get("bias") else False)
        self.conv1.build(conv1_input_shape)
        self.conv1.set_weights([to_numpy(t) for t in (
            torch_conv1_params["weight"].transpose(2, 3, 1, 0),
            # shape [Cout, Cin, h, w] in torch, [h, w, Cin, Cout] in keras
        )])
        conv1_output_shape = self.conv1.compute_output_shape(conv1_input_shape)

        torch_bn1_params = ffn_layers.get("01_BatchNorm2d")
        self.bn1 = BatchNormalization(epsilon=torch_bn1_params["eps"], momentum=torch_bn1_params["momentum"])
        self.bn1.build(conv1_output_shape)
        self.bn1.set_weights([to_numpy(t) for t in (
            torch_bn1_params["weight"],
            torch_bn1_params["bias"],
            torch_bn1_params["running_mean"],
            torch_bn1_params["running_var"]
        )])
        self.relu = ReLU()

        torch_conv2_params = ffn_layers["03_Conv2d"]
        self.conv2_padding = ZeroPadding2D(padding=torch_conv2_params["padding"])
        conv2_input_shape = self.conv2_padding.compute_output_shape(conv1_output_shape)
        self.conv2 = Conv2D(filters=torch_conv2_params["out_channels"],
                            kernel_size=torch_conv2_params["kernel_size"], strides=torch_conv2_params["stride"],
                            padding="valid", use_bias=True if torch_conv2_params.get("bias") else False)
        self.conv2.build(conv2_input_shape)
        self.conv2.set_weights([to_numpy(t) for t in (
            torch_conv2_params["weight"].transpose(2, 3, 1, 0),
            # shape [Cout, Cin, h, w] in torch, [h, w, Cin, Cout] in keras
        )])
        conv2_output_shape = self.conv2.compute_output_shape(conv2_input_shape)
        torch_bn2_params = ffn_layers["04_BatchNorm2d"]
        self.bn2 = BatchNormalization(epsilon=torch_bn2_params["eps"], momentum=torch_bn2_params["momentum"])
        self.bn2.build(conv2_output_shape)
        self.bn2.set_weights([to_numpy(t) for t in (
            torch_bn2_params["weight"],
            torch_bn2_params["bias"],
            torch_bn2_params["running_mean"],
            torch_bn2_params["running_var"]
        )])

        # the output shape of this block is the output shape of the final convolutional layer
        self.output_shape = conv2_output_shape

        self.downsample = None
        if downsample_layers is not None:
            torch_ds_conv1_params = downsample_layers["00_Conv2d"]
            torch_ds_bn1_params = downsample_layers["01_BatchNorm2d"]
            assert torch_ds_conv1_params["stride"] != 1 or in_channels != out_channels, "Downsampling will have no effect."
            ds_conv1_padding = ZeroPadding2D(padding=torch_ds_conv1_params["padding"])
            ds_conv1_input_shape = ds_conv1_padding.compute_output_shape(input_shape)
            ds_conv1 = Conv2D(filters=torch_ds_conv1_params["out_channels"],
                                kernel_size=torch_ds_conv1_params["kernel_size"],
                                strides=torch_ds_conv1_params["stride"],
                                padding="valid", use_bias=True if torch_ds_conv1_params.get("bias") else False)
            ds_conv1.build(ds_conv1_input_shape)
            ds_conv1.set_weights([to_numpy(t) for t in (
                torch_ds_conv1_params["weight"].transpose(2, 3, 1, 0),
                # shape [Cout, Cin, h, w] in torch, [h, w, Cin, Cout] in keras
            )])
            ds_conv1_output_shape = ds_conv1.compute_output_shape(ds_conv1_input_shape)

            ds_bn1 = BatchNormalization(epsilon=torch_ds_bn1_params["eps"], momentum=torch_ds_bn1_params["momentum"])
            ds_bn1.build(ds_conv1_output_shape)
            ds_bn1.set_weights([to_numpy(t) for t in (
                torch_ds_bn1_params["weight"],
                torch_ds_bn1_params["bias"],
                torch_ds_bn1_params["running_mean"],
                torch_ds_bn1_params["running_var"]
            )])
            self.downsample = Sequential([ds_conv1_padding, ds_conv1, ds_bn1])
            self.downsample.build(input_shape)

    def call(self, x, training=False):
        """
        Forward pass of the residual layer.
        :param x:           Batch of samples.
        :param training:    Flag denoting if the network is being trained.
        :return:            Batch of inferences.
        """
        identity = x

        out = self.conv1(self.conv1_padding(x, training=training), training=training)
        out = self.bn1(out, training=training)
        out = self.relu(out, training=training)

        out = self.conv2(self.conv2_padding(out, training=training), training=training)
        out = self.bn2(out, training=training)

        if self.downsample is not None:
            identity = self.downsample(x, training=training)

        out += identity
        return self.relu(out, training=training)

@hydra.main(config_path="./config", version_base=None)
def main(cfg: Config):
    pth_path = Path(cfg.save_parameters.model_path) / f"{cfg.save_parameters.model_name}.pth"
    print(f"Loading pth dict from path: {pth_path}")
    pth_dict = torch.load(pth_path, map_location='cpu')

    builder = KerasModelBuilder()
    model = builder.build(pth_dict["custom_state_dict"])
    model.build(input_shape=(None, 50, 50, 1))  # (batch, height, width, channels)

    # Build the model by calling it once with a dummy input.
    # Shape (batch, channels, height, width) -> (batch, height, width, channels)
    test_samples = pth_dict["test_samples"].transpose(0, 2, 3, 1)
    dummy_input = tf.convert_to_tensor(test_samples)
    test_sample_outputs = model(dummy_input, training=False)

    print(model.summary(expand_nested=True))

    torch_trace = pth_dict.get("trace")
    if torch_trace is not None:
        # compare the trace of the torch model with the Keras model
        _, keras_trace = model.call_trace(dummy_input, training=False)
        print(f"Keys in torch trace: {torch_trace.keys()}")
        print(f"Keys in keras trace: {keras_trace.keys()}")

    gt_test_sample_outputs = pth_dict["test_sample_outputs"]
    print(f"Ran initial forward inference with samples: \n{test_sample_outputs}")
    outputs_match = np.allclose(test_sample_outputs, gt_test_sample_outputs)
    print(f"Do the outputs of the PyTorch and TF model match? {outputs_match}")

    print(model.summary(expand_nested=True))

    # Convert the model to TFLite
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()

    # Save the TFLite model
    with open("test_number_model.tflite", "wb") as f:
        f.write(tflite_model)
    print("Saved model to test_number_model.tflite.")

    # Load the model
    interpreter = Interpreter(model_path="test_number_model.tflite")
    interpreter.allocate_tensors()
    print("Loaded TFLite model.")

    # Get input and output details
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Prepare input data (must match dtype and shape)
    input_data = tf.random.normal([1, 50, 50, 1]).numpy().astype(input_details[0]['dtype'])
    interpreter.set_tensor(input_details[0]['index'], input_data)

    # Run inference
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]['index'])

    print("TFLite output shape:", output.shape)

if __name__ == "__main__":
    main()