"""
This is a TFLite model that initializes a custom residual network. It is not meant to be trained. Instead,
the model parameters should come from the equivalent PyTorch model whose training script is given in 'NumberModel.py'.

This script is meant to generate a TFLite model that can be used on Android.
"""
# tensorflow imports
from keras.models import Sequential
from keras.layers import Input, Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization, ReLU, GlobalAveragePooling2D, ZeroPadding2D
# from keras import backend as K
import tensorflow as tf
# from tensorflow.python.tools import freeze_graph
# from tensorflow.python.tools import optimize_for_inference_lib
# from tensorflow.math import confusion_matrix
from ai_edge_litert.interpreter import Interpreter

# other imports
import numpy as np
import os, argparse, torch
from torch import Tensor, nn
from typing import *
import hydra
from config.config_schema import Config, Architecture

print("TensorFlow version:", tf.__version__)
print("GPU available:", tf.config.list_physical_devices('GPU'))

to_numpy = lambda x : x.detach().cpu().numpy() if isinstance(x, Tensor) else x

class BasicBlock(tf.keras.Model):
    def __init__(self, input_shape: Tuple[int,...], in_channels: int, out_channels: int, block_params: dict, stride=1):
        super().__init__()

        # unpack the feed forward network and the downsample that may optionally occur for the
        # residual connection
        ffn_layers = block_params.get("ffn")
        downsample_layers = block_params.get("downsample", None) # this attribute is optional

        torch_conv1_params = ffn_layers.get("1_Conv2d")
        self.conv1_padding = ZeroPadding2D(padding=torch_conv1_params.get("padding"))
        conv1_input_shape = self.conv1_padding.compute_output_shape(input_shape)
        self.conv1 = Conv2D(filters=torch_conv1_params.get("out_channels"),
                            kernel_size=torch_conv1_params.get("kernel_size"), strides=torch_conv1_params.get("stride"),
                            padding="valid", use_bias=True if torch_conv1_params.get("bias") else False)
        self.conv1.build(conv1_input_shape)
        self.conv1.set_weights([to_numpy(t) for t in (
            torch_conv1_params.get("weight").transpose(2, 3, 1, 0),
            # shape [Cout, Cin, h, w] in torch, [h, w, Cin, Cout] in keras
        )])
        conv1_output_shape = self.conv1.compute_output_shape(conv1_input_shape)

        torch_bn1_params = ffn_layers.get("2_BatchNorm2d")
        self.bn1 = BatchNormalization(epsilon=torch_bn1_params.get("eps"), momentum=torch_bn1_params.get("momentum"))
        self.bn1.build(conv1_output_shape)
        self.bn1.set_weights([to_numpy(t) for t in (
            torch_bn1_params.get("weight"),
            torch_bn1_params.get("bias"),
            torch_bn1_params.get("running_mean"),
            torch_bn1_params.get("running_var")
        )])
        self.relu = ReLU()

        torch_conv2_params = ffn_layers.get("4_Conv2d")
        self.conv2_padding = ZeroPadding2D(padding=torch_conv2_params.get("padding"))
        conv2_input_shape = self.conv2_padding.compute_output_shape(conv1_output_shape)
        self.conv2 = Conv2D(filters=torch_conv2_params.get("out_channels"),
                            kernel_size=torch_conv2_params.get("kernel_size"), strides=torch_conv2_params.get("stride"),
                            padding="valid", use_bias=True if torch_conv2_params.get("bias") else False)
        self.conv2.build(conv2_input_shape)
        self.conv2.set_weights([to_numpy(t) for t in (
            torch_conv2_params.get("weight").transpose(2, 3, 1, 0),
            # shape [Cout, Cin, h, w] in torch, [h, w, Cin, Cout] in keras
        )])
        conv2_output_shape = self.conv2.compute_output_shape(conv2_input_shape)
        torch_bn2_params = ffn_layers.get("5_BatchNorm2d")
        self.bn2 = BatchNormalization(epsilon=torch_bn2_params.get("eps"), momentum=torch_bn2_params.get("momentum"))
        self.bn2.build(conv2_output_shape)
        self.bn2.set_weights([to_numpy(t) for t in (
            torch_bn2_params.get("weight"),
            torch_bn2_params.get("bias"),
            torch_bn2_params.get("running_mean"),
            torch_bn2_params.get("running_var")
        )])

        # the output shape of this block is the output shape of the final convolutional layer
        self.output_shape = conv2_output_shape

        self.downsample = None
        if downsample_layers is not None:
            torch_ds_conv1_params = downsample_layers.get("0_Conv2d")
            torch_ds_bn1_params = downsample_layers.get("1_BatchNorm2d")
            assert torch_ds_conv1_params.get("stride") != 1 or in_channels != out_channels, "Downsampling will have no effect."
            ds_conv1_padding = ZeroPadding2D(padding=torch_ds_conv1_params.get("padding"))
            ds_conv1_input_shape = ds_conv1_padding.compute_output_shape(input_shape)
            ds_conv1 = Conv2D(filters=torch_ds_conv1_params.get("out_channels"),
                                kernel_size=torch_ds_conv1_params.get("kernel_size"),
                                strides=torch_ds_conv1_params.get("stride"),
                                padding="valid", use_bias=True if torch_ds_conv1_params.get("bias") else False)
            ds_conv1.build(ds_conv1_input_shape)
            ds_conv1.set_weights([to_numpy(t) for t in (
                torch_ds_conv1_params.get("weight").transpose(2, 3, 1, 0),
                # shape [Cout, Cin, h, w] in torch, [h, w, Cin, Cout] in keras
            )])
            ds_conv1_output_shape = ds_conv1.compute_output_shape(ds_conv1_input_shape)

            ds_bn1 = BatchNormalization(epsilon=torch_ds_bn1_params.get("eps"), momentum=torch_ds_bn1_params.get("momentum"))
            ds_bn1.build(ds_conv1_output_shape)
            ds_bn1.set_weights([to_numpy(t) for t in (
                torch_ds_bn1_params.get("weight"),
                torch_ds_bn1_params.get("bias"),
                torch_ds_bn1_params.get("running_mean"),
                torch_ds_bn1_params.get("running_var")
            )])
            self.downsample = Sequential([ds_conv1_padding, ds_conv1, ds_bn1])
            self.downsample.build(input_shape)

    def call(self, x, training=False):
        identity = x

        out = self.conv1(self.conv1_padding(x))
        out = self.bn1(out, training=training)
        out = self.relu(out)

        out = self.conv2(self.conv2_padding(out))
        out = self.bn2(out, training=training)

        if self.downsample is not None:
            identity = self.downsample(x, training=training)

        out += identity
        return self.relu(out)


class NumberModel(tf.keras.Model):
    """This is a Keras model meant to be initialized from a PyTorch model and not trained."""
    def __init__(self, weight_dict: dict, input_shape=(None, 50, 50, 1), num_classes=10):
        super().__init__()

        self.input_shape = input_shape
        self.in_channels = 64

        # build the first few convolutional layers
        torch_conv1_params = weight_dict.get("1_Conv2d")
        self.conv1_padding = ZeroPadding2D(padding=torch_conv1_params.get("padding"))
        conv1_input_shape = self.conv1_padding.compute_output_shape(input_shape)
        self.conv1 = Conv2D(filters=torch_conv1_params.get("out_channels"),
                            kernel_size=torch_conv1_params.get("kernel_size"), strides=torch_conv1_params.get("stride"),
                            padding="valid", use_bias=True if torch_conv1_params.get("bias") else False)
        self.conv1.build(conv1_input_shape)
        self.conv1.set_weights([to_numpy(t) for t in (
            torch_conv1_params.get("weight").transpose(2, 3, 1, 0),
        # shape [Cout, Cin, h, w] in torch, [h, w, Cin, Cout] in keras
        )])
        conv1_output_shape = self.conv1.compute_output_shape(conv1_input_shape)
        torch_bn1_params = weight_dict.get("2_BatchNorm2d")
        self.bn1 = BatchNormalization(epsilon=torch_bn1_params.get("eps"), momentum=torch_bn1_params.get("momentum"))
        self.bn1.build(conv1_output_shape)
        self.bn1.set_weights([to_numpy(t) for t in (
            torch_bn1_params.get("weight"),
            torch_bn1_params.get("bias"),
            torch_bn1_params.get("running_mean"),
            torch_bn1_params.get("running_var")
        )])
        self.relu = ReLU()
        torch_maxpool_params = weight_dict.get("4_MaxPool2d")
        self.maxpool_padding = ZeroPadding2D(padding=torch_maxpool_params.get("padding"))
        maxpool_input_shape = self.maxpool_padding.compute_output_shape(conv1_output_shape)
        self.maxpool = MaxPooling2D(pool_size=torch_maxpool_params.get("kernel_size"),
                                    strides=torch_maxpool_params.get("stride"), padding="valid")
        self.maxpool.build(maxpool_input_shape)

        # build the residual layers
        res_input_shape = maxpool_input_shape # the input shape to the first residual layer
        for i, res_key in zip(range(1, 5), (f"{j}_ResidualLayer" for j in range(5, 9))):
            torch_reslayer_params = weight_dict.get(res_key)
            num_blocks = torch_reslayer_params.get("NumberBlocks")
            out_channels = torch_reslayer_params.get("OutChannels")
            stride = torch_reslayer_params.get("Stride")
            res_layer, new_input_shape = self._make_layer(input_shape=res_input_shape, out_channels=out_channels,
                                                        layer_params=torch_reslayer_params.get("ResidualLayerModel"),
                                                        blocks=num_blocks, stride=stride)
            res_layer.build(res_input_shape)
            res_input_shape = new_input_shape
            setattr(self, f"layer{i}", res_layer)

        # build the final layers
        self.global_pool = GlobalAveragePooling2D()
        self.global_pool.build(res_input_shape)
        global_pool_output_shape = self.global_pool.compute_output_shape(res_input_shape)

        torch_fc_params = weight_dict.get("11_Linear")
        self.fc = Dense(num_classes)
        self.fc.build(global_pool_output_shape)
        self.fc.set_weights([to_numpy(t) for t in (
            torch_fc_params.get("weight").transpose(),
            torch_fc_params.get("bias")
        )])

    def _make_layer(self, input_shape: Tuple[int,...], out_channels: int, blocks: int, layer_params: dict,
                    stride: int = 1) -> Tuple[Sequential, Tuple[int,...]]:
        """
        Creates a residual layer using residual building blocks.
        :param input_shape:     The input shape to this residual layer.
        :param out_channels:    Number of output channels at the end of this layer.
        :param blocks:          Number of residual blocks in this layer.
        :param layer_params:    The parameters to use for this layer.
        :param stride:          The stride of the first residual block.
        :return:                Sequence of residual blocks and the final output shape.
        """
        res_blocks = Sequential() # store the blocks sequentially
        # initialize the first block of the layer
        init_block_params = layer_params.get("1_ResidualBasicBlock")
        init_block = BasicBlock(input_shape=input_shape, in_channels=self.in_channels, out_channels=out_channels,
                                block_params=init_block_params, stride=stride)
        res_blocks.add(init_block)
        output_shape = init_block.output_shape # in case only one block is used
        # initialize the remaining blocks of the layer
        for i in range(1, blocks):
            res_block_params = layer_params.get(f"{i+1}_ResidualBasicBlock")
            block = BasicBlock(input_shape=output_shape, in_channels=self.in_channels,
                                             out_channels=out_channels, block_params=res_block_params)
            output_shape= block.output_shape
            res_blocks.add(block)

        # update the input channels for the next layer
        self.in_channels = out_channels
        return res_blocks, output_shape

    def call(self, x, training=False):
        x = self.conv1(self.conv1_padding(x))
        x = self.bn1(x, training=training)
        x = self.relu(x)
        x = self.maxpool(self.maxpool_padding(x))

        x = self.layer1(x, training=training)
        x = self.layer2(x, training=training)
        x = self.layer3(x, training=training)
        x = self.layer4(x, training=training)

        x = self.global_pool(x)
        return self.fc(x)

@hydra.main(config_path="./config", version_base=None)
def main(cfg: Config):
    # pth_path = args_dict.get("torch_model_path")
    pth_path = cfg.save_parameters.model_path
    # keras_model_name = args_dict.get("model_name")
    keras_model_name = cfg.save_parameters.model_name
    pth_dict = torch.load(pth_path, map_location='cpu')
    model = NumberModel(pth_dict["custom_state_dict"])
    model.build(input_shape=(None, 50, 50, 1)) # (batch, height, width, channels)
    # Build the model by calling it once with a dummy input.
    # Shape (batch, channels, height, width) -> (batch, height, width, channels)
    test_samples = pth_dict["test_samples"].transpose(0, 2, 3, 1)
    dummy_input = tf.convert_to_tensor(test_samples)
    test_sample_outputs = model.call(dummy_input, training=False)
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
    # ## BEGIN PROGRAM ARGUMENTS ##
    # parser = argparse.ArgumentParser()
    # # Build arguments
    # parser.add_argument("torch_model_path", type=str,
    #                     help="Path of the PyTorch model to initialize from.")
    # parser.add_argument("model_name", type=str,
    #                     help="Name of the Keras model to save as.")
    # # Parse arguments
    # args = parser.parse_args()
    # args_dict = vars(args)
    # ## END PROGRAM ARGUMENTS
    #
    # ## RUN MAIN PROGRAM
    main()