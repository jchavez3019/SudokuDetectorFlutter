"""
Handles model tracing, serialization, and conversion to TFLite edge formats.
"""
import logging
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from typing import Any, Tuple

from src.architectures.parser import get_explicit_model, to_numpy
from src.facades.hydra_types import TrainingSettings, SaveParametersSettings

log = logging.getLogger(__name__)


def forward_trace(net: nn.Module, x: torch.Tensor) -> Tuple[torch.Tensor, dict]:
    """
    Forward pass of the model where each layer's output is recorded.

    :param net: The neural network module.
    :param x:   Batch of samples.
    :return:    Batch of inferences and trace of each layer's results.
    """
    trace = {}
    module_dict = nn.ModuleDict(net.model._modules)
    for layer_name, layer_fn in module_dict.items():
        x = layer_fn(x)
        trace[layer_name] = to_numpy(x)
    return x, trace


def save_model(net: nn.Module, architecture: Any, training: TrainingSettings,
               save_parameters: SaveParametersSettings, device: str) -> None:
    """
    Saves the PyTorch model state dict and attempts to convert it to a TFLite edge model.

    :param net:                 Network model to save.
    :param architecture:        Architecture configuration object.
    :param training:            Configuration object containing training hyperparameters.
    :param save_parameters:     Settings dictating where and how to save the model.
    :param device:              The hardware device the tensors currently reside on.
    """
    custom_state_dict = get_explicit_model(net)

    # FIX: Read image_dimensions from the instantiated model (net), not the config struct
    torch_test_samples = torch.rand(1, *net.image_dimensions).to(dtype=torch.float32, device=device)

    pth_dict = {
        "custom_state_dict": custom_state_dict,
        "lrate": training.lrate,
        "loss_fn": None,
        "image_dim": net.image_dimensions,  # FIX: updated here as well
        "out_size": 10,
        "test_samples": to_numpy(torch_test_samples),
    }

    if save_parameters.trace_sample:
        pth_dict["trace"] = forward_trace(net, torch_test_samples)[1]

    net.eval()
    output = to_numpy(net(torch_test_samples))
    pth_dict["test_sample_outputs"] = output
    pth_dict["state_dict"] = net.state_dict()

    # Ensure output directory exists
    Path(save_parameters.model_path).mkdir(parents=True, exist_ok=True)
    save_path = Path(save_parameters.model_path) / f"{save_parameters.model_name}.pth"
    torch.save(pth_dict, save_path)
    log.info(f"Torch model saved to path: {save_path}")

    # TFLite Conversion Attempt
    try:
        import litert_torch
        log.info("Starting TFLite conversion...")

        net_cpu = net.cpu()
        test_samples_cpu = torch_test_samples.cpu()

        edge_model = litert_torch.convert(
            net_cpu.eval(),
            (test_samples_cpu,)
        )

        tflite_path = str(save_path).replace(".pth", ".tflite")
        edge_model.export(tflite_path)
        log.info(f"✓ TFLite model saved to: {tflite_path}")

        # Optional Diagnostic block
        from ai_edge_litert.interpreter import Interpreter
        interpreter = Interpreter(model_path=tflite_path)
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        expected_elements = np.prod(input_details[0]['shape'])

        if expected_elements != 2500:
            log.warning(
                f"⚠️ MISMATCH CONFIRMED! The TFLite model expects {expected_elements} elements, but Flutter provides 2500 (50x50).")
        else:
            log.info(f"✓ Input dimensions match! Model should work in Flutter.")

    except Exception as e:
        log.error(f"✗ TFLite conversion failed: {e}")