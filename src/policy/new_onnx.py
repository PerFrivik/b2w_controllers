#!/usr/bin/env python3

import torch
import os

def export_jit_to_onnx(jit_model_path, onnx_output_path):
    # Load the JIT model
    model = torch.jit.load(jit_model_path)
    model.eval()

    # Create a dummy input matching your model’s input shape
    # Here we assume a batch size of 1 and 60 input features
    dummy_input = torch.randn(1, 60, dtype=torch.float)

    # Export to ONNX
    torch.onnx.export(
        model,
        dummy_input,
        onnx_output_path,
        export_params=True,      # Store the trained parameter weights
        opset_version=12,        # ONNX opset version (change as needed)
        do_constant_folding=True,
        input_names=["input"],   # Optional: Name your model’s input
        output_names=["output"], # Optional: Name your model’s output
        dynamic_axes=None        # Set to { ... } if you want dynamic shapes
    )

    print(f"Model exported to {onnx_output_path}")

if __name__ == "__main__":
    # Example usage:
    #  python export_jit_to_onnx.py
    # Adjust paths as needed
    # jit_path = "test.pt"
    # onnx_path = "my_model.onnx"
    script_dir = os.path.dirname(os.path.abspath(__file__))
    jit_path = os.path.join(script_dir, "policy_blind_b2w.pt")
    onnx_path = os.path.join(script_dir, "gg.onnx")
    export_jit_to_onnx(jit_path, onnx_path)
