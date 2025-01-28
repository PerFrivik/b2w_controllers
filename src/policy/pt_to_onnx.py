import torch
import os

# Define model and ONNX paths
script_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(script_dir, "policy_blind_b2w.pt")
onnx_path = os.path.join(script_dir, "policy_blind_b2w.onnx")

# Load the model
model = torch.jit.load(model_path)
model.eval()  # Ensure the model is in evaluation mode

# ✅ Corrected dummy input (sequence_length=1, batch_size=1, input_features=60)
dummy_input = torch.randn(1, 60)

# Export to ONNX
torch.onnx.export(
    model, 
    dummy_input, 
    onnx_path,
    export_params=True,          # Store trained parameters in the ONNX model
    opset_version=11,            # ONNX version
    do_constant_folding=True,    # Optimize constant foldable operations
    input_names=["input"],       # Input name
    output_names=["output"],     # Output name
    dynamic_axes={               # Allow dynamic batch sizes
        "input": {1: "batch_size"}, 
        "output": {1: "batch_size"}
    }
)

print(f"ONNX model exported successfully to: {onnx_path}")
