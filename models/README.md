# Models Directory

This directory contains ONNX model files that can be used with the Deep C++ inference framework.

## Exporting PyTorch Models to ONNX

### Simple Transformer Model

```python
import torch
import torch.nn as nn

class SimpleTransformer(nn.Module):
    def __init__(self, d_model=768, nhead=12, num_layers=1):
        super().__init__()
        self.d_model = d_model
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        self.fc = nn.Linear(d_model, d_model)
        
    def forward(self, x):
        x = self.transformer(x)
        return self.fc(x)

# Create and export model
model = SimpleTransformer()
dummy_input = torch.randn(1, 512, 768)  # batch_size=1, seq_len=512, d_model=768

torch.onnx.export(
    model,
    dummy_input,
    "models/simple_transformer.onnx",
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={
        "input": {0: "batch_size", 1: "seq_len"},
        "output": {0: "batch_size", 1: "seq_len"}
    },
    opset_version=17,
    do_constant_folding=True
)
```

### Mamba Model (with custom SSM scan)

```python
# First install mamba-ssm: pip install mamba-ssm
import torch
from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel

# Create a small Mamba model
model = MambaLMHeadModel(
    d_model=768,
    n_layer=1,
    vocab_size=50257,  # GPT-2 vocab size
    ssm_cfg={},
    rms_norm=True,
    initializer_cfg={},
    fused_add_norm=False,
    residual_in_fp32=True,
)

# Export to ONNX (note: this might require custom ops)
dummy_input = torch.randint(0, 50257, (1, 512))  # batch_size=1, seq_len=512

torch.onnx.export(
    model,
    dummy_input,
    "models/mamba_model.onnx",
    input_names=["input_ids"],
    output_names=["logits"],
    dynamic_axes={
        "input_ids": {0: "batch_size", 1: "seq_len"},
        "logits": {0: "batch_size", 1: "seq_len"}
    },
    opset_version=17,
    do_constant_folding=True
)
```

### Quantization

After exporting to ONNX, you can quantize the model for better CPU performance:

```python
import onnx
from onnxruntime.quantization import quantize_dynamic, QuantType

# Quantize the model
quantize_dynamic(
    "models/simple_transformer.onnx",
    "models/simple_transformer_int8.onnx",
    weight_type=QuantType.QInt8,
    extra_options={
        "ActivationSymmetric": True,
        "WeightSymmetric": True,
    }
)
```

## Testing Models

Use the main inference application to test your models:

```bash
# Build the project first
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j

# Run inference
./build/run_infer models/simple_transformer.onnx --batch-size 1 --seq-len 512 --num-runs 10

# Test with different configurations
./build/run_infer models/simple_transformer_int8.onnx --threads 4 --num-runs 100
```

## Model Requirements

- Models should be exported with opset version 17 or compatible
- Input/output names should be descriptive
- Dynamic axes help with different batch sizes and sequence lengths
- FP32 models work out-of-the-box; INT8 quantized models provide better performance

## Custom Operations

If your model uses custom operations (like Mamba's selective scan), make sure:

1. The custom op is registered in `src/custom_ops.cc`
2. The kernel implementation exists in `src/kernels/`
3. The op domain matches what's used in the ONNX export

Example custom op usage in PyTorch export:
```python
# Register custom op domain during export
torch.onnx.export(
    model,
    dummy_input,
    "models/custom_model.onnx",
    custom_opsets={"ryan_ops": 1},  # matches domain in custom_ops.cc
    # ... other parameters
)
``` 