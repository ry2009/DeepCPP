#!/usr/bin/env python3
"""
Simple test script to create a basic ONNX model for testing the Deep C++ framework.
"""

import torch
import torch.nn as nn

def create_simple_model():
    """Create a simple feedforward model."""
    class SimpleModel(nn.Module):
        def __init__(self, d_model=768):
            super().__init__()
            self.linear1 = nn.Linear(d_model, d_model * 2)
            self.relu = nn.ReLU()
            self.linear2 = nn.Linear(d_model * 2, d_model)
            self.layer_norm = nn.LayerNorm(d_model)
            
        def forward(self, x):
            x_orig = x
            x = self.linear1(x)
            x = self.relu(x)
            x = self.linear2(x)
            x = x + x_orig  # residual connection
            x = self.layer_norm(x)
            return x
    
    return SimpleModel()

def create_transformer_model():
    """Create a small transformer model."""
    class SimpleTransformer(nn.Module):
        def __init__(self, d_model=768, nhead=12, num_layers=2):
            super().__init__()
            self.d_model = d_model
            encoder_layer = nn.TransformerEncoderLayer(
                d_model, nhead, 
                batch_first=True,
                activation='relu'
            )
            self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
            self.output_proj = nn.Linear(d_model, d_model)
            
        def forward(self, x):
            x = self.transformer(x)
            return self.output_proj(x)
    
    return SimpleTransformer()

def export_model(model, model_name, batch_size=1, seq_len=512, d_model=768):
    """Export a PyTorch model to ONNX."""
    model.eval()
    
    # Create dummy input
    dummy_input = torch.randn(batch_size, seq_len, d_model)
    
    # Export to ONNX
    output_path = f"models/{model_name}.onnx"
    
    with torch.no_grad():
        torch.onnx.export(
            model,
            dummy_input,
            output_path,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={
                "input": {0: "batch_size", 1: "seq_len"},
                "output": {0: "batch_size", 1: "seq_len"}
            },
            opset_version=17,
            do_constant_folding=True,
            verbose=False
        )
    
    print(f"Exported {model_name} to {output_path}")
    
    # Test the model in PyTorch for comparison
    with torch.no_grad():
        output = model(dummy_input)
        print(f"PyTorch output shape: {output.shape}")
        print(f"PyTorch output mean: {output.mean().item():.6f}")
        print(f"PyTorch output std: {output.std().item():.6f}")
    
    return output_path

def main():
    print("Creating test models for Deep C++ framework...")
    
    # Create models directory if it doesn't exist
    import os
    os.makedirs("models", exist_ok=True)
    
    # Export simple feedforward model
    print("\n1. Creating simple feedforward model...")
    simple_model = create_simple_model()
    export_model(simple_model, "simple_feedforward")
    
    # Export transformer model
    print("\n2. Creating transformer model...")
    transformer_model = create_transformer_model()
    export_model(transformer_model, "simple_transformer")
    
    # Export a minimal model for basic testing
    print("\n3. Creating minimal ReLU model...")
    class MinimalModel(nn.Module):
        def forward(self, x):
            return torch.relu(x)
    
    minimal_model = MinimalModel()
    export_model(minimal_model, "minimal_relu", seq_len=64, d_model=128)
    
    print("\n✅ All test models exported successfully!")
    print("\nOnce the C++ build completes, you can test with:")
    print("  ./build/run_infer models/minimal_relu.onnx")
    print("  ./build/run_infer models/simple_feedforward.onnx") 
    print("  ./build/run_infer models/simple_transformer.onnx")

if __name__ == "__main__":
    main() 