#!/usr/bin/env python3

import torch
import torch.nn as nn
import sys

class SimpleModel(nn.Module):
    """A simple model for testing the Deep C++ framework"""
    
    def __init__(self, d_model=768):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_model)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(d_model, d_model)
        
    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        return x

class SimpleTransformer(nn.Module):
    """A simple transformer for testing"""
    
    def __init__(self, d_model=768, nhead=12, num_layers=1):
        super().__init__()
        self.d_model = d_model
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        self.fc = nn.Linear(d_model, d_model)
        
    def forward(self, x):
        x = self.transformer(x)
        return self.fc(x)

def create_model(model_type="simple"):
    """Create and export test models"""
    
    if model_type == "simple":
        model = SimpleModel(768)
        model_name = "simple_model.onnx"
        print("Creating simple linear model...")
    elif model_type == "transformer":
        model = SimpleTransformer(768, 12, 1)
        model_name = "simple_transformer.onnx"
        print("Creating simple transformer model...")
    else:
        print(f"Unknown model type: {model_type}")
        return
    
    # Set to eval mode
    model.eval()
    
    # Create dummy input
    batch_size = 1
    seq_len = 512
    d_model = 768
    dummy_input = torch.randn(batch_size, seq_len, d_model)
    
    print(f"Input shape: {dummy_input.shape}")
    
    # Test forward pass
    with torch.no_grad():
        output = model(dummy_input)
        print(f"Output shape: {output.shape}")
    
    # Export to ONNX
    output_path = f"models/{model_name}"
    
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
        verbose=True
    )
    
    print(f"Model exported to: {output_path}")
    return output_path

if __name__ == "__main__":
    model_type = sys.argv[1] if len(sys.argv) > 1 else "simple"
    create_model(model_type) 