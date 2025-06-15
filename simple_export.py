#!/usr/bin/env python3
import torch
import torch.nn as nn
import onnx

class SimpleTransformer(nn.Module):
    def __init__(self, vocab_size=1000, d_model=512, nhead=8, num_layers=6):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = nn.Embedding(1024, d_model)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dim_feedforward=d_model*4,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        self.lm_head = nn.Linear(d_model, vocab_size)
        
    def forward(self, input_ids):
        B, L = input_ids.shape
        
        # Embeddings
        x = self.embedding(input_ids)
        pos_ids = torch.arange(L, device=input_ids.device).unsqueeze(0).expand(B, -1)
        x = x + self.pos_embedding(pos_ids)
        
        # Transformer
        x = self.transformer(x)
        
        # Output
        return self.lm_head(x)

def export_simple_model():
    model = SimpleTransformer(vocab_size=1000, d_model=512, nhead=8, num_layers=6)
    model.eval()
    
    # Create dummy input
    dummy_input = torch.randint(0, 1000, (1, 128))  # batch=1, seq_len=128
    
    # Export to ONNX
    torch.onnx.export(
        model,
        dummy_input,
        "models/simple_transformer.onnx",
        input_names=['input_ids'],
        output_names=['logits'],
        dynamic_axes={
            'input_ids': {0: 'batch_size', 1: 'sequence_length'},
            'logits': {0: 'batch_size', 1: 'sequence_length'}
        },
        opset_version=17
    )
    
    # Verify
    onnx_model = onnx.load("models/simple_transformer.onnx")
    onnx.checker.check_model(onnx_model)
    print("✓ Simple transformer exported successfully")

if __name__ == "__main__":
    export_simple_model() 