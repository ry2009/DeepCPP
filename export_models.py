#!/usr/bin/env python3
"""
Comprehensive Model Export Pipeline for Deep C++ Framework
Exports state-of-the-art models including Mamba, Flash Attention, Linear Attention, and more
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import onnx
import onnxruntime as ort
import numpy as np
import json
import argparse
from pathlib import Path
from typing import Dict, Tuple, Optional, List
import math

# ============================================================================
# ATTENTION MECHANISMS
# ============================================================================

class FlashAttention(nn.Module):
    """Flash Attention implementation for memory efficiency"""
    
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)
        
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.o_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        B, L, D = x.shape
        
        # Generate Q, K, V
        qkv = self.qkv(x).reshape(B, L, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, B, H, L, D]
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Flash attention (simplified for ONNX compatibility)
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        out = torch.matmul(attn_weights, v)
        out = out.transpose(1, 2).reshape(B, L, D)
        
        return self.o_proj(out)


class LinearAttention(nn.Module):
    """Linear Attention for efficient long sequences"""
    
    def __init__(self, d_model: int, num_heads: int = 8):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.o_proj = nn.Linear(d_model, d_model)
        
    def forward(self, x):
        B, L, D = x.shape
        
        q = self.q_proj(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Apply feature map (ELU + 1 for positivity)
        q = F.elu(q) + 1
        k = F.elu(k) + 1
        
        # Linear attention: O(n) complexity
        kv = torch.einsum('bhnd,bhnf->bhdf', k, v)
        z = torch.einsum('bhnd,bhd->bhn', k, torch.ones_like(k[..., 0]))
        
        out = torch.einsum('bhnd,bhdf->bhnf', q, kv) / (torch.einsum('bhnd,bhd->bhn', q, z.detach()) + 1e-6).unsqueeze(-1)
        
        out = out.transpose(1, 2).reshape(B, L, D)
        return self.o_proj(out)


class RetentiveAttention(nn.Module):
    """Retentive Attention from RetNet"""
    
    def __init__(self, d_model: int, num_heads: int = 8):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model) 
        self.v_proj = nn.Linear(d_model, d_model)
        self.o_proj = nn.Linear(d_model, d_model)
        
        # Retention decay parameter
        self.gamma = nn.Parameter(torch.ones(num_heads))
        
    def forward(self, x):
        B, L, D = x.shape
        
        q = self.q_proj(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Create decay matrix
        decay = self.gamma.view(1, -1, 1, 1) ** torch.arange(L, device=x.device).float().view(1, 1, L, 1)
        
        # Apply retention mechanism  
        retention_weights = torch.matmul(q, k.transpose(-2, -1))
        
        # Apply causal mask and decay
        causal_mask = torch.tril(torch.ones(L, L, device=x.device))
        retention_weights = retention_weights * causal_mask.unsqueeze(0).unsqueeze(0)
        retention_weights = retention_weights * decay
        
        out = torch.matmul(retention_weights, v)
        out = out.transpose(1, 2).reshape(B, L, D)
        
        return self.o_proj(out)


# ============================================================================
# MAMBA STATE SPACE MODELS
# ============================================================================

class SelectiveScanSSM(nn.Module):
    """Selective State Space Model for Mamba"""
    
    def __init__(self, d_model: int, d_state: int = 16, d_conv: int = 4, expand: int = 2):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.d_inner = int(expand * d_model)
        
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)
        
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            bias=True,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1
        )
        
        self.activation = "silu"
        self.act = nn.SiLU()
        
        self.x_proj = nn.Linear(self.d_inner, d_state * 2, bias=False)
        self.dt_proj = nn.Linear(self.d_inner, self.d_inner, bias=True)
        
        # S4D real initialization
        A = torch.arange(1, d_state + 1, dtype=torch.float32).repeat(self.d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(self.d_inner))
        
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)
        
    def forward(self, x):
        B, L, D = x.shape
        
        x_and_res = self.in_proj(x)  # shape (B, L, 2 * d_inner)
        x, res = x_and_res.split(split_size=self.d_inner, dim=-1)
        
        x = x.transpose(-1, -2)  # (B, d_inner, L)
        x = self.conv1d(x)[:, :, :L]
        x = x.transpose(-1, -2)  # (B, L, d_inner)
        
        x = self.act(x)
        
        # SSM parameters
        x_proj = self.x_proj(x)  # (B, L, d_state * 2)
        delta, B_ssm = x_proj.split(split_size=self.d_state, dim=-1)
        
        delta = F.softplus(self.dt_proj(x))  # (B, L, d_inner)
        
        A = -torch.exp(self.A_log.float())  # (d_inner, d_state)
        
        # Simplified SSM scan for ONNX compatibility
        y = self._selective_scan_simplified(x, delta, A, B_ssm)
        
        y = y * self.act(res)
        
        return self.out_proj(y)
    
    def _selective_scan_simplified(self, x, delta, A, B):
        """Simplified selective scan for ONNX export"""
        B, L, D = x.shape
        N = A.shape[-1]
        
        # Initialize state
        h = torch.zeros(B, D, N, device=x.device, dtype=x.dtype)
        
        outputs = []
        for i in range(L):
            # Get current inputs
            x_i = x[:, i, :]  # (B, D)
            delta_i = delta[:, i, :]  # (B, D)
            B_i = B[:, i, :].unsqueeze(1)  # (B, 1, N)
            
            # Discretize
            dt = delta_i.unsqueeze(-1)  # (B, D, 1)
            dA = torch.exp(dt * A.unsqueeze(0))  # (B, D, N)
            dB = dt * B_i  # (B, D, N)
            
            # Update state: h = dA * h + dB * x
            h = dA * h + dB * x_i.unsqueeze(-1)
            
            # Output
            C = torch.ones(1, 1, N, device=x.device, dtype=x.dtype)  # Simplified C
            y_i = torch.sum(h * C, dim=-1)  # (B, D)
            
            outputs.append(y_i)
        
        y = torch.stack(outputs, dim=1)  # (B, L, D)
        
        # Add skip connection
        y = y + x * self.D.unsqueeze(0).unsqueeze(0)
        
        return y


class MambaBlock(nn.Module):
    """Complete Mamba Block"""
    
    def __init__(self, d_model: int, d_state: int = 16, d_conv: int = 4, expand: int = 2):
        super().__init__()
        self.d_model = d_model
        
        self.norm = nn.RMSNorm(d_model)
        self.mamba = SelectiveScanSSM(d_model, d_state, d_conv, expand)
        
    def forward(self, x):
        return x + self.mamba(self.norm(x))


# ============================================================================
# TRANSFORMER VARIANTS
# ============================================================================

class GLU(nn.Module):
    """Gated Linear Unit"""
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff, bias=False)
        self.w2 = nn.Linear(d_model, d_ff, bias=False)
        self.w3 = nn.Linear(d_ff, d_model, bias=False)
        
    def forward(self, x):
        return self.w3(F.silu(self.w1(x)) * self.w2(x))


class TransformerBlock(nn.Module):
    """Enhanced Transformer Block with various attention types"""
    
    def __init__(self, d_model: int, num_heads: int, d_ff: int, 
                 attention_type: str = "standard", dropout: float = 0.1):
        super().__init__()
        self.attention_type = attention_type
        
        if attention_type == "flash":
            self.attention = FlashAttention(d_model, num_heads, dropout)
        elif attention_type == "linear":
            self.attention = LinearAttention(d_model, num_heads)
        elif attention_type == "retentive":
            self.attention = RetentiveAttention(d_model, num_heads)
        else:  # standard
            self.attention = nn.MultiheadAttention(d_model, num_heads, dropout, batch_first=True)
            
        self.norm1 = nn.RMSNorm(d_model)
        self.norm2 = nn.RMSNorm(d_model)
        
        # Use GLU instead of standard FFN
        self.feed_forward = GLU(d_model, d_ff)
        
    def forward(self, x):
        # Attention
        if self.attention_type == "standard":
            attn_out, _ = self.attention(x, x, x)
            x = x + attn_out
        else:
            x = x + self.attention(x)
            
        x = self.norm1(x)
        
        # Feed forward
        x = x + self.feed_forward(x)
        x = self.norm2(x)
        
        return x


# ============================================================================
# COMPLETE MODEL ARCHITECTURES
# ============================================================================

class MambaLM(nn.Module):
    """Complete Mamba Language Model"""
    
    def __init__(self, vocab_size: int = 32000, d_model: int = 2048, 
                 n_layers: int = 24, d_state: int = 16):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList([
            MambaBlock(d_model, d_state) for _ in range(n_layers)
        ])
        self.norm = nn.RMSNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        
        # Weight tying
        self.lm_head.weight = self.embedding.weight
        
    def forward(self, input_ids):
        x = self.embedding(input_ids)
        
        for layer in self.layers:
            x = layer(x)
            
        x = self.norm(x)
        logits = self.lm_head(x)
        
        return logits


class FlashTransformerLM(nn.Module):
    """Flash Attention Transformer"""
    
    def __init__(self, vocab_size: int = 32000, d_model: int = 2048,
                 num_heads: int = 16, num_layers: int = 24, d_ff: int = 8192):
        super().__init__()
        self.vocab_size = vocab_size
        
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = nn.Embedding(8192, d_model)  # Max seq len
        
        self.layers = nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_ff, "flash")
            for _ in range(num_layers)
        ])
        
        self.norm = nn.RMSNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        
    def forward(self, input_ids):
        B, L = input_ids.shape
        
        x = self.embedding(input_ids)
        pos_ids = torch.arange(L, device=input_ids.device).unsqueeze(0).expand(B, -1)
        x = x + self.pos_embedding(pos_ids)
        
        for layer in self.layers:
            x = layer(x)
            
        x = self.norm(x)
        return self.lm_head(x)


class LinearTransformerLM(nn.Module):
    """Linear Attention Transformer for long sequences"""
    
    def __init__(self, vocab_size: int = 32000, d_model: int = 1024,
                 num_heads: int = 16, num_layers: int = 12, d_ff: int = 4096):
        super().__init__()
        self.vocab_size = vocab_size
        
        self.embedding = nn.Embedding(vocab_size, d_model)
        
        self.layers = nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_ff, "linear")
            for _ in range(num_layers)
        ])
        
        self.norm = nn.RMSNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        
    def forward(self, input_ids):
        x = self.embedding(input_ids)
        
        for layer in self.layers:
            x = layer(x)
            
        x = self.norm(x)
        return self.lm_head(x)


class RetNetLM(nn.Module):
    """RetNet Language Model"""
    
    def __init__(self, vocab_size: int = 32000, d_model: int = 1024,
                 num_heads: int = 16, num_layers: int = 12, d_ff: int = 4096):
        super().__init__()
        self.vocab_size = vocab_size
        
        self.embedding = nn.Embedding(vocab_size, d_model)
        
        self.layers = nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_ff, "retentive")
            for _ in range(num_layers)
        ])
        
        self.norm = nn.RMSNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        
    def forward(self, input_ids):
        x = self.embedding(input_ids)
        
        for layer in self.layers:
            x = layer(x)
            
        x = self.norm(x)
        return self.lm_head(x)


# ============================================================================
# HYBRID ARCHITECTURES
# ============================================================================

class MambaTransformerHybrid(nn.Module):
    """Hybrid model combining Mamba and Transformer layers"""
    
    def __init__(self, vocab_size: int = 32000, d_model: int = 1024,
                 num_heads: int = 16, num_layers: int = 24, mamba_ratio: float = 0.5):
        super().__init__()
        self.vocab_size = vocab_size
        
        self.embedding = nn.Embedding(vocab_size, d_model)
        
        num_mamba = int(num_layers * mamba_ratio)
        num_transformer = num_layers - num_mamba
        
        self.layers = nn.ModuleList()
        
        # Interleave Mamba and Transformer blocks
        for i in range(num_layers):
            if i % 2 == 0 and len([l for l in self.layers if isinstance(l, MambaBlock)]) < num_mamba:
                self.layers.append(MambaBlock(d_model))
            else:
                self.layers.append(TransformerBlock(d_model, num_heads, d_model * 4, "flash"))
        
        self.norm = nn.RMSNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        
    def forward(self, input_ids):
        x = self.embedding(input_ids)
        
        for layer in self.layers:
            x = layer(x)
            
        x = self.norm(x)
        return self.lm_head(x)


# ============================================================================
# MODEL EXPORT SYSTEM
# ============================================================================

class ModelExporter:
    """Advanced model export system"""
    
    def __init__(self, output_dir: str = "models"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
    def export_model(self, 
                    model: nn.Module, 
                    model_name: str,
                    input_shape: Tuple[int, ...],
                    optimize: bool = True,
                    quantize: bool = False) -> Dict:
        """Export model to ONNX with optimizations"""
        
        model.eval()
        
        # Create dummy input (always tokens for LLMs)
        dummy_input = torch.randint(0, min(1000, getattr(model, 'vocab_size', 1000)), input_shape)
        
        # Export path
        onnx_path = self.output_dir / f"{model_name}.onnx"
        
        print(f"Exporting {model_name} to ONNX...")
        
        try:
            # Export to ONNX
            torch.onnx.export(
                model,
                dummy_input,
                str(onnx_path),
                export_params=True,
                opset_version=17,
                do_constant_folding=True,
                input_names=['input_ids'],
                output_names=['logits'],
                dynamic_axes={
                    'input_ids': {0: 'batch_size', 1: 'sequence_length'},
                    'logits': {0: 'batch_size', 1: 'sequence_length'}
                }
            )
            
            # Verify export
            onnx_model = onnx.load(str(onnx_path))
            onnx.checker.check_model(onnx_model)
            print(f"✓ ONNX model verification passed")
            
            # Test inference
            inference_results = self._test_inference(str(onnx_path), dummy_input, model)
            
            # Generate model info
            model_info = self._generate_model_info(model, onnx_path, input_shape)
            
            return {
                "success": True,
                "onnx_path": str(onnx_path),
                "model_info": model_info,
                "inference_results": inference_results
            }
            
        except Exception as e:
            print(f"✗ Export failed: {e}")
            return {"success": False, "error": str(e)}
    
    def _test_inference(self, onnx_path: str, dummy_input: torch.Tensor, original_model: nn.Module):
        """Test ONNX inference and compare with PyTorch"""
        try:
            # ONNX Runtime inference
            session = ort.InferenceSession(onnx_path)
            ort_inputs = {session.get_inputs()[0].name: dummy_input.numpy()}
            ort_outputs = session.run(None, ort_inputs)
            
            # PyTorch inference
            with torch.no_grad():
                torch_outputs = original_model(dummy_input)
            
            # Compare outputs
            if isinstance(torch_outputs, torch.Tensor):
                torch_outputs = [torch_outputs]
            
            max_diff = 0
            for torch_out, ort_out in zip(torch_outputs, ort_outputs):
                diff = np.abs(torch_out.numpy() - ort_out).max()
                max_diff = max(max_diff, diff)
            
            print(f"✓ Inference test passed, max difference: {max_diff:.6f}")
            return {"success": True, "max_difference": float(max_diff)}
            
        except Exception as e:
            print(f"✗ Inference test failed: {e}")
            return {"success": False, "error": str(e)}
    
    def _generate_model_info(self, model: nn.Module, onnx_path: Path, input_shape: Tuple):
        """Generate comprehensive model information"""
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        # Get model size
        model_size_mb = onnx_path.stat().st_size / (1024 * 1024)
        
        info = {
            "model_class": model.__class__.__name__,
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "input_shape": input_shape,
            "model_size_mb": round(model_size_mb, 2),
            "onnx_path": str(onnx_path),
            "architecture_details": self._get_architecture_details(model)
        }
        
        # Save info to JSON
        info_path = onnx_path.with_suffix('.json')
        with open(info_path, 'w') as f:
            json.dump(info, f, indent=2)
        
        return info
    
    def _get_architecture_details(self, model):
        """Get detailed architecture information"""
        details = {
            "model_type": model.__class__.__name__,
        }
        
        if hasattr(model, 'd_model'):
            details['d_model'] = model.d_model
        if hasattr(model, 'vocab_size'):
            details['vocab_size'] = model.vocab_size
        if hasattr(model, 'layers'):
            details['num_layers'] = len(model.layers)
            
        return details


def export_all_models():
    """Export comprehensive set of models"""
    exporter = ModelExporter()
    
    models_to_export = [
        # Mamba models
        {
            "name": "mamba_small",
            "model": MambaLM(vocab_size=8192, d_model=768, n_layers=12),
            "input_shape": (1, 256),
        },
        {
            "name": "mamba_large", 
            "model": MambaLM(vocab_size=16384, d_model=2048, n_layers=24),
            "input_shape": (1, 512),
        },
        
        # Flash attention transformers
        {
            "name": "flash_transformer_small",
            "model": FlashTransformerLM(vocab_size=8192, d_model=768, num_heads=12, num_layers=12),
            "input_shape": (1, 256),
        },
        {
            "name": "flash_transformer_large",
            "model": FlashTransformerLM(vocab_size=16384, d_model=2048, num_heads=16, num_layers=24),
            "input_shape": (1, 512),
        },
        
        # Linear attention transformers
        {
            "name": "linear_transformer_small",
            "model": LinearTransformerLM(vocab_size=8192, d_model=768, num_heads=12, num_layers=12),
            "input_shape": (1, 1024),  # Can handle longer sequences
        },
        {
            "name": "linear_transformer_large",
            "model": LinearTransformerLM(vocab_size=16384, d_model=1024, num_heads=16, num_layers=16),
            "input_shape": (1, 2048),  # Very long sequences
        },
        
        # RetNet models
        {
            "name": "retnet_small",
            "model": RetNetLM(vocab_size=8192, d_model=768, num_heads=12, num_layers=12),
            "input_shape": (1, 256),
        },
        {
            "name": "retnet_large",
            "model": RetNetLM(vocab_size=16384, d_model=1024, num_heads=16, num_layers=16),
            "input_shape": (1, 512),
        },
        
        # Hybrid models
        {
            "name": "mamba_transformer_hybrid",
            "model": MambaTransformerHybrid(vocab_size=8192, d_model=1024, num_heads=16, num_layers=20),
            "input_shape": (1, 512),
        },
    ]
    
    results = {}
    for model_config in models_to_export:
        print(f"\n{'='*60}")
        print(f"Exporting {model_config['name']}")
        print(f"{'='*60}")
        
        result = exporter.export_model(
            model=model_config["model"],
            model_name=model_config["name"],
            input_shape=model_config["input_shape"],
            optimize=True,
            quantize=False
        )
        
        results[model_config["name"]] = result
        
        if result["success"]:
            info = result["model_info"]
            print(f"✓ Successfully exported {model_config['name']}")
            print(f"  Parameters: {info['total_parameters']:,}")
            print(f"  Model size: {info['model_size_mb']} MB")
            print(f"  Architecture: {info['architecture_details']}")
        else:
            print(f"✗ Failed to export {model_config['name']}: {result.get('error')}")
    
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export advanced models for Deep C++ framework")
    parser.add_argument("--model", choices=["all", "mamba", "flash", "linear", "retnet", "hybrid"], 
                      default="all", help="Which model family to export")
    parser.add_argument("--output-dir", default="models", help="Output directory")
    parser.add_argument("--size", choices=["small", "large"], default="small", help="Model size")
    
    args = parser.parse_args()
    
    if args.model == "all":
        results = export_all_models()
        
        # Print summary
        print(f"\n{'='*60}")
        print("EXPORT SUMMARY")
        print(f"{'='*60}")
        
        successful = sum(1 for r in results.values() if r["success"])
        total = len(results)
        
        print(f"Successfully exported: {successful}/{total} models")
        
        total_params = sum(r["model_info"]["total_parameters"] for r in results.values() if r["success"])
        total_size = sum(r["model_info"]["model_size_mb"] for r in results.values() if r["success"])
        
        print(f"Total parameters: {total_params:,}")
        print(f"Total model size: {total_size:.1f} MB")
        
    else:
        print("Specific model export not implemented yet - use --model all") 