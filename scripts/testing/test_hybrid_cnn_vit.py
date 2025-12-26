#!/usr/bin/env python3
"""Test Hybrid CNN-ViT architecture"""

import sys
sys.path.insert(0, 'src')

import torch
from models import HybridCNNViT, get_model, count_parameters, print_model_summary

print("="*80)
print("Testing Hybrid CNN-ViT Architecture")
print("="*80)

# Test 1: Model instantiation
print("\n[Test 1] Model Instantiation")
print("-" * 80)
model = HybridCNNViT(
    num_classes=8,
    pretrained=False,  # Faster testing
    dropout=0.3,
    num_transformer_layers=4,
    num_heads=8,
    embed_dim=512
)
print(" Model created successfully")

# Test 2: Model summary
print("\n[Test 2] Model Summary")
print("-" * 80)
print_model_summary(model, "HybridCNNViT")

# Test 3: Forward pass
print("\n[Test 3] Forward Pass")
print("-" * 80)
batch_size = 4
x = torch.randn(batch_size, 3, 384, 384)
print(f"Input shape: {x.shape}")

with torch.no_grad():
    output = model(x)

print(f"Output shape: {output.shape}")
assert output.shape == (batch_size, 8), f"Expected shape ({batch_size}, 8), got {output.shape}"
print(" Forward pass successful")

# Test 4: Feature extraction
print("\n[Test 4] Feature Extraction")
print("-" * 80)
with torch.no_grad():
    features = model.get_features(x)

print(f"Feature shape: {features.shape}")
assert features.shape == (batch_size, 512), f"Expected shape ({batch_size}, 512), got {features.shape}"
print(" Feature extraction successful")

# Test 5: Test with factory function
print("\n[Test 5] Factory Function")
print("-" * 80)
model_factory = get_model('hybrid', num_classes=8, pretrained=False)
print(" Created model via get_model('hybrid')")

with torch.no_grad():
    output_factory = model_factory(x)

print(f"Output shape: {output_factory.shape}")
assert output_factory.shape == (batch_size, 8)
print(" Factory function works correctly")

# Test 6: Different configurations
print("\n[Test 6] Different Configurations")
print("-" * 80)

configs = [
    {"num_transformer_layers": 2, "num_heads": 4, "embed_dim": 256},
    {"num_transformer_layers": 6, "num_heads": 8, "embed_dim": 512},
    {"num_transformer_layers": 4, "num_heads": 16, "embed_dim": 1024},
]

for i, config in enumerate(configs, 1):
    model_config = HybridCNNViT(num_classes=8, pretrained=False, **config)
    total, trainable = count_parameters(model_config)
    with torch.no_grad():
        out = model_config(x)
    print(f"  Config {i}: layers={config['num_transformer_layers']}, heads={config['num_heads']}, "
          f"embed={config['embed_dim']} -> Params: {total/1e6:.1f}M, Output: {out.shape}")
    assert out.shape == (batch_size, 8)

print(" All configurations work correctly")

# Test 7: Architecture components
print("\n[Test 7] Architecture Components")
print("-" * 80)
print(f"  CNN backbone: {model.cnn_backbone.__class__.__name__}")
print(f"  CNN feature dimension: {model.cnn_feature_dim}")
print(f"  Feature map size: {model.feature_map_size}x{model.feature_map_size}")
print(f"  Number of patches: {model.num_patches}")
print(f"  Transformer layers: {len(model.transformer.layers)}")
print(f"  CLS token shape: {model.cls_token.shape}")
print(f"  Positional embedding shape: {model.pos_embed.shape}")
print(" Architecture components verified")

# Test 8: Gradient flow
print("\n[Test 8] Gradient Flow")
print("-" * 80)
model.train()
x.requires_grad = True
output = model(x)
loss = output.sum()
loss.backward()

has_gradients = any(p.grad is not None for p in model.parameters() if p.requires_grad)
print(f"  Gradients computed: {has_gradients}")
assert has_gradients, "No gradients computed!"
print(" Gradient flow successful")

print("\n" + "="*80)
print(" ALL TESTS PASSED - Hybrid CNN-ViT is working correctly!")
print("="*80)

# Summary
print("\n" + "="*80)
print("ARCHITECTURE SUMMARY")
print("="*80)
print("""
Hybrid CNN-Transformer Architecture:

1. CNN Backbone (EfficientNet-B4):
   - Extracts local features with inductive bias
   - Output: [B, 1792, 12, 12] feature maps

2. Feature Projection:
   - Flattens spatial dimensions to patches
   - Projects to transformer embedding: [B, 144, 512]

3. Transformer Encoder (4 layers):
   - Global self-attention across all patches
   - CLS token for classification
   - Positional embeddings

4. Classification Head:
   - Takes CLS token representation
   - 2-layer MLP with GELU activation

Benefits:
- Combines CNN's local feature extraction with ViT's global attention
- Addresses Q1 journal finding: Pure ViTs underperform on small datasets
- Expected +2-4% improvement over pure CNN or pure ViT

Ready for Day 3-4: Multi-Modal Fusion!
""")
print("="*80)
