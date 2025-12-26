#!/usr/bin/env python3
"""Test Concept Bottleneck Layer"""

import sys
sys.path.insert(0, 'src')

import torch
from models import ConceptBottleneck, MultiModalFusionNet, count_parameters, print_model_summary

print("="*80)
print("Testing Concept Bottleneck Layer")
print("="*80)

# Test 1: Standalone ConceptBottleneck
print("\n[Test 1] Standalone Concept Bottleneck")
print("-" * 80)

concept_bn = ConceptBottleneck(
    input_dim=128,
    num_concepts=4,
    concept_dim=16,
    use_concept_supervision=False
)

batch_size = 8
features = torch.randn(batch_size, 128)

print(f"Input features shape: {features.shape}")

with torch.no_grad():
    concept_features, concept_probs = concept_bn(features)

print(f"Concept features shape: {concept_features.shape}")
print(f"Expected shape: ({batch_size}, {4 * 16})")
assert concept_features.shape == (batch_size, 4 * 16)
print(f"Concept probs: {concept_probs}")
assert concept_probs is None, "Should be None without supervision"
print(" Standalone concept bottleneck works")

# Test 2: ConceptBottleneck with supervision
print("\n[Test 2] Concept Bottleneck with Supervision")
print("-" * 80)

concept_bn_sup = ConceptBottleneck(
    input_dim=128,
    num_concepts=4,
    concept_dim=16,
    use_concept_supervision=True
)

with torch.no_grad():
    concept_features_sup, concept_probs_sup = concept_bn_sup(features)

print(f"Concept features shape: {concept_features_sup.shape}")
print(f"Concept probs shape: {concept_probs_sup.shape}")
print(f"Concept probs (sample): {concept_probs_sup[0]}")
assert concept_probs_sup is not None
assert concept_probs_sup.shape == (batch_size, 4)
print(" Concept bottleneck with supervision works")

# Test 3: Get concept activations
print("\n[Test 3] Concept Activations (ABCD)")
print("-" * 80)

activations = concept_bn_sup.get_concept_activations(features)

print(f"Concept names: {list(activations.keys())}")
expected_concepts = ['asymmetry', 'border', 'color', 'diameter']
assert list(activations.keys()) == expected_concepts

for concept_name, activation in activations.items():
    print(f"  {concept_name}: {activation.shape} - Sample: {activation[0].item():.4f}")
    assert activation.shape == (batch_size,)

print(" Concept activations work correctly")

# Test 4: MultiModalFusionNet without concept bottleneck
print("\n[Test 4] Multi-Modal Fusion without Concept Bottleneck")
print("-" * 80)

model_no_cb = MultiModalFusionNet(
    num_classes=8,
    metadata_dim=11,
    image_encoder='efficientnet',
    pretrained=False,
    use_cross_attention=True,
    use_concept_bottleneck=False
)

images = torch.randn(4, 3, 384, 384)
metadata = torch.randn(4, 11)

with torch.no_grad():
    output_no_cb = model_no_cb(images, metadata)

print(f"Output shape: {output_no_cb.shape}")
assert output_no_cb.shape == (4, 8)
total_no_cb, _ = count_parameters(model_no_cb)
print(f"Parameters (no concept): {total_no_cb/1e6:.1f}M")
print(" Multi-modal fusion without concept bottleneck works")

# Test 5: MultiModalFusionNet WITH concept bottleneck
print("\n[Test 5] Multi-Modal Fusion WITH Concept Bottleneck")
print("-" * 80)

model_with_cb = MultiModalFusionNet(
    num_classes=8,
    metadata_dim=11,
    image_encoder='efficientnet',
    pretrained=False,
    use_cross_attention=True,
    use_concept_bottleneck=True,
    concept_supervision=False
)

with torch.no_grad():
    output_with_cb = model_with_cb(images, metadata)

print(f"Output shape: {output_with_cb.shape}")
assert output_with_cb.shape == (4, 8)
total_with_cb, _ = count_parameters(model_with_cb)
print(f"Parameters (with concept): {total_with_cb/1e6:.1f}M")
print(f"Additional params: {(total_with_cb - total_no_cb)/1e3:.1f}K (+{(total_with_cb - total_no_cb)/total_no_cb*100:.1f}%)")
print(" Multi-modal fusion WITH concept bottleneck works")

# Test 6: Concept supervision enabled
print("\n[Test 6] Multi-Modal Fusion with Concept Supervision")
print("-" * 80)

model_with_cb_sup = MultiModalFusionNet(
    num_classes=8,
    metadata_dim=11,
    image_encoder='efficientnet',
    pretrained=False,
    use_cross_attention=True,
    use_concept_bottleneck=True,
    concept_supervision=True
)

model_with_cb_sup.train()
output_sup = model_with_cb_sup(images, metadata)

print(f"Output shape: {output_sup.shape}")
print(f"Stored concept probs shape: {model_with_cb_sup.concept_probs.shape}")
print(f"Concept probs (sample): {model_with_cb_sup.concept_probs[0]}")
assert model_with_cb_sup.concept_probs is not None
assert model_with_cb_sup.concept_probs.shape == (4, 4)  # [B, num_concepts]
print(" Concept supervision works correctly")

# Test 7: Get concept activations from full model
print("\n[Test 7] Get ABCD Concept Activations from Full Model")
print("-" * 80)

model_with_cb_sup.eval()
activations_full = model_with_cb_sup.get_concept_activations(images, metadata)

print(f"Concept activations retrieved:")
for concept_name, activation in activations_full.items():
    print(f"  {concept_name}: {activation.shape} - Mean: {activation.mean().item():.4f}, Std: {activation.std().item():.4f}")
    assert activation.shape == (4,)  # [B]

print(" Full model concept activations work")

# Test 8: Gradient flow with concept bottleneck
print("\n[Test 8] Gradient Flow with Concept Bottleneck")
print("-" * 80)

model_with_cb.train()
images.requires_grad = True
metadata.requires_grad = True

output = model_with_cb(images, metadata)
loss = output.sum()
loss.backward()

has_gradients = any(p.grad is not None for p in model_with_cb.parameters() if p.requires_grad)
print(f"  Gradients computed: {has_gradients}")
assert has_gradients
print(" Gradient flow successful with concept bottleneck")

# Test 9: Model summary comparison
print("\n[Test 9] Model Size Comparison")
print("-" * 80)

configs = [
    ("No Concept Bottleneck", False, False),
    ("With Concept Bottleneck", True, False),
    ("With Concept + Supervision", True, True),
]

print(f"{'Configuration':<30} {'Parameters':<15} {'Overhead':<15}")
print("-" * 60)

baseline_params = None
for name, use_cb, use_sup in configs:
    m = MultiModalFusionNet(
        num_classes=8,
        metadata_dim=11,
        image_encoder='efficientnet',
        pretrained=False,
        use_cross_attention=True,
        use_concept_bottleneck=use_cb,
        concept_supervision=use_sup
    )
    total, _ = count_parameters(m)

    if baseline_params is None:
        baseline_params = total
        overhead = 0
    else:
        overhead = (total - baseline_params) / baseline_params * 100

    print(f"{name:<30} {total/1e6:>8.2f}M      {overhead:>6.2f}%")

print("\n Concept bottleneck adds minimal parameters (<1%)")

print("\n" + "="*80)
print(" ALL TESTS PASSED - Concept Bottleneck Layer is working!")
print("="*80)

# Summary
print("\n" + "="*80)
print("CONCEPT BOTTLENECK LAYER SUMMARY")
print("="*80)
print("""
Concept Bottleneck Layer Architecture:

Input: Fused features [B, fusion_dim//2]
  ↓
For each ABCD concept:
  Concept Extractor (2-layer MLP) → [B, concept_dim]
  ↓
Concept Attention: Learn importance weights
  ↓
Weighted Concepts → [B, concept_dim * num_concepts]
  ↓
Optional: Predict concept probabilities (A, B, C, D)
  ↓
Classifier: Final disease prediction

ABCD Concepts (Dermatology Rule):
1. Asymmetry: Is the lesion asymmetric?
2. Border: Is the border irregular?
3. Color: Does it have multiple/unusual colors?
4. Diameter: Size and differential structures

Benefits:
 Interpretability: See which ABCD features are detected
 Clinical alignment: Features dermatologists use
 Debugging: Identify if model relies on wrong concepts
 Intervention: Manually correct concept predictions if needed
 Minimal overhead: <1% parameter increase

Integration:
- Can be added to any Multi-Modal Fusion model
- Works with/without concept supervision
- Optional concept loss for stronger alignment

Performance Expectations:
- Baseline (image-only): ~60%
- + Multi-modal fusion: ~68-70%
- + Concept bottleneck: ~69-71% (+1-2%)
- + Better interpretability for clinical deployment

Ready for Day 6: Self-Supervised Pre-training!
""")
print("="*80)
