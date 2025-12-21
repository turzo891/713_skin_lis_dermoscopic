#!/usr/bin/env python3
"""
Quick test to verify EfficientNetDualPath model works correctly.
"""

import sys
sys.path.insert(0, 'src')

import torch
from models import EfficientNetDualPath, get_model, count_parameters

def test_dualpath_model():
    print("="*80)
    print("Testing EfficientNet Dual-Path Model")
    print("="*80)

    # Create model
    print("\n1. Creating model...")
    model = get_model('efficientnet_dualpath', num_classes=8, pretrained=False)
    print("✓ Model created successfully")

    # Count parameters
    total, trainable = count_parameters(model)
    print(f"\n2. Model Parameters:")
    print(f"   Total: {total:,}")
    print(f"   Trainable: {trainable:,}")

    # Test forward pass
    print("\n3. Testing forward pass...")
    batch_size = 4
    dummy_input = torch.randn(batch_size, 3, 384, 384)

    model.eval()
    with torch.no_grad():
        output = model(dummy_input)

    print(f"   Input shape: {dummy_input.shape}")
    print(f"   Output shape: {output.shape}")
    assert output.shape == (batch_size, 8), f"Expected shape (4, 8), got {output.shape}"
    print("✓ Forward pass successful")

    # Test feature extraction
    print("\n4. Testing feature extraction...")
    with torch.no_grad():
        features = model.get_features(dummy_input)
    print(f"   Feature shape: {features.shape}")
    print(f"   Expected: GAP (1792) + GMP (1792) = 3584 features")
    assert features.shape == (batch_size, 3584), f"Expected (4, 3584), got {features.shape}"
    print("✓ Feature extraction successful")

    # Test architecture components
    print("\n5. Architecture Components:")
    print(f"   Backbone: {type(model.backbone).__name__}")
    print(f"   GAP path: {len(model.gap_path)} layers")
    print(f"   GMP path: {len(model.gmp_path)} layers")
    print(f"   Classifier: {len(model.classifier)} layers")

    print("\n" + "="*80)
    print("✓ All tests passed! Model is ready to use.")
    print("="*80)

    # Print comparison
    print("\n6. Comparison with Regular EfficientNet:")
    regular_model = get_model('efficientnet', num_classes=8, pretrained=False)
    reg_total, reg_trainable = count_parameters(regular_model)

    print(f"\n   Regular EfficientNet:")
    print(f"   - Parameters: {reg_total:,}")
    print(f"\n   Dual-Path EfficientNet:")
    print(f"   - Parameters: {total:,}")
    print(f"\n   Difference: +{total - reg_total:,} parameters")
    print(f"   Ratio: {total / reg_total:.2f}x")

    return True

if __name__ == "__main__":
    try:
        test_dualpath_model()
    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
