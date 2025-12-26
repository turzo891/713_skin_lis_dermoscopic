#!/usr/bin/env python3
"""Test multi-modal data loader"""

import sys
sys.path.insert(0, 'src')

from multimodal_dataloader import create_multimodal_data_loaders
import torch

# Paths
data_path = 'data/ISIC2019/ISIC_2019_Training_Input/ISIC_2019_Training_Input'
csv_path = 'data/ISIC2019/ISIC_2019_Training_GroundTruth.csv'
metadata_path = 'data/ISIC2019/ISIC_2019_Training_Metadata.csv'

print("Testing Multi-Modal Data Loader...")
print("="*80)

# Create loaders
train_loader, val_loader, test_loader, label_encoder, metadata_dim = create_multimodal_data_loaders(
    data_path=data_path,
    csv_path=csv_path,
    metadata_path=metadata_path,
    batch_size=8,
    image_size=384,
    num_workers=0  # Use 0 for testing
)

print("\n" + "="*80)
print("Testing Data Loading...")
print("="*80)

# Test one batch
for images, metadata, labels in train_loader:
    print(f"\nBatch shapes:")
    print(f"  Images:   {images.shape}")
    print(f"  Metadata: {metadata.shape}")
    print(f"  Labels:   {labels.shape}")

    print(f"\nMetadata sample (first item):")
    print(f"  {metadata[0]}")

    print(f"\nImage stats:")
    print(f"  Min: {images.min():.3f}")
    print(f"  Max: {images.max():.3f}")
    print(f"  Mean: {images.mean():.3f}")

    print(f"\nLabel distribution in batch:")
    for i in range(8):
        if i < len(labels):
            print(f"  {label_encoder[labels[i].item()]}: {labels[i].item()}")

    break

print("\n" + "="*80)
print(" Multi-Modal Data Loader Test PASSED!")
print("="*80)
