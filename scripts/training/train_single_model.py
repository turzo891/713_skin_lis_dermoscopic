#!/usr/bin/env python3
"""
Train a single model with comprehensive logging
Usage: python3 train_single_model.py --model <model_name>
"""

import os
import sys
import argparse
from pathlib import Path

# Add src to path
sys.path.insert(0, 'src')

# Import from train_with_logging
from train_with_logging import (
    setup_output_directory,
    save_training_info,
    save_final_results,
    save_predictions,
    save_confusion_matrix
)
from data_loader import create_data_loaders
from models import get_model
from train import train
from evaluate import get_predictions, calculate_metrics
from utils import set_seed, get_device
import torch
import pandas as pd
from datetime import datetime


def main():
    parser = argparse.ArgumentParser(description="Train a single model")
    parser.add_argument('--model', type=str, required=True,
                       choices=['resnet50', 'efficientnet', 'densenet', 'vit', 'swin'],
                       help='Model to train')
    parser.add_argument('--data_path', type=str, default='data/ISIC2019',
                       help='Path to dataset')
    parser.add_argument('--csv_path', type=str,
                       default='data/ISIC2019/ISIC_2019_Training_GroundTruth.csv',
                       help='Path to metadata CSV')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=64,
                       help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4,
                       help='Learning rate')
    parser.add_argument('--image_size', type=int, default=224,
                       help='Image size')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--use_focal_loss', action='store_true',
                       help='Use Focal Loss instead of CrossEntropy')
    parser.add_argument('--focal_gamma', type=float, default=2.0,
                       help='Gamma parameter for Focal Loss')
    parser.add_argument('--patience', type=int, default=10,
                       help='Early stopping patience')
    args = parser.parse_args()

    print("="*80)
    print(f"TRAINING: {args.model.upper()}")
    print("="*80)

    # Setup
    set_seed(args.seed)
    device = get_device()
    print(f"\n✓ Device: {device}")

    # Create output directories
    dirs = setup_output_directory(args.model)
    print(f"✓ Output directory: {dirs['root']}")

    # Training configuration
    config = {
        'model': args.model,
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'lr': args.lr,
        'image_size': args.image_size,
        'seed': args.seed,
        'use_amp': True,
        'use_focal_loss': args.use_focal_loss if hasattr(args, 'use_focal_loss') else False,
        'focal_gamma': args.focal_gamma if hasattr(args, 'focal_gamma') else 2.0,
        'patience': args.patience if hasattr(args, 'patience') else 10,
        'weight_decay': 0.01,
        'max_grad_norm': 1.0,
        'T_0': 10,
        'T_mult': 2
    }

    # Load data
    print("\n" + "="*80)
    print("LOADING DATA")
    print("="*80)
    train_loader, val_loader, test_loader, label_encoder = create_data_loaders(
        args.data_path, args.csv_path, args.batch_size, args.image_size
    )

    num_classes = len(label_encoder)
    class_names = list(label_encoder.keys())

    print(f"✓ Train samples: {len(train_loader.dataset)}")
    print(f"✓ Validation samples: {len(val_loader.dataset)}")
    print(f"✓ Test samples: {len(test_loader.dataset)}")
    print(f"✓ Number of classes: {num_classes}")
    print(f"✓ Classes: {', '.join(class_names)}")

    # Save training info
    save_training_info(dirs, config, args.model, {
        'train_size': len(train_loader.dataset),
        'val_size': len(val_loader.dataset),
        'test_size': len(test_loader.dataset),
        'num_classes': num_classes,
        'class_names': class_names
    })

    # Create model
    print("\n" + "="*80)
    print("CREATING MODEL")
    print("="*80)
    model = get_model(args.model, num_classes=num_classes, pretrained=True).to(device)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"✓ Total parameters: {total_params:,}")
    print(f"✓ Trainable parameters: {trainable_params:,}")

    # Train
    print("\n" + "="*80)
    print("TRAINING")
    print("="*80)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.lr}")
    print("-"*80)

    history = train(model, train_loader, val_loader, config, device, str(dirs['checkpoints']))

    # Save training history
    history_df = pd.DataFrame(history)
    history_df.to_csv(dirs['metrics'] / 'training_history.csv', index=False)

    # Final evaluation
    print("\n" + "="*80)
    print("FINAL EVALUATION")
    print("="*80)

    # Load best model
    best_model_path = dirs['checkpoints'] / 'best_model.pth'
    checkpoint = torch.load(best_model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # Evaluate on all splits
    print("Evaluating on training set...")
    train_pred, train_true, train_prob = get_predictions(model, train_loader, device)
    train_metrics = calculate_metrics(train_true, train_pred, train_prob, class_names)

    print("Evaluating on validation set...")
    val_pred, val_true, val_prob = get_predictions(model, val_loader, device)
    val_metrics = calculate_metrics(val_true, val_pred, val_prob, class_names)

    print("Evaluating on test set...")
    test_pred, test_true, test_prob = get_predictions(model, test_loader, device)
    test_metrics = calculate_metrics(test_true, test_pred, test_prob, class_names)

    # Save predictions
    print("\nSaving predictions...")
    save_predictions(dirs, 'train', train_true, train_pred, train_prob, class_names)
    save_predictions(dirs, 'val', val_true, val_pred, val_prob, class_names)
    save_predictions(dirs, 'test', test_true, test_pred, test_prob, class_names)

    # Save confusion matrices
    print("Saving confusion matrices...")
    save_confusion_matrix(dirs, 'train', train_true, train_pred, class_names)
    save_confusion_matrix(dirs, 'val', val_true, val_pred, class_names)
    save_confusion_matrix(dirs, 'test', test_true, test_pred, class_names)

    # Save final results
    save_final_results(dirs, train_metrics, val_metrics, test_metrics, class_names)

    print("\n" + "="*80)
    print("TRAINING COMPLETE!")
    print("="*80)
    print(f"✓ All outputs saved to: {dirs['root']}")
    print(f"\nTest Results:")
    print(f"  - Accuracy: {test_metrics.get('accuracy', 0):.4f}")
    print(f"  - F1 Score (Macro): {test_metrics.get('f1_macro', 0):.4f}")
    print(f"  - F1 Score (Weighted): {test_metrics.get('f1_weighted', 0):.4f}")
    print(f"  - AUC-ROC: {test_metrics.get('auc_roc', 0):.4f}")
    print("="*80)


if __name__ == "__main__":
    main()
