#!/usr/bin/env python3
"""
Enhanced training script with comprehensive output logging
"""

import os
import sys
import json
import argparse
from datetime import datetime
from pathlib import Path
import pandas as pd
import torch

# Add src to path
sys.path.insert(0, 'src')

from data_loader import create_data_loaders
from models import get_model
from train import train
from evaluate import evaluate_model, get_predictions, calculate_metrics
from utils import set_seed, get_device, load_config

def setup_output_directory(model_name):
    """Create comprehensive output directory structure"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(f"models/{model_name}_{timestamp}")

    dirs = {
        'root': output_dir,
        'checkpoints': output_dir / 'checkpoints',
        'logs': output_dir / 'logs',
        'metrics': output_dir / 'metrics',
        'predictions': output_dir / 'predictions',
        'config': output_dir / 'config'
    }

    for dir_path in dirs.values():
        dir_path.mkdir(parents=True, exist_ok=True)

    return dirs

def save_training_info(dirs, config, model_name, data_info):
    """Save training configuration and dataset info"""
    info = {
        'model_name': model_name,
        'timestamp': datetime.now().isoformat(),
        'config': config,
        'dataset_info': data_info,
        'device': str(get_device())
    }

    with open(dirs['config'] / 'training_info.json', 'w') as f:
        json.dump(info, f, indent=2)

def save_epoch_metrics(dirs, epoch, metrics, split='train'):
    """Save metrics for each epoch"""
    metrics_file = dirs['metrics'] / f'{split}_metrics.jsonl'

    epoch_data = {
        'epoch': epoch,
        'timestamp': datetime.now().isoformat(),
        **metrics
    }

    with open(metrics_file, 'a') as f:
        f.write(json.dumps(epoch_data) + '\n')

def save_predictions(dirs, split_name, true_labels, pred_labels, probabilities, class_names, epoch=None):
    """Save predictions with probabilities for a data split"""
    suffix = f"_epoch{epoch}" if epoch is not None else ""
    pred_file = dirs['predictions'] / f'{split_name}_predictions{suffix}.csv'

    pred_df = pd.DataFrame({
        'true_label': true_labels,
        'true_class': [class_names[i] for i in true_labels],
        'predicted_label': pred_labels,
        'predicted_class': [class_names[i] for i in pred_labels],
        **{f'prob_{class_names[i]}': probabilities[:, i] for i in range(len(class_names))}
    })

    pred_df.to_csv(pred_file, index=False)
    return pred_file

def save_confusion_matrix(dirs, split_name, true_labels, pred_labels, class_names):
    """Save confusion matrix as CSV"""
    from sklearn.metrics import confusion_matrix

    cm = confusion_matrix(true_labels, pred_labels)
    cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)

    cm_file = dirs['metrics'] / f'{split_name}_confusion_matrix.csv'
    cm_df.to_csv(cm_file)

    return cm_file

def save_final_results(dirs, train_metrics, val_metrics, test_metrics, class_names):
    """Save final comprehensive results"""
    results = {
        'training': train_metrics,
        'validation': val_metrics,
        'test': test_metrics,
        'timestamp': datetime.now().isoformat()
    }

    # Save as JSON
    with open(dirs['root'] / 'final_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    # Save summary as CSV for easy viewing
    results_df = pd.DataFrame({
        'Split': ['Train', 'Validation', 'Test'],
        'Accuracy': [
            train_metrics.get('accuracy', 0),
            val_metrics.get('accuracy', 0),
            test_metrics.get('accuracy', 0)
        ],
        'F1_Macro': [
            train_metrics.get('f1_macro', 0),
            val_metrics.get('f1_macro', 0),
            test_metrics.get('f1_macro', 0)
        ],
        'F1_Weighted': [
            train_metrics.get('f1_weighted', 0),
            val_metrics.get('f1_weighted', 0),
            test_metrics.get('f1_weighted', 0)
        ],
        'Precision': [
            train_metrics.get('precision_macro', 0),
            val_metrics.get('precision_macro', 0),
            test_metrics.get('precision_macro', 0)
        ],
        'Recall': [
            train_metrics.get('recall_macro', 0),
            val_metrics.get('recall_macro', 0),
            test_metrics.get('recall_macro', 0)
        ],
        'AUC_ROC': [
            train_metrics.get('auc_roc', 0),
            val_metrics.get('auc_roc', 0),
            test_metrics.get('auc_roc', 0)
        ]
    })
    results_df.to_csv(dirs['root'] / 'final_results.csv', index=False)

    # Save per-class metrics
    per_class_metrics = []
    for split_name, metrics in [('Train', train_metrics), ('Val', val_metrics), ('Test', test_metrics)]:
        if 'per_class_f1' in metrics:
            for idx, f1_score in enumerate(metrics['per_class_f1']):
                class_name = class_names[idx] if idx < len(class_names) else f'class_{idx}'
                per_class_metrics.append({
                    'Split': split_name,
                    'Class': class_name,
                    'F1_Score': f1_score,
                    'Precision': metrics['per_class_precision'][idx] if 'per_class_precision' in metrics else 0,
                    'Recall': metrics['per_class_recall'][idx] if 'per_class_recall' in metrics else 0
                })

    if per_class_metrics:
        per_class_df = pd.DataFrame(per_class_metrics)
        per_class_df.to_csv(dirs['root'] / 'per_class_metrics.csv', index=False)

    print("\n" + "=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)
    print(results_df.to_string(index=False))
    print("=" * 60)

def main():
    parser = argparse.ArgumentParser(description="Train with comprehensive logging")
    parser.add_argument('--model', type=str, required=True,
                       choices=['resnet50', 'efficientnet', 'densenet', 'vit', 'swin'])
    parser.add_argument('--data_path', type=str, default='data/HAM10000')
    parser.add_argument('--csv_path', type=str, default='data/HAM10000/HAM10000_metadata.csv')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--image_size', type=int, default=224)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    # Setup
    set_seed(args.seed)
    device = get_device()
    print(f"\nUsing device: {device}")

    # Create output directories
    dirs = setup_output_directory(args.model)
    print(f"Output directory: {dirs['root']}")

    # Training configuration
    config = {
        'model': args.model,
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'lr': args.lr,
        'image_size': args.image_size,
        'seed': args.seed,
        'use_amp': True,
        'use_focal_loss': False,
        'patience': 10,
        'weight_decay': 0.01,
        'max_grad_norm': 1.0,
        'T_0': 10,
        'T_mult': 2
    }

    # Load data
    print("\nLoading dataset...")
    train_loader, val_loader, test_loader, label_encoder = create_data_loaders(
        args.data_path, args.csv_path, args.batch_size, args.image_size
    )

    # Determine number of classes from label encoder
    num_classes = len(label_encoder)
    class_names = list(label_encoder.keys())

    print(f"Dataset split: Train={len(train_loader.dataset)}, "
          f"Val={len(val_loader.dataset)}, Test={len(test_loader.dataset)}")
    print(f"Number of classes: {num_classes}")

    # Save training info
    save_training_info(dirs, config, args.model, {
        'train_size': len(train_loader.dataset),
        'val_size': len(val_loader.dataset),
        'test_size': len(test_loader.dataset),
        'num_classes': num_classes,
        'class_names': class_names
    })

    # Create model
    print(f"\nCreating {args.model} model...")
    model = get_model(args.model, num_classes=num_classes, pretrained=True).to(device)

    # Save model architecture summary
    with open(dirs['config'] / 'model_summary.txt', 'w') as f:
        f.write(f"Model: {args.model}\n")
        f.write(f"Number of classes: {num_classes}\n")
        f.write(f"Pretrained: True\n\n")
        f.write(str(model))

    # Redirect stdout to log file
    log_file = open(dirs['logs'] / 'training.log', 'w')
    class TeeOutput:
        def __init__(self, *files):
            self.files = files
        def write(self, obj):
            for f in self.files:
                f.write(obj)
                f.flush()
        def flush(self):
            for f in self.files:
                f.flush()

    sys.stdout = TeeOutput(sys.stdout, log_file)

    # Train
    print(f"\nStarting training at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    history = train(model, train_loader, val_loader, config, device, str(dirs['checkpoints']))

    # Save training history
    history_df = pd.DataFrame(history)
    history_df.to_csv(dirs['metrics'] / 'training_history.csv', index=False)

    # Final evaluation
    print("\n" + "=" * 60)
    print("FINAL EVALUATION")
    print("=" * 60)

    # Load best model
    best_model_path = dirs['checkpoints'] / 'best_model.pth'
    checkpoint = torch.load(best_model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])

    # Evaluate on all splits
    print("\nEvaluating on training set...")
    train_pred, train_true, train_prob = get_predictions(model, train_loader, device)
    train_metrics = calculate_metrics(train_true, train_pred, train_prob, class_names)

    print("\nEvaluating on validation set...")
    val_pred, val_true, val_prob = get_predictions(model, val_loader, device)
    val_metrics = calculate_metrics(val_true, val_pred, val_prob, class_names)

    print("\nEvaluating on test set...")
    test_pred, test_true, test_prob = get_predictions(model, test_loader, device)
    test_metrics = calculate_metrics(test_true, test_pred, test_prob, class_names)

    # Save predictions for all splits
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

    # Close log file
    log_file.close()

    print(f"\n\nAll outputs saved to: {dirs['root']}")
    print("\nSaved files:")
    for file in sorted(dirs['root'].rglob('*')):
        if file.is_file():
            print(f"  - {file.relative_to(dirs['root'])}")

if __name__ == "__main__":
    main()
