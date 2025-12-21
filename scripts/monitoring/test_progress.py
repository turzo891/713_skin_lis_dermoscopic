#!/usr/bin/env python3
"""
Progress testing script for trained models
Checks training status, loads models, and evaluates performance
"""

import os
import sys
import json
from pathlib import Path
from datetime import datetime
import pandas as pd
import torch
import numpy as np
from tabulate import tabulate

# Add src to path
sys.path.insert(0, 'src')

from data_loader import create_data_loaders
from models import get_model
from evaluate import get_predictions, calculate_metrics
from utils import get_device, set_seed


class ModelProgressChecker:
    """Check progress of model training and evaluation"""

    def __init__(self, models_dir='models'):
        self.models_dir = Path(models_dir)
        self.device = get_device()
        set_seed(42)

    def find_trained_models(self):
        """Find all trained models in the models directory"""
        trained_models = {}

        if not self.models_dir.exists():
            print(f"⚠️  Models directory not found: {self.models_dir}")
            return trained_models

        # Look for model directories
        for model_dir in self.models_dir.iterdir():
            if model_dir.is_dir():
                # Check for best_model.pth or in checkpoints subdirectory
                best_model_path = model_dir / 'best_model.pth'
                checkpoint_path = model_dir / 'checkpoints' / 'best_model.pth'

                if best_model_path.exists():
                    trained_models[model_dir.name] = {
                        'path': best_model_path,
                        'dir': model_dir
                    }
                elif checkpoint_path.exists():
                    trained_models[model_dir.name] = {
                        'path': checkpoint_path,
                        'dir': model_dir
                    }

        return trained_models

    def extract_model_type(self, model_name):
        """Extract base model type from directory name"""
        # Handle timestamped names like "resnet50_20251220_190809"
        for base_model in ['resnet50', 'efficientnet', 'densenet', 'vit', 'swin']:
            if model_name.lower().startswith(base_model):
                return base_model
        return model_name

    def load_training_history(self, model_dir):
        """Load training history if available"""
        history_files = [
            model_dir / 'metrics' / 'training_history.csv',
            model_dir / 'training_history.csv'
        ]

        for history_file in history_files:
            if history_file.exists():
                try:
                    return pd.read_csv(history_file)
                except:
                    pass
        return None

    def load_final_results(self, model_dir):
        """Load final results if available"""
        results_files = [
            model_dir / 'final_results.json',
            model_dir / 'results.json'
        ]

        for results_file in results_files:
            if results_file.exists():
                try:
                    with open(results_file, 'r') as f:
                        return json.load(f)
                except:
                    pass
        return None

    def get_model_info(self, model_path):
        """Get information from model checkpoint"""
        try:
            checkpoint = torch.load(model_path, map_location='cpu')
            info = {
                'epoch': checkpoint.get('epoch', 'N/A'),
                'val_acc': checkpoint.get('metrics', {}).get('val_acc', 'N/A'),
                'val_loss': checkpoint.get('val_loss', 'N/A'),
            }
            return info
        except Exception as e:
            return {'error': str(e)}

    def evaluate_model(self, model_name, model_path, data_path, csv_path):
        """Load and evaluate a trained model"""
        try:
            print(f"\n{'='*60}")
            print(f"Evaluating: {model_name}")
            print(f"{'='*60}")

            # Get base model type
            base_model = self.extract_model_type(model_name)

            # Load data
            print("Loading data...")
            _, val_loader, test_loader, label_encoder = create_data_loaders(
                data_path, csv_path, batch_size=32, image_size=224
            )

            num_classes = len(label_encoder)
            class_names = list(label_encoder.keys())

            # Load model
            print(f"Loading {base_model} model...")
            model = get_model(base_model, num_classes=num_classes, pretrained=False)

            checkpoint = torch.load(model_path, map_location=self.device)
            model.load_state_dict(checkpoint['model_state_dict'])
            model = model.to(self.device)
            model.eval()

            # Evaluate on validation set
            print("Evaluating on validation set...")
            val_pred, val_true, val_prob = get_predictions(model, val_loader, self.device)
            val_metrics = calculate_metrics(val_true, val_pred, val_prob, class_names)

            # Evaluate on test set
            print("Evaluating on test set...")
            test_pred, test_true, test_prob = get_predictions(model, test_loader, self.device)
            test_metrics = calculate_metrics(test_true, test_pred, test_prob, class_names)

            return {
                'val_metrics': val_metrics,
                'test_metrics': test_metrics,
                'num_classes': num_classes,
                'class_names': class_names
            }

        except Exception as e:
            print(f"❌ Error evaluating {model_name}: {str(e)}")
            return None

    def display_summary(self, trained_models):
        """Display summary of all trained models"""
        print("\n" + "="*80)
        print("TRAINED MODELS SUMMARY")
        print("="*80)

        summary_data = []
        for model_name, model_info in trained_models.items():
            model_dir = model_info['dir']
            checkpoint_info = self.get_model_info(model_info['path'])

            # Load final results if available
            final_results = self.load_final_results(model_dir)

            # Get test accuracy
            test_acc = 'N/A'
            if final_results and 'test' in final_results:
                test_acc = f"{final_results['test'].get('accuracy', 0):.4f}"

            val_acc = checkpoint_info.get('val_acc', 'N/A')
            if isinstance(val_acc, (float, int)):
                val_acc = f"{val_acc:.4f}"

            summary_data.append({
                'Model': model_name,
                'Epoch': checkpoint_info.get('epoch', 'N/A'),
                'Val Acc': val_acc,
                'Test Acc': test_acc,
                'Path': str(model_info['path'].relative_to(self.models_dir))
            })

        if summary_data:
            df = pd.DataFrame(summary_data)
            print(tabulate(df, headers='keys', tablefmt='grid', showindex=False))
        else:
            print("No trained models found.")

    def display_detailed_results(self, model_name, results):
        """Display detailed results for a model"""
        if not results:
            return

        print(f"\n{'='*60}")
        print(f"DETAILED RESULTS: {model_name}")
        print(f"{'='*60}")

        # Overall metrics
        metrics_data = []
        for split_name, split_key in [('Validation', 'val_metrics'), ('Test', 'test_metrics')]:
            if split_key in results:
                metrics = results[split_key]
                metrics_data.append({
                    'Split': split_name,
                    'Accuracy': f"{metrics.get('accuracy', 0):.4f}",
                    'F1 (Macro)': f"{metrics.get('f1_macro', 0):.4f}",
                    'F1 (Weighted)': f"{metrics.get('f1_weighted', 0):.4f}",
                    'Precision': f"{metrics.get('precision_macro', 0):.4f}",
                    'Recall': f"{metrics.get('recall_macro', 0):.4f}",
                    'AUC-ROC': f"{metrics.get('auc_roc', 0):.4f}"
                })

        if metrics_data:
            df = pd.DataFrame(metrics_data)
            print("\nOverall Metrics:")
            print(tabulate(df, headers='keys', tablefmt='grid', showindex=False))

        # Per-class metrics for test set
        if 'test_metrics' in results and 'per_class_f1' in results['test_metrics']:
            test_metrics = results['test_metrics']
            class_names = results.get('class_names', [])

            per_class_data = []
            for idx, class_name in enumerate(class_names):
                per_class_data.append({
                    'Class': class_name,
                    'F1': f"{test_metrics['per_class_f1'][idx]:.4f}",
                    'Precision': f"{test_metrics['per_class_precision'][idx]:.4f}",
                    'Recall': f"{test_metrics['per_class_recall'][idx]:.4f}"
                })

            if per_class_data:
                df = pd.DataFrame(per_class_data)
                print("\nPer-Class Metrics (Test Set):")
                print(tabulate(df, headers='keys', tablefmt='grid', showindex=False))

    def compare_models(self, data_path, csv_path, quick=True):
        """Compare all trained models"""
        trained_models = self.find_trained_models()

        if not trained_models:
            print("❌ No trained models found!")
            return

        print(f"\n📊 Found {len(trained_models)} trained model(s)")
        self.display_summary(trained_models)

        if not quick:
            # Detailed evaluation
            print("\n" + "="*80)
            print("DETAILED EVALUATION")
            print("="*80)

            for model_name, model_info in trained_models.items():
                results = self.evaluate_model(
                    model_name,
                    model_info['path'],
                    data_path,
                    csv_path
                )

                if results:
                    self.display_detailed_results(model_name, results)

    def check_model_status(self, model_type):
        """Check if a specific model type has been trained"""
        trained_models = self.find_trained_models()

        for model_name in trained_models.keys():
            if model_type.lower() in model_name.lower():
                return True, model_name

        return False, None


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Check training progress")
    parser.add_argument('--data_path', type=str, default='data/ISIC2019',
                       help='Path to dataset')
    parser.add_argument('--csv_path', type=str, default='data/ISIC2019/ISIC_2019_Training_GroundTruth.csv',
                       help='Path to metadata CSV')
    parser.add_argument('--models_dir', type=str, default='models',
                       help='Directory containing trained models')
    parser.add_argument('--detailed', action='store_true',
                       help='Run detailed evaluation (slower)')
    parser.add_argument('--model', type=str, default=None,
                       help='Evaluate specific model only')
    args = parser.parse_args()

    checker = ModelProgressChecker(args.models_dir)

    print("="*80)
    print("MODEL TRAINING PROGRESS CHECKER")
    print("="*80)
    print(f"Device: {checker.device}")
    print(f"Models directory: {args.models_dir}")
    print(f"Dataset: {args.data_path}")

    if args.model:
        # Check specific model
        trained_models = checker.find_trained_models()
        model_info = trained_models.get(args.model)

        if model_info:
            results = checker.evaluate_model(
                args.model,
                model_info['path'],
                args.data_path,
                args.csv_path
            )
            if results:
                checker.display_detailed_results(args.model, results)
        else:
            print(f"❌ Model '{args.model}' not found!")
            print(f"\nAvailable models: {', '.join(trained_models.keys())}")
    else:
        # Compare all models
        checker.compare_models(args.data_path, args.csv_path, quick=not args.detailed)

    # Show what models still need training
    print("\n" + "="*80)
    print("TRAINING STATUS")
    print("="*80)

    all_models = ['resnet50', 'efficientnet', 'densenet', 'vit', 'swin']
    trained_models = checker.find_trained_models()

    trained_types = set()
    for model_name in trained_models.keys():
        base_type = checker.extract_model_type(model_name)
        trained_types.add(base_type)

    status_data = []
    for model_type in all_models:
        status = '✅ Trained' if model_type in trained_types else '⏳ Pending'
        status_data.append({'Model Type': model_type, 'Status': status})

    df = pd.DataFrame(status_data)
    print(tabulate(df, headers='keys', tablefmt='grid', showindex=False))

    pending = [m for m in all_models if m not in trained_types]
    if pending:
        print(f"\n📝 Models pending training: {', '.join(pending)}")
    else:
        print("\n✨ All models trained!")


if __name__ == "__main__":
    main()
