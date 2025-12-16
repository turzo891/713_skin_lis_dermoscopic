"""
Main experiment runner for skin cancer classification project.
"""

import os
import argparse
import json
from typing import Dict, List
import numpy as np
import torch

from data_loader import create_data_loaders, create_kfold_loaders
from models import get_model, EnsembleModel, print_model_summary
from train import train
from evaluate import evaluate_model, get_predictions, calculate_metrics
from xai_methods import XAIExplainer, compare_xai_methods, confidence_increase, faithfulness_metric
from visualize import (
    plot_confusion_matrix, plot_roc_curves, plot_model_comparison,
    plot_xai_grid, plot_training_history, plot_class_distribution
)
from utils import set_seed, get_device, load_config, save_config, CLASS_NAMES


def run_experiment_1_model_comparison(config: Dict, data_dir: str, output_dir: str) -> Dict:
    """Experiment 1: Compare all model architectures."""
    print("\n" + "=" * 60)
    print("Experiment 1: Model Comparison")
    print("=" * 60)

    device = get_device()
    model_names = ['resnet50', 'efficientnet', 'densenet', 'vit', 'swin']
    all_results = {}

    if config.get('use_kfold', True):
        n_folds = config.get('n_folds', 5)
        fold_loaders = create_kfold_loaders(
            data_dir, config.get('csv_path'), n_folds,
            config['batch_size'], config['image_size']
        )

        for model_name in model_names:
            print(f"\nTraining {model_name}...")
            fold_metrics = []

            for fold, (train_loader, val_loader) in enumerate(fold_loaders):
                print(f"\n--- Fold {fold + 1}/{n_folds} ---")
                model = get_model(model_name, num_classes=7, pretrained=True).to(device)
                fold_dir = os.path.join(output_dir, model_name, f'fold_{fold + 1}')

                history = train(model, train_loader, val_loader, config, device, fold_dir)

                # Evaluate
                y_pred, y_true, y_prob = get_predictions(model, val_loader, device)
                metrics = calculate_metrics(y_true, y_pred, y_prob, CLASS_NAMES)
                fold_metrics.append(metrics)

            # Average across folds
            avg_metrics = {}
            for key in fold_metrics[0].keys():
                if isinstance(fold_metrics[0][key], (int, float)):
                    values = [m[key] for m in fold_metrics]
                    avg_metrics[key] = np.mean(values)
                    avg_metrics[f'{key}_std'] = np.std(values)

            all_results[model_name] = avg_metrics
            print(f"{model_name}: Acc={avg_metrics['accuracy']:.4f}+-{avg_metrics['accuracy_std']:.4f}")

    else:
        train_loader, val_loader, test_loader, _ = create_data_loaders(
            data_dir, config.get('csv_path'), config['batch_size'], config['image_size']
        )

        for model_name in model_names:
            print(f"\nTraining {model_name}...")
            model = get_model(model_name, num_classes=7, pretrained=True).to(device)
            model_dir = os.path.join(output_dir, model_name)

            train(model, train_loader, val_loader, config, device, model_dir)

            # Evaluate on test set
            y_pred, y_true, y_prob = get_predictions(model, test_loader, device)
            metrics = calculate_metrics(y_true, y_pred, y_prob, CLASS_NAMES)
            all_results[model_name] = metrics

            print(f"{model_name}: Acc={metrics['accuracy']:.4f}, F1={metrics['f1_macro']:.4f}")

    # Save results
    with open(os.path.join(output_dir, 'experiment1_results.json'), 'w') as f:
        json.dump(all_results, f, indent=2, default=str)

    # Plot comparison
    plot_model_comparison(all_results, save_path=os.path.join(output_dir, 'model_comparison.png'))

    return all_results


def run_experiment_2_xai_comparison(config: Dict, data_dir: str, model_dir: str, output_dir: str) -> Dict:
    """Experiment 2: Compare XAI methods for each model."""
    print("\n" + "=" * 60)
    print("Experiment 2: XAI Methods Comparison")
    print("=" * 60)

    device = get_device()
    _, _, test_loader, _ = create_data_loaders(
        data_dir, config.get('csv_path'), config['batch_size'], config['image_size']
    )

    model_names = ['resnet50', 'efficientnet', 'densenet', 'vit', 'swin']
    xai_methods_cnn = ['gradcam', 'ig', 'saliency', 'occlusion', 'lime']
    xai_methods_vit = ['gradcam', 'ig', 'attention', 'occlusion']

    all_results = {}

    # Get sample images
    sample_images = []
    sample_labels = []
    for images, labels in test_loader:
        sample_images.extend(images[:5])
        sample_labels.extend(labels[:5].numpy())
        if len(sample_images) >= 10:
            break
    sample_images = sample_images[:10]

    for model_name in model_names:
        print(f"\nAnalyzing {model_name}...")
        model_path = os.path.join(model_dir, model_name, 'best_model.pth')

        if not os.path.exists(model_path):
            print(f"Model not found: {model_path}")
            continue

        model = get_model(model_name, num_classes=7, pretrained=False).to(device)
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()

        methods = xai_methods_vit if model_name in ['vit', 'swin'] else xai_methods_cnn
        explainer = XAIExplainer(model, device)

        model_results = {}
        xai_maps = {m: [] for m in methods}

        for idx, image in enumerate(sample_images):
            image = image.unsqueeze(0).to(device)

            with torch.no_grad():
                output = model(image)
                pred_class = output.argmax(dim=1).item()

            for method in methods:
                try:
                    attr_map, _ = explainer.explain(image, method, target_class=pred_class)
                    xai_maps[method].append(attr_map)

                    # Calculate metrics
                    ci = confidence_increase(model, image, attr_map, pred_class, device)
                    mif, lif = faithfulness_metric(model, image, attr_map, pred_class, device)

                    if method not in model_results:
                        model_results[method] = {'ci': [], 'faithfulness': []}
                    model_results[method]['ci'].append(ci)
                    model_results[method]['faithfulness'].append(mif[0] - mif[-1])

                except Exception as e:
                    print(f"Error with {method}: {e}")

        # Average metrics
        for method in model_results:
            model_results[method] = {
                'ci_mean': np.mean(model_results[method]['ci']),
                'ci_std': np.std(model_results[method]['ci']),
                'faithfulness_mean': np.mean(model_results[method]['faithfulness']),
            }

        all_results[model_name] = model_results

        # Save XAI visualizations
        xai_dir = os.path.join(output_dir, 'xai', model_name)
        os.makedirs(xai_dir, exist_ok=True)

        # Convert sample images for visualization
        sample_images_np = []
        for img in sample_images[:3]:
            img_np = img.permute(1, 2, 0).numpy()
            img_np = img_np * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
            img_np = np.clip(img_np, 0, 1)
            sample_images_np.append(img_np)

        xai_maps_subset = {m: xai_maps[m][:3] for m in methods if xai_maps[m]}
        plot_xai_grid(sample_images_np, xai_maps_subset, os.path.join(xai_dir, 'xai_grid.png'))

    # Save results
    with open(os.path.join(output_dir, 'experiment2_results.json'), 'w') as f:
        json.dump(all_results, f, indent=2, default=str)

    return all_results


def run_experiment_3_external_validation(config: Dict, model_dir: str, output_dir: str) -> Dict:
    """Experiment 3: External dataset validation."""
    print("\n" + "=" * 60)
    print("Experiment 3: External Validation")
    print("=" * 60)

    device = get_device()
    model_names = ['resnet50', 'efficientnet', 'densenet', 'vit', 'swin']
    datasets = ['HAM10000', 'ISIC2019', 'PH2']

    all_results = {}

    for model_name in model_names:
        model_path = os.path.join(model_dir, model_name, 'best_model.pth')
        if not os.path.exists(model_path):
            continue

        model = get_model(model_name, num_classes=7, pretrained=False).to(device)
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])

        model_results = {}
        for dataset in datasets:
            dataset_path = config.get(f'{dataset.lower()}_path')
            if dataset_path and os.path.exists(dataset_path):
                print(f"Evaluating {model_name} on {dataset}...")
                _, _, test_loader, _ = create_data_loaders(
                    dataset_path, batch_size=config['batch_size'], image_size=config['image_size']
                )
                y_pred, y_true, y_prob = get_predictions(model, test_loader, device)
                metrics = calculate_metrics(y_true, y_pred, y_prob, CLASS_NAMES)
                model_results[dataset] = {
                    'accuracy': metrics['accuracy'],
                    'f1_macro': metrics['f1_macro']
                }

        all_results[model_name] = model_results

    with open(os.path.join(output_dir, 'experiment3_results.json'), 'w') as f:
        json.dump(all_results, f, indent=2)

    return all_results


def run_experiment_4_ensemble(config: Dict, data_dir: str, model_dir: str, output_dir: str) -> Dict:
    """Experiment 4: Ensemble models."""
    print("\n" + "=" * 60)
    print("Experiment 4: Ensemble Models")
    print("=" * 60)

    device = get_device()
    _, _, test_loader, _ = create_data_loaders(
        data_dir, config.get('csv_path'), config['batch_size'], config['image_size']
    )

    # Load best CNN and best ViT
    models_to_ensemble = []
    for model_name in ['efficientnet', 'vit']:
        model_path = os.path.join(model_dir, model_name, 'best_model.pth')
        if os.path.exists(model_path):
            model = get_model(model_name, num_classes=7, pretrained=False).to(device)
            checkpoint = torch.load(model_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            model.eval()
            models_to_ensemble.append(model)

    if len(models_to_ensemble) < 2:
        print("Not enough models for ensemble")
        return {}

    results = {}

    # Average ensemble
    ensemble_avg = EnsembleModel(models_to_ensemble, method='average').to(device)
    y_pred, y_true, y_prob = get_predictions(ensemble_avg, test_loader, device)
    metrics = calculate_metrics(y_true, y_pred, y_prob, CLASS_NAMES)
    results['average_ensemble'] = {'accuracy': metrics['accuracy'], 'f1_macro': metrics['f1_macro']}

    # Weighted ensemble
    ensemble_weighted = EnsembleModel(models_to_ensemble, weights=[0.6, 0.4], method='weighted').to(device)
    y_pred, y_true, y_prob = get_predictions(ensemble_weighted, test_loader, device)
    metrics = calculate_metrics(y_true, y_pred, y_prob, CLASS_NAMES)
    results['weighted_ensemble'] = {'accuracy': metrics['accuracy'], 'f1_macro': metrics['f1_macro']}

    # Voting ensemble
    ensemble_voting = EnsembleModel(models_to_ensemble, method='voting').to(device)
    y_pred, y_true, y_prob = get_predictions(ensemble_voting, test_loader, device)
    metrics = calculate_metrics(y_true, y_pred, y_prob, CLASS_NAMES)
    results['voting_ensemble'] = {'accuracy': metrics['accuracy'], 'f1_macro': metrics['f1_macro']}

    print(f"Average Ensemble: Acc={results['average_ensemble']['accuracy']:.4f}")
    print(f"Weighted Ensemble: Acc={results['weighted_ensemble']['accuracy']:.4f}")
    print(f"Voting Ensemble: Acc={results['voting_ensemble']['accuracy']:.4f}")

    with open(os.path.join(output_dir, 'experiment4_results.json'), 'w') as f:
        json.dump(results, f, indent=2)

    return results


def main():
    parser = argparse.ArgumentParser(description="Run experiments")
    parser.add_argument('--config', type=str, default='configs/config.yaml')
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default='./results')
    parser.add_argument('--model_dir', type=str, default='./models')
    parser.add_argument('--experiment', type=str, default='all',
                       choices=['all', '1', '2', '3', '4'])
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    config = load_config(args.config) if os.path.exists(args.config) else {
        'epochs': 50, 'batch_size': 32, 'lr': 1e-4, 'image_size': 224,
        'use_kfold': False, 'n_folds': 5, 'patience': 10
    }

    if args.experiment in ['all', '1']:
        run_experiment_1_model_comparison(config, args.data_path, args.output_dir)

    if args.experiment in ['all', '2']:
        run_experiment_2_xai_comparison(config, args.data_path, args.model_dir, args.output_dir)

    if args.experiment in ['all', '3']:
        run_experiment_3_external_validation(config, args.model_dir, args.output_dir)

    if args.experiment in ['all', '4']:
        run_experiment_4_ensemble(config, args.data_path, args.model_dir, args.output_dir)

    print("\nAll experiments completed!")


if __name__ == "__main__":
    main()
