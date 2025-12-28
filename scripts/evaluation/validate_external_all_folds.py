#!/usr/bin/env python3
"""
External Validation Across All Cross-Validation Folds

Evaluates all trained model folds on external validation datasets (PH2)
and aggregates results with confidence intervals.

Usage:
    python validate_external_all_folds.py \
        --model_pattern "models/swin_fold*" \
        --model_name swin \
        --dataset ph2 \
        --data_dir data/PH2 \
        --output_dir results/external_validation
"""

import argparse
import os
import sys
from pathlib import Path
import json
import glob
import pandas as pd
import numpy as np
import torch
from scipy import stats
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Import validation functions
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from scripts.evaluation.validate_external_ph2 import (
    PH2Dataset, load_model, evaluate, compute_metrics, CLASSES
)
from torch.utils.data import DataLoader
from torchvision import transforms


def get_confidence_interval(data, confidence=0.95):
    """Calculate confidence interval for a list of values."""
    if len(data) < 2:
        return np.mean(data), np.mean(data), np.mean(data)

    n = len(data)
    mean = np.mean(data)
    se = stats.sem(data)
    h = se * stats.t.ppf((1 + confidence) / 2, n - 1)
    return mean, mean - h, mean + h


def aggregate_fold_results(all_metrics: list, output_dir: Path):
    """Aggregate results from all folds and compute statistics."""

    # Key metrics to aggregate
    metrics_to_aggregate = [
        'accuracy', 'balanced_accuracy', 'f1_weighted', 'f1_macro',
        'sensitivity_mel', 'specificity_mel', 'precision_mel', 'f1_mel',
        'auc_mel_vs_nv'
    ]

    aggregated = {}

    for metric in metrics_to_aggregate:
        values = [m.get(metric, 0) for m in all_metrics if metric in m]
        if values:
            mean, ci_low, ci_high = get_confidence_interval(values)
            std = np.std(values)
            aggregated[metric] = {
                'mean': float(mean),
                'std': float(std),
                'ci_low': float(ci_low),
                'ci_high': float(ci_high),
                'min': float(np.min(values)),
                'max': float(np.max(values)),
                'n_folds': len(values)
            }

    # Save aggregated results
    agg_path = output_dir / "aggregated_metrics.json"
    with open(agg_path, 'w') as f:
        json.dump(aggregated, f, indent=2)
    print(f"Saved aggregated metrics: {agg_path}")

    # Create summary table
    summary_data = []
    for metric, vals in aggregated.items():
        summary_data.append({
            'Metric': metric,
            'Mean': f"{vals['mean']:.4f}",
            'Std': f"{vals['std']:.4f}",
            '95% CI': f"[{vals['ci_low']:.4f}, {vals['ci_high']:.4f}]",
            'Range': f"[{vals['min']:.4f}, {vals['max']:.4f}]"
        })

    summary_df = pd.DataFrame(summary_data)
    summary_path = output_dir / "aggregated_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"Saved summary table: {summary_path}")

    return aggregated


def generate_aggregated_report(aggregated: dict, model_name: str, dataset_name: str,
                               n_folds: int, output_dir: Path):
    """Generate publication-ready aggregated report."""

    report_path = output_dir / f"aggregated_report_{dataset_name}.md"

    with open(report_path, 'w') as f:
        f.write(f"# Aggregated External Validation Results\n\n")
        f.write(f"## Configuration\n\n")
        f.write(f"- **Model:** {model_name}\n")
        f.write(f"- **Dataset:** {dataset_name.upper()}\n")
        f.write(f"- **Number of Folds:** {n_folds}\n\n")

        f.write("## Performance Metrics (Mean ± Std [95% CI])\n\n")
        f.write("| Metric | Value |\n")
        f.write("|--------|-------|\n")

        # Key metrics for publication
        key_metrics = [
            ('accuracy', 'Accuracy'),
            ('balanced_accuracy', 'Balanced Accuracy'),
            ('f1_weighted', 'F1 Score (weighted)'),
            ('sensitivity_mel', 'Sensitivity (Melanoma)'),
            ('specificity_mel', 'Specificity'),
            ('auc_mel_vs_nv', 'AUC-ROC'),
        ]

        for metric_key, metric_name in key_metrics:
            if metric_key in aggregated:
                vals = aggregated[metric_key]
                f.write(f"| {metric_name} | {vals['mean']:.4f} ± {vals['std']:.4f} "
                       f"[{vals['ci_low']:.4f}, {vals['ci_high']:.4f}] |\n")

        f.write("\n")

        # LaTeX table for publication
        f.write("## LaTeX Table (for publication)\n\n")
        f.write("```latex\n")
        f.write("\\begin{table}[h]\n")
        f.write("\\centering\n")
        f.write("\\caption{External Validation on PH2 Dataset}\n")
        f.write("\\begin{tabular}{lcc}\n")
        f.write("\\hline\n")
        f.write("Metric & Mean ± Std & 95\\% CI \\\\\n")
        f.write("\\hline\n")

        for metric_key, metric_name in key_metrics:
            if metric_key in aggregated:
                vals = aggregated[metric_key]
                f.write(f"{metric_name} & {vals['mean']:.3f} ± {vals['std']:.3f} & "
                       f"[{vals['ci_low']:.3f}, {vals['ci_high']:.3f}] \\\\\n")

        f.write("\\hline\n")
        f.write("\\end{tabular}\n")
        f.write("\\end{table}\n")
        f.write("```\n\n")

        # Interpretation
        f.write("## Interpretation\n\n")

        if 'auc_mel_vs_nv' in aggregated:
            auc_mean = aggregated['auc_mel_vs_nv']['mean']
            if auc_mean >= 0.90:
                f.write("- **Excellent** discriminative ability for melanoma detection (AUC >= 0.90)\n")
            elif auc_mean >= 0.80:
                f.write("- **Good** discriminative ability for melanoma detection (AUC >= 0.80)\n")
            else:
                f.write(f"- **Moderate** discriminative ability (AUC = {auc_mean:.3f})\n")

        if 'sensitivity_mel' in aggregated:
            sens_mean = aggregated['sensitivity_mel']['mean']
            f.write(f"- Melanoma sensitivity: {sens_mean:.1%} of melanomas correctly identified\n")

        f.write("\n")
        f.write("## Citation\n\n")
        f.write("PH2 Dataset:\n")
        f.write("> Mendonça et al. \"PH² - A dermoscopic image database for research and benchmarking\", ")
        f.write("35th IEEE EMBC, 2013.\n")

    print(f"Saved aggregated report: {report_path}")


def main():
    parser = argparse.ArgumentParser(description="External validation across all folds")
    parser.add_argument("--model_pattern", type=str, required=True,
                       help="Glob pattern for model directories (e.g., 'models/swin_fold*')")
    parser.add_argument("--model_name", type=str, required=True,
                       choices=['resnet50', 'efficientnet', 'densenet', 'vit', 'swin'],
                       help="Model architecture name")
    parser.add_argument("--dataset", type=str, default="ph2",
                       choices=['ph2'],
                       help="External validation dataset")
    parser.add_argument("--data_dir", type=str, required=True,
                       help="Dataset directory")
    parser.add_argument("--output_dir", type=str, default="results/external_validation",
                       help="Output directory for results")
    parser.add_argument("--image_size", type=int, default=224,
                       help="Image size for inference")
    parser.add_argument("--batch_size", type=int, default=16,
                       help="Batch size for inference")
    parser.add_argument("--device", type=str, default="cuda",
                       help="Device (cuda/cpu)")

    args = parser.parse_args()

    # Find model directories
    model_dirs = sorted(glob.glob(args.model_pattern))

    # Filter to only include directories with best_model.pth
    valid_model_dirs = [
        d for d in model_dirs
        if os.path.isdir(d) and os.path.exists(os.path.join(d, "best_model.pth"))
    ]

    if not valid_model_dirs:
        print(f"ERROR: No valid model directories found matching pattern: {args.model_pattern}")
        print("Ensure directories contain 'best_model.pth'")
        sys.exit(1)

    print("=" * 70)
    print("EXTERNAL VALIDATION ACROSS ALL FOLDS")
    print("=" * 70)
    print(f"Model pattern: {args.model_pattern}")
    print(f"Found {len(valid_model_dirs)} valid models:")
    for d in valid_model_dirs:
        print(f"  - {Path(d).name}")
    print(f"Dataset: {args.dataset.upper()}")
    print(f"Data directory: {args.data_dir}")
    print("=" * 70 + "\n")

    # Check device
    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        device = "cpu"

    # Setup output directory
    output_dir = Path(args.output_dir) / args.dataset
    output_dir.mkdir(parents=True, exist_ok=True)

    # Setup transforms
    transform = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Load dataset
    if args.dataset == 'ph2':
        dataset = PH2Dataset(data_dir=args.data_dir, transform=transform)
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")

    if len(dataset) == 0:
        print(f"ERROR: No images found in {args.data_dir}")
        sys.exit(1)

    dataloader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=4, pin_memory=True
    )

    # Evaluate each fold
    all_metrics = []
    all_predictions = []

    for i, model_dir in enumerate(tqdm(valid_model_dirs, desc="Evaluating folds")):
        print(f"\n{'='*50}")
        print(f"Fold {i+1}/{len(valid_model_dirs)}: {Path(model_dir).name}")
        print('='*50)

        # Load model
        model = load_model(model_dir, args.model_name, device, args.image_size)

        # Evaluate
        y_true, y_pred, y_prob, image_ids = evaluate(model, dataloader, device)

        # Compute metrics
        metrics = compute_metrics(y_true, y_pred, y_prob, CLASSES)
        metrics['fold'] = Path(model_dir).name
        all_metrics.append(metrics)

        # Store predictions
        fold_preds = pd.DataFrame({
            'fold': Path(model_dir).name,
            'image_id': image_ids,
            'true_label': y_true,
            'pred_label': y_pred,
            'prob_MEL': y_prob[:, 0],
            'prob_NV': y_prob[:, 1],
        })
        all_predictions.append(fold_preds)

        # Print summary
        print(f"  Accuracy: {metrics['accuracy']:.4f}")
        print(f"  Balanced Acc: {metrics['balanced_accuracy']:.4f}")
        if 'auc_mel_vs_nv' in metrics:
            print(f"  AUC: {metrics['auc_mel_vs_nv']:.4f}")

        # Save individual fold metrics
        fold_output_dir = output_dir / Path(model_dir).name
        fold_output_dir.mkdir(parents=True, exist_ok=True)
        with open(fold_output_dir / "metrics.json", 'w') as f:
            json.dump(metrics, f, indent=2)

        # Clean up model from GPU
        del model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # Aggregate results
    print("\n" + "=" * 70)
    print("AGGREGATED RESULTS")
    print("=" * 70)

    aggregated = aggregate_fold_results(all_metrics, output_dir)

    # Print summary
    for metric in ['accuracy', 'balanced_accuracy', 'auc_mel_vs_nv', 'sensitivity_mel', 'specificity_mel']:
        if metric in aggregated:
            vals = aggregated[metric]
            print(f"{metric}: {vals['mean']:.4f} ± {vals['std']:.4f} "
                  f"[{vals['ci_low']:.4f}, {vals['ci_high']:.4f}]")

    # Generate report
    generate_aggregated_report(
        aggregated, args.model_name, args.dataset,
        len(valid_model_dirs), output_dir
    )

    # Save all predictions
    all_preds_df = pd.concat(all_predictions, ignore_index=True)
    all_preds_df.to_csv(output_dir / "all_fold_predictions.csv", index=False)

    # Save all fold metrics
    metrics_df = pd.DataFrame(all_metrics)
    metrics_df.to_csv(output_dir / "all_fold_metrics.csv", index=False)

    print(f"\nResults saved to: {output_dir}")
    print("External validation complete!")


if __name__ == "__main__":
    main()
