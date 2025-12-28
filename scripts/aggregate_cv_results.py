#!/usr/bin/env python3
"""
Aggregate 10-Fold Cross-Validation Results

Collects results from all trained folds and computes:
- Mean ± std for all metrics
- Statistical significance tests
- Comparison visualizations
"""

import argparse
import json
from pathlib import Path
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from glob import glob

# Models and metrics
MODELS = ['swin', 'densenet', 'resnet50']
METRICS = ['accuracy', 'balanced_accuracy', 'f1_weighted', 'f1_macro', 'auc']


def find_fold_results(models_dir: Path, model_name: str):
    """Find all fold result files for a model."""
    results = {}

    # Find all model directories
    pattern = f"{model_name}_fold*"
    model_dirs = sorted(models_dir.glob(pattern))

    for model_dir in model_dirs:
        # Extract fold number
        dir_name = model_dir.name
        try:
            fold = int(dir_name.split('_fold')[1].split('_')[0])
        except (IndexError, ValueError):
            continue

        # Look for results file
        results_file = model_dir / 'final_results.json'
        if not results_file.exists():
            results_file = model_dir / 'best_results.json'

        if results_file.exists():
            with open(results_file) as f:
                data = json.load(f)
                results[fold] = data
        else:
            # Try to extract from training history
            history_file = model_dir / 'metrics' / 'training_history.csv'
            if history_file.exists():
                df = pd.read_csv(history_file)
                if len(df) > 0:
                    best_row = df.loc[df['val_balanced_accuracy'].idxmax()]
                    results[fold] = {
                        'accuracy': best_row.get('val_accuracy', 0),
                        'balanced_accuracy': best_row.get('val_balanced_accuracy', 0),
                        'f1_weighted': best_row.get('val_f1', 0),
                        'auc': best_row.get('val_auc', 0),
                    }

    return results


def aggregate_model_results(results: dict) -> dict:
    """Compute mean and std for a model's results across folds."""
    if not results:
        return {}

    # Collect metrics across folds
    metrics_data = {m: [] for m in METRICS}

    for fold, fold_results in results.items():
        for metric in METRICS:
            value = fold_results.get(metric, fold_results.get(f'val_{metric}', None))
            if value is not None:
                metrics_data[metric].append(value)

    # Compute statistics
    aggregated = {
        'n_folds': len(results),
        'folds': list(results.keys()),
    }

    for metric, values in metrics_data.items():
        if values:
            aggregated[f'{metric}_mean'] = np.mean(values)
            aggregated[f'{metric}_std'] = np.std(values)
            aggregated[f'{metric}_min'] = np.min(values)
            aggregated[f'{metric}_max'] = np.max(values)
            aggregated[f'{metric}_values'] = values

    return aggregated


def perform_statistical_tests(all_results: dict) -> dict:
    """Perform pairwise statistical tests between models."""
    tests = {}

    model_names = list(all_results.keys())

    for i, model1 in enumerate(model_names):
        for model2 in model_names[i+1:]:
            # Get balanced accuracy values
            values1 = all_results[model1].get('balanced_accuracy_values', [])
            values2 = all_results[model2].get('balanced_accuracy_values', [])

            if len(values1) >= 2 and len(values2) >= 2:
                # Paired t-test (if same folds)
                if len(values1) == len(values2):
                    t_stat, p_value = stats.ttest_rel(values1, values2)
                    test_type = 'paired_ttest'
                else:
                    # Independent t-test
                    t_stat, p_value = stats.ttest_ind(values1, values2)
                    test_type = 'independent_ttest'

                tests[f'{model1}_vs_{model2}'] = {
                    'test_type': test_type,
                    't_statistic': t_stat,
                    'p_value': p_value,
                    'significant_005': p_value < 0.05,
                    'significant_001': p_value < 0.01,
                }

    return tests


def create_comparison_plot(all_results: dict, output_dir: Path):
    """Create visualization comparing models across folds."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Prepare data
    models = []
    means = []
    stds = []
    all_values = []

    for model in MODELS:
        if model in all_results:
            models.append(model.title())
            means.append(all_results[model].get('balanced_accuracy_mean', 0) * 100)
            stds.append(all_results[model].get('balanced_accuracy_std', 0) * 100)
            values = all_results[model].get('balanced_accuracy_values', [])
            all_values.append([v * 100 for v in values])

    # (a) Bar chart with error bars
    ax = axes[0]
    colors = ['#E64B35', '#4DBBD5', '#00A087']
    bars = ax.bar(range(len(models)), means, yerr=stds,
                  color=colors[:len(models)], edgecolor='black', capsize=5)

    ax.set_xticks(range(len(models)))
    ax.set_xticklabels(models)
    ax.set_ylabel('Balanced Accuracy (%)')
    ax.set_title('(a) Model Comparison (Mean ± Std)', fontweight='bold')
    ax.set_ylim(80, 100)
    ax.grid(axis='y', alpha=0.3, linestyle='--')

    # Add value labels
    for bar, mean, std in zip(bars, means, stds):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + std + 0.5,
                f'{mean:.1f}±{std:.1f}', ha='center', va='bottom', fontsize=10)

    # (b) Box plot of fold results
    ax = axes[1]
    bp = ax.boxplot(all_values, labels=models, patch_artist=True)

    for patch, color in zip(bp['boxes'], colors[:len(models)]):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax.set_ylabel('Balanced Accuracy (%)')
    ax.set_title('(b) Distribution Across Folds', fontweight='bold')
    ax.set_ylim(80, 100)
    ax.grid(axis='y', alpha=0.3, linestyle='--')

    plt.tight_layout()

    # Save
    fig.savefig(output_dir / 'cv_comparison.pdf', format='pdf', dpi=300, bbox_inches='tight')
    fig.savefig(output_dir / 'cv_comparison.png', format='png', dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Saved: cv_comparison.pdf/png")


def create_fold_heatmap(all_results: dict, output_dir: Path):
    """Create heatmap showing performance across folds."""
    # Prepare data
    data = []
    for model in MODELS:
        if model in all_results:
            values = all_results[model].get('balanced_accuracy_values', [])
            folds = all_results[model].get('folds', [])
            for fold, value in zip(folds, values):
                data.append({
                    'Model': model.title(),
                    'Fold': f'Fold {fold}',
                    'Balanced Accuracy': value * 100
                })

    if not data:
        return

    df = pd.DataFrame(data)
    pivot = df.pivot(index='Model', columns='Fold', values='Balanced Accuracy')

    # Sort columns
    cols = sorted(pivot.columns, key=lambda x: int(x.split()[-1]))
    pivot = pivot[cols]

    fig, ax = plt.subplots(figsize=(12, 4))

    sns.heatmap(pivot, annot=True, fmt='.1f', cmap='RdYlGn',
                vmin=85, vmax=95, ax=ax, cbar_kws={'label': 'Balanced Accuracy (%)'})

    ax.set_title('Model Performance Across Folds', fontweight='bold')

    plt.tight_layout()

    fig.savefig(output_dir / 'cv_heatmap.pdf', format='pdf', dpi=300, bbox_inches='tight')
    fig.savefig(output_dir / 'cv_heatmap.png', format='png', dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Saved: cv_heatmap.pdf/png")


def main():
    parser = argparse.ArgumentParser(description='Aggregate 10-fold CV results')
    parser.add_argument('--models_dir', type=str, default='models',
                       help='Directory containing model folders')
    parser.add_argument('--output_dir', type=str, default='results/cv_results',
                       help='Output directory for aggregated results')

    args = parser.parse_args()

    models_dir = Path(args.models_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("="*60)
    print("AGGREGATING 10-FOLD CROSS-VALIDATION RESULTS")
    print("="*60)

    # Collect results for each model
    all_results = {}
    detailed_results = []

    for model in MODELS:
        print(f"\nProcessing {model}...")

        fold_results = find_fold_results(models_dir, model)

        if fold_results:
            aggregated = aggregate_model_results(fold_results)
            all_results[model] = aggregated

            print(f"  Found {len(fold_results)} folds: {sorted(fold_results.keys())}")
            if 'balanced_accuracy_mean' in aggregated:
                print(f"  Balanced Accuracy: {aggregated['balanced_accuracy_mean']*100:.2f} ± {aggregated['balanced_accuracy_std']*100:.2f}%")

            # Add to detailed results
            for fold, results in fold_results.items():
                row = {'model': model, 'fold': fold}
                row.update(results)
                detailed_results.append(row)
        else:
            print(f"  No results found")

    # Create summary DataFrame
    summary_data = []
    for model, results in all_results.items():
        row = {'model': model, 'n_folds': results.get('n_folds', 0)}
        for metric in METRICS:
            row[f'{metric}_mean'] = results.get(f'{metric}_mean', None)
            row[f'{metric}_std'] = results.get(f'{metric}_std', None)
        summary_data.append(row)

    summary_df = pd.DataFrame(summary_data)

    # Save summary
    summary_df.to_csv(output_dir / 'cv_summary.csv', index=False)
    print(f"\nSaved: cv_summary.csv")

    # Save detailed results
    if detailed_results:
        detailed_df = pd.DataFrame(detailed_results)
        detailed_df.to_csv(output_dir / 'cv_detailed.csv', index=False)
        print(f"Saved: cv_detailed.csv")

    # Perform statistical tests
    if len(all_results) >= 2:
        tests = perform_statistical_tests(all_results)

        with open(output_dir / 'cv_statistics.json', 'w') as f:
            json.dump(tests, f, indent=2)
        print(f"Saved: cv_statistics.json")

        # Print test results
        print("\nStatistical Tests:")
        for comparison, result in tests.items():
            sig = "***" if result['significant_001'] else ("*" if result['significant_005'] else "")
            print(f"  {comparison}: p={result['p_value']:.4f} {sig}")

    # Create visualizations
    if all_results:
        print("\nGenerating visualizations...")
        create_comparison_plot(all_results, output_dir)
        create_fold_heatmap(all_results, output_dir)

    # Print summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"\n{'Model':<15} {'Folds':<8} {'Balanced Acc (Mean ± Std)':<25}")
    print("-"*50)
    for model, results in all_results.items():
        n_folds = results.get('n_folds', 0)
        mean = results.get('balanced_accuracy_mean', 0) * 100
        std = results.get('balanced_accuracy_std', 0) * 100
        print(f"{model:<15} {n_folds:<8} {mean:.2f} ± {std:.2f}%")

    print("\n" + "="*60)
    print(f"Results saved to: {output_dir}")
    print("="*60)


if __name__ == '__main__':
    main()
