"""
Evaluation module for skin cancer classification models.
"""

import os
import json
import argparse
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    cohen_kappa_score, roc_curve, precision_recall_curve
)
from scipy import stats
from tqdm import tqdm

from models import get_model
from data_loader import create_data_loaders
from utils import get_device, load_checkpoint, CLASS_NAMES


def get_predictions(
    model: nn.Module,
    data_loader: DataLoader,
    device: torch.device
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Get model predictions."""
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for images, labels in tqdm(data_loader, desc="Evaluating"):
            images = images.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)

            all_probs.extend(probs.cpu().numpy())
            all_preds.extend(outputs.argmax(dim=1).cpu().numpy())
            all_labels.extend(labels.numpy())

    return np.array(all_preds), np.array(all_labels), np.array(all_probs)


def calculate_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    class_names: Dict[int, str]
) -> Dict:
    """Calculate comprehensive classification metrics."""
    num_classes = len(class_names)

    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision_macro': precision_score(y_true, y_pred, average='macro', zero_division=0),
        'recall_macro': recall_score(y_true, y_pred, average='macro', zero_division=0),
        'f1_macro': f1_score(y_true, y_pred, average='macro', zero_division=0),
        'precision_weighted': precision_score(y_true, y_pred, average='weighted', zero_division=0),
        'recall_weighted': recall_score(y_true, y_pred, average='weighted', zero_division=0),
        'f1_weighted': f1_score(y_true, y_pred, average='weighted', zero_division=0),
        'cohen_kappa': cohen_kappa_score(y_true, y_pred),
    }

    # Per-class metrics
    precision_per_class = precision_score(y_true, y_pred, average=None, zero_division=0)
    recall_per_class = recall_score(y_true, y_pred, average=None, zero_division=0)
    f1_per_class = f1_score(y_true, y_pred, average=None, zero_division=0)

    for i, name in class_names.items():
        if i < len(precision_per_class):
            metrics[f'precision_{name}'] = precision_per_class[i]
            metrics[f'recall_{name}'] = recall_per_class[i]
            metrics[f'f1_{name}'] = f1_per_class[i]

    # AUC-ROC (One-vs-Rest)
    try:
        auc_per_class = []
        for i in range(num_classes):
            y_true_binary = (y_true == i).astype(int)
            if len(np.unique(y_true_binary)) > 1:
                auc = roc_auc_score(y_true_binary, y_prob[:, i])
                auc_per_class.append(auc)
                metrics[f'auc_{class_names[i]}'] = auc
        metrics['auc_macro'] = np.mean(auc_per_class) if auc_per_class else 0.0
    except Exception:
        metrics['auc_macro'] = 0.0

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    metrics['confusion_matrix'] = cm.tolist()

    return metrics


def paired_ttest(scores1: List[float], scores2: List[float]) -> Tuple[float, float]:
    """Perform paired t-test between two sets of scores."""
    t_stat, p_value = stats.ttest_rel(scores1, scores2)
    return t_stat, p_value


def wilcoxon_test(scores1: List[float], scores2: List[float]) -> Tuple[float, float]:
    """Perform Wilcoxon signed-rank test."""
    try:
        stat, p_value = stats.wilcoxon(scores1, scores2)
    except ValueError:
        stat, p_value = 0.0, 1.0
    return stat, p_value


def mcnemar_test(y_true: np.ndarray, pred1: np.ndarray, pred2: np.ndarray) -> Tuple[float, float]:
    """Perform McNemar's test for classification comparison."""
    correct1 = (pred1 == y_true)
    correct2 = (pred2 == y_true)

    b = np.sum(correct1 & ~correct2)  # Model 1 correct, Model 2 wrong
    c = np.sum(~correct1 & correct2)  # Model 1 wrong, Model 2 correct

    if b + c == 0:
        return 0.0, 1.0

    chi2 = (abs(b - c) - 1) ** 2 / (b + c)
    p_value = 1 - stats.chi2.cdf(chi2, df=1)
    return chi2, p_value


def bootstrap_confidence_interval(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric_fn,
    n_bootstrap: int = 1000,
    confidence: float = 0.95
) -> Tuple[float, float, float]:
    """Calculate bootstrap confidence interval for a metric."""
    scores = []
    n = len(y_true)

    for _ in range(n_bootstrap):
        indices = np.random.choice(n, n, replace=True)
        score = metric_fn(y_true[indices], y_pred[indices])
        scores.append(score)

    lower = np.percentile(scores, (1 - confidence) / 2 * 100)
    upper = np.percentile(scores, (1 + confidence) / 2 * 100)
    return np.mean(scores), lower, upper


def generate_latex_table(metrics_dict: Dict[str, Dict], output_path: str) -> str:
    """Generate LaTeX table from metrics dictionary."""
    models = list(metrics_dict.keys())
    metric_names = ['accuracy', 'precision_macro', 'recall_macro', 'f1_macro', 'auc_macro', 'cohen_kappa']

    latex = "\\begin{table}[h]\n\\centering\n\\caption{Model Comparison Results}\n"
    latex += "\\begin{tabular}{l" + "c" * len(metric_names) + "}\n\\hline\n"
    latex += "Model & " + " & ".join([m.replace('_', ' ').title() for m in metric_names]) + " \\\\\n\\hline\n"

    for model in models:
        row = [model]
        for metric in metric_names:
            value = metrics_dict[model].get(metric, 0)
            row.append(f"{value:.4f}")
        latex += " & ".join(row) + " \\\\\n"

    latex += "\\hline\n\\end{tabular}\n\\end{table}"

    with open(output_path, 'w') as f:
        f.write(latex)

    return latex


def generate_markdown_report(metrics: Dict, model_name: str) -> str:
    """Generate markdown report from metrics."""
    report = f"# Evaluation Report: {model_name}\n\n"
    report += "## Overall Metrics\n\n"
    report += "| Metric | Value |\n|--------|-------|\n"

    for key in ['accuracy', 'precision_macro', 'recall_macro', 'f1_macro', 'auc_macro', 'cohen_kappa']:
        if key in metrics:
            report += f"| {key.replace('_', ' ').title()} | {metrics[key]:.4f} |\n"

    report += "\n## Per-Class Metrics\n\n"
    report += "| Class | Precision | Recall | F1 | AUC |\n|-------|-----------|--------|----|----||\n"

    for i, name in CLASS_NAMES.items():
        p = metrics.get(f'precision_{name}', 0)
        r = metrics.get(f'recall_{name}', 0)
        f = metrics.get(f'f1_{name}', 0)
        a = metrics.get(f'auc_{name}', 0)
        report += f"| {name} | {p:.4f} | {r:.4f} | {f:.4f} | {a:.4f} |\n"

    return report


def evaluate_model(
    model_path: str,
    model_name: str,
    data_loader: DataLoader,
    device: torch.device,
    output_dir: str
) -> Dict:
    """Evaluate a single model."""
    model = get_model(model_name, num_classes=7, pretrained=False).to(device)
    model, _, _ = load_checkpoint(model_path, model)

    y_pred, y_true, y_prob = get_predictions(model, data_loader, device)
    metrics = calculate_metrics(y_true, y_pred, y_prob, CLASS_NAMES)

    # Save metrics
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, f'{model_name}_metrics.json'), 'w') as f:
        json.dump({k: v if not isinstance(v, np.ndarray) else v.tolist() for k, v in metrics.items()}, f, indent=2)

    # Save predictions
    np.save(os.path.join(output_dir, f'{model_name}_predictions.npy'), y_pred)
    np.save(os.path.join(output_dir, f'{model_name}_probabilities.npy'), y_prob)
    np.save(os.path.join(output_dir, f'{model_name}_labels.npy'), y_true)

    # Generate report
    report = generate_markdown_report(metrics, model_name)
    with open(os.path.join(output_dir, f'{model_name}_report.md'), 'w') as f:
        f.write(report)

    return metrics


def main():
    parser = argparse.ArgumentParser(description="Evaluate skin cancer classification models")
    parser.add_argument('--model_dir', type=str, required=True, help='Directory containing model checkpoints')
    parser.add_argument('--data_path', type=str, required=True, help='Path to test dataset')
    parser.add_argument('--csv_path', type=str, default=None)
    parser.add_argument('--output', type=str, default='./results/metrics')
    parser.add_argument('--batch_size', type=int, default=32)
    args = parser.parse_args()

    device = get_device()
    _, _, test_loader, _ = create_data_loaders(args.data_path, args.csv_path, args.batch_size)

    all_metrics = {}
    model_names = ['resnet50', 'efficientnet', 'densenet', 'vit', 'swin']

    for model_name in model_names:
        model_path = os.path.join(args.model_dir, model_name, 'best_model.pth')
        if os.path.exists(model_path):
            print(f"\nEvaluating {model_name}...")
            metrics = evaluate_model(model_path, model_name, test_loader, device, args.output)
            all_metrics[model_name] = metrics
            print(f"Accuracy: {metrics['accuracy']:.4f}, F1: {metrics['f1_macro']:.4f}")

    if all_metrics:
        generate_latex_table(all_metrics, os.path.join(args.output, 'comparison_table.tex'))
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
