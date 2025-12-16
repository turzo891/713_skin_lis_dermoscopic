"""
Generate LaTeX and Markdown tables from experiment results.
"""

import os
import json
import argparse
from typing import Dict, List, Optional
import pandas as pd
import numpy as np


def load_results(results_dir: str) -> Dict:
    """Load all experiment results."""
    results = {}
    for filename in os.listdir(results_dir):
        if filename.endswith('.json'):
            with open(os.path.join(results_dir, filename), 'r') as f:
                results[filename.replace('.json', '')] = json.load(f)
    return results


def generate_dataset_stats_table(data_info: Dict, output_path: str) -> str:
    """Generate Table 1: Dataset Statistics."""
    latex = r"""\begin{table}[h]
\centering
\caption{Dataset Statistics}
\label{tab:dataset_stats}
\begin{tabular}{lrrr}
\hline
\textbf{Class} & \textbf{Train} & \textbf{Test} & \textbf{Percentage} \\
\hline
"""
    class_names = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']
    full_names = {
        'akiec': 'Actinic Keratoses',
        'bcc': 'Basal Cell Carcinoma',
        'bkl': 'Benign Keratosis',
        'df': 'Dermatofibroma',
        'mel': 'Melanoma',
        'nv': 'Melanocytic Nevi',
        'vasc': 'Vascular Lesions'
    }

    total = sum(data_info.get('train_counts', {}).values()) + sum(data_info.get('test_counts', {}).values())

    for name in class_names:
        train = data_info.get('train_counts', {}).get(name, 0)
        test = data_info.get('test_counts', {}).get(name, 0)
        pct = (train + test) / total * 100 if total > 0 else 0
        latex += f"{full_names[name]} & {train} & {test} & {pct:.1f}\\% \\\\\n"

    latex += r"""\hline
\end{tabular}
\end{table}"""

    with open(output_path, 'w') as f:
        f.write(latex)
    return latex


def generate_model_comparison_table(results: Dict, output_path: str) -> str:
    """Generate Table 2: Model Comparison Results."""
    latex = r"""\begin{table*}[h]
\centering
\caption{Model Comparison Results on HAM10000 Dataset}
\label{tab:model_comparison}
\begin{tabular}{lcccccc}
\hline
\textbf{Model} & \textbf{Accuracy} & \textbf{Precision} & \textbf{Recall} & \textbf{F1-Score} & \textbf{AUC} & \textbf{Params (M)} \\
\hline
"""
    model_params = {
        'resnet50': 25.6, 'efficientnet': 19.3, 'densenet': 20.0,
        'vit': 86.6, 'swin': 87.8
    }

    metrics_list = []
    for model, metrics in results.items():
        acc = metrics.get('accuracy', 0)
        prec = metrics.get('precision_macro', 0)
        rec = metrics.get('recall_macro', 0)
        f1 = metrics.get('f1_macro', 0)
        auc = metrics.get('auc_macro', 0)
        params = model_params.get(model, 0)
        metrics_list.append((model, acc, prec, rec, f1, auc, params))

    # Find best values for bolding
    best = {
        'acc': max(m[1] for m in metrics_list),
        'prec': max(m[2] for m in metrics_list),
        'rec': max(m[3] for m in metrics_list),
        'f1': max(m[4] for m in metrics_list),
        'auc': max(m[5] for m in metrics_list)
    }

    for model, acc, prec, rec, f1, auc, params in metrics_list:
        acc_str = f"\\textbf{{{acc:.4f}}}" if acc == best['acc'] else f"{acc:.4f}"
        prec_str = f"\\textbf{{{prec:.4f}}}" if prec == best['prec'] else f"{prec:.4f}"
        rec_str = f"\\textbf{{{rec:.4f}}}" if rec == best['rec'] else f"{rec:.4f}"
        f1_str = f"\\textbf{{{f1:.4f}}}" if f1 == best['f1'] else f"{f1:.4f}"
        auc_str = f"\\textbf{{{auc:.4f}}}" if auc == best['auc'] else f"{auc:.4f}"

        model_display = model.replace('_', '-').title()
        latex += f"{model_display} & {acc_str} & {prec_str} & {rec_str} & {f1_str} & {auc_str} & {params:.1f} \\\\\n"

    latex += r"""\hline
\end{tabular}
\end{table*}"""

    with open(output_path, 'w') as f:
        f.write(latex)
    return latex


def generate_xai_comparison_table(results: Dict, output_path: str) -> str:
    """Generate Table 4: XAI Methods Comparison."""
    latex = r"""\begin{table}[h]
\centering
\caption{XAI Methods Comparison (CNN Models)}
\label{tab:xai_comparison}
\begin{tabular}{lccc}
\hline
\textbf{Method} & \textbf{CI} $\uparrow$ & \textbf{Faithfulness} $\uparrow$ & \textbf{Sparsity} $\uparrow$ \\
\hline
"""
    methods = ['gradcam', 'ig', 'saliency', 'occlusion', 'lime']

    for method in methods:
        ci = results.get(method, {}).get('ci_mean', 0)
        faith = results.get(method, {}).get('faithfulness_mean', 0)
        sparse = results.get(method, {}).get('sparsity_mean', 0.5)
        latex += f"{method.upper()} & {ci:.4f} & {faith:.4f} & {sparse:.4f} \\\\\n"

    latex += r"""\hline
\end{tabular}
\end{table}"""

    with open(output_path, 'w') as f:
        f.write(latex)
    return latex


def generate_ensemble_table(results: Dict, output_path: str) -> str:
    """Generate Table 6: Ensemble Results."""
    latex = r"""\begin{table}[h]
\centering
\caption{Ensemble Model Results}
\label{tab:ensemble}
\begin{tabular}{lcc}
\hline
\textbf{Method} & \textbf{Accuracy} & \textbf{F1-Score} \\
\hline
"""
    for method, metrics in results.items():
        acc = metrics.get('accuracy', 0)
        f1 = metrics.get('f1_macro', 0)
        method_display = method.replace('_', ' ').title()
        latex += f"{method_display} & {acc:.4f} & {f1:.4f} \\\\\n"

    latex += r"""\hline
\end{tabular}
\end{table}"""

    with open(output_path, 'w') as f:
        f.write(latex)
    return latex


def generate_markdown_summary(all_results: Dict, output_path: str) -> str:
    """Generate markdown summary of all results."""
    md = "# Experiment Results Summary\n\n"

    if 'experiment1_results' in all_results:
        md += "## Model Comparison\n\n"
        md += "| Model | Accuracy | F1-Score | AUC |\n"
        md += "|-------|----------|----------|-----|\n"
        for model, metrics in all_results['experiment1_results'].items():
            acc = metrics.get('accuracy', 0)
            f1 = metrics.get('f1_macro', 0)
            auc = metrics.get('auc_macro', 0)
            md += f"| {model} | {acc:.4f} | {f1:.4f} | {auc:.4f} |\n"
        md += "\n"

    if 'experiment4_results' in all_results:
        md += "## Ensemble Results\n\n"
        md += "| Method | Accuracy | F1-Score |\n"
        md += "|--------|----------|----------|\n"
        for method, metrics in all_results['experiment4_results'].items():
            acc = metrics.get('accuracy', 0)
            f1 = metrics.get('f1_macro', 0)
            md += f"| {method} | {acc:.4f} | {f1:.4f} |\n"

    with open(output_path, 'w') as f:
        f.write(md)
    return md


def generate_csv_results(all_results: Dict, output_dir: str):
    """Generate CSV files for all results."""
    os.makedirs(output_dir, exist_ok=True)

    if 'experiment1_results' in all_results:
        df = pd.DataFrame(all_results['experiment1_results']).T
        df.to_csv(os.path.join(output_dir, 'model_comparison.csv'))

    if 'experiment4_results' in all_results:
        df = pd.DataFrame(all_results['experiment4_results']).T
        df.to_csv(os.path.join(output_dir, 'ensemble_results.csv'))


def main():
    parser = argparse.ArgumentParser(description="Generate tables from results")
    parser.add_argument('--results_dir', type=str, default='./results')
    parser.add_argument('--output', type=str, default='./tables')
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    # Load results
    results = load_results(args.results_dir)

    # Generate tables
    if 'experiment1_results' in results:
        generate_model_comparison_table(
            results['experiment1_results'],
            os.path.join(args.output, 'table_model_comparison.tex')
        )

    if 'experiment2_results' in results:
        for model, xai_results in results['experiment2_results'].items():
            generate_xai_comparison_table(
                xai_results,
                os.path.join(args.output, f'table_xai_{model}.tex')
            )

    if 'experiment4_results' in results:
        generate_ensemble_table(
            results['experiment4_results'],
            os.path.join(args.output, 'table_ensemble.tex')
        )

    # Generate summary
    generate_markdown_summary(results, os.path.join(args.output, 'summary.md'))
    generate_csv_results(results, args.output)

    print(f"Tables generated in {args.output}")


if __name__ == "__main__":
    main()
