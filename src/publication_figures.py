"""
Publication-quality figure generation for research papers.
Follows IEEE/Nature/Medical journal standards.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import seaborn as sns
from typing import Dict, List, Optional, Tuple
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve


# Publication style settings
def set_publication_style():
    """Set matplotlib style for publication-quality figures."""
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams.update({
        # Figure
        'figure.figsize': (7, 5),  # Single column width
        'figure.dpi': 300,
        'savefig.dpi': 300,
        'savefig.format': 'pdf',
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.1,
        
        # Font
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'DejaVu Serif'],
        'font.size': 10,
        'axes.labelsize': 11,
        'axes.titlesize': 12,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 9,
        
        # Lines
        'lines.linewidth': 1.5,
        'lines.markersize': 6,
        
        # Axes
        'axes.linewidth': 1,
        'axes.spines.top': False,
        'axes.spines.right': False,
        
        # Grid
        'grid.linewidth': 0.5,
        'grid.alpha': 0.3,
        
        # Legend
        'legend.frameon': True,
        'legend.framealpha': 0.8,
    })


# Color palettes for consistency
NATURE_COLORS = ['#E64B35', '#4DBBD5', '#00A087', '#3C5488', '#F39B7F', '#8491B4', '#91D1C2']
IEEE_COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2']
MEDICAL_COLORS = ['#0072B2', '#D55E00', '#009E73', '#CC79A7', '#F0E442', '#56B4E9', '#E69F00']


def create_model_comparison_figure(
    metrics_dict: Dict[str, Dict],
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Create comprehensive model comparison figure.
    
    Args:
        metrics_dict: {model_name: {accuracy, precision, recall, f1, auc}}
        save_path: Path to save figure
    """
    set_publication_style()
    
    models = list(metrics_dict.keys())
    metrics = ['accuracy', 'precision_macro', 'recall_macro', 'f1_macro', 'auc_macro']
    metric_labels = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC-ROC']
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = np.arange(len(metrics))
    width = 0.15
    
    for i, (model, color) in enumerate(zip(models, NATURE_COLORS)):
        values = [metrics_dict[model].get(m, 0) for m in metrics]
        stds = [metrics_dict[model].get(f'{m}_std', 0) for m in metrics]
        
        bars = ax.bar(x + i * width, values, width, label=model.upper(), 
                     color=color, edgecolor='black', linewidth=0.5)
        
        # Add error bars if available
        if any(s > 0 for s in stds):
            ax.errorbar(x + i * width, values, yerr=stds, fmt='none', 
                       color='black', capsize=3, capthick=1, linewidth=1)
    
    ax.set_ylabel('Score')
    ax.set_ylim(0, 1.05)
    ax.set_xticks(x + width * (len(models) - 1) / 2)
    ax.set_xticklabels(metric_labels, rotation=15, ha='right')
    ax.legend(loc='upper right', ncol=2)
    ax.set_title('Model Performance Comparison')
    
    # Add horizontal line at 0.9 for reference
    ax.axhline(y=0.9, color='gray', linestyle='--', alpha=0.5, linewidth=1)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        fig.savefig(save_path.replace('.png', '.pdf'), bbox_inches='tight')
    
    return fig


def create_confusion_matrix_grid(
    confusion_matrices: Dict[str, np.ndarray],
    class_names: List[str],
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Create grid of confusion matrices for multiple models.
    """
    set_publication_style()
    
    n_models = len(confusion_matrices)
    n_cols = min(3, n_models)
    n_rows = (n_models + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4.5 * n_rows))
    if n_models == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = axes.reshape(1, -1)
    
    for idx, (model_name, cm) in enumerate(confusion_matrices.items()):
        row, col = idx // n_cols, idx % n_cols
        ax = axes[row, col]
        
        # Normalize
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues',
                   xticklabels=class_names, yticklabels=class_names,
                   ax=ax, cbar=True, square=True,
                   annot_kws={'size': 8})
        
        ax.set_xlabel('Predicted')
        ax.set_ylabel('True')
        ax.set_title(model_name.upper())
    
    # Remove unused axes
    for idx in range(n_models, n_rows * n_cols):
        row, col = idx // n_cols, idx % n_cols
        fig.delaxes(axes[row, col])
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        fig.savefig(save_path.replace('.png', '.pdf'), bbox_inches='tight')
    
    return fig


def create_roc_curves_figure(
    roc_data: Dict[str, Tuple[np.ndarray, np.ndarray, float]],
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Create ROC curves comparison figure.
    
    Args:
        roc_data: {model_name: (fpr, tpr, auc_score)}
    """
    set_publication_style()
    
    fig, ax = plt.subplots(figsize=(7, 6))
    
    for (model_name, (fpr, tpr, auc_score)), color in zip(roc_data.items(), NATURE_COLORS):
        ax.plot(fpr, tpr, color=color, linewidth=2,
               label=f'{model_name.upper()} (AUC = {auc_score:.3f})')
    
    # Diagonal
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1, alpha=0.7)
    
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('Receiver Operating Characteristic (ROC) Curves')
    ax.legend(loc='lower right')
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.02])
    
    # Add grid
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        fig.savefig(save_path.replace('.png', '.pdf'), bbox_inches='tight')
    
    return fig


def create_xai_comparison_figure(
    original_image: np.ndarray,
    xai_maps: Dict[str, np.ndarray],
    prediction: str,
    confidence: float,
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Create XAI methods comparison figure for publication.
    """
    set_publication_style()
    
    n_methods = len(xai_maps)
    fig = plt.figure(figsize=(2.5 * (n_methods + 1), 3))
    gs = GridSpec(1, n_methods + 1, figure=fig)
    
    # Original image
    ax0 = fig.add_subplot(gs[0, 0])
    ax0.imshow(original_image)
    ax0.set_title(f'Original\n({prediction}, {confidence:.1%})', fontsize=10)
    ax0.axis('off')
    
    # XAI methods
    for i, (method_name, attr_map) in enumerate(xai_maps.items(), 1):
        ax = fig.add_subplot(gs[0, i])
        ax.imshow(original_image)
        im = ax.imshow(attr_map, cmap='jet', alpha=0.5)
        ax.set_title(method_name.upper(), fontsize=10)
        ax.axis('off')
    
    # Add colorbar
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    plt.colorbar(im, cax=cbar_ax, label='Attribution')
    
    plt.tight_layout(rect=[0, 0, 0.9, 1])
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        fig.savefig(save_path.replace('.png', '.pdf'), bbox_inches='tight')
    
    return fig


def create_training_curves_figure(
    histories: Dict[str, Dict[str, List[float]]],
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Create training curves figure.
    
    Args:
        histories: {model_name: {train_loss, val_loss, train_acc, val_acc}}
    """
    set_publication_style()
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    for (model_name, history), color in zip(histories.items(), NATURE_COLORS):
        epochs = range(1, len(history['train_loss']) + 1)
        
        # Loss
        axes[0].plot(epochs, history['train_loss'], '-', color=color, alpha=0.5)
        axes[0].plot(epochs, history['val_loss'], '-', color=color, 
                    label=model_name.upper(), linewidth=2)
        
        # Accuracy
        axes[1].plot(epochs, history['train_acc'], '-', color=color, alpha=0.5)
        axes[1].plot(epochs, history['val_acc'], '-', color=color,
                    label=model_name.upper(), linewidth=2)
    
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training and Validation Loss')
    axes[0].legend()
    
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title('Training and Validation Accuracy')
    axes[1].legend()
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        fig.savefig(save_path.replace('.png', '.pdf'), bbox_inches='tight')
    
    return fig


def create_per_class_performance_figure(
    per_class_metrics: Dict[str, Dict[str, float]],
    class_names: List[str],
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Create per-class performance comparison.
    
    Args:
        per_class_metrics: {model: {class: f1_score}}
    """
    set_publication_style()
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    models = list(per_class_metrics.keys())
    x = np.arange(len(class_names))
    width = 0.8 / len(models)
    
    for i, (model, color) in enumerate(zip(models, NATURE_COLORS)):
        values = [per_class_metrics[model].get(c, 0) for c in class_names]
        ax.bar(x + i * width, values, width, label=model.upper(), color=color, edgecolor='black')
    
    ax.set_ylabel('F1-Score')
    ax.set_xlabel('Lesion Type')
    ax.set_xticks(x + width * (len(models) - 1) / 2)
    ax.set_xticklabels(class_names, rotation=45, ha='right')
    ax.legend(loc='upper right')
    ax.set_ylim(0, 1.05)
    ax.set_title('Per-Class F1-Score Comparison')
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        fig.savefig(save_path.replace('.png', '.pdf'), bbox_inches='tight')
    
    return fig


def create_uncertainty_figure(
    confidence: np.ndarray,
    uncertainty: np.ndarray,
    correct: np.ndarray,
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Create uncertainty analysis figure.
    """
    set_publication_style()
    
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    
    # 1. Confidence vs Uncertainty scatter
    colors = ['#2ca02c' if c else '#d62728' for c in correct]
    axes[0].scatter(confidence, uncertainty, c=colors, alpha=0.5, s=20)
    axes[0].set_xlabel('Confidence')
    axes[0].set_ylabel('Uncertainty')
    axes[0].set_title('Confidence vs Uncertainty')
    
    # Legend
    correct_patch = mpatches.Patch(color='#2ca02c', label='Correct')
    incorrect_patch = mpatches.Patch(color='#d62728', label='Incorrect')
    axes[0].legend(handles=[correct_patch, incorrect_patch])
    
    # 2. Uncertainty distribution
    axes[1].hist(uncertainty[correct], bins=30, alpha=0.7, label='Correct', color='#2ca02c')
    axes[1].hist(uncertainty[~correct], bins=30, alpha=0.7, label='Incorrect', color='#d62728')
    axes[1].set_xlabel('Uncertainty')
    axes[1].set_ylabel('Count')
    axes[1].set_title('Uncertainty Distribution')
    axes[1].legend()
    
    # 3. Selective prediction (risk-coverage)
    sorted_idx = np.argsort(uncertainty)
    sorted_correct = correct[sorted_idx]
    coverage = np.arange(1, len(correct) + 1) / len(correct)
    accuracy = np.cumsum(sorted_correct) / np.arange(1, len(correct) + 1)
    
    axes[2].plot(coverage, accuracy, color='#1f77b4', linewidth=2)
    axes[2].fill_between(coverage, accuracy, alpha=0.3)
    axes[2].set_xlabel('Coverage')
    axes[2].set_ylabel('Accuracy')
    axes[2].set_title('Selective Prediction Curve')
    axes[2].set_xlim(0, 1)
    axes[2].set_ylim(0.5, 1.02)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        fig.savefig(save_path.replace('.png', '.pdf'), bbox_inches='tight')
    
    return fig


def generate_all_publication_figures(
    results_dir: str,
    output_dir: str
) -> None:
    """
    Generate all publication figures from saved results.
    """
    set_publication_style()
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Generating publication figures in {output_dir}")
    print("Note: This function requires saved results from experiments.")
    print("Run experiments first to generate results, then call this function.")
