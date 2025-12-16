"""
Visualization utilities for skin cancer classification project.
"""

import os
from typing import Dict, List, Optional, Tuple
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve
import torch


def set_plot_style():
    """Set publication-quality plot style."""
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams.update({
        'font.size': 12,
        'axes.labelsize': 14,
        'axes.titlesize': 16,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.fontsize': 12,
        'figure.figsize': (10, 8),
        'figure.dpi': 150,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight'
    })


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str],
    normalize: bool = True,
    save_path: Optional[str] = None,
    title: str = "Confusion Matrix"
) -> plt.Figure:
    """Plot confusion matrix heatmap."""
    set_plot_style()

    cm = confusion_matrix(y_true, y_pred)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='.2f' if normalize else 'd', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names, ax=ax)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title(title)

    if save_path:
        fig.savefig(save_path)
        fig.savefig(save_path.replace('.png', '.pdf'))

    return fig


def plot_roc_curves(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    class_names: List[str],
    save_path: Optional[str] = None
) -> plt.Figure:
    """Plot ROC curves for all classes."""
    set_plot_style()

    n_classes = len(class_names)
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()

    colors = plt.cm.Set2(np.linspace(0, 1, n_classes))

    for i, (name, color) in enumerate(zip(class_names, colors)):
        y_true_binary = (y_true == i).astype(int)
        if len(np.unique(y_true_binary)) < 2:
            continue

        fpr, tpr, _ = roc_curve(y_true_binary, y_prob[:, i])
        roc_auc = auc(fpr, tpr)

        axes[i].plot(fpr, tpr, color=color, lw=2, label=f'AUC = {roc_auc:.3f}')
        axes[i].plot([0, 1], [0, 1], 'k--', lw=1)
        axes[i].set_xlabel('False Positive Rate')
        axes[i].set_ylabel('True Positive Rate')
        axes[i].set_title(f'{name}')
        axes[i].legend(loc='lower right')

    # Remove unused subplot
    if n_classes < len(axes):
        for j in range(n_classes, len(axes)):
            fig.delaxes(axes[j])

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path)
        fig.savefig(save_path.replace('.png', '.pdf'))

    return fig


def plot_model_comparison(
    metrics_dict: Dict[str, Dict],
    metric_names: List[str] = None,
    save_path: Optional[str] = None
) -> plt.Figure:
    """Plot bar chart comparing models."""
    set_plot_style()

    if metric_names is None:
        metric_names = ['accuracy', 'precision_macro', 'recall_macro', 'f1_macro']

    models = list(metrics_dict.keys())
    x = np.arange(len(metric_names))
    width = 0.8 / len(models)

    fig, ax = plt.subplots(figsize=(12, 6))
    colors = plt.cm.Set2(np.linspace(0, 1, len(models)))

    for i, (model, color) in enumerate(zip(models, colors)):
        values = [metrics_dict[model].get(m, 0) for m in metric_names]
        ax.bar(x + i * width, values, width, label=model, color=color)

    ax.set_xlabel('Metrics')
    ax.set_ylabel('Score')
    ax.set_title('Model Comparison')
    ax.set_xticks(x + width * (len(models) - 1) / 2)
    ax.set_xticklabels([m.replace('_', ' ').title() for m in metric_names])
    ax.legend()
    ax.set_ylim(0, 1)

    if save_path:
        fig.savefig(save_path)

    return fig


def plot_xai_comparison(
    image: np.ndarray,
    xai_maps: Dict[str, np.ndarray],
    save_path: Optional[str] = None,
    title: str = "XAI Methods Comparison"
) -> plt.Figure:
    """Plot XAI methods side by side."""
    set_plot_style()

    n_methods = len(xai_maps) + 1
    fig, axes = plt.subplots(1, n_methods, figsize=(4 * n_methods, 4))

    # Original image
    axes[0].imshow(image)
    axes[0].set_title('Original')
    axes[0].axis('off')

    # XAI methods
    for i, (method, attr_map) in enumerate(xai_maps.items(), 1):
        if attr_map is not None:
            axes[i].imshow(image)
            axes[i].imshow(attr_map, cmap='jet', alpha=0.5)
        axes[i].set_title(method.upper())
        axes[i].axis('off')

    fig.suptitle(title)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path)
        fig.savefig(save_path.replace('.png', '.pdf'))

    return fig


def plot_xai_grid(
    images: List[np.ndarray],
    xai_results: Dict[str, List[np.ndarray]],
    save_path: Optional[str] = None
) -> plt.Figure:
    """Plot grid of images and their XAI explanations."""
    set_plot_style()

    methods = list(xai_results.keys())
    n_images = len(images)
    n_cols = len(methods) + 1

    fig, axes = plt.subplots(n_images, n_cols, figsize=(3 * n_cols, 3 * n_images))

    for i, image in enumerate(images):
        # Original
        axes[i, 0].imshow(image)
        if i == 0:
            axes[i, 0].set_title('Original')
        axes[i, 0].axis('off')

        # XAI methods
        for j, method in enumerate(methods, 1):
            attr_map = xai_results[method][i]
            axes[i, j].imshow(image)
            if attr_map is not None:
                axes[i, j].imshow(attr_map, cmap='jet', alpha=0.5)
            if i == 0:
                axes[i, j].set_title(method.upper())
            axes[i, j].axis('off')

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path)
        fig.savefig(save_path.replace('.png', '.pdf'))

    return fig


def plot_training_history(
    history: Dict[str, List[float]],
    save_path: Optional[str] = None
) -> plt.Figure:
    """Plot training history (loss and accuracy)."""
    set_plot_style()

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Loss
    epochs = range(1, len(history['train_loss']) + 1)
    axes[0].plot(epochs, history['train_loss'], 'b-', label='Train Loss')
    axes[0].plot(epochs, history['val_loss'], 'r-', label='Val Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training and Validation Loss')
    axes[0].legend()

    # Accuracy
    axes[1].plot(epochs, history['train_acc'], 'b-', label='Train Acc')
    axes[1].plot(epochs, history['val_acc'], 'r-', label='Val Acc')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title('Training and Validation Accuracy')
    axes[1].legend()

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path)

    return fig


def plot_class_distribution(
    labels: List[int],
    class_names: List[str],
    save_path: Optional[str] = None,
    title: str = "Class Distribution"
) -> plt.Figure:
    """Plot class distribution bar chart."""
    set_plot_style()

    counts = np.bincount(labels, minlength=len(class_names))

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = plt.cm.Set2(np.linspace(0, 1, len(class_names)))
    bars = ax.bar(class_names, counts, color=colors)

    ax.set_xlabel('Class')
    ax.set_ylabel('Count')
    ax.set_title(title)

    # Add count labels on bars
    for bar, count in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 50,
                str(count), ha='center', va='bottom')

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path)

    return fig


def plot_xai_metrics_heatmap(
    metrics_df,
    save_path: Optional[str] = None,
    title: str = "XAI Methods Comparison"
) -> plt.Figure:
    """Plot heatmap of XAI metrics."""
    set_plot_style()

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(metrics_df, annot=True, fmt='.3f', cmap='RdYlGn', ax=ax)
    ax.set_title(title)

    if save_path:
        fig.savefig(save_path)
        fig.savefig(save_path.replace('.png', '.pdf'))

    return fig


def plot_attention_heads(
    attention_weights: List[torch.Tensor],
    layer_idx: int = -1,
    save_path: Optional[str] = None
) -> plt.Figure:
    """Visualize attention heads from Vision Transformer."""
    set_plot_style()

    attn = attention_weights[layer_idx].squeeze().cpu().numpy()
    n_heads = attn.shape[0]

    n_cols = 4
    n_rows = (n_heads + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))
    axes = axes.flatten()

    for i in range(n_heads):
        axes[i].imshow(attn[i], cmap='viridis')
        axes[i].set_title(f'Head {i + 1}')
        axes[i].axis('off')

    for j in range(n_heads, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path)

    return fig
