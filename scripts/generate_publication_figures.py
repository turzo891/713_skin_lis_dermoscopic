#!/usr/bin/env python3
"""
Generate Publication-Quality Figures for Skin Cancer Classification Paper

Creates comprehensive EDA and result visualizations following academic standards:
- Vector graphics (PDF/SVG) for scalability
- Proper fonts and sizing
- Color-blind friendly palettes
- Statistical annotations

Output: results/figures/
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from scipy import stats
import json
import warnings
warnings.filterwarnings('ignore')

# Set publication-quality defaults
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.titlesize': 14,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1,
})

# Color palettes (colorblind-friendly)
CLASS_COLORS = {
    'MEL': '#E64B35',   # Red - Melanoma (dangerous)
    'NV': '#4DBBD5',    # Cyan - Nevus (benign)
    'BCC': '#00A087',   # Teal - BCC
    'BKL': '#3C5488',   # Blue - BKL
    'AK': '#F39B7F',    # Salmon - AK
    'SCC': '#8491B4',   # Purple-gray - SCC
    'VASC': '#91D1C2',  # Light teal - Vascular
    'DF': '#DC91A4',    # Pink - DF
}

SKIN_TONE_COLORS = {
    0: '#8B4513',  # Very Dark
    1: '#A0522D',  # Dark
    2: '#CD853F',  # Medium-Dark
    3: '#DEB887',  # Medium
    4: '#F5DEB3',  # Medium-Light
    5: '#FFEFD5',  # Very Light
}

MODEL_COLORS = {
    'Swin': '#E64B35',
    'DenseNet': '#4DBBD5',
    'ResNet50': '#00A087',
}

CLASS_NAMES = {
    'MEL': 'Melanoma',
    'NV': 'Melanocytic Nevus',
    'BCC': 'Basal Cell Carcinoma',
    'BKL': 'Benign Keratosis',
    'AK': 'Actinic Keratosis',
    'SCC': 'Squamous Cell Carcinoma',
    'VASC': 'Vascular Lesion',
    'DF': 'Dermatofibroma'
}

SKIN_TONE_NAMES = {
    0: 'Very Dark (0)',
    1: 'Dark (1)',
    2: 'Medium-Dark (2)',
    3: 'Medium (3)',
    4: 'Medium-Light (4)',
    5: 'Very Light (5)'
}


def load_data(data_dir: Path):
    """Load all metadata files."""
    combined = pd.read_csv(data_dir / "combined" / "master_metadata.csv")

    # Load MILK10k metadata for skin tone info
    milk_meta = pd.read_csv(data_dir / "MILK10k" / "metadata" / "lesion_metadata.csv")

    return combined, milk_meta


def fig1_dataset_overview(combined: pd.DataFrame, output_dir: Path):
    """
    Figure 1: Dataset Overview
    - (a) Class distribution bar chart
    - (b) Dataset source comparison
    - (c) Class imbalance ratio
    """
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    # (a) Class distribution
    ax = axes[0]
    class_counts = combined['diagnosis'].value_counts()
    classes = ['MEL', 'NV', 'BCC', 'BKL', 'AK', 'SCC', 'VASC', 'DF']
    counts = [class_counts.get(c, 0) for c in classes]
    colors = [CLASS_COLORS[c] for c in classes]

    bars = ax.bar(range(len(classes)), counts, color=colors, edgecolor='black', linewidth=0.5)
    ax.set_xticks(range(len(classes)))
    ax.set_xticklabels(classes, rotation=45, ha='right')
    ax.set_ylabel('Number of Samples')
    ax.set_title('(a) Class Distribution', fontweight='bold')
    ax.set_xlabel('Diagnosis Class')

    # Add count labels
    for bar, count in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 200,
                f'{count:,}', ha='center', va='bottom', fontsize=8)

    ax.set_ylim(0, max(counts) * 1.15)
    ax.grid(axis='y', alpha=0.3, linestyle='--')

    # (b) Dataset source comparison
    ax = axes[1]
    source_class = combined.groupby(['dataset_source', 'diagnosis']).size().unstack(fill_value=0)

    x = np.arange(len(classes))
    width = 0.35

    isic_counts = [source_class.loc['ISIC2019', c] if 'ISIC2019' in source_class.index and c in source_class.columns else 0 for c in classes]
    milk_counts = [source_class.loc['MILK10k', c] if 'MILK10k' in source_class.index and c in source_class.columns else 0 for c in classes]

    ax.bar(x - width/2, isic_counts, width, label='ISIC2019', color='#3C5488', edgecolor='black', linewidth=0.5)
    ax.bar(x + width/2, milk_counts, width, label='MILK10k', color='#E64B35', edgecolor='black', linewidth=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels(classes, rotation=45, ha='right')
    ax.set_ylabel('Number of Samples')
    ax.set_title('(b) Distribution by Dataset Source', fontweight='bold')
    ax.set_xlabel('Diagnosis Class')
    ax.legend(loc='upper right')
    ax.grid(axis='y', alpha=0.3, linestyle='--')

    # (c) Class imbalance ratio
    ax = axes[2]
    max_count = max(counts)
    min_count = min(counts)
    ratios = [max_count / c for c in counts]

    bars = ax.bar(range(len(classes)), ratios, color=colors, edgecolor='black', linewidth=0.5)
    ax.set_xticks(range(len(classes)))
    ax.set_xticklabels(classes, rotation=45, ha='right')
    ax.set_ylabel('Imbalance Ratio (max/class)')
    ax.set_title('(c) Class Imbalance Ratios', fontweight='bold')
    ax.set_xlabel('Diagnosis Class')
    ax.axhline(y=1, color='green', linestyle='--', alpha=0.7, label='Balanced')
    ax.grid(axis='y', alpha=0.3, linestyle='--')

    # Add ratio labels
    for bar, ratio in zip(bars, ratios):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{ratio:.1f}x', ha='center', va='bottom', fontsize=8)

    plt.tight_layout()

    # Save in multiple formats
    fig.savefig(output_dir / 'fig1_dataset_overview.pdf', format='pdf')
    fig.savefig(output_dir / 'fig1_dataset_overview.png', format='png')
    plt.close()
    print(f"Saved: fig1_dataset_overview.pdf/png")


def fig2_skin_tone_distribution(combined: pd.DataFrame, output_dir: Path):
    """
    Figure 2: Skin Tone Analysis
    - (a) Skin tone distribution
    - (b) Skin tone by class heatmap
    """
    # Filter to MILK10k with skin tone
    milk_df = combined[(combined['dataset_source'] == 'MILK10k') & (combined['skin_tone'].notna())].copy()

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # (a) Skin tone distribution
    ax = axes[0]
    tone_counts = milk_df['skin_tone'].value_counts().sort_index()
    tones = [0, 1, 2, 3, 4, 5]
    counts = [tone_counts.get(t, 0) for t in tones]
    colors = [SKIN_TONE_COLORS[t] for t in tones]
    labels = [SKIN_TONE_NAMES[t] for t in tones]

    bars = ax.bar(range(len(tones)), counts, color=colors, edgecolor='black', linewidth=1)
    ax.set_xticks(range(len(tones)))
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.set_ylabel('Number of Samples')
    ax.set_title('(a) Skin Tone Distribution (MILK10k)', fontweight='bold')
    ax.set_xlabel('Skin Tone (Monk Scale)')

    # Add percentage labels
    total = sum(counts)
    for bar, count in zip(bars, counts):
        pct = count / total * 100
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 50,
                f'{pct:.1f}%', ha='center', va='bottom', fontsize=9)

    ax.set_ylim(0, max(counts) * 1.15)
    ax.grid(axis='y', alpha=0.3, linestyle='--')

    # (b) Skin tone by class heatmap
    ax = axes[1]

    # Create cross-tabulation
    cross_tab = pd.crosstab(milk_df['diagnosis'], milk_df['skin_tone'], normalize='index') * 100
    classes = ['MEL', 'NV', 'BCC', 'BKL', 'AK', 'SCC', 'VASC', 'DF']
    cross_tab = cross_tab.reindex(classes)
    cross_tab = cross_tab.reindex(columns=[0, 1, 2, 3, 4, 5], fill_value=0)

    im = ax.imshow(cross_tab.values, cmap='YlOrBr', aspect='auto')

    ax.set_xticks(range(6))
    ax.set_xticklabels(['0', '1', '2', '3', '4', '5'])
    ax.set_yticks(range(len(classes)))
    ax.set_yticklabels(classes)
    ax.set_xlabel('Skin Tone')
    ax.set_ylabel('Diagnosis Class')
    ax.set_title('(b) Class Distribution by Skin Tone (%)', fontweight='bold')

    # Add text annotations
    for i in range(len(classes)):
        for j in range(6):
            val = cross_tab.iloc[i, j] if j < cross_tab.shape[1] else 0
            color = 'white' if val > 50 else 'black'
            ax.text(j, i, f'{val:.1f}', ha='center', va='center', color=color, fontsize=8)

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Percentage (%)')

    plt.tight_layout()

    fig.savefig(output_dir / 'fig2_skin_tone_distribution.pdf', format='pdf')
    fig.savefig(output_dir / 'fig2_skin_tone_distribution.png', format='png')
    plt.close()
    print(f"Saved: fig2_skin_tone_distribution.pdf/png")


def fig3_methodology_flowchart(output_dir: Path):
    """
    Figure 3: Methodology Flowchart
    Creates a visual representation of the training pipeline.
    """
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 8)
    ax.axis('off')

    # Define box style
    box_style = dict(boxstyle='round,pad=0.3', facecolor='lightblue', edgecolor='black', linewidth=1.5)
    arrow_style = dict(arrowstyle='->', color='black', linewidth=1.5)

    # Data sources
    ax.text(1.5, 7, 'ISIC2019\n(25,331 images)', ha='center', va='center', fontsize=10, bbox=dict(boxstyle='round,pad=0.3', facecolor='#E8F4FD', edgecolor='black'))
    ax.text(3.5, 7, 'MILK10k\n(5,137 images)', ha='center', va='center', fontsize=10, bbox=dict(boxstyle='round,pad=0.3', facecolor='#FDE8E8', edgecolor='black'))

    # Arrows to merge
    ax.annotate('', xy=(2.5, 6), xytext=(1.5, 6.5), arrowprops=arrow_style)
    ax.annotate('', xy=(2.5, 6), xytext=(3.5, 6.5), arrowprops=arrow_style)

    # Combined dataset
    ax.text(2.5, 5.5, 'Combined Dataset\n(30,468 images, 8 classes)', ha='center', va='center', fontsize=10, bbox=dict(boxstyle='round,pad=0.3', facecolor='#E8FDE8', edgecolor='black', linewidth=2))

    # Arrow to preprocessing
    ax.annotate('', xy=(2.5, 4.5), xytext=(2.5, 5), arrowprops=arrow_style)

    # Preprocessing
    ax.text(2.5, 4, 'Data Preprocessing\n• Resize (224/384)\n• Normalize\n• Augmentation', ha='center', va='center', fontsize=9, bbox=box_style)

    # Arrow to stratification
    ax.annotate('', xy=(2.5, 2.8), xytext=(2.5, 3.3), arrowprops=arrow_style)

    # Stratified sampling
    ax.text(2.5, 2.3, 'Skin-Tone-Aware\nStratified Sampling', ha='center', va='center', fontsize=10, bbox=dict(boxstyle='round,pad=0.3', facecolor='#FFF3E0', edgecolor='orange', linewidth=2), fontweight='bold')

    # Arrow to models
    ax.annotate('', xy=(6.5, 2.3), xytext=(4, 2.3), arrowprops=arrow_style)

    # Models box
    ax.text(8, 2.3, 'Deep Learning Models\n• Swin Transformer\n• DenseNet201\n• ResNet50', ha='center', va='center', fontsize=10, bbox=dict(boxstyle='round,pad=0.3', facecolor='#E8E8FD', edgecolor='purple', linewidth=2))

    # Training details
    ax.text(8, 4.5, 'Training Configuration\n• Focal Loss (γ=2.0)\n• Class Weighting\n• 10-Fold CV\n• Early Stopping', ha='center', va='center', fontsize=9, bbox=box_style)

    # Arrow from training to models
    ax.annotate('', xy=(8, 3.3), xytext=(8, 3.8), arrowprops=arrow_style)

    # Arrow to evaluation
    ax.annotate('', xy=(11.5, 2.3), xytext=(9.8, 2.3), arrowprops=arrow_style)

    # Evaluation
    ax.text(12.5, 2.3, 'Evaluation\n• Fairness Analysis\n• External Validation\n• XAI (Grad-CAM)', ha='center', va='center', fontsize=10, bbox=dict(boxstyle='round,pad=0.3', facecolor='#E8FDF0', edgecolor='green', linewidth=2))

    # Title
    ax.text(7, 7.5, 'Fairness-Aware Skin Cancer Classification Pipeline', ha='center', va='center', fontsize=14, fontweight='bold')

    plt.tight_layout()

    fig.savefig(output_dir / 'fig3_methodology_flowchart.pdf', format='pdf')
    fig.savefig(output_dir / 'fig3_methodology_flowchart.png', format='png')
    plt.close()
    print(f"Saved: fig3_methodology_flowchart.pdf/png")


def fig4_model_comparison(output_dir: Path):
    """
    Figure 4: Model Performance Comparison
    - (a) Overall metrics comparison
    - (b) Fairness comparison
    """
    # Data from evaluation results
    models = ['Swin', 'DenseNet', 'ResNet50']

    metrics = {
        'Balanced Accuracy': [90.8, 89.8, 90.3],
        'F1 (weighted)': [93.5, 93.0, 93.6],
        'AUC': [99.1, 98.9, 98.9],
    }

    fairness = {
        'Fairness Gap (%)': [6.7, 8.9, 10.4],
    }

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # (a) Overall metrics
    ax = axes[0]
    x = np.arange(len(models))
    width = 0.25

    for i, (metric, values) in enumerate(metrics.items()):
        offset = (i - 1) * width
        bars = ax.bar(x + offset, values, width, label=metric, edgecolor='black', linewidth=0.5)

        # Add value labels
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                    f'{val:.1f}', ha='center', va='bottom', fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.set_ylabel('Score (%)')
    ax.set_title('(a) Model Performance Comparison', fontweight='bold')
    ax.set_ylim(85, 102)
    ax.legend(loc='lower right')
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.axhline(y=90, color='gray', linestyle=':', alpha=0.5)

    # (b) Fairness comparison
    ax = axes[1]
    gaps = fairness['Fairness Gap (%)']
    colors = ['#00A087' if g < 7 else '#F39B7F' if g < 10 else '#E64B35' for g in gaps]

    bars = ax.bar(models, gaps, color=colors, edgecolor='black', linewidth=1)

    ax.axhline(y=5, color='green', linestyle='--', linewidth=2, label='Fair (<5%)')
    ax.axhline(y=10, color='red', linestyle='--', linewidth=2, label='Unfair (≥10%)')
    ax.fill_between([-0.5, 2.5], 0, 5, color='green', alpha=0.1)
    ax.fill_between([-0.5, 2.5], 5, 10, color='orange', alpha=0.1)
    ax.fill_between([-0.5, 2.5], 10, 15, color='red', alpha=0.1)

    ax.set_ylabel('Balanced Accuracy Gap (%)')
    ax.set_title('(b) Fairness Comparison (Skin Tone Gap)', fontweight='bold')
    ax.set_ylim(0, 12)
    ax.legend(loc='upper right')
    ax.grid(axis='y', alpha=0.3, linestyle='--')

    # Add value labels
    for bar, val in zip(bars, gaps):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
                f'{val:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')

    plt.tight_layout()

    fig.savefig(output_dir / 'fig4_model_comparison.pdf', format='pdf')
    fig.savefig(output_dir / 'fig4_model_comparison.png', format='png')
    plt.close()
    print(f"Saved: fig4_model_comparison.pdf/png")


def fig5_fairness_by_skin_tone(output_dir: Path):
    """
    Figure 5: Detailed Fairness Analysis by Skin Tone
    """
    # Data from fairness evaluation
    skin_tones = ['Very Dark\n(0-1)', 'Dark\n(2)', 'Medium\n(3)', 'Light\n(4)', 'Very Light\n(5)']

    swin_acc = [95.4, 93.7, 90.3, 88.8, 90.1]
    densenet_acc = [94.1, 94.9, 88.9, 88.7, 86.0]
    resnet_acc = [94.7, 95.1, 90.3, 88.0, 84.7]

    samples = [103, 487, 3131, 1050, 366]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # (a) Balanced accuracy by skin tone
    ax = axes[0]
    x = np.arange(len(skin_tones))
    width = 0.25

    ax.bar(x - width, swin_acc, width, label='Swin', color=MODEL_COLORS['Swin'], edgecolor='black')
    ax.bar(x, densenet_acc, width, label='DenseNet', color=MODEL_COLORS['DenseNet'], edgecolor='black')
    ax.bar(x + width, resnet_acc, width, label='ResNet50', color=MODEL_COLORS['ResNet50'], edgecolor='black')

    ax.set_xticks(x)
    ax.set_xticklabels(skin_tones)
    ax.set_ylabel('Balanced Accuracy (%)')
    ax.set_xlabel('Skin Tone Group')
    ax.set_title('(a) Model Performance by Skin Tone', fontweight='bold')
    ax.set_ylim(80, 100)
    ax.legend(loc='lower left')
    ax.grid(axis='y', alpha=0.3, linestyle='--')

    # Add trend annotation
    ax.annotate('', xy=(4.2, 86), xytext=(0.2, 95),
                arrowprops=dict(arrowstyle='->', color='red', lw=2, ls='--'))
    ax.text(2.5, 83, 'Performance decreases\nfor lighter skin tones', ha='center', fontsize=9, color='red', style='italic')

    # (b) Sample distribution
    ax = axes[1]
    colors = [SKIN_TONE_COLORS[i] for i in [0, 2, 3, 4, 5]]

    wedges, texts, autotexts = ax.pie(samples, labels=skin_tones, autopct='%1.1f%%',
                                       colors=colors, explode=[0.05, 0, 0, 0, 0],
                                       startangle=90, textprops={'fontsize': 9})

    ax.set_title('(b) Sample Distribution by Skin Tone', fontweight='bold')

    # Highlight imbalance
    ax.text(0, -1.4, 'Note: Very Dark skin tones (0-1) represent only 2% of data',
            ha='center', fontsize=9, style='italic', color='red')

    plt.tight_layout()

    fig.savefig(output_dir / 'fig5_fairness_by_skin_tone.pdf', format='pdf')
    fig.savefig(output_dir / 'fig5_fairness_by_skin_tone.png', format='png')
    plt.close()
    print(f"Saved: fig5_fairness_by_skin_tone.pdf/png")


def fig6_external_validation(output_dir: Path):
    """
    Figure 6: External Validation Results (HAM10000)
    """
    models = ['Swin', 'DenseNet', 'ResNet50']

    # HAM10000 results
    ham_acc = [82.7, 85.8, 87.8]
    ham_bal_acc = [85.1, 84.2, 84.8]
    ham_f1 = [84.2, 87.1, 88.8]

    # MILK10k results (for comparison)
    milk_bal_acc = [90.8, 89.8, 90.3]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # (a) HAM10000 metrics
    ax = axes[0]
    x = np.arange(len(models))
    width = 0.25

    ax.bar(x - width, ham_acc, width, label='Accuracy', color='#3C5488', edgecolor='black')
    ax.bar(x, ham_bal_acc, width, label='Balanced Acc', color='#E64B35', edgecolor='black')
    ax.bar(x + width, ham_f1, width, label='F1 (weighted)', color='#00A087', edgecolor='black')

    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.set_ylabel('Score (%)')
    ax.set_title('(a) External Validation on HAM10000', fontweight='bold')
    ax.set_ylim(75, 95)
    ax.legend(loc='lower right')
    ax.grid(axis='y', alpha=0.3, linestyle='--')

    # (b) MILK10k vs HAM10000 comparison
    ax = axes[1]
    x = np.arange(len(models))
    width = 0.35

    ax.bar(x - width/2, milk_bal_acc, width, label='MILK10k (Skin Tone)', color='#E64B35', edgecolor='black')
    ax.bar(x + width/2, ham_bal_acc, width, label='HAM10000 (External)', color='#4DBBD5', edgecolor='black')

    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.set_ylabel('Balanced Accuracy (%)')
    ax.set_title('(b) Performance: MILK10k vs HAM10000', fontweight='bold')
    ax.set_ylim(80, 95)
    ax.legend(loc='lower right')
    ax.grid(axis='y', alpha=0.3, linestyle='--')

    # Add delta annotations
    for i, (m, h) in enumerate(zip(milk_bal_acc, ham_bal_acc)):
        delta = m - h
        ax.annotate(f'Δ={delta:.1f}%', xy=(i, min(m, h) - 1),
                   ha='center', fontsize=9, color='gray')

    plt.tight_layout()

    fig.savefig(output_dir / 'fig6_external_validation.pdf', format='pdf')
    fig.savefig(output_dir / 'fig6_external_validation.png', format='png')
    plt.close()
    print(f"Saved: fig6_external_validation.pdf/png")


def fig7_eda_statistics(combined: pd.DataFrame, output_dir: Path):
    """
    Figure 7: Exploratory Data Analysis Statistics
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # (a) Age distribution by class
    ax = axes[0, 0]
    age_data = combined[combined['age'].notna()].copy()

    classes = ['MEL', 'NV', 'BCC', 'BKL', 'AK', 'SCC', 'VASC', 'DF']
    age_by_class = [age_data[age_data['diagnosis'] == c]['age'].dropna().values for c in classes]

    bp = ax.boxplot(age_by_class, labels=classes, patch_artist=True)
    for patch, cls in zip(bp['boxes'], classes):
        patch.set_facecolor(CLASS_COLORS[cls])
        patch.set_alpha(0.7)

    ax.set_ylabel('Age (years)')
    ax.set_xlabel('Diagnosis Class')
    ax.set_title('(a) Age Distribution by Class', fontweight='bold')
    ax.grid(axis='y', alpha=0.3, linestyle='--')

    # (b) Gender distribution
    ax = axes[0, 1]
    gender_counts = combined.groupby(['diagnosis', 'sex']).size().unstack(fill_value=0)
    gender_counts = gender_counts.reindex(classes)

    x = np.arange(len(classes))
    width = 0.35

    male_counts = gender_counts['male'].values if 'male' in gender_counts.columns else np.zeros(len(classes))
    female_counts = gender_counts['female'].values if 'female' in gender_counts.columns else np.zeros(len(classes))

    ax.bar(x - width/2, male_counts, width, label='Male', color='#4DBBD5', edgecolor='black')
    ax.bar(x + width/2, female_counts, width, label='Female', color='#E64B35', edgecolor='black')

    ax.set_xticks(x)
    ax.set_xticklabels(classes, rotation=45, ha='right')
    ax.set_ylabel('Number of Samples')
    ax.set_xlabel('Diagnosis Class')
    ax.set_title('(b) Gender Distribution by Class', fontweight='bold')
    ax.legend()
    ax.grid(axis='y', alpha=0.3, linestyle='--')

    # (c) Anatomical site distribution
    ax = axes[1, 0]
    site_counts = combined['anatom_site_general'].value_counts().head(8)

    colors = plt.cm.Set3(np.linspace(0, 1, len(site_counts)))
    bars = ax.barh(range(len(site_counts)), site_counts.values, color=colors, edgecolor='black')
    ax.set_yticks(range(len(site_counts)))
    ax.set_yticklabels(site_counts.index)
    ax.set_xlabel('Number of Samples')
    ax.set_title('(c) Anatomical Site Distribution', fontweight='bold')
    ax.grid(axis='x', alpha=0.3, linestyle='--')

    # (d) Dataset statistics table
    ax = axes[1, 1]
    ax.axis('off')

    stats_data = [
        ['Total Samples', f'{len(combined):,}'],
        ['ISIC2019 Samples', f'{len(combined[combined["dataset_source"]=="ISIC2019"]):,}'],
        ['MILK10k Samples', f'{len(combined[combined["dataset_source"]=="MILK10k"]):,}'],
        ['Number of Classes', '8'],
        ['Samples with Skin Tone', f'{combined["skin_tone"].notna().sum():,}'],
        ['Max Class (NV)', f'{combined["diagnosis"].value_counts()["NV"]:,}'],
        ['Min Class (DF)', f'{combined["diagnosis"].value_counts()["DF"]:,}'],
        ['Imbalance Ratio', f'{combined["diagnosis"].value_counts().max() / combined["diagnosis"].value_counts().min():.1f}:1'],
    ]

    table = ax.table(cellText=stats_data, colLabels=['Statistic', 'Value'],
                     loc='center', cellLoc='left', colColours=['#E8E8E8', '#E8E8E8'])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)
    ax.set_title('(d) Dataset Statistics Summary', fontweight='bold', pad=20)

    plt.tight_layout()

    fig.savefig(output_dir / 'fig7_eda_statistics.pdf', format='pdf')
    fig.savefig(output_dir / 'fig7_eda_statistics.png', format='png')
    plt.close()
    print(f"Saved: fig7_eda_statistics.pdf/png")


def generate_all_figures():
    """Generate all publication figures."""
    data_dir = Path("data")
    output_dir = Path("results/figures")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("="*60)
    print("GENERATING PUBLICATION FIGURES")
    print("="*60)

    # Load data
    print("\nLoading data...")
    combined, milk_meta = load_data(data_dir)
    print(f"Loaded {len(combined)} samples from combined dataset")

    # Generate figures
    print("\nGenerating figures...")

    fig1_dataset_overview(combined, output_dir)
    fig2_skin_tone_distribution(combined, output_dir)
    fig3_methodology_flowchart(output_dir)
    fig4_model_comparison(output_dir)
    fig5_fairness_by_skin_tone(output_dir)
    fig6_external_validation(output_dir)
    fig7_eda_statistics(combined, output_dir)

    print("\n" + "="*60)
    print(f"All figures saved to: {output_dir}")
    print("="*60)


if __name__ == "__main__":
    generate_all_figures()
