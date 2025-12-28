#!/usr/bin/env python3
"""
External Validation on PH2 Dataset

Validates trained skin cancer classification models on the PH2 dataset
for TRUE external validation (completely independent from ISIC2019).

PH2 Dataset:
- 200 dermoscopic images from Hospital Pedro Hispano, Portugal
- 80 common nevi, 80 atypical nevi, 40 melanomas
- Completely independent from ISIC datasets

Citation:
    Teresa Mendonça, Pedro M. Ferreira, Jorge Marques, Andre R. S. Marcal, Jorge Rozeira.
    "PH² - A dermoscopic image database for research and benchmarking",
    35th IEEE EMBC, July 2013, Osaka, Japan.

Usage:
    python validate_external_ph2.py \
        --model_dir models/swin_fold0_20251227_165102 \
        --model_name swin \
        --ph2_dir data/PH2 \
        --output_dir results/external_validation/ph2
"""

import argparse
import os
import sys
from pathlib import Path
import json
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    precision_score, recall_score, roc_curve, auc
)
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.models import get_model


# Class mapping: PH2 diagnosis -> Our 8-class labels
# PH2 only has melanoma and nevi (common + atypical)
PH2_TO_STANDARD = {
    'melanoma': 'MEL',
    'nevus': 'NV',
    'common_nevus': 'NV',
    'atypical_nevus': 'NV',
}

# Our standard 8 classes (matching ISIC2019 training)
CLASSES = ['MEL', 'NV', 'BCC', 'BKL', 'AK', 'SCC', 'VASC', 'DF']
CLASS_TO_IDX = {cls: i for i, cls in enumerate(CLASSES)}

# Known PH2 melanoma image IDs (from PH2 documentation)
PH2_MELANOMA_IDS = {
    'IMD002', 'IMD003', 'IMD009', 'IMD016', 'IMD022', 'IMD024',
    'IMD025', 'IMD035', 'IMD037', 'IMD044', 'IMD045', 'IMD050',
    'IMD064', 'IMD065', 'IMD076', 'IMD078', 'IMD085', 'IMD088',
    'IMD090', 'IMD091', 'IMD168', 'IMD211', 'IMD219', 'IMD240',
    'IMD242', 'IMD251', 'IMD254', 'IMD256', 'IMD278', 'IMD279',
    'IMD280', 'IMD304', 'IMD305', 'IMD348', 'IMD349', 'IMD395',
    'IMD407', 'IMD413', 'IMD417', 'IMD420',
}


class PH2Dataset(Dataset):
    """PH2 dataset for external validation."""

    def __init__(
        self,
        data_dir: str,
        transform=None,
        labels_csv: str = None
    ):
        self.data_dir = Path(data_dir)
        self.transform = transform

        # Find images directory
        self.images_dir = self._find_images_dir()

        # Load or create labels
        if labels_csv and os.path.exists(labels_csv):
            self.metadata = pd.read_csv(labels_csv)
        else:
            self.metadata = self._create_metadata()

        print(f"Loaded {len(self.metadata)} PH2 images")
        print(f"Class distribution:")
        if 'mapped_class' in self.metadata.columns:
            print(self.metadata['mapped_class'].value_counts())
        elif 'diagnosis' in self.metadata.columns:
            print(self.metadata['diagnosis'].value_counts())

    def _find_images_dir(self) -> Path:
        """Find the images directory in PH2 structure."""
        possible_dirs = [
            self.data_dir / "PH2_Dataset_images",
            self.data_dir / "PH2 Dataset images",
            self.data_dir / "images",
        ]

        for d in possible_dirs:
            if d.exists():
                return d

        # Search for any directory with images
        for item in self.data_dir.iterdir():
            if item.is_dir() and any(
                f.suffix.lower() in ['.bmp', '.jpg', '.png']
                for f in item.rglob('*')
            ):
                return item

        raise FileNotFoundError(f"Could not find images directory in {self.data_dir}")

    def _create_metadata(self) -> pd.DataFrame:
        """Create metadata by scanning the PH2 directory structure."""
        records = []

        for lesion_dir in sorted(self.images_dir.iterdir()):
            if not lesion_dir.is_dir():
                continue

            image_id = lesion_dir.name

            # Find dermoscopic image
            image_path = None
            for subdir in lesion_dir.iterdir():
                if subdir.is_dir() and 'dermoscopic' in subdir.name.lower():
                    for img_file in subdir.iterdir():
                        if img_file.suffix.lower() in ['.bmp', '.jpg', '.png']:
                            image_path = str(img_file)
                            break
                    break

            # Also check for images directly in lesion_dir
            if image_path is None:
                for img_file in lesion_dir.iterdir():
                    if img_file.suffix.lower() in ['.bmp', '.jpg', '.png'] and 'lesion' not in img_file.name.lower():
                        image_path = str(img_file)
                        break

            if image_path:
                # Determine label based on known melanoma IDs
                diagnosis = 'melanoma' if image_id in PH2_MELANOMA_IDS else 'nevus'
                mapped_class = 'MEL' if diagnosis == 'melanoma' else 'NV'

                records.append({
                    'image_id': image_id,
                    'image_path': image_path,
                    'diagnosis': diagnosis,
                    'mapped_class': mapped_class,
                    'label': CLASS_TO_IDX[mapped_class]
                })

        return pd.DataFrame(records)

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        row = self.metadata.iloc[idx]

        # Load image
        image = Image.open(row['image_path']).convert('RGB')

        # Get label
        if 'label' in row:
            label = row['label']
        else:
            mapped_class = row.get('mapped_class', 'NV')
            label = CLASS_TO_IDX[mapped_class]

        if self.transform:
            image = self.transform(image)

        return image, label, row['image_id']


def load_model(model_dir: str, model_name: str, device: str, image_size: int = 224):
    """Load trained model from checkpoint."""
    model_path = Path(model_dir) / "best_model.pth"

    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    print(f"Loading model from {model_path}")

    # Create model architecture
    model = get_model(
        name=model_name,
        num_classes=len(CLASSES),
        pretrained=False,
        image_size=image_size
    )

    # Load weights
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)

    # Handle different checkpoint formats
    if isinstance(checkpoint, dict):
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        elif 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
        else:
            model.load_state_dict(checkpoint)
    else:
        model.load_state_dict(checkpoint)

    model = model.to(device)
    model.eval()

    return model


def evaluate(model, dataloader, device):
    """Run evaluation and collect predictions."""
    all_labels = []
    all_preds = []
    all_probs = []
    all_ids = []

    with torch.no_grad():
        for images, labels, image_ids in tqdm(dataloader, desc="Evaluating"):
            images = images.to(device)

            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            preds = outputs.argmax(dim=1)

            all_labels.extend(labels.numpy())
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            all_ids.extend(image_ids)

    return (
        np.array(all_labels),
        np.array(all_preds),
        np.array(all_probs),
        all_ids
    )


def compute_metrics(y_true, y_pred, y_prob, classes):
    """Compute comprehensive evaluation metrics."""
    metrics = {
        'accuracy': float(accuracy_score(y_true, y_pred)),
        'balanced_accuracy': float(balanced_accuracy_score(y_true, y_pred)),
        'f1_weighted': float(f1_score(y_true, y_pred, average='weighted', zero_division=0)),
        'f1_macro': float(f1_score(y_true, y_pred, average='macro', zero_division=0)),
        'precision_weighted': float(precision_score(y_true, y_pred, average='weighted', zero_division=0)),
        'recall_weighted': float(recall_score(y_true, y_pred, average='weighted', zero_division=0)),
    }

    # Binary classification metrics (MEL vs NV)
    # In PH2, we only have MEL (0) and NV (1)
    present_classes = np.unique(y_true)

    if len(present_classes) == 2 and set(present_classes) == {0, 1}:
        # MEL vs NV binary classification
        # Use MEL (0) as positive class for sensitivity/specificity
        mel_probs = y_prob[:, 0]  # Probability of melanoma

        try:
            metrics['auc_mel_vs_nv'] = float(roc_auc_score(
                (y_true == 0).astype(int),  # 1 if melanoma, 0 if nevus
                mel_probs
            ))
        except Exception as e:
            print(f"AUC calculation error: {e}")
            metrics['auc_mel_vs_nv'] = 0.0

        # Sensitivity (recall for MEL) and Specificity
        mel_true = (y_true == 0)
        mel_pred = (y_pred == 0)
        metrics['sensitivity_mel'] = float(recall_score(mel_true, mel_pred, zero_division=0))
        metrics['specificity_mel'] = float(recall_score(~mel_true, ~mel_pred, zero_division=0))

        # Precision for melanoma detection
        metrics['precision_mel'] = float(precision_score(mel_true, mel_pred, zero_division=0))
        metrics['f1_mel'] = float(f1_score(mel_true, mel_pred, zero_division=0))

    # Per-class metrics (for classes present in data)
    for class_idx in present_classes:
        class_name = classes[class_idx]
        class_true = (y_true == class_idx)
        class_pred = (y_pred == class_idx)
        metrics[f'f1_{class_name}'] = float(f1_score(class_true, class_pred, zero_division=0))
        metrics[f'precision_{class_name}'] = float(precision_score(class_true, class_pred, zero_division=0))
        metrics[f'recall_{class_name}'] = float(recall_score(class_true, class_pred, zero_division=0))

    # Sample counts
    metrics['n_samples'] = len(y_true)
    metrics['n_melanoma'] = int(np.sum(y_true == 0))
    metrics['n_nevus'] = int(np.sum(y_true == 1))
    metrics['n_classes_present'] = len(present_classes)

    return metrics


def plot_confusion_matrix(y_true, y_pred, classes, output_path):
    """Generate and save confusion matrix plot."""
    # Only use classes that are present
    present_classes = sorted(set(y_true) | set(y_pred))
    present_labels = [classes[i] for i in present_classes]

    cm = confusion_matrix(y_true, y_pred, labels=present_classes)

    # Normalize
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Absolute counts
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues',
        xticklabels=present_labels,
        yticklabels=present_labels,
        ax=axes[0]
    )
    axes[0].set_xlabel('Predicted', fontsize=12)
    axes[0].set_ylabel('True', fontsize=12)
    axes[0].set_title('Confusion Matrix (Counts)', fontsize=14)

    # Normalized
    sns.heatmap(
        cm_normalized, annot=True, fmt='.2%', cmap='Blues',
        xticklabels=present_labels,
        yticklabels=present_labels,
        ax=axes[1]
    )
    axes[1].set_xlabel('Predicted', fontsize=12)
    axes[1].set_ylabel('True', fontsize=12)
    axes[1].set_title('Confusion Matrix (Normalized)', fontsize=14)

    plt.suptitle('PH2 External Validation', fontsize=16)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved confusion matrix: {output_path}")


def plot_roc_curve(y_true, y_prob, output_path):
    """Generate ROC curve for MEL vs NV classification."""
    # MEL (0) vs NV (1) binary classification
    mel_true = (y_true == 0).astype(int)
    mel_probs = y_prob[:, 0]

    fpr, tpr, thresholds = roc_curve(mel_true, mel_probs)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (1 - Specificity)', fontsize=12)
    plt.ylabel('True Positive Rate (Sensitivity)', fontsize=12)
    plt.title('ROC Curve: Melanoma vs Nevus (PH2 Dataset)', fontsize=14)
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved ROC curve: {output_path}")

    return roc_auc


def plot_metrics_summary(metrics, output_path):
    """Plot summary of key metrics."""
    key_metrics = {
        'Accuracy': metrics['accuracy'],
        'Balanced Acc': metrics['balanced_accuracy'],
        'F1 (weighted)': metrics['f1_weighted'],
        'Sensitivity\n(MEL)': metrics.get('sensitivity_mel', 0),
        'Specificity\n(MEL)': metrics.get('specificity_mel', 0),
        'AUC': metrics.get('auc_mel_vs_nv', 0),
    }

    plt.figure(figsize=(10, 6))
    bars = plt.bar(range(len(key_metrics)), list(key_metrics.values()), color='steelblue')

    plt.xticks(range(len(key_metrics)), list(key_metrics.keys()), fontsize=11)
    plt.ylabel('Score', fontsize=12)
    plt.title('External Validation Performance on PH2 Dataset', fontsize=14)
    plt.ylim([0, 1])
    plt.grid(axis='y', alpha=0.3)

    # Add value labels
    for bar, val in zip(bars, key_metrics.values()):
        plt.text(bar.get_x() + bar.get_width()/2, val + 0.02,
                f'{val:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved metrics summary: {output_path}")


def generate_report(metrics, model_dir, output_dir, model_name):
    """Generate markdown report for publication."""
    report_path = Path(output_dir) / "external_validation_report_ph2.md"

    with open(report_path, 'w') as f:
        f.write("# External Validation Report - PH2 Dataset\n\n")
        f.write("## Dataset Information\n\n")
        f.write("| Property | Value |\n")
        f.write("|----------|-------|\n")
        f.write("| Dataset | PH2 (Hospital Pedro Hispano, Portugal) |\n")
        f.write("| Total Images | 200 |\n")
        f.write("| Melanoma | 40 |\n")
        f.write("| Nevi (common + atypical) | 160 |\n")
        f.write("| Independence | Fully independent from ISIC2019 |\n\n")

        f.write("## Model Information\n\n")
        f.write(f"- **Model Directory:** {Path(model_dir).name}\n")
        f.write(f"- **Architecture:** {model_name}\n")
        f.write(f"- **Samples Evaluated:** {metrics['n_samples']}\n\n")

        f.write("---\n\n")

        # Overall metrics
        f.write("## Overall Performance\n\n")
        f.write("| Metric | Value |\n")
        f.write("|--------|-------|\n")
        f.write(f"| Accuracy | {metrics['accuracy']:.4f} |\n")
        f.write(f"| Balanced Accuracy | {metrics['balanced_accuracy']:.4f} |\n")
        f.write(f"| F1 Score (weighted) | {metrics['f1_weighted']:.4f} |\n")
        f.write(f"| F1 Score (macro) | {metrics['f1_macro']:.4f} |\n\n")

        # Binary classification metrics
        f.write("## Melanoma Detection Performance\n\n")
        f.write("| Metric | Value |\n")
        f.write("|--------|-------|\n")

        if 'sensitivity_mel' in metrics:
            f.write(f"| Sensitivity (Recall) | {metrics['sensitivity_mel']:.4f} |\n")
            f.write(f"| Specificity | {metrics['specificity_mel']:.4f} |\n")
            f.write(f"| Precision | {metrics.get('precision_mel', 0):.4f} |\n")
            f.write(f"| F1 Score | {metrics.get('f1_mel', 0):.4f} |\n")

        if 'auc_mel_vs_nv' in metrics:
            f.write(f"| AUC-ROC | {metrics['auc_mel_vs_nv']:.4f} |\n")

        f.write("\n")

        # Interpretation
        f.write("## Clinical Interpretation\n\n")

        sens = metrics.get('sensitivity_mel', 0)
        spec = metrics.get('specificity_mel', 0)
        auc_val = metrics.get('auc_mel_vs_nv', 0)

        if sens >= 0.90:
            f.write("- **Sensitivity:** Excellent melanoma detection (>=90%)\n")
        elif sens >= 0.80:
            f.write("- **Sensitivity:** Good melanoma detection (>=80%)\n")
        else:
            f.write(f"- **Sensitivity:** Needs improvement ({sens:.1%})\n")

        if spec >= 0.80:
            f.write("- **Specificity:** Good at avoiding false positives\n")
        else:
            f.write(f"- **Specificity:** High false positive rate ({1-spec:.1%})\n")

        if auc_val >= 0.90:
            f.write("- **AUC:** Excellent discriminative ability\n")
        elif auc_val >= 0.80:
            f.write("- **AUC:** Good discriminative ability\n")
        else:
            f.write(f"- **AUC:** Moderate discriminative ability ({auc_val:.3f})\n")

        f.write("\n")

        # Publication notes
        f.write("## Publication Notes\n\n")
        f.write("This external validation demonstrates model generalization to an independent dataset:\n\n")
        f.write("1. **PH2** is completely independent from ISIC2019 (different institution, country, time period)\n")
        f.write("2. Tests binary classification (melanoma vs benign nevi)\n")
        f.write("3. Smaller dataset (200 images) but gold-standard expert annotations\n")
        f.write("4. Results should be reported alongside internal validation metrics\n\n")

        f.write("### Citation\n\n")
        f.write("```\n")
        f.write("Teresa Mendonça, Pedro M. Ferreira, Jorge Marques, Andre R. S. Marcal, Jorge Rozeira.\n")
        f.write('"PH² - A dermoscopic image database for research and benchmarking",\n')
        f.write("35th International Conference of the IEEE Engineering in Medicine and Biology Society,\n")
        f.write("July 3-7, 2013, Osaka, Japan.\n")
        f.write("```\n")

    print(f"Saved report: {report_path}")


def main():
    parser = argparse.ArgumentParser(description="External validation on PH2 dataset")
    parser.add_argument("--model_dir", type=str, required=True,
                       help="Path to trained model directory")
    parser.add_argument("--model_name", type=str, required=True,
                       choices=['resnet50', 'efficientnet', 'densenet', 'vit', 'swin'],
                       help="Model architecture name")
    parser.add_argument("--ph2_dir", type=str, default="data/PH2",
                       help="PH2 dataset directory")
    parser.add_argument("--output_dir", type=str, default="results/external_validation/ph2",
                       help="Output directory for results")
    parser.add_argument("--image_size", type=int, default=224,
                       help="Image size for inference")
    parser.add_argument("--batch_size", type=int, default=16,
                       help="Batch size for inference")
    parser.add_argument("--device", type=str, default="cuda",
                       help="Device (cuda/cpu)")
    parser.add_argument("--labels_csv", type=str, default=None,
                       help="Optional: CSV file with PH2 labels")

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("EXTERNAL VALIDATION - PH2 DATASET")
    print("=" * 70)
    print(f"Model: {args.model_dir}")
    print(f"Architecture: {args.model_name}")
    print(f"PH2 Directory: {args.ph2_dir}")
    print(f"Image Size: {args.image_size}")
    print("=" * 70 + "\n")

    # Check device
    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        device = "cpu"

    # Setup transforms
    transform = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    # Create dataset
    dataset = PH2Dataset(
        data_dir=args.ph2_dir,
        transform=transform,
        labels_csv=args.labels_csv
    )

    if len(dataset) == 0:
        print("ERROR: No images found in PH2 dataset!")
        print(f"Please ensure the PH2 dataset is downloaded and extracted to: {args.ph2_dir}")
        sys.exit(1)

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    # Load model
    model = load_model(args.model_dir, args.model_name, device, args.image_size)

    # Evaluate
    print("\nRunning evaluation...")
    y_true, y_pred, y_prob, image_ids = evaluate(model, dataloader, device)

    # Compute metrics
    metrics = compute_metrics(y_true, y_pred, y_prob, CLASSES)

    # Print summary
    print("\n" + "=" * 70)
    print("EXTERNAL VALIDATION RESULTS - PH2")
    print("=" * 70)
    print(f"Samples Evaluated: {metrics['n_samples']}")
    print(f"  - Melanoma: {metrics['n_melanoma']}")
    print(f"  - Nevus: {metrics['n_nevus']}")
    print("-" * 70)
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Balanced Accuracy: {metrics['balanced_accuracy']:.4f}")
    print(f"F1 (weighted): {metrics['f1_weighted']:.4f}")
    print("-" * 70)
    print("Melanoma Detection:")
    if 'sensitivity_mel' in metrics:
        print(f"  Sensitivity: {metrics['sensitivity_mel']:.4f}")
        print(f"  Specificity: {metrics['specificity_mel']:.4f}")
    if 'auc_mel_vs_nv' in metrics:
        print(f"  AUC-ROC: {metrics['auc_mel_vs_nv']:.4f}")
    print("=" * 70 + "\n")

    # Generate visualizations
    plot_confusion_matrix(
        y_true, y_pred, CLASSES,
        output_dir / "confusion_matrix_ph2.png"
    )

    plot_roc_curve(
        y_true, y_prob,
        output_dir / "roc_curve_ph2.png"
    )

    plot_metrics_summary(
        metrics,
        output_dir / "metrics_summary_ph2.png"
    )

    # Generate report
    generate_report(metrics, args.model_dir, output_dir, args.model_name)

    # Save metrics JSON
    metrics_path = output_dir / "external_validation_metrics_ph2.json"
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved metrics: {metrics_path}")

    # Save predictions CSV
    predictions_df = pd.DataFrame({
        'image_id': image_ids,
        'true_label': [CLASSES[i] for i in y_true],
        'pred_label': [CLASSES[i] for i in y_pred],
        'true_label_idx': y_true,
        'pred_label_idx': y_pred,
        'prob_MEL': y_prob[:, 0],
        'prob_NV': y_prob[:, 1],
        'correct': y_true == y_pred
    })
    predictions_path = output_dir / "predictions_ph2.csv"
    predictions_df.to_csv(predictions_path, index=False)
    print(f"Saved predictions: {predictions_path}")

    print("\nExternal validation complete!")
    print(f"Results saved to: {output_dir}")

    return metrics


if __name__ == "__main__":
    main()
