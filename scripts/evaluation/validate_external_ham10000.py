#!/usr/bin/env python3
"""
External Validation on HAM10000 Dataset

Validates trained skin cancer classification models on the HAM10000 dataset
to test generalization beyond the training distribution.

IMPORTANT: HAM10000 is a subset of ISIC2019, so images may overlap with training data.
This script identifies and excludes overlapping images for true external validation.

Usage:
    python validate_external_ham10000.py \
        --model_dir models/densenet_fold5_20251224_045753 \
        --model_name densenet \
        --output_dir results/external_validation
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
    roc_auc_score, confusion_matrix, classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.models import get_model


# Class mapping: HAM10000 diagnosis codes -> Our 8-class labels
HAM10000_TO_STANDARD = {
    'akiec': 'AK',   # Actinic Keratoses
    'bcc': 'BCC',    # Basal Cell Carcinoma
    'bkl': 'BKL',    # Benign Keratosis
    'df': 'DF',      # Dermatofibroma
    'mel': 'MEL',    # Melanoma
    'nv': 'NV',      # Melanocytic Nevi
    'vasc': 'VASC',  # Vascular lesions
}

# Our standard 8 classes (HAM10000 doesn't have SCC)
CLASSES = ['MEL', 'NV', 'BCC', 'BKL', 'AK', 'SCC', 'VASC', 'DF']
CLASS_TO_IDX = {cls: i for i, cls in enumerate(CLASSES)}


class HAM10000Dataset(Dataset):
    """HAM10000 dataset for external validation."""

    def __init__(
        self,
        metadata_path: str,
        images_dir: str,
        transform=None,
        exclude_ids: set = None
    ):
        self.metadata = pd.read_csv(metadata_path)
        self.images_dir = Path(images_dir)
        self.transform = transform

        # Map diagnoses to standard labels
        self.metadata['diagnosis'] = self.metadata['dx'].map(HAM10000_TO_STANDARD)

        # Filter out unknown diagnoses
        self.metadata = self.metadata[self.metadata['diagnosis'].notna()].copy()

        # Exclude overlapping IDs if provided
        if exclude_ids is not None:
            original_count = len(self.metadata)
            self.metadata = self.metadata[
                ~self.metadata['image_id'].isin(exclude_ids)
            ].copy()
            excluded_count = original_count - len(self.metadata)
            print(f"Excluded {excluded_count} overlapping images")

        # Map to class indices
        self.metadata['label'] = self.metadata['diagnosis'].map(CLASS_TO_IDX)

        # Find image paths (in part_1 or part_2)
        self.metadata['image_path'] = self.metadata['image_id'].apply(
            self._find_image_path
        )

        # Filter to existing images only
        self.metadata = self.metadata[self.metadata['image_path'].notna()].copy()
        self.metadata = self.metadata.reset_index(drop=True)

        print(f"Loaded {len(self.metadata)} HAM10000 images")
        print(f"Class distribution:")
        print(self.metadata['diagnosis'].value_counts())

    def _find_image_path(self, image_id: str) -> str:
        """Find image in either part_1 or part_2."""
        for part in ['HAM10000_images_part_1', 'HAM10000_images_part_2']:
            path = self.images_dir / part / f"{image_id}.jpg"
            if path.exists():
                return str(path)
        return None

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        row = self.metadata.iloc[idx]

        image = Image.open(row['image_path']).convert('RGB')
        label = row['label']

        if self.transform:
            image = self.transform(image)

        return image, label, row['image_id']


def get_overlapping_ids(isic2019_path: str, ham10000_metadata: pd.DataFrame) -> set:
    """
    Find image IDs that overlap between HAM10000 and ISIC2019 training data.
    HAM10000 images have ISIC_ prefixed IDs that may be in ISIC2019.
    """
    overlapping = set()

    if not os.path.exists(isic2019_path):
        print("WARNING: ISIC2019 metadata not found. Cannot check for overlaps.")
        return overlapping

    isic2019 = pd.read_csv(isic2019_path)
    isic2019_ids = set(isic2019['image'].values)

    for image_id in ham10000_metadata['image_id']:
        if image_id in isic2019_ids:
            overlapping.add(image_id)

    print(f"Found {len(overlapping)} overlapping images with ISIC2019")
    return overlapping


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
    checkpoint = torch.load(model_path, map_location=device)

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
    }

    # AUC-ROC (multiclass)
    try:
        # Only include classes present in y_true
        present_classes = np.unique(y_true)
        if len(present_classes) >= 2:
            # Filter probabilities to present classes
            metrics['auc_macro'] = float(roc_auc_score(
                y_true, y_prob,
                multi_class='ovr',
                average='macro',
                labels=present_classes
            ))
        else:
            metrics['auc_macro'] = 0.0
    except Exception as e:
        print(f"AUC calculation warning: {e}")
        metrics['auc_macro'] = 0.0

    # Per-class metrics
    f1_per_class = f1_score(y_true, y_pred, average=None, zero_division=0, labels=range(len(classes)))
    for i, cls in enumerate(classes):
        if i < len(f1_per_class):
            metrics[f'f1_{cls}'] = float(f1_per_class[i])

    # Sample counts
    metrics['n_samples'] = len(y_true)
    metrics['n_classes_present'] = len(np.unique(y_true))

    return metrics


def plot_confusion_matrix(y_true, y_pred, classes, output_path):
    """Generate and save confusion matrix plot."""
    # Only use classes that are present
    present_classes = sorted(set(y_true) | set(y_pred))
    present_labels = [classes[i] for i in present_classes]

    cm = confusion_matrix(y_true, y_pred, labels=present_classes)

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues',
        xticklabels=present_labels,
        yticklabels=present_labels
    )
    plt.xlabel('Predicted', fontsize=12)
    plt.ylabel('True', fontsize=12)
    plt.title('Confusion Matrix - HAM10000 External Validation', fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved confusion matrix: {output_path}")


def plot_class_performance(metrics, classes, output_path):
    """Plot per-class F1 scores."""
    f1_scores = []
    labels = []

    for cls in classes:
        key = f'f1_{cls}'
        if key in metrics:
            f1_scores.append(metrics[key])
            labels.append(cls)

    plt.figure(figsize=(10, 6))
    bars = plt.bar(range(len(labels)), f1_scores, color='steelblue')

    plt.xticks(range(len(labels)), labels, rotation=45, ha='right')
    plt.ylabel('F1 Score', fontsize=12)
    plt.xlabel('Class', fontsize=12)
    plt.title('Per-Class F1 Scores - HAM10000 External Validation', fontsize=14)
    plt.ylim([0, 1])
    plt.grid(axis='y', alpha=0.3)

    # Add value labels
    for bar, val in zip(bars, f1_scores):
        plt.text(bar.get_x() + bar.get_width()/2, val + 0.02,
                f'{val:.3f}', ha='center', va='bottom', fontsize=10)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved class performance plot: {output_path}")


def generate_report(metrics, model_dir, output_dir, exclude_overlaps):
    """Generate markdown report."""
    report_path = Path(output_dir) / "external_validation_report.md"

    with open(report_path, 'w') as f:
        f.write("# External Validation Report - HAM10000 Dataset\n\n")
        f.write(f"**Model:** {Path(model_dir).name}\n")
        f.write(f"**Validation Dataset:** HAM10000 (10,015 images)\n")
        f.write(f"**Overlap Exclusion:** {'Yes' if exclude_overlaps else 'No'}\n")
        f.write(f"**Samples Evaluated:** {metrics['n_samples']:,}\n\n")
        f.write("---\n\n")

        # Overall metrics
        f.write("## Overall Performance\n\n")
        f.write("| Metric | Value |\n")
        f.write("|--------|-------|\n")
        f.write(f"| Accuracy | {metrics['accuracy']:.4f} |\n")
        f.write(f"| Balanced Accuracy | {metrics['balanced_accuracy']:.4f} |\n")
        f.write(f"| F1 (weighted) | {metrics['f1_weighted']:.4f} |\n")
        f.write(f"| F1 (macro) | {metrics['f1_macro']:.4f} |\n")
        f.write(f"| AUC-ROC (macro) | {metrics['auc_macro']:.4f} |\n\n")

        # Per-class metrics
        f.write("## Per-Class Performance\n\n")
        f.write("| Class | F1 Score |\n")
        f.write("|-------|----------|\n")

        for cls in CLASSES:
            key = f'f1_{cls}'
            if key in metrics:
                f.write(f"| {cls} | {metrics[key]:.4f} |\n")

        f.write("\n")

        # Interpretation
        f.write("## Interpretation\n\n")

        bal_acc = metrics['balanced_accuracy']
        if bal_acc >= 0.75:
            f.write("The model shows **strong generalization** to the external HAM10000 dataset.\n\n")
        elif bal_acc >= 0.65:
            f.write("The model shows **moderate generalization** to the external HAM10000 dataset.\n\n")
        else:
            f.write("The model shows **limited generalization** to the external HAM10000 dataset.\n\n")

        f.write("### Key Observations:\n\n")
        f.write("1. HAM10000 contains 7 of our 8 classes (missing SCC)\n")
        f.write("2. Class distribution differs from training data (NV dominant at 67%)\n")
        f.write("3. Images may have different acquisition characteristics\n\n")

        # Recommendations
        f.write("## Recommendations\n\n")
        if bal_acc < 0.70:
            f.write("- Consider domain adaptation techniques\n")
            f.write("- Review misclassified cases for systematic errors\n")
            f.write("- May need to include HAM10000 in training with domain labels\n")
        else:
            f.write("- Model is suitable for deployment consideration\n")
            f.write("- Continue monitoring performance on diverse datasets\n")
            f.write("- Consider additional external validation on other datasets\n")

    print(f"Saved report: {report_path}")


def main():
    parser = argparse.ArgumentParser(description="External validation on HAM10000")
    parser.add_argument("--model_dir", type=str, required=True,
                       help="Path to trained model directory")
    parser.add_argument("--model_name", type=str, required=True,
                       choices=['resnet50', 'efficientnet', 'densenet', 'vit', 'swin'],
                       help="Model architecture name")
    parser.add_argument("--ham10000_dir", type=str, default="data/HAM10000",
                       help="HAM10000 dataset directory")
    parser.add_argument("--isic2019_metadata", type=str,
                       default="data/ISIC2019/ISIC_2019_Training_GroundTruth.csv",
                       help="ISIC2019 metadata for overlap detection")
    parser.add_argument("--output_dir", type=str, default="results/external_validation",
                       help="Output directory for results")
    parser.add_argument("--image_size", type=int, default=224,
                       help="Image size for inference")
    parser.add_argument("--batch_size", type=int, default=32,
                       help="Batch size for inference")
    parser.add_argument("--exclude_overlaps", action="store_true",
                       help="Exclude images that overlap with ISIC2019 training")
    parser.add_argument("--device", type=str, default="cuda",
                       help="Device (cuda/cpu)")

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("="*70)
    print("EXTERNAL VALIDATION - HAM10000 DATASET")
    print("="*70)
    print(f"Model: {args.model_dir}")
    print(f"Architecture: {args.model_name}")
    print(f"HAM10000: {args.ham10000_dir}")
    print(f"Image Size: {args.image_size}")
    print(f"Exclude Overlaps: {args.exclude_overlaps}")
    print("="*70 + "\n")

    # Check device
    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        device = "cpu"

    # Load HAM10000 metadata
    ham_metadata_path = Path(args.ham10000_dir) / "HAM10000_metadata.csv"
    ham_metadata = pd.read_csv(ham_metadata_path)

    # Check for overlaps with training data
    exclude_ids = set()
    if args.exclude_overlaps:
        exclude_ids = get_overlapping_ids(args.isic2019_metadata, ham_metadata)

    # Setup transforms (same as training validation)
    transform = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    # Create dataset
    dataset = HAM10000Dataset(
        metadata_path=ham_metadata_path,
        images_dir=args.ham10000_dir,
        transform=transform,
        exclude_ids=exclude_ids if args.exclude_overlaps else None
    )

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
    print("\n" + "="*70)
    print("EXTERNAL VALIDATION RESULTS")
    print("="*70)
    print(f"Samples Evaluated: {metrics['n_samples']:,}")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Balanced Accuracy: {metrics['balanced_accuracy']:.4f}")
    print(f"F1 (weighted): {metrics['f1_weighted']:.4f}")
    print(f"F1 (macro): {metrics['f1_macro']:.4f}")
    print(f"AUC-ROC (macro): {metrics['auc_macro']:.4f}")
    print("="*70 + "\n")

    # Generate visualizations
    plot_confusion_matrix(
        y_true, y_pred, CLASSES,
        output_dir / "confusion_matrix_ham10000.png"
    )

    plot_class_performance(
        metrics, CLASSES,
        output_dir / "class_performance_ham10000.png"
    )

    # Generate report
    generate_report(metrics, args.model_dir, output_dir, args.exclude_overlaps)

    # Save metrics JSON
    metrics_path = output_dir / "external_validation_metrics.json"
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved metrics: {metrics_path}")

    # Save predictions CSV
    predictions_df = pd.DataFrame({
        'image_id': image_ids,
        'true_label': [CLASSES[i] for i in y_true],
        'pred_label': [CLASSES[i] for i in y_pred],
        'correct': y_true == y_pred
    })
    predictions_path = output_dir / "predictions_ham10000.csv"
    predictions_df.to_csv(predictions_path, index=False)
    print(f"Saved predictions: {predictions_path}")

    print("\nExternal validation complete!")


if __name__ == "__main__":
    main()
