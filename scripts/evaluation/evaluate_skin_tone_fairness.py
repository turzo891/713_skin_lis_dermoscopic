#!/usr/bin/env python3
"""
Skin Tone Fairness Evaluation

Evaluates model performance across different skin tones using MILK10k subset.
Implements fairness metrics and generates comprehensive reports.

Based on:
- Nature Digital Medicine paper on skin tone evaluation (2025)
- MILK10k skin tone labels (0-5 scale: 0=very dark, 5=very light)

Usage:
    python evaluate_skin_tone_fairness.py \\
        --model_dir models/resnet50_fold0 \\
        --data_dir data/combined \\
        --output_dir results/fairness
"""

import argparse
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from typing import Dict, List, Tuple
import json
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.models import get_model
from torch.utils.data import Dataset, DataLoader
from PIL import Image


class SkinToneFairnessEvaluator:
    """Evaluates model fairness across skin tones"""

    def __init__(
        self,
        model_path: str,
        model_name: str,
        data_dir: str,
        output_dir: str,
        image_size: int = 224,
        batch_size: int = 32,
        device: str = "cuda"
    ):
        self.model_path = Path(model_path)
        self.model_name = model_name
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.image_size = image_size
        self.batch_size = batch_size
        self.device = device

        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Target 8 classes
        self.classes = ["MEL", "NV", "BCC", "BKL", "AK", "SCC", "VASC", "DF"]

        # Skin tone groups (based on MILK10k 0-5 scale)
        self.skin_tone_groups = {
            "Very Dark (0-1)": [0, 1],
            "Dark (2)": [2],
            "Medium (3)": [3],
            "Light (4)": [4],
            "Very Light (5)": [5],
        }

        # Setup transforms
        from torchvision import transforms
        self.transform = transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    def load_metadata(self) -> pd.DataFrame:
        """Load master metadata"""
        metadata_path = self.data_dir / "master_metadata.csv"

        if not metadata_path.exists():
            raise FileNotFoundError(f"Metadata not found: {metadata_path}")

        df = pd.read_csv(metadata_path)

        # Filter to MILK10k subset with skin tone data
        df = df[
            (df["dataset_source"] == "MILK10k") &
            (df["skin_tone"].notna())
        ].copy()

        print(f"Loaded {len(df)} MILK10k samples with skin tone data")

        # Add skin tone group
        df["skin_tone_group"] = df["skin_tone"].apply(self._assign_skin_tone_group)

        return df

    def _assign_skin_tone_group(self, tone: float) -> str:
        """Assign skin tone to group"""
        for group_name, tones in self.skin_tone_groups.items():
            if tone in tones:
                return group_name
        return "Unknown"

    def load_model(self):
        """Load trained model"""
        # Check for best_model.pth in directory
        if self.model_path.is_dir():
            checkpoint_path = self.model_path / "best_model.pth"
        else:
            checkpoint_path = self.model_path

        print(f"Loading model from {checkpoint_path}...")

        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Model not found: {checkpoint_path}")

        # Create model architecture
        model = get_model(
            name=self.model_name,
            num_classes=len(self.classes),
            pretrained=False,
            image_size=self.image_size
        )

        # Load weights
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

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

        model = model.to(self.device)
        model.eval()

        print(f"Model loaded successfully: {self.model_name}")
        return model

    def predict_on_subset(self, df: pd.DataFrame, model) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Run inference on subset of data

        Returns:
            y_true: Ground truth labels
            y_pred: Predicted labels
            y_prob: Prediction probabilities
        """
        print(f"Running predictions on {len(df)} samples...")

        # Create simple dataset
        class MILK10kSubset(Dataset):
            def __init__(self, dataframe, data_dir, classes, transform):
                self.df = dataframe.reset_index(drop=True)
                self.data_dir = Path(data_dir)
                self.classes = classes
                self.class_to_idx = {cls: i for i, cls in enumerate(classes)}
                self.transform = transform

            def __len__(self):
                return len(self.df)

            def __getitem__(self, idx):
                row = self.df.iloc[idx]

                # Use image_path from metadata if available
                if 'image_path' in row and pd.notna(row['image_path']):
                    image_path = self.data_dir / row['image_path']
                else:
                    # Fallback: try dermoscopic subfolder first
                    image_path = self.data_dir / "MILK10k" / "images" / "dermoscopic" / f"{row['image_id']}.jpg"

                # Try alternative paths
                if not image_path.exists():
                    image_path = self.data_dir / "MILK10k" / "images" / f"{row['image_id']}.jpg"
                if not image_path.exists():
                    image_path = self.data_dir / "MILK10k" / f"{row['image_id']}.jpg"
                if not image_path.exists():
                    # Return placeholder if image not found
                    print(f"Warning: Image not found: {row['image_id']}")
                    image = Image.new('RGB', (224, 224), (128, 128, 128))
                else:
                    image = Image.open(image_path).convert('RGB')

                if self.transform:
                    image = self.transform(image)

                label = self.class_to_idx.get(row['diagnosis'], 0)
                return image, label

        # Create dataset and dataloader
        dataset = MILK10kSubset(df, self.data_dir.parent, self.classes, self.transform)
        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )

        # Run inference
        all_labels = []
        all_preds = []
        all_probs = []

        model.eval()
        with torch.no_grad():
            for images, labels in tqdm(dataloader, desc="Evaluating"):
                images = images.to(self.device)

                outputs = model(images)
                probs = torch.softmax(outputs, dim=1)
                preds = outputs.argmax(dim=1)

                all_labels.extend(labels.numpy())
                all_preds.extend(preds.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())

        y_true = np.array(all_labels)
        y_pred = np.array(all_preds)
        y_prob = np.array(all_probs)

        print(f"Predictions complete. Accuracy: {(y_true == y_pred).mean():.4f}")

        return y_true, y_pred, y_prob

    def compute_metrics_by_skin_tone(
        self,
        df: pd.DataFrame,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_prob: np.ndarray
    ) -> pd.DataFrame:
        """Compute performance metrics stratified by skin tone"""
        print("\nComputing metrics by skin tone...")

        results = []

        # Overall metrics (all skin tones)
        overall_metrics = self._compute_metrics(y_true, y_pred, y_prob)
        overall_metrics["skin_tone_group"] = "Overall"
        overall_metrics["n_samples"] = len(df)
        results.append(overall_metrics)

        # By skin tone group
        for group_name in self.skin_tone_groups.keys():
            mask = df["skin_tone_group"] == group_name

            if mask.sum() == 0:
                continue

            y_true_group = y_true[mask]
            y_pred_group = y_pred[mask]
            y_prob_group = y_prob[mask]

            metrics = self._compute_metrics(y_true_group, y_pred_group, y_prob_group)
            metrics["skin_tone_group"] = group_name
            metrics["n_samples"] = mask.sum()

            results.append(metrics)

        df_results = pd.DataFrame(results)

        return df_results

    def _compute_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_prob: np.ndarray
    ) -> Dict:
        """Compute standard classification metrics"""

        metrics = {
            "accuracy": accuracy_score(y_true, y_pred),
            "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
            "f1_weighted": f1_score(y_true, y_pred, average="weighted", zero_division=0),
            "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
        }

        # AUC (handle multiclass)
        try:
            if len(np.unique(y_true)) > 1:
                metrics["auc_ovr"] = roc_auc_score(
                    y_true, y_prob, multi_class="ovr", average="macro"
                )
            else:
                metrics["auc_ovr"] = np.nan
        except:
            metrics["auc_ovr"] = np.nan

        # Per-class F1 scores
        f1_per_class = f1_score(y_true, y_pred, average=None, zero_division=0)
        for i, cls in enumerate(self.classes):
            if i < len(f1_per_class):
                metrics[f"f1_{cls}"] = f1_per_class[i]

        return metrics

    def compute_fairness_metrics(self, df_results: pd.DataFrame) -> Dict:
        """Compute fairness metrics"""
        print("\nComputing fairness metrics...")

        # Exclude overall row
        df_groups = df_results[df_results["skin_tone_group"] != "Overall"].copy()

        if len(df_groups) == 0:
            return {}

        fairness = {}

        # 1. Performance gap (max - min across groups)
        for metric in ["balanced_accuracy", "f1_weighted", "auc_ovr"]:
            if metric in df_groups.columns:
                values = df_groups[metric].dropna()
                if len(values) > 0:
                    fairness[f"{metric}_gap"] = float(values.max() - values.min())
                    fairness[f"{metric}_ratio"] = float(values.min() / values.max()) if values.max() > 0 else 0

        # 2. Statistical significance test (Kruskal-Wallis)
        # Note: This requires raw predictions per sample, not aggregated metrics
        # For now, report descriptive statistics

        fairness["n_groups_evaluated"] = len(df_groups)
        fairness["total_samples"] = int(df_groups["n_samples"].sum())

        # 3. Fairness threshold checks
        # Common threshold: <5% gap in balanced accuracy
        bal_acc_gap = fairness.get("balanced_accuracy_gap", 0)
        if bal_acc_gap < 0.05:
            fairness["fairness_status"] = "Fair (gap < 5%)"
        elif bal_acc_gap < 0.10:
            fairness["fairness_status"] = "Moderately Fair (gap < 10%)"
        else:
            fairness["fairness_status"] = "Unfair (gap >= 10%)"

        return fairness

    def plot_fairness_results(
        self,
        df_results: pd.DataFrame,
        fairness: Dict
    ):
        """Create fairness visualization plots"""
        print("\nGenerating fairness plots...")

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        # Exclude overall for some plots
        df_groups = df_results[df_results["skin_tone_group"] != "Overall"].copy()

        # 1. Balanced Accuracy by Skin Tone
        ax1 = axes[0, 0]
        groups = df_groups["skin_tone_group"].values
        bal_acc = df_groups["balanced_accuracy"].values

        colors = plt.cm.viridis(np.linspace(0, 1, len(groups)))
        bars = ax1.bar(range(len(groups)), bal_acc, color=colors)

        ax1.set_xticks(range(len(groups)))
        ax1.set_xticklabels(groups, rotation=45, ha="right")
        ax1.set_ylabel("Balanced Accuracy", fontsize=12)
        ax1.set_title("Balanced Accuracy by Skin Tone", fontsize=14, fontweight="bold")
        ax1.set_ylim([0, 1])
        ax1.axhline(y=0.9, color="green", linestyle="--", alpha=0.5, label="Target (90%)")
        ax1.axhline(y=0.8, color="orange", linestyle="--", alpha=0.5, label="Acceptable (80%)")
        ax1.legend()
        ax1.grid(axis="y", alpha=0.3)

        # Add value labels
        for i, (bar, val) in enumerate(zip(bars, bal_acc)):
            ax1.text(i, val + 0.02, f"{val:.3f}", ha="center", va="bottom", fontsize=10)

        # 2. F1 Score by Skin Tone
        ax2 = axes[0, 1]
        f1_weighted = df_groups["f1_weighted"].values

        bars = ax2.bar(range(len(groups)), f1_weighted, color=colors)

        ax2.set_xticks(range(len(groups)))
        ax2.set_xticklabels(groups, rotation=45, ha="right")
        ax2.set_ylabel("F1 Score (weighted)", fontsize=12)
        ax2.set_title("F1 Score by Skin Tone", fontsize=14, fontweight="bold")
        ax2.set_ylim([0, 1])
        ax2.grid(axis="y", alpha=0.3)

        # Add value labels
        for i, (bar, val) in enumerate(zip(bars, f1_weighted)):
            ax2.text(i, val + 0.02, f"{val:.3f}", ha="center", va="bottom", fontsize=10)

        # 3. Per-class F1 scores by skin tone (heatmap)
        ax3 = axes[1, 0]

        # Extract per-class F1 scores
        f1_matrix = []
        for _, row in df_groups.iterrows():
            f1_scores = [row.get(f"f1_{cls}", 0) for cls in self.classes]
            f1_matrix.append(f1_scores)

        f1_matrix = np.array(f1_matrix)

        im = ax3.imshow(f1_matrix.T, cmap="RdYlGn", aspect="auto", vmin=0, vmax=1)
        ax3.set_xticks(range(len(groups)))
        ax3.set_yticks(range(len(self.classes)))
        ax3.set_xticklabels(groups, rotation=45, ha="right")
        ax3.set_yticklabels(self.classes)
        ax3.set_title("Per-Class F1 Scores by Skin Tone", fontsize=14, fontweight="bold")

        # Add colorbar
        cbar = plt.colorbar(im, ax=ax3)
        cbar.set_label("F1 Score", rotation=270, labelpad=15)

        # 4. Sample size distribution
        ax4 = axes[1, 1]
        n_samples = df_groups["n_samples"].values

        bars = ax4.bar(range(len(groups)), n_samples, color=colors)

        ax4.set_xticks(range(len(groups)))
        ax4.set_xticklabels(groups, rotation=45, ha="right")
        ax4.set_ylabel("Number of Samples", fontsize=12)
        ax4.set_title("Sample Distribution by Skin Tone", fontsize=14, fontweight="bold")
        ax4.grid(axis="y", alpha=0.3)

        # Add value labels
        for i, (bar, val) in enumerate(zip(bars, n_samples)):
            ax4.text(i, val + max(n_samples)*0.01, f"{int(val):,}",
                    ha="center", va="bottom", fontsize=10)

        plt.tight_layout()

        # Save plot
        plot_path = self.output_dir / "skin_tone_fairness.png"
        plt.savefig(plot_path, dpi=300, bbox_inches="tight")
        print(f"Saved plot: {plot_path}")

        plt.close()

    def generate_report(
        self,
        df_results: pd.DataFrame,
        fairness: Dict
    ):
        """Generate markdown report"""
        print("\nGenerating fairness report...")

        report_path = self.output_dir / "fairness_report.md"

        with open(report_path, "w") as f:
            f.write("# Skin Tone Fairness Evaluation Report\n\n")
            f.write(f"**Model:** {self.model_path.name}\n")
            f.write(f"**Dataset:** MILK10k subset\n\n")
            f.write("---\n\n")

            # Summary
            f.write("## Summary\n\n")
            f.write(f"- **Total Samples Evaluated:** {fairness.get('total_samples', 0):,}\n")
            f.write(f"- **Number of Skin Tone Groups:** {fairness.get('n_groups_evaluated', 0)}\n")
            f.write(f"- **Fairness Status:** {fairness.get('fairness_status', 'Unknown')}\n\n")

            # Performance gaps
            f.write("## Fairness Metrics\n\n")
            f.write("| Metric | Gap (max-min) | Ratio (min/max) | Status |\n")
            f.write("|--------|---------------|-----------------|--------|\n")

            for metric in ["balanced_accuracy", "f1_weighted", "auc_ovr"]:
                gap = fairness.get(f"{metric}_gap", np.nan)
                ratio = fairness.get(f"{metric}_ratio", np.nan)

                if not np.isnan(gap):
                    if gap < 0.05:
                        status = " Fair"
                    elif gap < 0.10:
                        status = " Moderate"
                    else:
                        status = " Unfair"

                    f.write(f"| {metric} | {gap:.4f} | {ratio:.4f} | {status} |\n")

            f.write("\n")

            # Performance by skin tone table
            f.write("## Performance by Skin Tone\n\n")
            f.write("| Skin Tone Group | Samples | Balanced Acc | F1 (weighted) | AUC (OvR) |\n")
            f.write("|-----------------|---------|--------------|---------------|----------|\n")

            for _, row in df_results.iterrows():
                group = row["skin_tone_group"]
                n = int(row["n_samples"])
                bal_acc = row.get("balanced_accuracy", np.nan)
                f1 = row.get("f1_weighted", np.nan)
                auc = row.get("auc_ovr", np.nan)

                f.write(f"| {group} | {n:,} | {bal_acc:.4f} | {f1:.4f} | {auc:.4f} |\n")

            f.write("\n")

            # Recommendations
            f.write("## Recommendations\n\n")

            bal_acc_gap = fairness.get("balanced_accuracy_gap", 0)

            if bal_acc_gap < 0.05:
                f.write(" **Model meets fairness criteria** (gap < 5%)\n\n")
                f.write("The model performs consistently across skin tones.\n")
            elif bal_acc_gap < 0.10:
                f.write(" **Model shows moderate fairness concerns** (gap 5-10%)\n\n")
                f.write("Consider:\n")
                f.write("1. Collecting more data for underrepresented skin tones\n")
                f.write("2. Applying skin tone-aware data augmentation\n")
                f.write("3. Using fairness-constrained training objectives\n")
            else:
                f.write(" **Model shows significant fairness issues** (gap > 10%)\n\n")
                f.write("Action required:\n")
                f.write("1. **Data Collection:** Increase samples for poorly performing skin tones\n")
                f.write("2. **Rebalancing:** Use skin tone-stratified sampling during training\n")
                f.write("3. **Model Debugging:** Investigate why certain skin tones perform worse\n")
                f.write("4. **Feature Analysis:** Check if model relies on skin color vs lesion features\n")

            f.write("\n")

            # Clinical implications
            f.write("## Clinical Implications\n\n")
            f.write("Based on the Nature Digital Medicine paper on skin tone assessment:\n\n")
            f.write("- **Melanoma Detection:** Critical to maintain high sensitivity across all skin tones\n")
            f.write("- **Monk Skin Tone Scale:** MILK10k uses 0-5 scale, better than Fitzpatrick for AI fairness\n")
            f.write("- **Bias Mitigation:** Essential for equitable healthcare AI deployment\n\n")

        print(f"Saved report: {report_path}")

    def save_results(self, df_results: pd.DataFrame, fairness: Dict):
        """Save numerical results"""
        # Save metrics table
        results_path = self.output_dir / "metrics_by_skin_tone.csv"
        df_results.to_csv(results_path, index=False)
        print(f"Saved metrics: {results_path}")

        # Save fairness metrics
        fairness_path = self.output_dir / "fairness_metrics.json"
        with open(fairness_path, "w") as f:
            json.dump(fairness, f, indent=2)
        print(f"Saved fairness metrics: {fairness_path}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate skin tone fairness")
    parser.add_argument("--model_path", type=str, required=True,
                       help="Path to trained model directory or checkpoint")
    parser.add_argument("--model_name", type=str, required=True,
                       choices=['resnet50', 'efficientnet', 'densenet', 'vit', 'swin'],
                       help="Model architecture name")
    parser.add_argument("--data_dir", type=str, default="data/combined",
                       help="Combined dataset directory")
    parser.add_argument("--output_dir", type=str, default="results/fairness",
                       help="Output directory for results")
    parser.add_argument("--image_size", type=int, default=224,
                       help="Image size for inference")
    parser.add_argument("--batch_size", type=int, default=32,
                       help="Batch size for inference")
    parser.add_argument("--device", type=str, default="cuda",
                       help="Device (cuda/cpu)")

    args = parser.parse_args()

    # Check device
    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        device = "cpu"

    print("="*70)
    print("SKIN TONE FAIRNESS EVALUATION")
    print("="*70)
    print(f"Model: {args.model_path}")
    print(f"Architecture: {args.model_name}")
    print(f"Data: {args.data_dir}")
    print(f"Image Size: {args.image_size}")
    print(f"Output: {args.output_dir}")
    print("="*70 + "\n")

    evaluator = SkinToneFairnessEvaluator(
        model_path=args.model_path,
        model_name=args.model_name,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        image_size=args.image_size,
        batch_size=args.batch_size,
        device=device
    )

    # Load metadata
    df = evaluator.load_metadata()

    # Load model
    model = evaluator.load_model()

    # Get predictions
    y_true, y_pred, y_prob = evaluator.predict_on_subset(df, model)

    # Compute metrics by skin tone
    df_results = evaluator.compute_metrics_by_skin_tone(df, y_true, y_pred, y_prob)

    # Compute fairness metrics
    fairness = evaluator.compute_fairness_metrics(df_results)

    # Print summary
    print("\n" + "="*70)
    print("FAIRNESS SUMMARY")
    print("="*70)
    print(f"Status: {fairness.get('fairness_status', 'Unknown')}")
    print(f"Balanced Accuracy Gap: {fairness.get('balanced_accuracy_gap', 0):.4f}")
    print("="*70 + "\n")

    # Generate plots
    evaluator.plot_fairness_results(df_results, fairness)

    # Generate report
    evaluator.generate_report(df_results, fairness)

    # Save results
    evaluator.save_results(df_results, fairness)

    print(" Fairness evaluation complete!\n")


if __name__ == "__main__":
    main()
