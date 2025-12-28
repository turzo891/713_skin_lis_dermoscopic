#!/usr/bin/env python3
"""
Generate Grad-CAM visualizations for skin cancer classification models.
Creates visualizations for samples across different classes and skin tones.
"""

import argparse
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.models import get_model
from src.xai_methods import GradCAMPlusPlus

# Constants
CLASSES = ['MEL', 'NV', 'BCC', 'BKL', 'AK', 'SCC', 'VASC', 'DF']
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
    0: 'Very Dark',
    1: 'Dark',
    2: 'Medium-Dark',
    3: 'Medium',
    4: 'Medium-Light',
    5: 'Very Light'
}


def load_model(model_path: str, model_name: str, device: str, image_size: int = 224):
    """Load trained model."""
    print(f"Loading model from {model_path}")

    model = get_model(
        name=model_name,
        num_classes=len(CLASSES),
        pretrained=False,
        image_size=image_size
    )

    checkpoint = torch.load(model_path, map_location=device)

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


def get_transform(image_size: int):
    """Get preprocessing transform."""
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])


def denormalize(tensor):
    """Denormalize image tensor."""
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    return tensor.cpu() * std + mean


def create_overlay(image: np.ndarray, heatmap: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """Create heatmap overlay on image."""
    import cv2

    # Create colored heatmap
    heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB) / 255.0

    # Resize if needed
    if heatmap_colored.shape[:2] != image.shape[:2]:
        heatmap_colored = cv2.resize(heatmap_colored, (image.shape[1], image.shape[0]))

    # Overlay
    overlay = alpha * heatmap_colored + (1 - alpha) * image
    return np.clip(overlay, 0, 1)


def generate_gradcam_for_sample(
    model,
    image_path: str,
    transform,
    device: str,
    true_label: str = None
) -> dict:
    """Generate Grad-CAM for a single image."""
    # Load image
    image = Image.open(image_path).convert('RGB')
    original_image = np.array(image) / 255.0

    # Transform
    input_tensor = transform(image).unsqueeze(0).to(device)

    # Get prediction
    with torch.no_grad():
        output = model(input_tensor)
        probs = torch.softmax(output, dim=1)
        pred_class = output.argmax(dim=1).item()
        confidence = probs[0, pred_class].item()

    # Generate Grad-CAM
    target_layer = model.get_cam_target_layer()
    gradcam = GradCAMPlusPlus(model, target_layer)
    heatmap = gradcam.generate(input_tensor, pred_class)

    # Create overlay
    # Resize original image to match input size
    import cv2
    original_resized = cv2.resize(original_image, (input_tensor.shape[3], input_tensor.shape[2]))
    overlay = create_overlay(original_resized, heatmap)

    return {
        'original': original_resized,
        'heatmap': heatmap,
        'overlay': overlay,
        'pred_class': CLASSES[pred_class],
        'pred_name': CLASS_NAMES[CLASSES[pred_class]],
        'confidence': confidence,
        'true_label': true_label
    }


def select_samples(metadata: pd.DataFrame, data_dir: Path, n_per_class: int = 2, n_per_tone: int = 2):
    """Select representative samples for visualization."""
    samples = []

    # Filter to MILK10k for skin tone visualization
    milk_df = metadata[
        (metadata['dataset_source'] == 'MILK10k') &
        (metadata['skin_tone'].notna())
    ].copy()

    # Sample by class
    print("Selecting samples by class...")
    for cls in CLASSES:
        cls_df = milk_df[milk_df['diagnosis'] == cls]
        if len(cls_df) > 0:
            n = min(n_per_class, len(cls_df))
            sampled = cls_df.sample(n=n, random_state=42)
            for _, row in sampled.iterrows():
                image_path = data_dir / row['image_path']
                if image_path.exists():
                    samples.append({
                        'path': str(image_path),
                        'class': cls,
                        'skin_tone': row['skin_tone'],
                        'type': 'by_class'
                    })

    # Sample by skin tone (for diversity)
    print("Selecting samples by skin tone...")
    for tone in [0, 1, 2, 3, 4, 5]:
        tone_df = milk_df[milk_df['skin_tone'] == tone]
        if len(tone_df) > 0:
            n = min(n_per_tone, len(tone_df))
            sampled = tone_df.sample(n=n, random_state=42)
            for _, row in sampled.iterrows():
                image_path = data_dir / row['image_path']
                if image_path.exists():
                    samples.append({
                        'path': str(image_path),
                        'class': row['diagnosis'],
                        'skin_tone': row['skin_tone'],
                        'type': 'by_skin_tone'
                    })

    return samples


def create_class_comparison_figure(results: list, output_path: str):
    """Create figure comparing Grad-CAM across classes."""
    if len(results) == 0:
        print(f"Skipping class comparison figure - no results")
        return

    # Group by class
    by_class = {}
    for r in results:
        if r['type'] == 'by_class':
            cls = r['true_label']
            if cls not in by_class:
                by_class[cls] = []
            by_class[cls].append(r)

    n_classes = len(by_class)
    if n_classes == 0:
        print(f"Skipping class comparison figure - no class results")
        return

    fig, axes = plt.subplots(n_classes, 3, figsize=(12, 4 * n_classes))
    if n_classes == 1:
        axes = axes.reshape(1, -1)

    for i, (cls, class_results) in enumerate(sorted(by_class.items())):
        r = class_results[0]  # Take first sample

        # Original
        axes[i, 0].imshow(r['original'])
        axes[i, 0].set_title(f"{CLASS_NAMES[cls]}\nTrue: {cls}", fontsize=10)
        axes[i, 0].axis('off')

        # Heatmap
        axes[i, 1].imshow(r['heatmap'], cmap='jet')
        axes[i, 1].set_title('Grad-CAM Heatmap', fontsize=10)
        axes[i, 1].axis('off')

        # Overlay
        axes[i, 2].imshow(r['overlay'])
        pred_correct = "✓" if r['pred_class'] == cls else "✗"
        axes[i, 2].set_title(f"Pred: {r['pred_class']} ({r['confidence']:.1%}) {pred_correct}", fontsize=10)
        axes[i, 2].axis('off')

    plt.suptitle('Grad-CAM Visualization by Disease Class', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def create_skin_tone_comparison_figure(results: list, output_path: str):
    """Create figure comparing Grad-CAM across skin tones."""
    if len(results) == 0:
        print(f"Skipping skin tone comparison figure - no results")
        return

    # Group by skin tone
    by_tone = {}
    for r in results:
        if r['type'] == 'by_skin_tone':
            tone = int(r['skin_tone'])
            if tone not in by_tone:
                by_tone[tone] = []
            by_tone[tone].append(r)

    n_tones = len(by_tone)
    if n_tones == 0:
        print(f"Skipping skin tone comparison figure - no skin tone results")
        return

    fig, axes = plt.subplots(n_tones, 3, figsize=(12, 4 * n_tones))
    if n_tones == 1:
        axes = axes.reshape(1, -1)

    for i, (tone, tone_results) in enumerate(sorted(by_tone.items())):
        r = tone_results[0]  # Take first sample

        # Original
        axes[i, 0].imshow(r['original'])
        axes[i, 0].set_title(f"Skin Tone: {SKIN_TONE_NAMES.get(tone, tone)}\nClass: {r['true_label']}", fontsize=10)
        axes[i, 0].axis('off')

        # Heatmap
        axes[i, 1].imshow(r['heatmap'], cmap='jet')
        axes[i, 1].set_title('Grad-CAM Heatmap', fontsize=10)
        axes[i, 1].axis('off')

        # Overlay
        axes[i, 2].imshow(r['overlay'])
        pred_correct = "✓" if r['pred_class'] == r['true_label'] else "✗"
        axes[i, 2].set_title(f"Pred: {r['pred_class']} ({r['confidence']:.1%}) {pred_correct}", fontsize=10)
        axes[i, 2].axis('off')

    plt.suptitle('Grad-CAM Visualization by Skin Tone', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def create_grid_figure(results: list, output_path: str, title: str, max_samples: int = 12):
    """Create a grid of Grad-CAM overlays."""
    if len(results) == 0:
        print(f"Skipping grid figure - no results")
        return

    n = min(len(results), max_samples)
    cols = 4
    rows = max(1, (n + cols - 1) // cols)

    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    axes = axes.flatten() if rows > 1 else [axes] if cols == 1 else axes

    for i in range(len(axes)):
        if i < n:
            r = results[i]
            axes[i].imshow(r['overlay'])
            pred_correct = "✓" if r['pred_class'] == r['true_label'] else "✗"
            axes[i].set_title(f"{r['true_label']} → {r['pred_class']} {pred_correct}\n{r['confidence']:.1%}", fontsize=9)
        axes[i].axis('off')

    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate Grad-CAM visualizations")
    parser.add_argument("--model_path", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--model_name", type=str, required=True,
                       choices=['resnet50', 'efficientnet', 'densenet', 'vit', 'swin'])
    parser.add_argument("--data_dir", type=str, default="data", help="Data directory")
    parser.add_argument("--output_dir", type=str, default="results/xai", help="Output directory")
    parser.add_argument("--image_size", type=int, default=224, help="Image size")
    parser.add_argument("--n_samples", type=int, default=2, help="Samples per class/tone")
    parser.add_argument("--device", type=str, default="cuda", help="Device")

    args = parser.parse_args()

    # Setup
    device = args.device if torch.cuda.is_available() else "cpu"
    output_dir = Path(args.output_dir) / args.model_name
    output_dir.mkdir(parents=True, exist_ok=True)
    data_dir = Path(args.data_dir)

    print("="*70)
    print("GRAD-CAM VISUALIZATION GENERATION")
    print("="*70)
    print(f"Model: {args.model_path}")
    print(f"Architecture: {args.model_name}")
    print(f"Output: {output_dir}")
    print("="*70)

    # Load model
    model = load_model(args.model_path, args.model_name, device, args.image_size)
    transform = get_transform(args.image_size)

    # Load metadata
    metadata_path = data_dir / "combined" / "master_metadata.csv"
    metadata = pd.read_csv(metadata_path)
    print(f"Loaded metadata with {len(metadata)} samples")

    # Select samples
    samples = select_samples(metadata, data_dir, n_per_class=args.n_samples, n_per_tone=args.n_samples)
    print(f"Selected {len(samples)} samples for visualization")

    # Generate Grad-CAM for each sample
    results = []
    for sample in tqdm(samples, desc="Generating Grad-CAM"):
        try:
            result = generate_gradcam_for_sample(
                model, sample['path'], transform, device, sample['class']
            )
            result['skin_tone'] = sample['skin_tone']
            result['type'] = sample['type']
            result['true_label'] = sample['class']
            results.append(result)
        except Exception as e:
            print(f"Error processing {sample['path']}: {e}")

    print(f"\nGenerated {len(results)} visualizations")

    # Create figures
    print("\nCreating comparison figures...")

    # By class
    create_class_comparison_figure(
        results,
        str(output_dir / "gradcam_by_class.png")
    )

    # By skin tone
    create_skin_tone_comparison_figure(
        results,
        str(output_dir / "gradcam_by_skin_tone.png")
    )

    # Grid of all overlays
    create_grid_figure(
        results,
        str(output_dir / "gradcam_grid.png"),
        f"Grad-CAM Visualizations - {args.model_name.upper()}"
    )

    # Save individual overlays
    individual_dir = output_dir / "individual"
    individual_dir.mkdir(exist_ok=True)
    for i, r in enumerate(results):
        plt.imsave(
            str(individual_dir / f"{i:03d}_{r['true_label']}_tone{int(r['skin_tone'])}.png"),
            r['overlay']
        )

    print(f"\nSaved {len(results)} individual visualizations to {individual_dir}")
    print("\nGrad-CAM generation complete!")


if __name__ == "__main__":
    main()
