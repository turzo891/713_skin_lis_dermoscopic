"""
Generate XAI visualizations using Occlusion Sensitivity.
This method is model-agnostic and works correctly with Swin Transformer.
"""

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import cv2
from tqdm import tqdm
import timm

# Add project root to path
sys.path.insert(0, '/home/spoof/adv_pat')

class OcclusionSensitivity:
    """Occlusion sensitivity analysis - model agnostic."""

    def __init__(self, model: nn.Module, patch_size: int = 28, stride: int = 14):
        self.model = model
        self.patch_size = patch_size
        self.stride = stride

    @torch.no_grad()
    def generate(self, input_tensor: torch.Tensor, target_class: int = None) -> np.ndarray:
        """Generate occlusion sensitivity heatmap."""
        self.model.eval()
        device = input_tensor.device
        _, _, h, w = input_tensor.shape

        # Get baseline prediction
        output = self.model(input_tensor)
        if target_class is None:
            target_class = output.argmax(dim=1).item()
        baseline_prob = F.softmax(output, dim=1)[0, target_class].item()

        sensitivity_map = np.zeros((h, w))
        count_map = np.zeros((h, w))

        # Slide occlusion window
        for i in range(0, h - self.patch_size + 1, self.stride):
            for j in range(0, w - self.patch_size + 1, self.stride):
                occluded = input_tensor.clone()
                # Occlude with gray (mean value)
                occluded[:, :, i:i+self.patch_size, j:j+self.patch_size] = 0.5

                output = self.model(occluded)
                prob = F.softmax(output, dim=1)[0, target_class].item()

                # Higher drop = more important region
                importance = baseline_prob - prob
                sensitivity_map[i:i+self.patch_size, j:j+self.patch_size] += importance
                count_map[i:i+self.patch_size, j:j+self.patch_size] += 1

        # Average overlapping regions
        sensitivity_map = sensitivity_map / (count_map + 1e-8)

        # Normalize to 0-1
        sensitivity_map = np.clip(sensitivity_map, 0, None)  # Keep only positive (important) regions
        if sensitivity_map.max() > 0:
            sensitivity_map = sensitivity_map / sensitivity_map.max()

        return sensitivity_map


def load_model(model_path: str, model_name: str = 'swin', num_classes: int = 8, device: torch.device = None):
    """Load trained model."""
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Import model classes from src
    from src.models import SwinTransformerModel, DenseNetModel, ResNet50Model

    if model_name == 'swin':
        model = SwinTransformerModel(
            num_classes=num_classes,
            pretrained=False,
            model_name='swin_base_patch4_window7_224',
            image_size=384
        )
    elif model_name == 'densenet':
        model = DenseNetModel(num_classes=num_classes, pretrained=False)
    elif model_name == 'resnet50':
        model = ResNet50Model(num_classes=num_classes, pretrained=False)
    else:
        raise ValueError(f"Unknown model: {model_name}")

    # Load weights
    checkpoint = torch.load(model_path, map_location=device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)

    model = model.to(device)
    model.eval()
    return model


def load_and_preprocess_image(image_path: str, image_size: int = 384):
    """Load and preprocess image for model input."""
    from torchvision import transforms

    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    image = Image.open(image_path).convert('RGB')
    tensor = transform(image).unsqueeze(0)
    return tensor, image


def create_heatmap_overlay(image: Image.Image, heatmap: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """Create heatmap overlay on original image."""
    # Resize heatmap to image size
    img_array = np.array(image.resize((heatmap.shape[1], heatmap.shape[0]))) / 255.0

    # Apply colormap
    heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB) / 255.0

    # Blend
    overlay = alpha * heatmap_colored + (1 - alpha) * img_array
    return np.clip(overlay, 0, 1)


def generate_xai_visualizations():
    """Main function to generate all XAI visualizations."""

    # Paths
    model_path = '/home/spoof/adv_pat/models/swin_fold5_20251225_040536/best_model.pth'
    metadata_path = '/home/spoof/adv_pat/data/combined/master_metadata.csv'
    images_root = '/home/spoof/adv_pat/data/'
    output_dir = '/home/spoof/adv_pat/results/xai/swin_occlusion'

    os.makedirs(output_dir, exist_ok=True)

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load model
    print("Loading Swin model...")
    model = load_model(model_path, 'swin', num_classes=8, device=device)

    # Load metadata
    df = pd.read_csv(metadata_path)
    df_with_skin_tone = df[df['skin_tone'].notna()].copy()

    # Class names
    # IMPORTANT: Must match training order from src/data_loader.py load_isic2019_data()
    class_names = ['MEL', 'NV', 'BCC', 'AK', 'BKL', 'DF', 'VASC', 'SCC']

    # Create occlusion sensitivity analyzer
    occluder = OcclusionSensitivity(model, patch_size=48, stride=24)

    # ========== 1. XAI Grid (3x4 grid of samples - showing original + overlay) ==========
    print("\n1. Generating XAI Grid...")
    fig, axes = plt.subplots(3, 8, figsize=(24, 12))  # 8 columns: 4 pairs of (original, overlay)

    # Sample 12 random images
    samples = df_with_skin_tone.sample(n=12, random_state=42)

    for idx, (_, row) in enumerate(tqdm(samples.iterrows(), total=12, desc="Grid")):
        row_idx = idx // 4
        col_idx = (idx % 4) * 2  # Each sample takes 2 columns

        ax_orig = axes[row_idx, col_idx]
        ax_overlay = axes[row_idx, col_idx + 1]

        # Load image
        img_path = os.path.join(images_root, row['image_path'])
        if not os.path.exists(img_path):
            ax_orig.set_title("Image not found")
            ax_orig.axis('off')
            ax_overlay.axis('off')
            continue

        tensor, orig_img = load_and_preprocess_image(img_path, 384)
        tensor = tensor.to(device)

        # Get prediction
        with torch.no_grad():
            output = model(tensor)
            pred_class = output.argmax(dim=1).item()
            confidence = F.softmax(output, dim=1)[0, pred_class].item()

        # Generate heatmap
        heatmap = occluder.generate(tensor, pred_class)

        # Create overlay
        overlay = create_heatmap_overlay(orig_img, heatmap, alpha=0.5)

        # Show original
        ax_orig.imshow(orig_img)
        true_label = row['diagnosis']
        pred_label = class_names[pred_class]
        status = "✓" if true_label == pred_label else "✗"
        ax_orig.set_title(f"Original\n{true_label}", fontsize=9)
        ax_orig.axis('off')

        # Show overlay
        ax_overlay.imshow(overlay)
        ax_overlay.set_title(f"XAI: {pred_label} {status}\n{confidence*100:.1f}%", fontsize=9)
        ax_overlay.axis('off')

    plt.suptitle('Occlusion Sensitivity Visualizations - SWIN (Original + Heatmap)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'xai_grid.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   Saved: {output_dir}/xai_grid.png")

    # ========== 2. XAI by Class (one per class - showing original + overlay) ==========
    print("\n2. Generating XAI by Class...")
    fig, axes = plt.subplots(2, 8, figsize=(24, 8))  # 8 columns: 4 pairs per row

    for idx, class_name in enumerate(tqdm(class_names, desc="By Class")):
        row_idx = idx // 4
        col_idx = (idx % 4) * 2

        ax_orig = axes[row_idx, col_idx]
        ax_overlay = axes[row_idx, col_idx + 1]

        # Get sample from this class
        class_samples = df_with_skin_tone[df_with_skin_tone['diagnosis'] == class_name]
        if len(class_samples) == 0:
            ax_orig.set_title(f"{class_name}\nNo samples")
            ax_orig.axis('off')
            ax_overlay.axis('off')
            continue

        sample = class_samples.sample(n=1, random_state=42).iloc[0]
        img_path = os.path.join(images_root, sample['image_path'])

        if not os.path.exists(img_path):
            ax_orig.set_title(f"{class_name}\nImage not found")
            ax_orig.axis('off')
            ax_overlay.axis('off')
            continue

        tensor, orig_img = load_and_preprocess_image(img_path, 384)
        tensor = tensor.to(device)

        # Get prediction
        with torch.no_grad():
            output = model(tensor)
            pred_class = output.argmax(dim=1).item()
            confidence = F.softmax(output, dim=1)[0, pred_class].item()

        # Generate heatmap
        heatmap = occluder.generate(tensor, pred_class)
        overlay = create_heatmap_overlay(orig_img, heatmap, alpha=0.5)

        # Show original
        ax_orig.imshow(orig_img)
        ax_orig.set_title(f"Original\n{class_name}", fontsize=10)
        ax_orig.axis('off')

        # Show overlay
        pred_label = class_names[pred_class]
        status = "✓" if class_name == pred_label else "✗"
        ax_overlay.imshow(overlay)
        ax_overlay.set_title(f"XAI: {pred_label} {status}\n{confidence*100:.1f}%", fontsize=10)
        ax_overlay.axis('off')

    plt.suptitle('Occlusion Sensitivity by Diagnostic Class - SWIN (Original + Heatmap)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'xai_by_class.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   Saved: {output_dir}/xai_by_class.png")

    # ========== 3. XAI by Skin Tone (showing original + overlay) ==========
    print("\n3. Generating XAI by Skin Tone...")
    fig, axes = plt.subplots(2, 6, figsize=(24, 10))  # 6 columns: 3 pairs per row

    skin_tone_labels = ['Very Dark (0)', 'Dark (1)', 'Medium-Dark (2)',
                        'Medium (3)', 'Medium-Light (4)', 'Light (5)']

    for idx, tone in enumerate(tqdm(range(6), desc="By Skin Tone")):
        row_idx = idx // 3
        col_idx = (idx % 3) * 2

        ax_orig = axes[row_idx, col_idx]
        ax_overlay = axes[row_idx, col_idx + 1]

        # Get sample from this skin tone
        tone_samples = df_with_skin_tone[df_with_skin_tone['skin_tone'] == tone]
        if len(tone_samples) == 0:
            ax_orig.set_title(f"{skin_tone_labels[idx]}\nNo samples")
            ax_orig.axis('off')
            ax_overlay.axis('off')
            continue

        sample = tone_samples.sample(n=1, random_state=42).iloc[0]
        img_path = os.path.join(images_root, sample['image_path'])

        if not os.path.exists(img_path):
            ax_orig.set_title(f"{skin_tone_labels[idx]}\nImage not found")
            ax_orig.axis('off')
            ax_overlay.axis('off')
            continue

        tensor, orig_img = load_and_preprocess_image(img_path, 384)
        tensor = tensor.to(device)

        # Get prediction
        with torch.no_grad():
            output = model(tensor)
            pred_class = output.argmax(dim=1).item()
            confidence = F.softmax(output, dim=1)[0, pred_class].item()

        # Generate heatmap
        heatmap = occluder.generate(tensor, pred_class)
        overlay = create_heatmap_overlay(orig_img, heatmap, alpha=0.5)

        # Show original
        true_label = sample['diagnosis']
        ax_orig.imshow(orig_img)
        ax_orig.set_title(f"Original: {skin_tone_labels[idx]}\n{true_label}", fontsize=10)
        ax_orig.axis('off')

        # Show overlay
        pred_label = class_names[pred_class]
        status = "✓" if true_label == pred_label else "✗"
        ax_overlay.imshow(overlay)
        ax_overlay.set_title(f"XAI: {pred_label} {status}\n{confidence*100:.1f}%", fontsize=10)
        ax_overlay.axis('off')

    plt.suptitle('Occlusion Sensitivity by Skin Tone - SWIN (Original + Heatmap)\n(Verifying model focuses on lesion, not skin color)',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'xai_by_skin_tone.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   Saved: {output_dir}/xai_by_skin_tone.png")

    # ========== 4. Summary Statistics ==========
    print("\n4. Computing XAI Statistics...")

    # Sample images for statistics
    stat_samples = df_with_skin_tone.sample(n=min(50, len(df_with_skin_tone)), random_state=42)

    lesion_focus_scores = []

    for _, row in tqdm(stat_samples.iterrows(), total=len(stat_samples), desc="Stats"):
        img_path = os.path.join(images_root, row['image_path'])
        if not os.path.exists(img_path):
            continue

        tensor, orig_img = load_and_preprocess_image(img_path, 384)
        tensor = tensor.to(device)

        with torch.no_grad():
            output = model(tensor)
            pred_class = output.argmax(dim=1).item()

        heatmap = occluder.generate(tensor, pred_class)

        # Calculate focus score (how concentrated is the attention)
        # Higher = more focused on specific region
        h, w = heatmap.shape
        center_region = heatmap[h//4:3*h//4, w//4:3*w//4]
        edge_region_mean = (heatmap.sum() - center_region.sum()) / (h*w - center_region.size)
        center_mean = center_region.mean()

        if edge_region_mean > 0:
            focus_score = center_mean / (edge_region_mean + 1e-8)
        else:
            focus_score = center_mean * 10

        lesion_focus_scores.append(min(focus_score, 10))  # Cap at 10

    avg_focus = np.mean(lesion_focus_scores)
    std_focus = np.std(lesion_focus_scores)

    print(f"\n   XAI Statistics:")
    print(f"   - Average Focus Score: {avg_focus:.2f} (higher = more focused on center/lesion)")
    print(f"   - Std Dev: {std_focus:.2f}")
    print(f"   - Interpretation: {'Good lesion focus' if avg_focus > 1.5 else 'Needs investigation'}")

    # Save statistics
    stats = {
        'method': 'Occlusion Sensitivity',
        'model': 'Swin Transformer',
        'patch_size': 48,
        'stride': 24,
        'avg_focus_score': float(avg_focus),
        'std_focus_score': float(std_focus),
        'n_samples': len(lesion_focus_scores)
    }

    import json
    with open(os.path.join(output_dir, 'xai_statistics.json'), 'w') as f:
        json.dump(stats, f, indent=2)

    print(f"\n✓ All XAI visualizations saved to: {output_dir}")
    return output_dir


if __name__ == '__main__':
    generate_xai_visualizations()
