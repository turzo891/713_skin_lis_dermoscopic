"""
Explainable AI (XAI) methods for skin cancer classification.
Includes Grad-CAM, SHAP, LIME, Integrated Gradients, and more.
"""

import os
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import matplotlib.pyplot as plt
import cv2

try:
    from captum.attr import (
        IntegratedGradients, Saliency, GuidedBackprop,
        LayerGradCam, LayerAttribution, NoiseTunnel
    )
    CAPTUM_AVAILABLE = True
except ImportError:
    CAPTUM_AVAILABLE = False

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

try:
    from lime import lime_image
    LIME_AVAILABLE = True
except ImportError:
    LIME_AVAILABLE = False


class GradCAMPlusPlus:
    """Grad-CAM++ implementation for CNN visualization."""

    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0].detach()

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def generate(
        self,
        input_tensor: torch.Tensor,
        target_class: Optional[int] = None
    ) -> np.ndarray:
        """Generate Grad-CAM++ heatmap."""
        self.model.eval()
        output = self.model(input_tensor)

        if target_class is None:
            target_class = output.argmax(dim=1).item()

        self.model.zero_grad()
        one_hot = torch.zeros_like(output)
        one_hot[0, target_class] = 1
        output.backward(gradient=one_hot, retain_graph=True)

        # Grad-CAM++ weights
        gradients = self.gradients
        activations = self.activations

        b, k, u, v = gradients.size()
        alpha_num = gradients.pow(2)
        alpha_denom = gradients.pow(2).mul(2) + activations.mul(gradients.pow(3)).view(b, k, -1).sum(-1, keepdim=True).view(b, k, 1, 1)
        alpha_denom = torch.where(alpha_denom != 0.0, alpha_denom, torch.ones_like(alpha_denom))
        alphas = alpha_num.div(alpha_denom + 1e-7)

        positive_gradients = F.relu(output[0, target_class].exp() * gradients)
        weights = (alphas * positive_gradients).view(b, k, -1).sum(-1)

        cam = torch.sum(weights.view(b, k, 1, 1) * activations, dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = F.interpolate(cam, size=input_tensor.shape[2:], mode='bilinear', align_corners=False)
        cam = cam.squeeze().cpu().numpy()

        # Normalize
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam


class OcclusionSensitivity:
    """Occlusion sensitivity analysis."""

    def __init__(self, model: nn.Module, patch_size: int = 32, stride: int = 8):
        self.model = model
        self.patch_size = patch_size
        self.stride = stride

    @torch.no_grad()
    def generate(
        self,
        input_tensor: torch.Tensor,
        target_class: Optional[int] = None
    ) -> np.ndarray:
        """Generate occlusion sensitivity heatmap."""
        self.model.eval()
        device = input_tensor.device
        _, _, h, w = input_tensor.shape

        output = self.model(input_tensor)
        if target_class is None:
            target_class = output.argmax(dim=1).item()
        baseline_prob = F.softmax(output, dim=1)[0, target_class].item()

        sensitivity_map = np.zeros((h, w))
        count_map = np.zeros((h, w))

        for i in range(0, h - self.patch_size + 1, self.stride):
            for j in range(0, w - self.patch_size + 1, self.stride):
                occluded = input_tensor.clone()
                occluded[:, :, i:i+self.patch_size, j:j+self.patch_size] = 0

                output = self.model(occluded)
                prob = F.softmax(output, dim=1)[0, target_class].item()

                sensitivity_map[i:i+self.patch_size, j:j+self.patch_size] += (baseline_prob - prob)
                count_map[i:i+self.patch_size, j:j+self.patch_size] += 1

        sensitivity_map = sensitivity_map / (count_map + 1e-8)
        sensitivity_map = (sensitivity_map - sensitivity_map.min()) / (sensitivity_map.max() - sensitivity_map.min() + 1e-8)
        return sensitivity_map


class AttentionRollout:
    """Attention rollout for Vision Transformers."""

    def __init__(self, model: nn.Module, head_fusion: str = 'mean', discard_ratio: float = 0.9):
        self.model = model
        self.head_fusion = head_fusion
        self.discard_ratio = discard_ratio

    def generate(self, input_tensor: torch.Tensor) -> np.ndarray:
        """Generate attention rollout visualization."""
        self.model.eval()

        if not hasattr(self.model, 'get_attention_weights'):
            raise ValueError("Model must have get_attention_weights method")

        attention_weights = self.model.get_attention_weights(input_tensor)

        # Fuse heads
        result = torch.eye(attention_weights[0].size(-1))
        with torch.no_grad():
            for attention in attention_weights:
                if self.head_fusion == 'mean':
                    attention_heads_fused = attention.mean(dim=1)
                elif self.head_fusion == 'max':
                    attention_heads_fused = attention.max(dim=1)[0]
                elif self.head_fusion == 'min':
                    attention_heads_fused = attention.min(dim=1)[0]
                else:
                    raise ValueError(f"Unknown head fusion: {self.head_fusion}")

                # Discard low attention
                flat = attention_heads_fused.view(attention_heads_fused.size(0), -1)
                _, indices = flat.topk(int(flat.size(-1) * self.discard_ratio), dim=-1, largest=False)
                flat.scatter_(-1, indices, 0)

                I = torch.eye(attention_heads_fused.size(-1))
                a = (attention_heads_fused + I) / 2
                a = a / a.sum(dim=-1, keepdim=True)
                result = torch.matmul(a, result)

        # Get attention to CLS token
        mask = result[0, 0, 1:]  # Remove CLS token
        width = int(np.sqrt(mask.size(0)))
        mask = mask.reshape(width, width).numpy()
        mask = cv2.resize(mask, (input_tensor.shape[3], input_tensor.shape[2]))
        mask = (mask - mask.min()) / (mask.max() - mask.min() + 1e-8)
        return mask


class XAIExplainer:
    """Unified interface for XAI methods."""

    def __init__(self, model: nn.Module, device: torch.device):
        self.model = model
        self.device = device

    def explain(
        self,
        image: torch.Tensor,
        method: str = 'gradcam',
        target_class: Optional[int] = None,
        **kwargs
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate explanation for an image.

        Args:
            image: Input image tensor (1, C, H, W)
            method: XAI method ('gradcam', 'gradcam++', 'ig', 'shap', 'lime', 'occlusion', 'attention')
            target_class: Target class for explanation
            **kwargs: Additional arguments for specific methods

        Returns:
            Tuple of (attribution_map, visualization)
        """
        image = image.to(self.device)
        self.model.eval()

        if method == 'gradcam' or method == 'gradcam++':
            target_layer = self.model.get_cam_target_layer()
            explainer = GradCAMPlusPlus(self.model, target_layer)
            attr_map = explainer.generate(image, target_class)

        elif method == 'ig' and CAPTUM_AVAILABLE:
            ig = IntegratedGradients(self.model)
            baseline = torch.zeros_like(image).to(self.device)
            attributions = ig.attribute(image, baseline, target=target_class, n_steps=kwargs.get('n_steps', 50))
            attr_map = attributions.squeeze().cpu().numpy()
            attr_map = np.abs(attr_map).sum(axis=0)
            attr_map = (attr_map - attr_map.min()) / (attr_map.max() - attr_map.min() + 1e-8)

        elif method == 'saliency' and CAPTUM_AVAILABLE:
            saliency = Saliency(self.model)
            attributions = saliency.attribute(image, target=target_class)
            attr_map = attributions.squeeze().cpu().numpy()
            attr_map = np.abs(attr_map).sum(axis=0)
            attr_map = (attr_map - attr_map.min()) / (attr_map.max() - attr_map.min() + 1e-8)

        elif method == 'occlusion':
            patch_size = kwargs.get('patch_size', 32)
            stride = kwargs.get('stride', 8)
            explainer = OcclusionSensitivity(self.model, patch_size, stride)
            attr_map = explainer.generate(image, target_class)

        elif method == 'attention':
            if hasattr(self.model, 'get_attention_weights'):
                explainer = AttentionRollout(self.model)
                attr_map = explainer.generate(image)
            else:
                raise ValueError("Attention method only available for Vision Transformers")

        elif method == 'lime' and LIME_AVAILABLE:
            explainer = lime_image.LimeImageExplainer()

            def predict_fn(images):
                images = torch.tensor(images).permute(0, 3, 1, 2).float().to(self.device)
                with torch.no_grad():
                    outputs = self.model(images)
                return F.softmax(outputs, dim=1).cpu().numpy()

            img_np = image.squeeze().cpu().permute(1, 2, 0).numpy()
            explanation = explainer.explain_instance(img_np, predict_fn, top_labels=7, hide_color=0, num_samples=kwargs.get('num_samples', 1000))
            _, mask = explanation.get_image_and_mask(target_class if target_class else explanation.top_labels[0], positive_only=True, num_features=5, hide_rest=False)
            attr_map = mask.astype(float)

        elif method == 'shap' and SHAP_AVAILABLE:
            background = torch.zeros(1, *image.shape[1:]).to(self.device)
            explainer = shap.DeepExplainer(self.model, background)
            shap_values = explainer.shap_values(image)
            if isinstance(shap_values, list):
                attr_map = np.abs(shap_values[target_class if target_class else 0]).squeeze().sum(axis=0)
            else:
                attr_map = np.abs(shap_values).squeeze().sum(axis=0)
            attr_map = (attr_map - attr_map.min()) / (attr_map.max() - attr_map.min() + 1e-8)

        else:
            raise ValueError(f"Unknown or unavailable method: {method}")

        # Create visualization
        visualization = self._create_overlay(image, attr_map)
        return attr_map, visualization

    def _create_overlay(self, image: torch.Tensor, attr_map: np.ndarray, alpha: float = 0.5) -> np.ndarray:
        """Create heatmap overlay on original image."""
        # Denormalize image
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(image.device)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(image.device)
        img = image * std + mean
        img = img.squeeze().cpu().permute(1, 2, 0).numpy()
        img = np.clip(img, 0, 1)

        # Create heatmap
        heatmap = cv2.applyColorMap(np.uint8(255 * attr_map), cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB) / 255.0

        # Resize if needed
        if heatmap.shape[:2] != img.shape[:2]:
            heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))

        # Overlay
        overlay = alpha * heatmap + (1 - alpha) * img
        return np.clip(overlay, 0, 1)


def compare_xai_methods(
    model: nn.Module,
    images: List[torch.Tensor],
    methods: List[str],
    device: torch.device,
    save_dir: Optional[str] = None
) -> Dict[str, List[np.ndarray]]:
    """Compare multiple XAI methods on a set of images."""
    explainer = XAIExplainer(model, device)
    results = {method: [] for method in methods}

    for i, image in enumerate(images):
        for method in methods:
            try:
                attr_map, viz = explainer.explain(image, method)
                results[method].append(attr_map)

                if save_dir:
                    os.makedirs(save_dir, exist_ok=True)
                    plt.imsave(os.path.join(save_dir, f'image_{i}_{method}.png'), viz)
            except Exception as e:
                print(f"Error with {method}: {e}")
                results[method].append(None)

    return results


# XAI Quantitative Metrics
def confidence_increase(
    model: nn.Module,
    image: torch.Tensor,
    attr_map: np.ndarray,
    target_class: int,
    device: torch.device,
    threshold: float = 0.5
) -> float:
    """Calculate Confidence Increase metric."""
    model.eval()
    image = image.to(device)

    with torch.no_grad():
        original_output = model(image)
        original_conf = F.softmax(original_output, dim=1)[0, target_class].item()

    # Create mask from attribution map
    mask = (attr_map > threshold * attr_map.max()).astype(float)
    mask = torch.tensor(mask).unsqueeze(0).unsqueeze(0).float().to(device)
    mask = F.interpolate(mask, size=image.shape[2:], mode='bilinear', align_corners=False)

    masked_image = image * mask

    with torch.no_grad():
        masked_output = model(masked_image)
        masked_conf = F.softmax(masked_output, dim=1)[0, target_class].item()

    return masked_conf - original_conf


def faithfulness_metric(
    model: nn.Module,
    image: torch.Tensor,
    attr_map: np.ndarray,
    target_class: int,
    device: torch.device,
    num_steps: int = 10
) -> Tuple[List[float], List[float]]:
    """Calculate faithfulness by progressively masking important features."""
    model.eval()
    image = image.to(device)

    # Flatten and sort attribution values
    flat_attr = attr_map.flatten()
    sorted_indices = np.argsort(flat_attr)[::-1]  # Most important first

    mif_scores = []  # Most Important First
    lif_scores = []  # Least Important First

    step_size = len(sorted_indices) // num_steps

    for step in range(num_steps + 1):
        # MIF: mask top features progressively
        mif_mask = np.ones_like(flat_attr)
        mif_mask[sorted_indices[:step * step_size]] = 0
        mif_mask = mif_mask.reshape(attr_map.shape)
        mif_mask = torch.tensor(mif_mask).unsqueeze(0).unsqueeze(0).float().to(device)
        mif_mask = F.interpolate(mif_mask, size=image.shape[2:], mode='nearest')

        with torch.no_grad():
            output = model(image * mif_mask)
            conf = F.softmax(output, dim=1)[0, target_class].item()
            mif_scores.append(conf)

        # LIF: mask bottom features progressively
        lif_mask = np.ones_like(flat_attr)
        lif_mask[sorted_indices[-(step * step_size):]] = 0 if step > 0 else 1
        lif_mask = lif_mask.reshape(attr_map.shape)
        lif_mask = torch.tensor(lif_mask).unsqueeze(0).unsqueeze(0).float().to(device)
        lif_mask = F.interpolate(lif_mask, size=image.shape[2:], mode='nearest')

        with torch.no_grad():
            output = model(image * lif_mask)
            conf = F.softmax(output, dim=1)[0, target_class].item()
            lif_scores.append(conf)

    return mif_scores, lif_scores


def localization_iou(attr_map: np.ndarray, ground_truth_mask: np.ndarray, threshold: float = 0.5) -> float:
    """Calculate IoU between attribution map and ground truth mask."""
    binary_attr = (attr_map > threshold * attr_map.max()).astype(float)

    if binary_attr.shape != ground_truth_mask.shape:
        binary_attr = cv2.resize(binary_attr, (ground_truth_mask.shape[1], ground_truth_mask.shape[0]))

    intersection = np.logical_and(binary_attr, ground_truth_mask).sum()
    union = np.logical_or(binary_attr, ground_truth_mask).sum()

    return intersection / (union + 1e-8)


def sparsity_metric(attr_map: np.ndarray, threshold: float = 0.1) -> float:
    """Calculate sparsity of attribution map."""
    binary = (attr_map > threshold * attr_map.max()).astype(float)
    return 1 - (binary.sum() / binary.size)
