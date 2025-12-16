"""
Model definitions for skin cancer classification.
Includes CNN (ResNet, EfficientNet, DenseNet) and Vision Transformers (ViT, Swin).
"""

from typing import Dict, List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from torchvision import models


def count_parameters(model: nn.Module) -> Tuple[int, int]:
    """Count total and trainable parameters."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


class ResNet50Model(nn.Module):
    """ResNet50 for skin lesion classification."""

    def __init__(self, num_classes: int = 7, pretrained: bool = True, dropout: float = 0.5):
        super().__init__()
        weights = models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        self.backbone = models.resnet50(weights=weights)
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(nn.Dropout(dropout), nn.Linear(in_features, num_classes))
        self.num_classes = num_classes

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        modules = list(self.backbone.children())[:-1]
        feature_extractor = nn.Sequential(*modules)
        return feature_extractor(x).squeeze(-1).squeeze(-1)

    def get_cam_target_layer(self) -> nn.Module:
        return self.backbone.layer4[-1]


class EfficientNetModel(nn.Module):
    """EfficientNet-B4 for skin lesion classification."""

    def __init__(self, num_classes: int = 7, pretrained: bool = True, dropout: float = 0.5,
                 model_name: str = 'efficientnet_b4'):
        super().__init__()
        self.model = timm.create_model(model_name, pretrained=pretrained, num_classes=num_classes, drop_rate=dropout)
        self.num_classes = num_classes

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        return self.model.forward_features(x).mean(dim=[-2, -1])

    def get_cam_target_layer(self) -> nn.Module:
        return self.model.conv_head


class DenseNetModel(nn.Module):
    """DenseNet201 for skin lesion classification."""

    def __init__(self, num_classes: int = 7, pretrained: bool = True, dropout: float = 0.5):
        super().__init__()
        weights = models.DenseNet201_Weights.IMAGENET1K_V1 if pretrained else None
        self.backbone = models.densenet201(weights=weights)
        in_features = self.backbone.classifier.in_features
        self.backbone.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(in_features, num_classes))
        self.num_classes = num_classes

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone.features(x)
        features = F.relu(features, inplace=True)
        return F.adaptive_avg_pool2d(features, (1, 1)).squeeze(-1).squeeze(-1)

    def get_cam_target_layer(self) -> nn.Module:
        return self.backbone.features.denseblock4.denselayer32


class ViTModel(nn.Module):
    """Vision Transformer (ViT-B/16) for skin lesion classification."""

    def __init__(self, num_classes: int = 7, pretrained: bool = True, dropout: float = 0.1,
                 model_name: str = 'vit_base_patch16_224'):
        super().__init__()
        self.model = timm.create_model(model_name, pretrained=pretrained, num_classes=num_classes, drop_rate=dropout)
        self.num_classes = num_classes

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        return self.model.forward_features(x)[:, 0]

    def get_attention_weights(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Extract attention weights from all layers."""
        attention_weights = []
        x = self.model.patch_embed(x)
        cls_token = self.model.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_token, x), dim=1)
        x = x + self.model.pos_embed
        x = self.model.pos_drop(x)

        for block in self.model.blocks:
            B, N, C = x.shape
            qkv = block.attn.qkv(block.norm1(x)).reshape(B, N, 3, block.attn.num_heads, C // block.attn.num_heads)
            qkv = qkv.permute(2, 0, 3, 1, 4)
            q, k, v = qkv[0], qkv[1], qkv[2]
            attn = (q @ k.transpose(-2, -1)) * block.attn.scale
            attn = attn.softmax(dim=-1)
            attention_weights.append(attn)
            x = x + block.drop_path1(block.ls1(block.attn(block.norm1(x))))
            x = x + block.drop_path2(block.ls2(block.mlp(block.norm2(x))))

        return attention_weights

    def get_cam_target_layer(self) -> nn.Module:
        return self.model.blocks[-1].norm1


class SwinTransformerModel(nn.Module):
    """Swin Transformer for skin lesion classification."""

    def __init__(self, num_classes: int = 7, pretrained: bool = True, dropout: float = 0.1,
                 model_name: str = 'swin_base_patch4_window7_224'):
        super().__init__()
        self.model = timm.create_model(model_name, pretrained=pretrained, num_classes=num_classes, drop_rate=dropout)
        self.num_classes = num_classes

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        return self.model.forward_features(x).mean(dim=1)

    def get_cam_target_layer(self) -> nn.Module:
        return self.model.layers[-1].blocks[-1].norm1


def get_model(name: str, num_classes: int = 7, pretrained: bool = True, dropout: float = 0.5) -> nn.Module:
    """Factory function to create models."""
    name = name.lower()
    if name == 'resnet50':
        return ResNet50Model(num_classes, pretrained, dropout)
    elif name in ['efficientnet', 'efficientnet_b4']:
        return EfficientNetModel(num_classes, pretrained, dropout)
    elif name in ['densenet', 'densenet201']:
        return DenseNetModel(num_classes, pretrained, dropout)
    elif name in ['vit', 'vit_base']:
        return ViTModel(num_classes, pretrained, dropout)
    elif name in ['swin', 'swin_base']:
        return SwinTransformerModel(num_classes, pretrained, dropout)
    else:
        raise ValueError(f"Unknown model: {name}")


def freeze_backbone(model: nn.Module, freeze: bool = True) -> None:
    """Freeze or unfreeze model backbone."""
    if isinstance(model, (ResNet50Model, DenseNetModel)):
        for param in model.backbone.parameters():
            param.requires_grad = not freeze
        if hasattr(model.backbone, 'fc'):
            for param in model.backbone.fc.parameters():
                param.requires_grad = True
        else:
            for param in model.backbone.classifier.parameters():
                param.requires_grad = True
    elif isinstance(model, (EfficientNetModel, ViTModel, SwinTransformerModel)):
        for name, param in model.model.named_parameters():
            if 'head' in name or 'classifier' in name or 'fc' in name:
                param.requires_grad = True
            else:
                param.requires_grad = not freeze


def print_model_summary(model: nn.Module, model_name: str = "") -> None:
    """Print model summary with parameter counts."""
    total, trainable = count_parameters(model)
    print(f"\n{'='*60}\nModel: {model_name}\n{'='*60}")
    print(f"Total parameters: {total:,}")
    print(f"Trainable parameters: {trainable:,}")
    print(f"Non-trainable parameters: {total - trainable:,}\n{'='*60}\n")


class EnsembleModel(nn.Module):
    """Ensemble of multiple models."""

    def __init__(self, models_list: List[nn.Module], weights: Optional[List[float]] = None, method: str = 'average'):
        super().__init__()
        self.models = nn.ModuleList(models_list)
        self.method = method
        self.weights = weights if weights else [1.0 / len(models_list)] * len(models_list)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outputs = [model(x) for model in self.models]
        if self.method == 'average':
            return torch.stack(outputs).mean(dim=0)
        elif self.method == 'weighted':
            return sum(w * o for w, o in zip(self.weights, outputs))
        elif self.method == 'voting':
            predictions = [o.argmax(dim=1) for o in outputs]
            stacked = torch.stack(predictions, dim=1)
            voted = torch.mode(stacked, dim=1).values
            return F.one_hot(voted, outputs[0].shape[1]).float()
        raise ValueError(f"Unknown method: {self.method}")
