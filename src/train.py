"""
Training pipeline for skin cancer classification models.
"""

import os
import argparse
from typing import Dict, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from tqdm import tqdm
import wandb

from data_loader import create_data_loaders, create_kfold_loaders, get_class_weights
from models import get_model, print_model_summary, freeze_backbone
from utils import (
    set_seed, setup_logging, load_config, save_config,
    get_device, save_checkpoint, load_checkpoint,
    EarlyStopping, AverageMeter
)


class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance."""

    def __init__(self, alpha: Optional[torch.Tensor] = None, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = nn.functional.cross_entropy(inputs, targets, weight=self.alpha, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        return focal_loss.mean()


def train_one_epoch(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    scaler: Optional[GradScaler] = None,
    max_grad_norm: float = 1.0
) -> Tuple[float, float]:
    """Train for one epoch."""
    model.train()
    loss_meter = AverageMeter()
    acc_meter = AverageMeter()

    pbar = tqdm(train_loader, desc="Training")
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()

        if scaler is not None:
            with autocast():
                outputs = model(images)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()

        preds = outputs.argmax(dim=1)
        acc = (preds == labels).float().mean()

        loss_meter.update(loss.item(), images.size(0))
        acc_meter.update(acc.item(), images.size(0))
        pbar.set_postfix(loss=loss_meter.avg, acc=acc_meter.avg)

    return loss_meter.avg, acc_meter.avg


@torch.no_grad()
def validate(
    model: nn.Module,
    val_loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device
) -> Tuple[float, float, Dict[int, float]]:
    """Validate model."""
    model.eval()
    loss_meter = AverageMeter()
    acc_meter = AverageMeter()
    class_correct = {}
    class_total = {}

    for images, labels in tqdm(val_loader, desc="Validation"):
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        preds = outputs.argmax(dim=1)
        acc = (preds == labels).float().mean()

        loss_meter.update(loss.item(), images.size(0))
        acc_meter.update(acc.item(), images.size(0))

        for label, pred in zip(labels.cpu().numpy(), preds.cpu().numpy()):
            if label not in class_total:
                class_total[label] = 0
                class_correct[label] = 0
            class_total[label] += 1
            if label == pred:
                class_correct[label] += 1

    per_class_acc = {k: class_correct[k] / class_total[k] for k in class_total}
    return loss_meter.avg, acc_meter.avg, per_class_acc


def train(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    config: Dict,
    device: torch.device,
    output_dir: str,
    use_wandb: bool = False
) -> Dict:
    """Full training loop."""
    os.makedirs(output_dir, exist_ok=True)

    # Loss function with class weights
    class_weights = get_class_weights([label for _, label in train_loader.dataset], 7).to(device)
    if config.get('use_focal_loss', False):
        criterion = FocalLoss(alpha=class_weights, gamma=config.get('focal_gamma', 2.0))
    else:
        criterion = nn.CrossEntropyLoss(weight=class_weights)

    # Optimizer with discriminative learning rates
    backbone_params = []
    classifier_params = []
    for name, param in model.named_parameters():
        if 'fc' in name or 'classifier' in name or 'head' in name:
            classifier_params.append(param)
        else:
            backbone_params.append(param)

    optimizer = torch.optim.AdamW([
        {'params': backbone_params, 'lr': config['lr'] * 0.1},
        {'params': classifier_params, 'lr': config['lr']}
    ], weight_decay=config.get('weight_decay', 0.01))

    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=config.get('T_0', 10), T_mult=config.get('T_mult', 2))
    scaler = GradScaler() if config.get('use_amp', True) and device.type == 'cuda' else None
    early_stopping = EarlyStopping(patience=config.get('patience', 10), mode='max')

    best_val_acc = 0.0
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

    for epoch in range(config['epochs']):
        print(f"\nEpoch {epoch + 1}/{config['epochs']}")

        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device, scaler, config.get('max_grad_norm', 1.0)
        )
        val_loss, val_acc, per_class_acc = validate(model, val_loader, criterion, device)
        scheduler.step()

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

        if use_wandb:
            wandb.log({'train_loss': train_loss, 'train_acc': train_acc, 'val_loss': val_loss, 'val_acc': val_acc,
                       'lr': optimizer.param_groups[0]['lr'], 'epoch': epoch})

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_checkpoint(model, optimizer, epoch, val_loss, {'val_acc': val_acc, 'per_class_acc': per_class_acc},
                          os.path.join(output_dir, 'best_model.pth'), scheduler)
            print(f"Saved best model with val_acc: {val_acc:.4f}")

        save_checkpoint(model, optimizer, epoch, val_loss, {'val_acc': val_acc},
                       os.path.join(output_dir, 'last_model.pth'), scheduler)

        if early_stopping(val_acc):
            print(f"Early stopping at epoch {epoch + 1}")
            break

    return history


def main():
    parser = argparse.ArgumentParser(description="Train skin cancer classification model")
    parser.add_argument('--model', type=str, default='resnet50', choices=['resnet50', 'efficientnet', 'densenet', 'vit', 'swin'])
    parser.add_argument('--data_path', type=str, required=True, help='Path to dataset')
    parser.add_argument('--csv_path', type=str, default=None, help='Path to metadata CSV')
    parser.add_argument('--output_dir', type=str, default='./models', help='Output directory')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--image_size', type=int, default=224)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume from')
    parser.add_argument('--use_wandb', action='store_true')
    parser.add_argument('--kfold', type=int, default=0, help='Number of folds for CV (0 = no CV)')
    args = parser.parse_args()

    set_seed(args.seed)
    device = get_device()
    print(f"Using device: {device}")

    config = {
        'model': args.model, 'epochs': args.epochs, 'batch_size': args.batch_size,
        'lr': args.lr, 'image_size': args.image_size, 'seed': args.seed,
        'use_amp': True, 'use_focal_loss': False, 'patience': 10, 'weight_decay': 0.01
    }

    if args.use_wandb:
        wandb.init(project="skin-cancer-xai", config=config)

    output_dir = os.path.join(args.output_dir, args.model)
    os.makedirs(output_dir, exist_ok=True)
    save_config(config, os.path.join(output_dir, 'config.yaml'))

    if args.kfold > 0:
        fold_loaders = create_kfold_loaders(args.data_path, args.csv_path, args.kfold, args.batch_size, args.image_size)
        for fold, (train_loader, val_loader) in enumerate(fold_loaders):
            print(f"\n{'='*50}\nTraining Fold {fold + 1}/{args.kfold}\n{'='*50}")
            model = get_model(args.model, num_classes=7, pretrained=True).to(device)
            fold_output_dir = os.path.join(output_dir, f'fold_{fold + 1}')
            train(model, train_loader, val_loader, config, device, fold_output_dir, args.use_wandb)
    else:
        train_loader, val_loader, test_loader, _ = create_data_loaders(
            args.data_path, args.csv_path, args.batch_size, args.image_size
        )
        model = get_model(args.model, num_classes=7, pretrained=True).to(device)
        print_model_summary(model, args.model)

        if args.resume:
            model, start_epoch, _ = load_checkpoint(args.resume, model)
            print(f"Resumed from epoch {start_epoch}")

        train(model, train_loader, val_loader, config, device, output_dir, args.use_wandb)

    if args.use_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
