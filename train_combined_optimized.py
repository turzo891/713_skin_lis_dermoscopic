#!/usr/bin/env python3
"""
Optimized Training Script for Combined MILK10k + ISIC2019 Dataset
Supports 10-fold CV, skin tone-aware sampling, and all optimization features.
"""

import sys
sys.path.insert(0, 'src')
sys.path.insert(0, 'scripts/training')

import os
import time
import argparse
from datetime import datetime
import json
import logging
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
try:
    from torch.amp import autocast, GradScaler
except ImportError:
    from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score,
    f1_score, roc_auc_score, classification_report,
    confusion_matrix
)

from models import get_model
from skin_tone_aware_sampler import SkinToneAwareDataset, SkinToneAwareSampler
from torchvision import transforms


class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance."""

    def __init__(self, alpha=None, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        ce_loss = nn.functional.cross_entropy(
            inputs, targets, weight=self.alpha, reduction='none'
        )
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        return focal_loss.mean()


class OptimizedTrainer:
    """Optimized trainer for combined dataset with all features."""

    def __init__(self, args):
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Create output directory
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.output_dir = Path(args.output_dir) / f"{args.model}_fold{args.fold}_{timestamp}"
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Setup logging
        self.setup_logging()

        # Initialize mixed precision scaler
        if args.use_amp:
            try:
                self.scaler = GradScaler('cuda')
            except TypeError:
                # Fallback for older PyTorch versions
                self.scaler = GradScaler()
        else:
            self.scaler = None

        # Load class weights if provided
        self.class_weights = None
        if args.class_weights_path:
            df_weights = pd.read_csv(args.class_weights_path)
            if args.use_focal_loss:
                weights = df_weights['focal_alpha'].values
            else:
                weights = df_weights['weight'].values
            self.class_weights = torch.FloatTensor(weights).to(self.device)

        self.print_config()

    def setup_logging(self):
        """Setup logging to file and console."""
        log_file = self.output_dir / 'training.log'

        # Create logger
        self.logger = logging.getLogger('TrainingLogger')
        self.logger.setLevel(logging.INFO)

        # Remove existing handlers
        self.logger.handlers = []

        # File handler
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)

        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)

        # Formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)

        # Add handlers
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)

        self.logger.info(f"Logging initialized. Log file: {log_file}")

    def print_config(self):
        """Print training configuration."""
        config_msg = [
            "="*80,
            "OPTIMIZED TRAINING CONFIGURATION",
            "="*80,
            f"Model: {self.args.model}",
            f"Dataset: Combined MILK10k + ISIC2019",
            f"Fold: {self.args.fold}",
            f"Batch size: {self.args.batch_size}",
            f"Effective batch size: {self.args.batch_size * self.args.accumulation_steps}",
            f"Epochs: {self.args.epochs}",
            f"Learning rate: {self.args.lr}",
            f"Workers: {self.args.num_workers}",
            f"Mixed precision: {self.args.use_amp}",
            f"Gradient accumulation: {self.args.accumulation_steps}",
            f"Prefetch factor: {self.args.prefetch_factor}",
            f"Focal loss: {self.args.use_focal_loss}",
            f"Skin tone sampling: {self.args.use_skin_tone_sampling}",
            f"Device: {self.device}",
            f"Output: {self.output_dir}",
            "="*80
        ]

        for msg in config_msg:
            print(msg)
            self.logger.info(msg)

    def create_data_loaders(self):
        """Create optimized data loaders with skin tone-aware sampling."""
        msg = "\nCreating data loaders..."
        print(msg)
        self.logger.info(msg)

        # Define transforms
        train_transform = transforms.Compose([
            transforms.Resize((self.args.image_size, self.args.image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(20),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        val_transform = transforms.Compose([
            transforms.Resize((self.args.image_size, self.args.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # Create datasets
        train_dataset = SkinToneAwareDataset(
            metadata_path=self.args.metadata_path,
            images_root=self.args.images_root,
            split='train',
            fold=self.args.fold,
            transform=train_transform
        )

        val_dataset = SkinToneAwareDataset(
            metadata_path=self.args.metadata_path,
            images_root=self.args.images_root,
            split='val',
            fold=self.args.fold,
            transform=val_transform
        )

        self.logger.info(f"Train samples: {len(train_dataset)}")
        self.logger.info(f"Val samples: {len(val_dataset)}")
        print(f"Train samples: {len(train_dataset)}")
        print(f"Val samples: {len(val_dataset)}")

        # Create train loader with optional skin tone-aware sampling
        if self.args.use_skin_tone_sampling:
            sampler = SkinToneAwareSampler(
                dataset=train_dataset,
                samples_per_epoch=self.args.samples_per_epoch or len(train_dataset)
            )
            train_loader = DataLoader(
                train_dataset,
                batch_size=self.args.batch_size,
                sampler=sampler,
                num_workers=self.args.num_workers,
                pin_memory=True,
                prefetch_factor=self.args.prefetch_factor,
                persistent_workers=True if self.args.num_workers > 0 else False
            )
        else:
            train_loader = DataLoader(
                train_dataset,
                batch_size=self.args.batch_size,
                shuffle=True,
                num_workers=self.args.num_workers,
                pin_memory=True,
                prefetch_factor=self.args.prefetch_factor,
                persistent_workers=True if self.args.num_workers > 0 else False
            )

        # Create validation loader
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.args.batch_size * 2,  # Larger batch for validation
            shuffle=False,
            num_workers=self.args.num_workers,
            pin_memory=True,
            prefetch_factor=self.args.prefetch_factor,
            persistent_workers=True if self.args.num_workers > 0 else False
        )

        return train_loader, val_loader

    def train_epoch(self, model, train_loader, criterion, optimizer, epoch):
        """Train for one epoch with all optimizations."""
        model.train()

        total_loss = 0
        all_preds = []
        all_labels = []

        pbar = tqdm(train_loader, desc=f'Epoch {epoch}/{self.args.epochs}')

        optimizer.zero_grad()

        for batch_idx, batch_data in enumerate(pbar):
            # Unpack batch (dataset returns 3 values: image, label, metadata)
            if len(batch_data) == 3:
                images, labels, _ = batch_data  # Ignore metadata
            else:
                images, labels = batch_data

            # Move to GPU (non-blocking for speed)
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            # Mixed precision training
            if self.args.use_amp:
                try:
                    # Try new API
                    with autocast('cuda'):
                        outputs = model(images)
                        loss = criterion(outputs, labels)
                        loss = loss / self.args.accumulation_steps
                except TypeError:
                    # Fallback to old API
                    with autocast():
                        outputs = model(images)
                        loss = criterion(outputs, labels)
                        loss = loss / self.args.accumulation_steps

                self.scaler.scale(loss).backward()

                # Update weights after accumulation
                if (batch_idx + 1) % self.args.accumulation_steps == 0:
                    self.scaler.step(optimizer)
                    self.scaler.update()
                    optimizer.zero_grad()
            else:
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss = loss / self.args.accumulation_steps

                loss.backward()

                if (batch_idx + 1) % self.args.accumulation_steps == 0:
                    optimizer.step()
                    optimizer.zero_grad()

            # Statistics
            total_loss += loss.item() * self.args.accumulation_steps
            preds = outputs.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            # Update progress bar
            pbar.set_postfix({
                'loss': f'{total_loss/(batch_idx+1):.4f}',
                'acc': f'{100.*np.mean(np.array(all_preds)==np.array(all_labels)):.2f}%'
            })

        avg_loss = total_loss / len(train_loader)
        acc = accuracy_score(all_labels, all_preds)

        return avg_loss, acc

    def validate(self, model, val_loader, criterion):
        """Validate the model with comprehensive metrics."""
        model.eval()

        # Check if validation set is empty
        if len(val_loader) == 0:
            msg = "WARNING: Validation set is empty! Check your fold number (valid: 0-9 for 10-fold CV)"
            print(msg)
            self.logger.warning(msg)
            # Return dummy metrics
            return {
                'loss': 999.0,
                'accuracy': 0.0,
                'balanced_accuracy': 0.0,
                'f1_weighted': 0.0,
                'f1_macro': 0.0,
                'auc_macro': 0.0
            }

        total_loss = 0
        all_preds = []
        all_labels = []
        all_probs = []

        with torch.no_grad():
            for batch_data in tqdm(val_loader, desc='Validating'):
                # Unpack batch (dataset returns 3 values: image, label, metadata)
                if len(batch_data) == 3:
                    images, labels, _ = batch_data  # Ignore metadata
                else:
                    images, labels = batch_data

                images = images.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)

                if self.args.use_amp:
                    try:
                        with autocast('cuda'):
                            outputs = model(images)
                            loss = criterion(outputs, labels)
                    except TypeError:
                        with autocast():
                            outputs = model(images)
                            loss = criterion(outputs, labels)
                else:
                    outputs = model(images)
                    loss = criterion(outputs, labels)

                total_loss += loss.item()
                preds = outputs.argmax(dim=1)
                probs = torch.softmax(outputs, dim=1)

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())

        avg_loss = total_loss / len(val_loader)

        # Compute metrics
        all_labels = np.array(all_labels)
        all_preds = np.array(all_preds)
        all_probs = np.array(all_probs)

        metrics = {
            'loss': avg_loss,
            'accuracy': accuracy_score(all_labels, all_preds),
            'balanced_accuracy': balanced_accuracy_score(all_labels, all_preds),
            'f1_weighted': f1_score(all_labels, all_preds, average='weighted'),
            'f1_macro': f1_score(all_labels, all_preds, average='macro')
        }

        # AUC-ROC (handle edge cases)
        try:
            num_classes = all_probs.shape[1]
            # Check if all classes are present in labels
            unique_labels = np.unique(all_labels)
            if len(unique_labels) >= 2:
                metrics['auc_macro'] = roc_auc_score(
                    all_labels, all_probs,
                    multi_class='ovr',
                    average='macro'
                )
            else:
                metrics['auc_macro'] = 0.0
        except Exception as e:
            # Log the error for debugging
            self.logger.warning(f"AUC calculation failed: {e}")
            metrics['auc_macro'] = 0.0

        return metrics

    def train(self):
        """Main training loop."""
        # Create data loaders
        train_loader, val_loader = self.create_data_loaders()

        # Create model
        msg = f"\nCreating {self.args.model} model..."
        print(msg)
        self.logger.info(msg)
        model = get_model(self.args.model, num_classes=8, pretrained=True, image_size=self.args.image_size)
        model = model.to(self.device)

        # Loss function
        if self.args.use_focal_loss:
            criterion = FocalLoss(
                alpha=self.class_weights,
                gamma=self.args.focal_gamma
            )
        else:
            criterion = nn.CrossEntropyLoss(weight=self.class_weights)

        # Optimizer
        optimizer = optim.AdamW(
            model.parameters(),
            lr=self.args.lr,
            weight_decay=self.args.weight_decay
        )

        # Scheduler
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.args.epochs
        )

        # Training history
        history = {
            'train_loss': [],
            'train_acc': [],
            'val_metrics': []
        }

        best_val_metric = 0
        best_epoch = 0
        patience_counter = 0

        msg = "\nStarting training...\n"
        print(msg)
        self.logger.info(msg)
        overall_start = time.time()

        # Training loop
        for epoch in range(1, self.args.epochs + 1):
            epoch_start = time.time()

            # Train
            train_loss, train_acc = self.train_epoch(
                model, train_loader, criterion, optimizer, epoch
            )

            # Validate
            val_metrics = self.validate(model, val_loader, criterion)

            # Update scheduler
            scheduler.step()

            # Save history
            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            history['val_metrics'].append(val_metrics)

            epoch_time = time.time() - epoch_start

            # Print epoch summary
            epoch_summary = [
                f"\nEpoch {epoch}/{self.args.epochs} - {epoch_time:.2f}s",
                f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}",
                f"  Val Loss: {val_metrics['loss']:.4f}",
                f"  Val Accuracy: {val_metrics['accuracy']:.4f}",
                f"  Val Balanced Acc: {val_metrics['balanced_accuracy']:.4f}",
                f"  Val F1 (weighted): {val_metrics['f1_weighted']:.4f}",
                f"  Val F1 (macro): {val_metrics['f1_macro']:.4f}",
                f"  Val AUC (macro): {val_metrics['auc_macro']:.4f}",
                f"  LR: {optimizer.param_groups[0]['lr']:.6f}"
            ]

            for msg in epoch_summary:
                print(msg)
                self.logger.info(msg)

            # Model selection based on validation metric
            current_metric = val_metrics[self.args.val_metric]

            if current_metric > best_val_metric:
                best_val_metric = current_metric
                best_epoch = epoch
                patience_counter = 0

                # Save best model
                checkpoint_path = self.output_dir / 'best_model.pth'
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_metrics': val_metrics,
                    'config': vars(self.args)
                }, checkpoint_path)
                msg = f"  Saved best model ({self.args.val_metric}: {current_metric:.4f})"
                print(msg)
                self.logger.info(msg)
            else:
                patience_counter += 1
                msg = f"  No improvement (patience: {patience_counter}/{self.args.early_stopping_patience})"
                print(msg)
                self.logger.info(msg)

            # Early stopping
            if patience_counter >= self.args.early_stopping_patience:
                msg = f"\nEarly stopping at epoch {epoch}"
                print(msg)
                self.logger.info(msg)
                break

        # Training complete
        total_time = time.time() - overall_start
        completion_msg = [
            "\n" + "="*80,
            "TRAINING COMPLETE",
            "="*80,
            f"Total time: {total_time/60:.2f} minutes",
            f"Best {self.args.val_metric}: {best_val_metric:.4f} (epoch {best_epoch})",
            f"Average time per epoch: {total_time/len(history['train_loss']):.2f} seconds",
            f"Model saved to: {self.output_dir}"
        ]

        for msg in completion_msg:
            print(msg)
            self.logger.info(msg)

        # Save history
        history_path = self.output_dir / 'training_history.json'
        with open(history_path, 'w') as f:
            json.dump(history, f, indent=2)

        # Save final results
        results = {
            'model': self.args.model,
            'fold': self.args.fold,
            'best_val_metric': best_val_metric,
            'val_metric_name': self.args.val_metric,
            'best_epoch': best_epoch,
            'total_time_minutes': total_time / 60,
            'configuration': vars(self.args)
        }

        results_path = self.output_dir / 'final_results.json'
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)

        msg = f"\nResults saved to: {results_path}"
        print(msg)
        self.logger.info(msg)


def main():
    parser = argparse.ArgumentParser(
        description='Optimized training for combined MILK10k + ISIC2019 dataset'
    )

    # Dataset
    parser.add_argument('--metadata_path', type=str, required=True,
                        help='Path to master_metadata.csv')
    parser.add_argument('--images_root', type=str, required=True,
                        help='Root directory for images')
    parser.add_argument('--class_weights_path', type=str, default=None,
                        help='Path to class_weights.csv')
    parser.add_argument('--fold', type=int, required=True,
                        help='Fold number (0-9)')

    # Model
    parser.add_argument('--model', type=str, default='resnet50',
                        choices=['resnet50', 'efficientnet', 'densenet', 'vit', 'swin'],
                        help='Model architecture')

    # Training parameters
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='Weight decay')
    parser.add_argument('--image_size', type=int, default=224,
                        help='Input image size')

    # Loss function
    parser.add_argument('--use_focal_loss', action='store_true',
                        help='Use Focal Loss instead of CrossEntropyLoss')
    parser.add_argument('--focal_gamma', type=float, default=2.0,
                        help='Focal loss gamma parameter')

    # Sampling
    parser.add_argument('--use_skin_tone_sampling', action='store_true',
                        help='Use skin tone-aware sampling')
    parser.add_argument('--samples_per_epoch', type=int, default=None,
                        help='Samples per epoch (for skin tone sampling)')

    # Validation and early stopping
    parser.add_argument('--val_metric', type=str, default='balanced_accuracy',
                        choices=['accuracy', 'balanced_accuracy', 'f1_weighted', 'f1_macro', 'auc_macro'],
                        help='Metric for model selection')
    parser.add_argument('--early_stopping_patience', type=int, default=10,
                        help='Early stopping patience')

    # Optimization
    parser.add_argument('--num_workers', type=int, default=8,
                        help='Number of DataLoader workers')
    parser.add_argument('--use_amp', action='store_true', default=True,
                        help='Use mixed precision training')
    parser.add_argument('--no_amp', action='store_false', dest='use_amp',
                        help='Disable mixed precision')
    parser.add_argument('--accumulation_steps', type=int, default=1,
                        help='Gradient accumulation steps')
    parser.add_argument('--prefetch_factor', type=int, default=2,
                        help='Number of batches to prefetch')

    # Output
    parser.add_argument('--output_dir', type=str, default='models',
                        help='Output directory')
    parser.add_argument('--random_seed', type=int, default=42,
                        help='Random seed')

    args = parser.parse_args()

    # Validate fold number (must be 0-9 for 10-fold CV)
    if args.fold < 0 or args.fold > 9:
        print(f"ERROR: Invalid fold number {args.fold}. Must be 0-9 for 10-fold cross-validation.")
        print("Usage: --fold 0  (valid values: 0, 1, 2, 3, 4, 5, 6, 7, 8, 9)")
        return

    # Set random seed
    torch.manual_seed(args.random_seed)
    np.random.seed(args.random_seed)

    # Create trainer
    trainer = OptimizedTrainer(args)

    # Train
    trainer.train()


if __name__ == '__main__':
    main()
