#!/usr/bin/env python3
"""
Optimized Training Script - Maximum CPU + GPU + RAM Usage

This script implements all optimization strategies to maximize resource utilization:
1. Mixed precision training (FP16)
2. Pinned memory
3. Parallel data loading
4. Gradient accumulation
5. Prefetching

Expected speedup: 2-3x faster than standard training
"""

import sys
sys.path.insert(0, 'src')

import os
import time
import argparse
from datetime import datetime
import json

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm

from data_loader import create_data_loaders
from models import get_model


class OptimizedTrainer:
    """Optimized trainer using CPU, GPU, and RAM efficiently."""

    def __init__(
        self,
        model_name: str = 'resnet50',
        dataset: str = 'ISIC2019',
        batch_size: int = 32,
        epochs: int = 50,
        learning_rate: float = 1e-4,
        num_workers: int = 8,
        use_amp: bool = True,
        accumulation_steps: int = 1,
        prefetch_factor: int = 2,
        device: str = 'cuda'
    ):
        self.model_name = model_name
        self.dataset = dataset
        self.batch_size = batch_size
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.num_workers = num_workers
        self.use_amp = use_amp
        self.accumulation_steps = accumulation_steps
        self.prefetch_factor = prefetch_factor
        self.device = device

        # Auto-tune parameters based on hardware
        self.auto_tune_parameters()

        # Create output directory
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.output_dir = f'models/{model_name}_optimized_{timestamp}'
        os.makedirs(self.output_dir, exist_ok=True)

        # Initialize mixed precision scaler
        self.scaler = GradScaler() if use_amp else None

        print("="*80)
        print("OPTIMIZED TRAINING CONFIGURATION")
        print("="*80)
        print(f"Model: {model_name}")
        print(f"Batch size: {batch_size}")
        print(f"Effective batch size: {batch_size * accumulation_steps}")
        print(f"Workers: {num_workers}")
        print(f"Mixed precision: {use_amp}")
        print(f"Prefetch factor: {prefetch_factor}")
        print(f"Device: {device}")
        print(f"Output: {self.output_dir}")
        print("="*80)

    def auto_tune_parameters(self):
        """Auto-tune parameters based on available hardware."""
        if torch.cuda.is_available():
            # Get GPU info
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"\nDetected GPU: {torch.cuda.get_device_name(0)} ({gpu_memory:.1f} GB)")

            # Auto-tune batch size based on GPU memory
            if self.use_amp:
                if gpu_memory >= 20:  # RTX 3090, A100
                    recommended_batch = 64
                elif gpu_memory >= 10:  # RTX 3080
                    recommended_batch = 32
                else:  # RTX 3070, smaller
                    recommended_batch = 16
            else:
                if gpu_memory >= 20:
                    recommended_batch = 32
                elif gpu_memory >= 10:
                    recommended_batch = 16
                else:
                    recommended_batch = 8

            if self.batch_size < recommended_batch:
                print(f"Recommendation: Increase batch_size to {recommended_batch} (you have {gpu_memory:.1f} GB VRAM)")

        # Auto-tune num_workers based on CPU cores
        import multiprocessing
        cpu_count = multiprocessing.cpu_count()
        if self.num_workers > cpu_count:
            print(f"Warning: num_workers ({self.num_workers}) > CPU cores ({cpu_count})")
            print(f"Setting num_workers = {cpu_count}")
            self.num_workers = cpu_count

    def create_data_loaders(self):
        """Create optimized data loaders."""
        print("\nCreating optimized data loaders...")

        # Dataset paths
        if self.dataset == 'ISIC2019':
            data_dir = 'data/ISIC2019/ISIC_2019_Training_Input/ISIC_2019_Training_Input'
            csv_path = 'data/ISIC2019/ISIC_2019_Training_GroundTruth.csv'
        else:
            raise ValueError(f"Unknown dataset: {self.dataset}")

        # Create base loaders
        train_loader, val_loader, test_loader, label_encoder = create_data_loaders(
            data_dir=data_dir,
            csv_path=csv_path,
            batch_size=self.batch_size,
            image_size=224,
            num_workers=0,  # We'll recreate with optimized settings
            val_split=0.15,
            test_split=0.15
        )

        # Get datasets from loaders
        train_dataset = train_loader.dataset
        val_dataset = val_loader.dataset
        test_dataset = test_loader.dataset

        # Create optimized loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,              # Fast CPU→GPU transfer
            prefetch_factor=self.prefetch_factor,  # Preload batches
            persistent_workers=True       # Keep workers alive
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            prefetch_factor=self.prefetch_factor,
            persistent_workers=True
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers // 2,  # Less workers for test
            pin_memory=True
        )

        print(f"Train batches: {len(train_loader)}")
        print(f"Val batches: {len(val_loader)}")
        print(f"Test batches: {len(test_loader)}")

        return train_loader, val_loader, test_loader, label_encoder

    def train_epoch(self, model, train_loader, criterion, optimizer, epoch):
        """Train for one epoch with all optimizations."""
        model.train()

        total_loss = 0
        correct = 0
        total = 0

        pbar = tqdm(train_loader, desc=f'Epoch {epoch}/{self.epochs}')

        # Timing
        data_time = 0
        batch_time = 0
        start_time = time.time()

        for batch_idx, (images, labels) in enumerate(pbar):
            data_time += time.time() - start_time

            # Move to GPU (non-blocking for speed)
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            # Mixed precision training
            if self.use_amp:
                with autocast():
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    loss = loss / self.accumulation_steps  # Normalize loss

                # Backward with gradient scaling
                self.scaler.scale(loss).backward()

                # Update weights after accumulation
                if (batch_idx + 1) % self.accumulation_steps == 0:
                    self.scaler.step(optimizer)
                    self.scaler.update()
                    optimizer.zero_grad()
            else:
                # Standard training
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss = loss / self.accumulation_steps

                loss.backward()

                if (batch_idx + 1) % self.accumulation_steps == 0:
                    optimizer.step()
                    optimizer.zero_grad()

            # Statistics
            total_loss += loss.item() * self.accumulation_steps
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            # Update progress bar
            batch_time += time.time() - start_time
            avg_batch_time = batch_time / (batch_idx + 1)
            avg_data_time = data_time / (batch_idx + 1)
            gpu_time = avg_batch_time - avg_data_time

            pbar.set_postfix({
                'loss': f'{total_loss/(batch_idx+1):.4f}',
                'acc': f'{100.*correct/total:.2f}%',
                'data': f'{avg_data_time*1000:.0f}ms',
                'gpu': f'{gpu_time*1000:.0f}ms'
            })

            start_time = time.time()

        avg_loss = total_loss / len(train_loader)
        acc = 100. * correct / total

        return avg_loss, acc

    def validate(self, model, val_loader, criterion):
        """Validate the model."""
        model.eval()

        total_loss = 0
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc='Validating'):
                images = images.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)

                if self.use_amp:
                    with autocast():
                        outputs = model(images)
                        loss = criterion(outputs, labels)
                else:
                    outputs = model(images)
                    loss = criterion(outputs, labels)

                total_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

        avg_loss = total_loss / len(val_loader)
        acc = 100. * correct / total

        return avg_loss, acc

    def train(self):
        """Main training loop."""
        # Create data loaders
        train_loader, val_loader, test_loader, label_encoder = self.create_data_loaders()

        # Create model
        print(f"\nCreating {self.model_name} model...")
        model = get_model(self.model_name, num_classes=8, pretrained=True)
        model = model.to(self.device)

        # Loss and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(model.parameters(), lr=self.learning_rate, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.epochs)

        # Training history
        history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'lr': []
        }

        best_val_acc = 0
        best_epoch = 0

        print("\nStarting optimized training...\n")
        overall_start = time.time()

        # Training loop
        for epoch in range(1, self.epochs + 1):
            epoch_start = time.time()

            # Train
            train_loss, train_acc = self.train_epoch(
                model, train_loader, criterion, optimizer, epoch
            )

            # Validate
            val_loss, val_acc = self.validate(model, val_loader, criterion)

            # Update scheduler
            scheduler.step()

            # Save history
            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)
            history['lr'].append(optimizer.param_groups[0]['lr'])

            epoch_time = time.time() - epoch_start

            # Print epoch summary
            print(f"\nEpoch {epoch}/{self.epochs} - {epoch_time:.2f}s")
            print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
            print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
            print(f"  LR: {optimizer.param_groups[0]['lr']:.6f}")

            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_epoch = epoch
                checkpoint_path = os.path.join(self.output_dir, 'best_model.pth')
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_acc': val_acc,
                    'val_loss': val_loss,
                }, checkpoint_path)
                print(f"  Saved best model (Val Acc: {val_acc:.2f}%)")

        # Training complete
        total_time = time.time() - overall_start
        print("\n" + "="*80)
        print("TRAINING COMPLETE")
        print("="*80)
        print(f"Total time: {total_time/60:.2f} minutes")
        print(f"Best validation accuracy: {best_val_acc:.2f}% (epoch {best_epoch})")
        print(f"Average time per epoch: {total_time/self.epochs:.2f} seconds")
        print(f"Model saved to: {self.output_dir}")

        # Save history
        history_path = os.path.join(self.output_dir, 'training_history.json')
        with open(history_path, 'w') as f:
            json.dump(history, f, indent=2)

        # Save final results
        results = {
            'model': self.model_name,
            'best_val_acc': best_val_acc,
            'best_epoch': best_epoch,
            'total_time_minutes': total_time / 60,
            'configuration': {
                'batch_size': self.batch_size,
                'effective_batch_size': self.batch_size * self.accumulation_steps,
                'num_workers': self.num_workers,
                'use_amp': self.use_amp,
                'prefetch_factor': self.prefetch_factor,
                'accumulation_steps': self.accumulation_steps
            }
        }

        results_path = os.path.join(self.output_dir, 'final_results.json')
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)

        print(f"\nResults saved to: {results_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Optimized training with maximum CPU+GPU+RAM utilization'
    )

    # Model and dataset
    parser.add_argument('--model', type=str, default='resnet50',
                        choices=['resnet50', 'efficientnet', 'densenet', 'vit', 'swin'],
                        help='Model architecture')
    parser.add_argument('--dataset', type=str, default='ISIC2019',
                        help='Dataset name')

    # Training parameters
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate')

    # Optimization parameters
    parser.add_argument('--num_workers', type=int, default=8,
                        help='Number of DataLoader workers (CPU cores)')
    parser.add_argument('--use_amp', action='store_true', default=True,
                        help='Use mixed precision training (FP16)')
    parser.add_argument('--no_amp', action='store_false', dest='use_amp',
                        help='Disable mixed precision training')
    parser.add_argument('--accumulation_steps', type=int, default=1,
                        help='Gradient accumulation steps')
    parser.add_argument('--prefetch_factor', type=int, default=2,
                        help='Number of batches to prefetch')

    # Device
    parser.add_argument('--device', type=str, default='cuda',
                        choices=['cuda', 'cpu'],
                        help='Device to use')

    args = parser.parse_args()

    # Create trainer
    trainer = OptimizedTrainer(
        model_name=args.model,
        dataset=args.dataset,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.lr,
        num_workers=args.num_workers,
        use_amp=args.use_amp,
        accumulation_steps=args.accumulation_steps,
        prefetch_factor=args.prefetch_factor,
        device=args.device
    )

    # Train
    trainer.train()


if __name__ == '__main__':
    main()
