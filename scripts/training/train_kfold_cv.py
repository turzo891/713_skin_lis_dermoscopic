#!/usr/bin/env python3
"""
10-Fold Stratified Cross-Validation Training Script

For robust evaluation of multi-modal skin cancer classification.
Implements stratified K-fold to ensure balanced class distribution in each fold.
"""

import sys
sys.path.insert(0, 'src')

import os
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, roc_auc_score
from tqdm import tqdm
import argparse
from datetime import datetime

from multimodal_dataloader import MultiModalSkinDataset
from models import MultiModalFusionNet
from torchvision import transforms


class KFoldCrossValidator:
    """10-Fold Stratified Cross-Validation for Multi-Modal Model"""

    def __init__(
        self,
        data_path: str,
        csv_path: str,
        metadata_path: str,
        model_config: dict,
        train_config: dict,
        n_folds: int = 10,
        save_dir: str = 'kfold_results',
        device: str = 'cuda'
    ):
        self.data_path = data_path
        self.csv_path = csv_path
        self.metadata_path = metadata_path
        self.model_config = model_config
        self.train_config = train_config
        self.n_folds = n_folds
        self.save_dir = save_dir
        self.device = device

        os.makedirs(save_dir, exist_ok=True)

        # Store results across all folds
        self.fold_results = []

    def create_fold_datasets(self, fold_idx, train_indices, val_indices):
        """Create train and validation datasets for a specific fold."""

        # Transformations
        train_transform = transforms.Compose([
            transforms.Resize((384, 384)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(20),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        val_transform = transforms.Compose([
            transforms.Resize((384, 384)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # Create full dataset
        full_dataset = MultiModalSkinDataset(
            csv_file=self.csv_path,
            metadata_file=self.metadata_path,
            img_dir=self.data_path,
            transform=None,
            train=True
        )

        # Create train dataset for this fold
        train_dataset = MultiModalSkinDataset(
            csv_file=self.csv_path,
            metadata_file=self.metadata_path,
            img_dir=self.data_path,
            transform=train_transform,
            train=True
        )
        train_dataset.data = train_dataset.data.iloc[train_indices].reset_index(drop=True)

        # Create validation dataset for this fold
        val_dataset = MultiModalSkinDataset(
            csv_file=self.csv_path,
            metadata_file=self.metadata_path,
            img_dir=self.data_path,
            transform=val_transform,
            train=False
        )
        val_dataset.data = val_dataset.data.iloc[val_indices].reset_index(drop=True)
        val_dataset.age_mean = train_dataset.age_mean
        val_dataset.age_std = train_dataset.age_std
        val_dataset.sex_encoder = train_dataset.sex_encoder
        val_dataset.location_encoder = train_dataset.location_encoder
        val_dataset.num_locations = train_dataset.num_locations
        val_dataset.apply_encoding()

        return train_dataset, val_dataset

    def train_one_epoch(self, model, train_loader, criterion, optimizer, epoch):
        """Train for one epoch."""
        model.train()
        total_loss = 0
        all_preds = []
        all_labels = []

        pbar = tqdm(train_loader, desc=f'Training Epoch {epoch}')
        for images, metadata, labels in pbar:
            images = images.to(self.device)
            metadata = metadata.to(self.device)
            labels = labels.to(self.device)

            optimizer.zero_grad()
            outputs = model(images, metadata)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            preds = outputs.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        avg_loss = total_loss / len(train_loader)
        acc = accuracy_score(all_labels, all_preds)

        return avg_loss, acc

    def validate(self, model, val_loader, criterion):
        """Validate the model."""
        model.eval()
        total_loss = 0
        all_preds = []
        all_labels = []
        all_probs = []

        with torch.no_grad():
            for images, metadata, labels in tqdm(val_loader, desc='Validating'):
                images = images.to(self.device)
                metadata = metadata.to(self.device)
                labels = labels.to(self.device)

                outputs = model(images, metadata)
                loss = criterion(outputs, labels)

                total_loss += loss.item()
                probs = torch.softmax(outputs, dim=1)
                preds = outputs.argmax(dim=1)

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())

        avg_loss = total_loss / len(val_loader)

        # Calculate metrics
        acc = accuracy_score(all_labels, all_preds)
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, all_preds, average='weighted', zero_division=0
        )

        # Per-class metrics
        precision_per_class, recall_per_class, f1_per_class, _ = precision_recall_fscore_support(
            all_labels, all_preds, average=None, zero_division=0
        )

        # Confusion matrix
        cm = confusion_matrix(all_labels, all_preds)

        metrics = {
            'loss': avg_loss,
            'accuracy': acc,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'precision_per_class': precision_per_class.tolist(),
            'recall_per_class': recall_per_class.tolist(),
            'f1_per_class': f1_per_class.tolist(),
            'confusion_matrix': cm.tolist()
        }

        return metrics

    def train_fold(self, fold_idx, train_dataset, val_dataset):
        """Train model for one fold."""
        print(f"\n{'='*80}")
        print(f"FOLD {fold_idx + 1}/{self.n_folds}")
        print(f"{'='*80}")
        print(f"Train samples: {len(train_dataset)}")
        print(f"Val samples: {len(val_dataset)}")

        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.train_config['batch_size'],
            shuffle=True,
            num_workers=self.train_config.get('num_workers', 4),
            pin_memory=True
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=self.train_config['batch_size'],
            shuffle=False,
            num_workers=self.train_config.get('num_workers', 4),
            pin_memory=True
        )

        # Create model
        model = MultiModalFusionNet(**self.model_config).to(self.device)

        # Loss and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(
            model.parameters(),
            lr=self.train_config['learning_rate'],
            weight_decay=self.train_config.get('weight_decay', 1e-4)
        )

        # Learning rate scheduler
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.train_config['epochs']
        )

        best_val_acc = 0
        best_metrics = None
        patience_counter = 0

        for epoch in range(1, self.train_config['epochs'] + 1):
            # Train
            train_loss, train_acc = self.train_one_epoch(
                model, train_loader, criterion, optimizer, epoch
            )

            # Validate
            val_metrics = self.validate(model, val_loader, criterion)

            print(f"\nEpoch {epoch}/{self.train_config['epochs']}")
            print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
            print(f"  Val Loss: {val_metrics['loss']:.4f}, Val Acc: {val_metrics['accuracy']:.4f}")
            print(f"  Val F1: {val_metrics['f1']:.4f}, Val Recall: {val_metrics['recall']:.4f}")

            # Save best model
            if val_metrics['accuracy'] > best_val_acc:
                best_val_acc = val_metrics['accuracy']
                best_metrics = val_metrics
                patience_counter = 0

                # Save model checkpoint
                checkpoint_path = os.path.join(
                    self.save_dir,
                    f'fold_{fold_idx + 1}_best_model.pth'
                )
                torch.save({
                    'fold': fold_idx,
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_metrics': val_metrics,
                }, checkpoint_path)
            else:
                patience_counter += 1

            # Early stopping
            if patience_counter >= self.train_config.get('patience', 10):
                print(f"\nEarly stopping at epoch {epoch}")
                break

            scheduler.step()

        return best_metrics

    def run_cross_validation(self):
        """Run complete K-fold cross-validation."""
        print(f"\n{'='*80}")
        print(f"STARTING {self.n_folds}-FOLD STRATIFIED CROSS-VALIDATION")
        print(f"{'='*80}\n")

        # Load full dataset to get labels
        full_dataset = MultiModalSkinDataset(
            csv_file=self.csv_path,
            metadata_file=self.metadata_path,
            img_dir=self.data_path,
            transform=None,
            train=True
        )

        # Get labels for stratification
        labels = full_dataset.data['label'].values
        indices = np.arange(len(labels))

        # Stratified K-Fold
        skf = StratifiedKFold(n_splits=self.n_folds, shuffle=True, random_state=42)

        for fold_idx, (train_idx, val_idx) in enumerate(skf.split(indices, labels)):
            # Create datasets for this fold
            train_dataset, val_dataset = self.create_fold_datasets(
                fold_idx, train_idx, val_idx
            )

            # Train this fold
            fold_metrics = self.train_fold(fold_idx, train_dataset, val_dataset)

            # Store results
            self.fold_results.append({
                'fold': fold_idx + 1,
                'train_size': len(train_dataset),
                'val_size': len(val_dataset),
                **fold_metrics
            })

        # Aggregate results
        self.save_results()
        self.print_summary()

    def save_results(self):
        """Save cross-validation results."""
        results_path = os.path.join(self.save_dir, 'kfold_cv_results.json')

        # Calculate aggregate statistics
        accuracies = [fold['accuracy'] for fold in self.fold_results]
        f1_scores = [fold['f1'] for fold in self.fold_results]
        precisions = [fold['precision'] for fold in self.fold_results]
        recalls = [fold['recall'] for fold in self.fold_results]

        aggregate_results = {
            'n_folds': self.n_folds,
            'fold_results': self.fold_results,
            'aggregate_metrics': {
                'accuracy_mean': np.mean(accuracies),
                'accuracy_std': np.std(accuracies),
                'f1_mean': np.mean(f1_scores),
                'f1_std': np.std(f1_scores),
                'precision_mean': np.mean(precisions),
                'precision_std': np.std(precisions),
                'recall_mean': np.mean(recalls),
                'recall_std': np.std(recalls),
            },
            'model_config': self.model_config,
            'train_config': self.train_config,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }

        with open(results_path, 'w') as f:
            json.dump(aggregate_results, f, indent=2)

        print(f"\n Results saved to: {results_path}")

    def print_summary(self):
        """Print cross-validation summary."""
        print(f"\n{'='*80}")
        print(f"CROSS-VALIDATION SUMMARY ({self.n_folds} FOLDS)")
        print(f"{'='*80}\n")

        accuracies = [fold['accuracy'] for fold in self.fold_results]
        f1_scores = [fold['f1'] for fold in self.fold_results]
        precisions = [fold['precision'] for fold in self.fold_results]
        recalls = [fold['recall'] for fold in self.fold_results]

        print(f"Accuracy:  {np.mean(accuracies):.4f} ± {np.std(accuracies):.4f}")
        print(f"F1 Score:  {np.mean(f1_scores):.4f} ± {np.std(f1_scores):.4f}")
        print(f"Precision: {np.mean(precisions):.4f} ± {np.std(precisions):.4f}")
        print(f"Recall:    {np.mean(recalls):.4f} ± {np.std(recalls):.4f}")

        print(f"\n{'='*80}")
        print("Per-Fold Results:")
        print(f"{'='*80}")
        print(f"{'Fold':<6} {'Accuracy':<10} {'F1':<10} {'Precision':<12} {'Recall':<10}")
        print(f"{'-'*80}")

        for fold in self.fold_results:
            print(f"{fold['fold']:<6} {fold['accuracy']:<10.4f} {fold['f1']:<10.4f} "
                  f"{fold['precision']:<12.4f} {fold['recall']:<10.4f}")

        print(f"{'='*80}\n")


def main():
    parser = argparse.ArgumentParser(description='10-Fold Cross-Validation Training')
    parser.add_argument('--data_path', type=str,
                        default='data/ISIC2019/ISIC_2019_Training_Input/ISIC_2019_Training_Input',
                        help='Path to image directory')
    parser.add_argument('--csv_path', type=str,
                        default='data/ISIC2019/ISIC_2019_Training_GroundTruth.csv',
                        help='Path to ground truth CSV')
    parser.add_argument('--metadata_path', type=str,
                        default='data/ISIC2019/ISIC_2019_Training_Metadata.csv',
                        help='Path to metadata CSV')
    parser.add_argument('--n_folds', type=int, default=10,
                        help='Number of folds')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of epochs per fold')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--image_encoder', type=str, default='efficientnet',
                        choices=['hybrid', 'efficientnet', 'resnet50'],
                        help='Image encoder type')
    parser.add_argument('--use_cross_attention', action='store_true',
                        help='Use cross-modal attention')
    parser.add_argument('--use_concept_bottleneck', action='store_true',
                        help='Use concept bottleneck layer')
    parser.add_argument('--save_dir', type=str, default='kfold_results',
                        help='Directory to save results')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (cuda/cpu)')

    args = parser.parse_args()

    # Model configuration
    model_config = {
        'num_classes': 8,
        'metadata_dim': 11,
        'image_encoder': args.image_encoder,
        'pretrained': True,
        'dropout': 0.3,
        'use_cross_attention': args.use_cross_attention,
        'use_concept_bottleneck': args.use_concept_bottleneck,
    }

    # Training configuration
    train_config = {
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'learning_rate': args.lr,
        'weight_decay': 1e-4,
        'patience': 10,
        'num_workers': 4
    }

    # Create cross-validator
    cv = KFoldCrossValidator(
        data_path=args.data_path,
        csv_path=args.csv_path,
        metadata_path=args.metadata_path,
        model_config=model_config,
        train_config=train_config,
        n_folds=args.n_folds,
        save_dir=args.save_dir,
        device=args.device
    )

    # Run cross-validation
    cv.run_cross_validation()


if __name__ == '__main__':
    main()
