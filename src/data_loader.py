"""
Data loading and preprocessing module for skin cancer classification.
"""

import os
from typing import Dict, List, Optional, Tuple, Callable
import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.model_selection import train_test_split, StratifiedKFold
import albumentations as A
from albumentations.pytorch import ToTensorV2

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class SkinLesionDataset(Dataset):
    """PyTorch Dataset for skin lesion images."""

    def __init__(
        self,
        image_paths: List[str],
        labels: List[int],
        transform: Optional[Callable] = None,
        return_path: bool = False
    ):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        self.return_path = return_path

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        image_path = self.image_paths[idx]
        label = self.labels[idx]

        image = Image.open(image_path).convert('RGB')
        image = np.array(image)

        if self.transform:
            transformed = self.transform(image=image)
            image = transformed['image']

        if self.return_path:
            return image, label, image_path
        return image, label


def get_train_transforms(image_size: int = 224) -> A.Compose:
    """Get training data augmentation pipeline."""
    return A.Compose([
        A.Resize(image_size, image_size),
        A.RandomRotate90(p=0.5),
        A.Flip(p=0.5),
        A.Transpose(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.15, rotate_limit=45, p=0.5),
        A.OneOf([
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=1.0),
            A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=1.0),
        ], p=0.5),
        A.OneOf([
            A.GaussNoise(var_limit=(10.0, 50.0), p=1.0),
            A.GaussianBlur(blur_limit=(3, 7), p=1.0),
        ], p=0.3),
        A.CoarseDropout(max_holes=8, max_height=image_size // 8, max_width=image_size // 8, fill_value=0, p=0.3),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2()
    ])


def get_val_transforms(image_size: int = 224) -> A.Compose:
    """Get validation/test data transform pipeline."""
    return A.Compose([
        A.Resize(image_size, image_size),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2()
    ])


def load_ham10000_data(
    data_dir: str,
    csv_path: Optional[str] = None
) -> Tuple[List[str], List[int], Dict[str, int]]:
    """Load HAM10000 dataset metadata."""
    label_encoder = {'akiec': 0, 'bcc': 1, 'bkl': 2, 'df': 3, 'mel': 4, 'nv': 5, 'vasc': 6}

    if csv_path and os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        image_paths = []
        labels = []

        for _, row in df.iterrows():
            image_id = row['image_id']
            label = label_encoder[row['dx']]

            possible_paths = [
                os.path.join(data_dir, f"{image_id}.jpg"),
                os.path.join(data_dir, 'HAM10000_images_part_1', f"{image_id}.jpg"),
                os.path.join(data_dir, 'HAM10000_images_part_2', f"{image_id}.jpg"),
            ]

            for path in possible_paths:
                if os.path.exists(path):
                    image_paths.append(path)
                    labels.append(label)
                    break
    else:
        image_paths = []
        labels = []
        for class_name, class_idx in label_encoder.items():
            class_dir = os.path.join(data_dir, class_name)
            if os.path.exists(class_dir):
                for img_name in os.listdir(class_dir):
                    if img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                        image_paths.append(os.path.join(class_dir, img_name))
                        labels.append(class_idx)

    return image_paths, labels, label_encoder


def get_class_weights(labels: List[int], num_classes: int = 7) -> torch.Tensor:
    """Calculate class weights for handling imbalanced data."""
    class_counts = np.bincount(labels, minlength=num_classes)
    total_samples = len(labels)
    class_weights = total_samples / (num_classes * class_counts + 1e-6)
    return torch.FloatTensor(class_weights)


def get_sample_weights(labels: List[int], num_classes: int = 7) -> List[float]:
    """Calculate sample weights for WeightedRandomSampler."""
    class_weights = get_class_weights(labels, num_classes)
    return [class_weights[label].item() for label in labels]


def create_data_loaders(
    data_dir: str,
    csv_path: Optional[str] = None,
    batch_size: int = 32,
    image_size: int = 224,
    num_workers: int = 4,
    val_split: float = 0.15,
    test_split: float = 0.15,
    use_weighted_sampler: bool = True,
    seed: int = 42
) -> Tuple[DataLoader, DataLoader, DataLoader, Dict[str, int]]:
    """Create train, validation, and test data loaders."""
    image_paths, labels, label_encoder = load_ham10000_data(data_dir, csv_path)

    train_val_paths, test_paths, train_val_labels, test_labels = train_test_split(
        image_paths, labels, test_size=test_split, stratify=labels, random_state=seed
    )

    val_ratio = val_split / (1 - test_split)
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        train_val_paths, train_val_labels, test_size=val_ratio, stratify=train_val_labels, random_state=seed
    )

    train_transform = get_train_transforms(image_size)
    val_transform = get_val_transforms(image_size)

    train_dataset = SkinLesionDataset(train_paths, train_labels, train_transform)
    val_dataset = SkinLesionDataset(val_paths, val_labels, val_transform)
    test_dataset = SkinLesionDataset(test_paths, test_labels, val_transform)

    if use_weighted_sampler:
        sample_weights = get_sample_weights(train_labels)
        sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(train_labels), replacement=True)
        shuffle = False
    else:
        sampler = None
        shuffle = True

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, sampler=sampler,
                              num_workers=num_workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=True)

    print(f"Dataset split: Train={len(train_dataset)}, Val={len(val_dataset)}, Test={len(test_dataset)}")
    return train_loader, val_loader, test_loader, label_encoder


def create_kfold_loaders(
    data_dir: str,
    csv_path: Optional[str] = None,
    n_folds: int = 5,
    batch_size: int = 32,
    image_size: int = 224,
    num_workers: int = 4,
    use_weighted_sampler: bool = True,
    seed: int = 42
) -> List[Tuple[DataLoader, DataLoader]]:
    """Create k-fold cross-validation data loaders."""
    image_paths, labels, _ = load_ham10000_data(data_dir, csv_path)
    image_paths = np.array(image_paths)
    labels = np.array(labels)

    train_transform = get_train_transforms(image_size)
    val_transform = get_val_transforms(image_size)

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    fold_loaders = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(image_paths, labels)):
        train_paths = image_paths[train_idx].tolist()
        train_labels = labels[train_idx].tolist()
        val_paths = image_paths[val_idx].tolist()
        val_labels = labels[val_idx].tolist()

        train_dataset = SkinLesionDataset(train_paths, train_labels, train_transform)
        val_dataset = SkinLesionDataset(val_paths, val_labels, val_transform)

        if use_weighted_sampler:
            sample_weights = get_sample_weights(train_labels)
            sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(train_labels), replacement=True)
            shuffle = False
        else:
            sampler = None
            shuffle = True

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, sampler=sampler,
                                  num_workers=num_workers, pin_memory=True, drop_last=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                                num_workers=num_workers, pin_memory=True)

        fold_loaders.append((train_loader, val_loader))
        print(f"Fold {fold + 1}: Train={len(train_dataset)}, Val={len(val_dataset)}")

    return fold_loaders
