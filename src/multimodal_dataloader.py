"""
Multi-Modal Data Loader for Skin Cancer Classification
Combines dermoscopic images with clinical metadata (age, sex, location)
"""

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import os
from typing import Tuple, Dict, Optional
from sklearn.preprocessing import LabelEncoder, StandardScaler


class MultiModalSkinDataset(Dataset):
    """
    Dataset that provides both images and clinical metadata.

    Metadata features:
    - age_approx: Patient age (normalized)
    - sex: Male/Female (encoded)
    - anatom_site_general: Body location (encoded)
    """

    def __init__(
        self,
        csv_file: str,
        metadata_file: str,
        img_dir: str,
        transform=None,
        train=True
    ):
        """
        Args:
            csv_file: Path to ground truth CSV
            metadata_file: Path to metadata CSV
            img_dir: Directory with images
            transform: Image transformations
            train: Whether this is training set (for fitting encoders)
        """
        self.img_dir = img_dir
        self.transform = transform
        self.train = train

        # Load labels
        self.labels_df = pd.read_csv(csv_file)
        self.labels_df['label'] = self.labels_df.iloc[:, 1:].values.argmax(axis=1)

        # Load metadata
        self.metadata_df = pd.read_csv(metadata_file)

        # Merge labels and metadata
        self.data = pd.merge(
            self.labels_df[['image', 'label']],
            self.metadata_df,
            on='image',
            how='left'
        )

        print(f"\n{'='*60}")
        print(f"Multi-Modal Dataset Initialized")
        print(f"{'='*60}")
        print(f"Total samples: {len(self.data)}")
        print(f"\nMetadata coverage:")
        print(f"  Age available: {(~self.data['age_approx'].isna()).sum()} / {len(self.data)} ({(~self.data['age_approx'].isna()).sum()/len(self.data)*100:.1f}%)")
        print(f"  Sex available: {(~self.data['sex'].isna()).sum()} / {len(self.data)} ({(~self.data['sex'].isna()).sum()/len(self.data)*100:.1f}%)")
        print(f"  Location available: {(~self.data['anatom_site_general'].isna()).sum()} / {len(self.data)} ({(~self.data['anatom_site_general'].isna()).sum()/len(self.data)*100:.1f}%)")

        # Prepare metadata encoders (only for training set initially)
        if self.train:
            self._prepare_metadata_encoding()

    def _prepare_metadata_encoding(self):
        """Prepare encoders and handle missing values."""

        # Age: Impute missing with median, normalize
        age_median = self.data['age_approx'].median()
        self.data['age_approx'] = self.data['age_approx'].fillna(age_median)

        if self.train:
            self.age_mean = self.data['age_approx'].mean()
            self.age_std = self.data['age_approx'].std()
        # Normalize age
        self.data['age_normalized'] = (self.data['age_approx'] - self.age_mean) / self.age_std

        # Sex: Encode as binary
        self.data['sex'] = self.data['sex'].fillna('unknown')
        if self.train:
            self.sex_encoder = LabelEncoder()
            self.data['sex_encoded'] = self.sex_encoder.fit_transform(self.data['sex'])
        else:
            self.data['sex_encoded'] = self.sex_encoder.transform(self.data['sex'])

        # Anatomical location: One-hot encoding
        self.data['anatom_site_general'] = self.data['anatom_site_general'].fillna('unknown')
        if self.train:
            self.location_encoder = LabelEncoder()
            self.data['location_encoded'] = self.location_encoder.fit_transform(self.data['anatom_site_general'])
            self.num_locations = len(self.location_encoder.classes_)
        else:
            self.data['location_encoded'] = self.location_encoder.transform(self.data['anatom_site_general'])

        print(f"\nMetadata encoding:")
        print(f"  Age range: {self.data['age_approx'].min():.0f} - {self.data['age_approx'].max():.0f} years")
        print(f"  Sex classes: {self.sex_encoder.classes_ if self.train else 'using pre-fitted'}")
        print(f"  Location classes ({self.num_locations if self.train else 'N/A'}): {self.location_encoder.classes_ if self.train else 'using pre-fitted'}")
        print(f"{'='*60}\n")

    def apply_encoding(self):
        """Apply pre-fitted encoders to the data (for val/test sets)."""
        # Age: Impute missing with median, normalize
        age_median = self.data['age_approx'].median()
        self.data['age_approx'] = self.data['age_approx'].fillna(age_median)
        self.data['age_normalized'] = (self.data['age_approx'] - self.age_mean) / self.age_std

        # Sex: Encode as binary
        self.data['sex'] = self.data['sex'].fillna('unknown')
        self.data['sex_encoded'] = self.sex_encoder.transform(self.data['sex'])

        # Anatomical location: One-hot encoding
        self.data['anatom_site_general'] = self.data['anatom_site_general'].fillna('unknown')
        self.data['location_encoded'] = self.location_encoder.transform(self.data['anatom_site_general'])

    def get_metadata_dim(self) -> int:
        """Return total dimension of metadata features."""
        # age (1) + sex (1) + location (one-hot)
        return 1 + 1 + (self.num_locations if hasattr(self, 'num_locations') else 0)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor, int]:
        """
        Returns:
            image: Transformed image tensor
            metadata: Clinical metadata tensor
            label: Class label
        """
        row = self.data.iloc[idx]

        # Load image
        img_name = row['image'] + '.jpg'
        img_path = os.path.join(self.img_dir, img_name)
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        # Prepare metadata tensor
        metadata_features = [
            row['age_normalized'],  # Normalized age
            row['sex_encoded'],     # Encoded sex
        ]

        # Add one-hot encoded location
        location_onehot = np.zeros(self.num_locations if hasattr(self, 'num_locations') else len(self.location_encoder.classes_))
        location_onehot[int(row['location_encoded'])] = 1.0
        metadata_features.extend(location_onehot)

        metadata = torch.FloatTensor(metadata_features)

        label = int(row['label'])

        return image, metadata, label


def create_multimodal_data_loaders(
    data_path: str,
    csv_path: str,
    metadata_path: str,
    batch_size: int = 32,
    image_size: int = 384,
    num_workers: int = 4
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader, torch.utils.data.DataLoader, Dict]:
    """
    Create multi-modal data loaders with clinical metadata.

    Args:
        data_path: Path to image directory
        csv_path: Path to ground truth CSV
        metadata_path: Path to metadata CSV
        batch_size: Batch size
        image_size: Image size for resize
        num_workers: Number of workers for data loading

    Returns:
        train_loader, val_loader, test_loader, label_encoder
    """
    from torchvision import transforms
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
    from sklearn.model_selection import train_test_split

    # Transformations
    train_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(20),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Create full dataset
    full_dataset = MultiModalSkinDataset(
        csv_file=csv_path,
        metadata_file=metadata_path,
        img_dir=data_path,
        transform=None,
        train=True
    )

    # Split into train/val/test (70/15/15)
    indices = list(range(len(full_dataset)))
    labels = full_dataset.data['label'].values

    train_idx, temp_idx = train_test_split(
        indices, test_size=0.3, stratify=labels, random_state=42
    )
    val_idx, test_idx = train_test_split(
        temp_idx, test_size=0.5, stratify=labels[temp_idx], random_state=42
    )

    # Create datasets with proper transforms
    train_dataset = MultiModalSkinDataset(
        csv_file=csv_path,
        metadata_file=metadata_path,
        img_dir=data_path,
        transform=train_transform,
        train=True
    )
    train_dataset.data = train_dataset.data.iloc[train_idx].reset_index(drop=True)

    val_dataset = MultiModalSkinDataset(
        csv_file=csv_path,
        metadata_file=metadata_path,
        img_dir=data_path,
        transform=val_transform,
        train=False
    )
    val_dataset.data = val_dataset.data.iloc[val_idx].reset_index(drop=True)
    val_dataset.age_mean = train_dataset.age_mean
    val_dataset.age_std = train_dataset.age_std
    val_dataset.sex_encoder = train_dataset.sex_encoder
    val_dataset.location_encoder = train_dataset.location_encoder
    val_dataset.num_locations = train_dataset.num_locations
    val_dataset.apply_encoding()  # Apply encoding with fitted encoders

    test_dataset = MultiModalSkinDataset(
        csv_file=csv_path,
        metadata_file=metadata_path,
        img_dir=data_path,
        transform=val_transform,
        train=False
    )
    test_dataset.data = test_dataset.data.iloc[test_idx].reset_index(drop=True)
    test_dataset.age_mean = train_dataset.age_mean
    test_dataset.age_std = train_dataset.age_std
    test_dataset.sex_encoder = train_dataset.sex_encoder
    test_dataset.location_encoder = train_dataset.location_encoder
    test_dataset.num_locations = train_dataset.num_locations
    test_dataset.apply_encoding()  # Apply encoding with fitted encoders

    # Create data loaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    # Create label encoder (class names)
    class_columns = ['MEL', 'NV', 'BCC', 'AK', 'BKL', 'DF', 'VASC', 'SCC']
    label_encoder = {i: name for i, name in enumerate(class_columns)}

    print(f"\nData Loaders Created:")
    print(f"  Train: {len(train_loader.dataset)} samples")
    print(f"  Val:   {len(val_loader.dataset)} samples")
    print(f"  Test:  {len(test_loader.dataset)} samples")
    print(f"  Metadata dim: {train_dataset.get_metadata_dim()}")
    print(f"  Classes: {len(label_encoder)}")

    return train_loader, val_loader, test_loader, label_encoder, train_dataset.get_metadata_dim()
