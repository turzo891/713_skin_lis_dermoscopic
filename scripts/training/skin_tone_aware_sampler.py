#!/usr/bin/env python3
"""
Skin Tone-Aware Sampling Strategy

Implements stratified sampling that balances BOTH class and skin tone distribution.

Key Features:
- Ensures minority classes are represented
- Ensures diverse skin tones in each batch
- Prevents skin tone bias during training
- Compatible with PyTorch DataLoader

Usage:
    from skin_tone_aware_sampler import SkinToneAwareDataset, SkinToneAwareSampler

    dataset = SkinToneAwareDataset(metadata_path="data/combined/master_metadata.csv", ...)
    sampler = SkinToneAwareSampler(dataset, samples_per_epoch=10000)
    loader = DataLoader(dataset, batch_size=32, sampler=sampler)
"""

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, Sampler
from pathlib import Path
from PIL import Image
from typing import Iterator, List, Optional, Dict
from collections import Counter
import warnings


class SkinToneAwareDataset(Dataset):
    """
    Dataset with skin tone metadata for stratified sampling.

    Handles both MILK10k (with skin tone) and ISIC2019 (without skin tone).
    """

    def __init__(
        self,
        metadata_path: str,
        images_root: str,
        split: str = "train",
        fold: int = 0,
        transform=None,
        target_classes: Optional[List[str]] = None
    ):
        """
        Args:
            metadata_path: Path to master_metadata.csv
            images_root: Root directory containing milk10k/ and isic2019/
            split: "train", "val", or "test"
            fold: Cross-validation fold (0-9 for 10-fold)
            transform: Image transformations
            target_classes: List of class names (default: 8 classes)
        """
        self.images_root = Path(images_root)
        self.transform = transform

        # Default 8 classes
        if target_classes is None:
            target_classes = ["MEL", "NV", "BCC", "BKL", "AK", "SCC", "VASC", "DF"]

        self.classes = target_classes
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}

        # Load and filter metadata
        df = pd.read_csv(metadata_path)

        if split == "test":
            df = df[df["split"] == "test"]
        elif split == "val":
            df = df[df["fold"] == fold]
        elif split == "train":
            df = df[(df["split"] == "train") & (df["fold"] != fold)]
        else:
            raise ValueError(f"Invalid split: {split}")

        self.metadata = df.reset_index(drop=True)

        # Create stratification groups
        self._create_stratification_groups()

        print(f"\nSkinToneAwareDataset:")
        print(f"  Split: {split}, Fold: {fold}")
        print(f"  Total samples: {len(self.metadata)}")
        print(f"  Stratification groups: {len(self.group_to_idx)}")
        self._print_distribution()

    def _create_stratification_groups(self):
        """Create groups for joint class + skin tone stratification"""

        # Group samples by (class, skin_tone)
        self.group_to_idx = {}  # group_id -> list of sample indices
        self.idx_to_group = {}  # sample_idx -> group_id

        for idx, row in self.metadata.iterrows():
            diagnosis = row["diagnosis"]
            skin_tone = row.get("skin_tone")

            # Handle missing skin tone (ISIC2019 samples)
            if pd.isna(skin_tone):
                # Group ISIC2019 samples by class only
                group_id = f"{diagnosis}_no_tone"
            else:
                # Group MILK10k samples by class + skin tone
                # Bin skin tones: 0-1 (dark), 2-3 (medium), 4-5 (light)
                if skin_tone <= 1:
                    tone_bin = "dark"
                elif skin_tone <= 3:
                    tone_bin = "medium"
                else:
                    tone_bin = "light"

                group_id = f"{diagnosis}_{tone_bin}"

            # Add to groups
            if group_id not in self.group_to_idx:
                self.group_to_idx[group_id] = []

            self.group_to_idx[group_id].append(idx)
            self.idx_to_group[idx] = group_id

    def _print_distribution(self):
        """Print class and skin tone distribution"""

        print("\n  Class distribution:")
        class_counts = self.metadata["diagnosis"].value_counts()
        for cls in self.classes:
            count = class_counts.get(cls, 0)
            pct = 100 * count / len(self.metadata) if len(self.metadata) > 0 else 0
            print(f"    {cls}: {count} ({pct:.1f}%)")

        # Skin tone distribution (MILK10k subset only)
        milk10k_with_tone = self.metadata[
            (self.metadata["dataset_source"] == "MILK10k") &
            (self.metadata["skin_tone"].notna())
        ]

        if len(milk10k_with_tone) > 0:
            print(f"\n  Skin tone distribution (MILK10k, n={len(milk10k_with_tone)}):")
            tone_counts = milk10k_with_tone["skin_tone"].value_counts().sort_index()
            for tone, count in tone_counts.items():
                pct = 100 * count / len(milk10k_with_tone)
                print(f"    Tone {tone}: {count} ({pct:.1f}%)")

        print(f"\n  Stratification groups:")
        print(f"    Total groups: {len(self.group_to_idx)}")
        for group_id, indices in sorted(self.group_to_idx.items(), key=lambda x: len(x[1]), reverse=True)[:10]:
            print(f"      {group_id}: {len(indices)} samples")

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        """Get image, label, and metadata"""

        row = self.metadata.iloc[idx]

        # Load image
        img_path = self.images_root / row["image_path"]
        image = Image.open(img_path).convert("RGB")

        # Apply transforms
        if self.transform:
            image = self.transform(image)

        # Get label
        label = self.class_to_idx[row["diagnosis"]]

        # Metadata
        metadata = {
            "image_id": row["image_id"],
            "dataset_source": row["dataset_source"],
            "skin_tone": row.get("skin_tone"),
            "group_id": self.idx_to_group[idx],
        }

        return image, label, metadata

    def get_group_id(self, idx: int) -> str:
        """Get stratification group ID for sample"""
        return self.idx_to_group[idx]

    def get_group_indices(self, group_id: str) -> List[int]:
        """Get all sample indices in a group"""
        return self.group_to_idx.get(group_id, [])

    def get_all_groups(self) -> List[str]:
        """Get list of all group IDs"""
        return list(self.group_to_idx.keys())


class SkinToneAwareSampler(Sampler):
    """
    Sampler that ensures balanced representation of both classes and skin tones.

    Strategy:
    1. Sample groups (class + skin tone combinations) with equal probability
    2. Within each group, sample uniformly
    3. This ensures minority classes AND minority skin tones are represented

    Benefits:
    - Prevents majority class (NV) domination
    - Prevents majority skin tone (light) domination
    - Improves fairness across skin tones
    """

    def __init__(
        self,
        dataset: SkinToneAwareDataset,
        samples_per_epoch: int,
        replacement: bool = True,
        random_seed: Optional[int] = None
    ):
        """
        Args:
            dataset: SkinToneAwareDataset instance
            samples_per_epoch: Number of samples to draw per epoch
            replacement: Sample with replacement (recommended for imbalanced data)
            random_seed: Random seed for reproducibility
        """
        super().__init__(dataset)

        self.dataset = dataset
        self.samples_per_epoch = samples_per_epoch
        self.replacement = replacement

        # Set random seed
        if random_seed is not None:
            self.generator = torch.Generator()
            self.generator.manual_seed(random_seed)
        else:
            self.generator = None

        # Get all groups
        self.groups = dataset.get_all_groups()
        self.n_groups = len(self.groups)

        print(f"\nSkinToneAwareSampler:")
        print(f"  Samples per epoch: {samples_per_epoch}")
        print(f"  Number of groups: {self.n_groups}")
        print(f"  Sampling with replacement: {replacement}")

    def __iter__(self) -> Iterator[int]:
        """
        Generate indices for one epoch.

        Strategy:
        1. For each sample to draw:
           a. Randomly select a group (equal probability)
           b. Randomly select a sample from that group
        """

        indices = []

        for _ in range(self.samples_per_epoch):
            # Step 1: Randomly select a group
            if self.generator:
                group_idx = torch.randint(
                    0, self.n_groups, (1,), generator=self.generator
                ).item()
            else:
                group_idx = np.random.randint(0, self.n_groups)

            group_id = self.groups[group_idx]

            # Step 2: Randomly select a sample from the group
            group_indices = self.dataset.get_group_indices(group_id)

            if len(group_indices) == 0:
                # Skip empty groups (shouldn't happen)
                continue

            if self.generator:
                sample_idx = group_indices[
                    torch.randint(
                        0, len(group_indices), (1,), generator=self.generator
                    ).item()
                ]
            else:
                sample_idx = group_indices[np.random.randint(0, len(group_indices))]

            indices.append(sample_idx)

        return iter(indices)

    def __len__(self) -> int:
        return self.samples_per_epoch


class StratifiedBatchSampler(Sampler):
    """
    Alternative sampler that creates batches with balanced classes and skin tones.

    Each batch contains roughly equal representation of all classes and skin tones.
    """

    def __init__(
        self,
        dataset: SkinToneAwareDataset,
        batch_size: int,
        batches_per_epoch: int,
        random_seed: Optional[int] = None
    ):
        """
        Args:
            dataset: SkinToneAwareDataset instance
            batch_size: Batch size
            batches_per_epoch: Number of batches per epoch
            random_seed: Random seed
        """
        super().__init__(dataset)

        self.dataset = dataset
        self.batch_size = batch_size
        self.batches_per_epoch = batches_per_epoch

        if random_seed is not None:
            self.generator = torch.Generator()
            self.generator.manual_seed(random_seed)
        else:
            self.generator = None

        self.groups = dataset.get_all_groups()
        self.n_groups = len(self.groups)

        print(f"\nStratifiedBatchSampler:")
        print(f"  Batch size: {batch_size}")
        print(f"  Batches per epoch: {batches_per_epoch}")
        print(f"  Samples per epoch: {batch_size * batches_per_epoch}")

    def __iter__(self) -> Iterator[List[int]]:
        """Generate batches"""

        for _ in range(self.batches_per_epoch):
            batch = []

            # Fill batch by sampling from each group
            samples_per_group = max(1, self.batch_size // self.n_groups)

            for group_id in self.groups:
                group_indices = self.dataset.get_group_indices(group_id)

                if len(group_indices) == 0:
                    continue

                # Sample from this group
                n_samples = min(samples_per_group, len(group_indices))

                if self.generator:
                    selected = torch.randint(
                        0, len(group_indices), (n_samples,), generator=self.generator
                    ).tolist()
                else:
                    selected = np.random.choice(
                        len(group_indices), size=n_samples, replace=False
                    ).tolist()

                batch.extend([group_indices[i] for i in selected])

                if len(batch) >= self.batch_size:
                    break

            # Shuffle batch
            if self.generator:
                perm = torch.randperm(len(batch), generator=self.generator).tolist()
            else:
                perm = np.random.permutation(len(batch)).tolist()

            batch = [batch[i] for i in perm[:self.batch_size]]

            yield batch

    def __len__(self) -> int:
        return self.batches_per_epoch


# Example usage
if __name__ == "__main__":
    import torchvision.transforms as transforms
    from torch.utils.data import DataLoader

    # Example transforms
    train_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(20),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Create dataset
    dataset = SkinToneAwareDataset(
        metadata_path="data/combined/master_metadata.csv",
        images_root="data",
        split="train",
        fold=0,
        transform=train_transforms
    )

    # Option 1: SkinToneAwareSampler (recommended)
    sampler = SkinToneAwareSampler(
        dataset=dataset,
        samples_per_epoch=10000,  # Adjust based on dataset size
        replacement=True,
        random_seed=42
    )

    loader = DataLoader(
        dataset,
        batch_size=32,
        sampler=sampler,
        num_workers=4,
        pin_memory=True
    )

    # Option 2: StratifiedBatchSampler (alternative)
    # batch_sampler = StratifiedBatchSampler(
    #     dataset=dataset,
    #     batch_size=32,
    #     batches_per_epoch=312,  # 10000 samples / 32 batch_size
    #     random_seed=42
    # )
    #
    # loader = DataLoader(
    #     dataset,
    #     batch_sampler=batch_sampler,
    #     num_workers=4,
    #     pin_memory=True
    # )

    print(f"\nDataLoader created:")
    print(f"  Samples per epoch: {len(loader.sampler)}")
    print(f"  Batches per epoch: {len(loader)}")

    # Test one batch
    images, labels, metadata = next(iter(loader))
    print(f"\nFirst batch:")
    print(f"  Images shape: {images.shape}")
    print(f"  Labels: {labels}")
    print(f"  Groups: {[m for m in metadata['group_id']]}")
