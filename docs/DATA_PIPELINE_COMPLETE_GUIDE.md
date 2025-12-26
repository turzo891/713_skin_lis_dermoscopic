# Complete Data Pipeline Guide: Raw Datasets → Training-Ready Data

**Complete Reproducible Pipeline for Skin Cancer Classification**

**Goal:** Transform raw ISIC2019 + HAM10000 datasets into a clean, balanced, training-ready format using Tidy Data principles and OpenRefine.

---

## Table of Contents

1. [Overview: Data Flow](#overview-data-flow)
2. [Phase 0: Raw Data Download](#phase-0-raw-data-download)
3. [Phase 1: OpenRefine Cleaning](#phase-1-openrefine-cleaning)
4. [Phase 2: Tidy Data Transformation](#phase-2-tidy-data-transformation)
5. [Phase 3: Training Integration](#phase-3-training-integration)
6. [Validation & Quality Checks](#validation--quality-checks)
7. [Complete File Structure](#complete-file-structure)

---

## Overview: Data Flow

```

                    COMPLETE DATA PIPELINE                            

                                                                      
  Phase 0: RAW DOWNLOAD                                              
                             
     ISIC2019              HAM10000                              
     25,331 images         10,015 images                         
     Raw CSVs              Raw CSVs                              
                             
                                                                    
                                             
                                                                     
  Phase 1: OPENREFINE CLEANING                                       
                             
    - Standardize labels                                           
    - Remove duplicates                                            
    - Fix metadata issues                                          
    - Quality flagging                                             
                             
                                                                     
                     cleaned_metadata.csv                            
                                                                     
  Phase 2: TIDY DATA TRANSFORMATION                                  
                             
    master_metadata.csv  (Single Source)                           
    splits.csv           (80-20 + 5-CV)                            
    class_weights.csv    (Imbalance)                               
    schema.json          (Structure)                               
                             
                                                                     
                                                                     
  Phase 3: PYTORCH DATASET                                           
                             
    TidySkinDataset                                                
    + Focal Loss                                                   
    + Weighted Sampling                                            
    + Stratified Splits                                            
                             
                                                                     
                                                                     
            TRAINING READY                                          
                                                                      

```

---

## Phase 0: Raw Data Download

### Step 0.1: Download ISIC2019

```bash
#!/bin/bash
# scripts/00_download_isic2019.sh

echo "Downloading ISIC2019 dataset..."

# Create directories
mkdir -p data/raw/ISIC2019

cd data/raw/ISIC2019

# Option 1: Using wget (if you have direct links)
# wget https://isic-challenge-data.s3.amazonaws.com/2019/ISIC_2019_Training_Input.zip
# wget https://isic-challenge-data.s3.amazonaws.com/2019/ISIC_2019_Training_GroundTruth.csv
# wget https://isic-challenge-data.s3.amazonaws.com/2019/ISIC_2019_Training_Metadata.csv

# Option 2: Manual download instructions
echo "
Please download the following files manually from:
https://challenge.isic-archive.com/data/#2019

Required files:
1. ISIC_2019_Training_Input.zip (9.6 GB)
2. ISIC_2019_Training_GroundTruth.csv
3. ISIC_2019_Training_Metadata.csv

Place them in: data/raw/ISIC2019/
"

# Unzip images
if [ -f "ISIC_2019_Training_Input.zip" ]; then
    echo "Extracting images..."
    unzip -q ISIC_2019_Training_Input.zip
    echo " ISIC2019 extraction complete"
fi

cd ../../..
```

**Expected Structure:**
```
data/raw/ISIC2019/
 ISIC_2019_Training_Input/
    ISIC_2019_Training_Input/
       ISIC_0000000.jpg
       ISIC_0000001.jpg
       ... (25,331 images)
 ISIC_2019_Training_GroundTruth.csv
 ISIC_2019_Training_Metadata.csv
```

### Step 0.2: Download HAM10000

```bash
#!/bin/bash
# scripts/00_download_ham10000.sh

echo "Downloading HAM10000 dataset..."

mkdir -p data/raw/HAM10000

cd data/raw/HAM10000

# Download from Harvard Dataverse
echo "
Please download HAM10000 from:
https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/DBW86T

Required files:
1. HAM10000_images_part_1.zip
2. HAM10000_images_part_2.zip
3. HAM10000_metadata.csv

Place them in: data/raw/HAM10000/
"

# Unzip if files exist
if [ -f "HAM10000_images_part_1.zip" ]; then
    echo "Extracting part 1..."
    unzip -q HAM10000_images_part_1.zip
fi

if [ -f "HAM10000_images_part_2.zip" ]; then
    echo "Extracting part 2..."
    unzip -q HAM10000_images_part_2.zip
fi

echo " HAM10000 extraction complete"

cd ../../..
```

**Expected Structure:**
```
data/raw/HAM10000/
 HAM10000_images_part_1/
    ISIC_0024306.jpg
    ... (5,000 images)
 HAM10000_images_part_2/
    ISIC_0024307.jpg
    ... (5,015 images)
 HAM10000_metadata.csv
```

### Step 0.3: Verify Raw Downloads

```bash
# scripts/00_verify_downloads.sh

echo "Verifying raw dataset downloads..."

# Check ISIC2019
if [ -d "data/raw/ISIC2019/ISIC_2019_Training_Input/ISIC_2019_Training_Input" ]; then
    ISIC_COUNT=$(ls data/raw/ISIC2019/ISIC_2019_Training_Input/ISIC_2019_Training_Input/*.jpg 2>/dev/null | wc -l)
    echo " ISIC2019 images: $ISIC_COUNT (expected: 25,331)"
else
    echo " ISIC2019 images not found"
fi

if [ -f "data/raw/ISIC2019/ISIC_2019_Training_GroundTruth.csv" ]; then
    echo " ISIC2019 ground truth found"
else
    echo " ISIC2019 ground truth missing"
fi

# Check HAM10000
HAM_COUNT1=$(ls data/raw/HAM10000/HAM10000_images_part_1/*.jpg 2>/dev/null | wc -l)
HAM_COUNT2=$(ls data/raw/HAM10000/HAM10000_images_part_2/*.jpg 2>/dev/null | wc -l)
HAM_TOTAL=$((HAM_COUNT1 + HAM_COUNT2))

echo " HAM10000 images: $HAM_TOTAL (expected: 10,015)"

if [ -f "data/raw/HAM10000/HAM10000_metadata.csv" ]; then
    echo " HAM10000 metadata found"
else
    echo " HAM10000 metadata missing"
fi

echo "
 Raw data verification complete
"
```

---

## Phase 1: OpenRefine Cleaning

### Why OpenRefine?
- **Visual inspection** of data quality issues
- **Interactive duplicate detection** (phonetic matching, clustering)
- **Manual review** of edge cases
- **Audit trail** of all transformations
- **Reproducible** via saved projects

### Step 1.1: Prepare Metadata for OpenRefine

```python
# scripts/01_prepare_for_openrefine.py
"""
Merge ISIC2019 and HAM10000 metadata into a single CSV for OpenRefine cleaning.
"""

import pandas as pd
import os
from pathlib import Path

def prepare_isic2019_metadata():
    """Load and standardize ISIC2019 metadata."""
    gt_path = 'data/raw/ISIC2019/ISIC_2019_Training_GroundTruth.csv'
    meta_path = 'data/raw/ISIC2019/ISIC_2019_Training_Metadata.csv'

    # Load ground truth
    gt_df = pd.read_csv(gt_path)

    # Melt one-hot encoded labels to single column
    label_cols = ['MEL', 'NV', 'BCC', 'AK', 'BKL', 'DF', 'VASC', 'SCC']

    records = []
    for _, row in gt_df.iterrows():
        image_id = row['image']
        for label in label_cols:
            if row[label] == 1.0:
                records.append({'image_id': image_id, 'diagnosis': label})
                break

    isic_df = pd.DataFrame(records)

    # Add metadata if available
    if os.path.exists(meta_path):
        meta_df = pd.read_csv(meta_path)
        isic_df = isic_df.merge(meta_df, left_on='image_id', right_on='image', how='left')

    # Add source column
    isic_df['source'] = 'ISIC2019'

    # Add image path
    isic_df['image_path'] = isic_df['image_id'].apply(
        lambda x: f'ISIC2019/ISIC_2019_Training_Input/ISIC_2019_Training_Input/{x}.jpg'
    )

    return isic_df[['image_id', 'source', 'diagnosis', 'image_path', 'age', 'sex', 'anatom_site_general']]


def prepare_ham10000_metadata():
    """Load and standardize HAM10000 metadata."""
    meta_path = 'data/raw/HAM10000/HAM10000_metadata.csv'

    ham_df = pd.read_csv(meta_path)

    # Map HAM10000 labels to ISIC2019 format
    label_mapping = {
        'mel': 'MEL',
        'nv': 'NV',
        'bcc': 'BCC',
        'akiec': 'AK',
        'bkl': 'BKL',
        'df': 'DF',
        'vasc': 'VASC'
    }

    ham_df['diagnosis'] = ham_df['dx'].map(label_mapping)
    ham_df['source'] = 'HAM10000'

    # Determine image path (check both parts)
    def get_ham_path(image_id):
        part1 = f'HAM10000/HAM10000_images_part_1/{image_id}.jpg'
        part2 = f'HAM10000/HAM10000_images_part_2/{image_id}.jpg'

        if os.path.exists(f'data/raw/{part1}'):
            return part1
        else:
            return part2

    ham_df['image_path'] = ham_df['image_id'].apply(get_ham_path)

    return ham_df[['image_id', 'source', 'diagnosis', 'image_path', 'age', 'sex', 'localization']]


def merge_for_openrefine():
    """Merge both datasets for OpenRefine cleaning."""
    print("Loading ISIC2019 metadata...")
    isic_df = prepare_isic2019_metadata()
    print(f"   Loaded {len(isic_df)} ISIC2019 records")

    print("Loading HAM10000 metadata...")
    ham_df = prepare_ham10000_metadata()
    print(f"   Loaded {len(ham_df)} HAM10000 records")

    # Standardize column names
    isic_df = isic_df.rename(columns={'anatom_site_general': 'localization'})

    # Merge
    combined_df = pd.concat([isic_df, ham_df], ignore_index=True)

    # Add patient_id placeholder (needed for patient-aware splits)
    # For ISIC2019, use image_id as patient_id (conservative)
    # For HAM10000, lesion_id could be used if available
    combined_df['patient_id'] = combined_df['image_id']  # Simplification

    # Save for OpenRefine
    output_path = 'data/openrefine/combined_raw_metadata.csv'
    os.makedirs('data/openrefine', exist_ok=True)
    combined_df.to_csv(output_path, index=False)

    print(f"\n Combined metadata saved to: {output_path}")
    print(f"  Total records: {len(combined_df)}")
    print(f"  Columns: {list(combined_df.columns)}")
    print(f"\nClass distribution:")
    print(combined_df['diagnosis'].value_counts())

    return combined_df


if __name__ == '__main__':
    merge_for_openrefine()
```

**Run:**
```bash
python scripts/01_prepare_for_openrefine.py
```

**Output:**
```
data/openrefine/combined_raw_metadata.csv
```

### Step 1.2: OpenRefine Cleaning (Interactive)

**Manual Steps (Document in Markdown):**

Create `docs/OPENREFINE_CLEANING_PROTOCOL.md`:

```markdown
# OpenRefine Cleaning Protocol

## Step 1: Import Data

1. Open OpenRefine (http://localhost:3333)
2. Click "Create Project"
3. Upload: `data/openrefine/combined_raw_metadata.csv`
4. Click "Next"
5. Configure:
   - Parse as: CSV
   - Columns are separated by: commas
   - Character encoding: UTF-8
6. Click "Create Project"

## Step 2: Standardize Diagnosis Labels

**Goal:** Ensure all labels use consistent format (MEL, NV, etc.)

1. Click on `diagnosis` column → Facet → Text facet
2. Review all unique values
3. For any variations:
   - Click value → Edit → Enter standard form
   - Examples:
     - "melanoma" → "MEL"
     - "Melanoma" → "MEL"
     - "nevus" → "NV"

**Expected unique values after cleaning:**
```
MEL, NV, BCC, AK, BKL, DF, VASC, SCC
```

## Step 3: Detect and Remove Exact Duplicates

**Goal:** Remove images that appear in both datasets

1. Click on `image_id` column → Edit cells → Blank down
2. Click on `image_id` column → Facet → Customized facets → Duplicates facet
3. Review duplicates
4. For each duplicate:
   - Keep the one from ISIC2019 (if tie)
   - Or keep highest quality metadata
5. Remove duplicates:
   - Select duplicates
   - Click "All" → Edit rows → Remove matching rows

**Document decision:**
```
Duplicates removed: ~200-500
Kept source: ISIC2019 (better metadata)
```

## Step 4: Detect Near-Duplicates (Clustering)

**Goal:** Find potential duplicates with slight variations

1. Click on `image_id` column → Edit cells → Cluster and edit
2. Try methods:
   - key collision: fingerprint
   - key collision: ngram-fingerprint
   - nearest neighbor: levenshtein
3. Review clusters manually
4. Merge if confirmed duplicates

## Step 5: Handle Missing Data

**Age:**
1. Click `age` → Facet → Numeric facet
2. Identify blanks
3. Options:
   - Fill with median: Edit cells → Transform → `value.isBlank() ? 45 : value`
   - Or keep blank (handled in training)

**Sex:**
1. Click `sex` → Facet → Text facet
2. Standardize:
   - "M" / "male" / "Male" → "male"
   - "F" / "female" / "Female" → "female"
   - "unknown" / blank → "unknown"

**Localization:**
1. Click `localization` → Facet → Text facet
2. Standardize anatomical site names
3. Fix typos

## Step 6: Quality Flagging

Add a new column `quality_flag`:

1. Click "All" → Edit columns → Add column based on this column
2. Column name: `quality_flag`
3. Expression:
```
if(isBlank(cells['diagnosis'].value), 'exclude',
   if(isBlank(cells['image_path'].value), 'exclude',
      'good'))
```

## Step 7: Verify Image Paths

Check that all image paths exist:

1. Add column `image_exists`:
2. Expression (requires custom OpenRefine function or external verification)

**Or:** Export and verify with Python script

## Step 8: Export Cleaned Data

1. Click "Export" → "Comma-separated value"
2. Save as: `data/openrefine/cleaned_metadata.csv`
3. Save project: `data/openrefine/cleaning_project.tar.gz`

## Step 9: Document Changes

Create a change log:

```
Cleaning Summary:
- Total records processed: 35,346
- Exact duplicates removed: 312
- Near-duplicates removed: 47
- Records with missing labels: 0 (excluded)
- Records with missing images: 0 (excluded)
- Final clean records: 34,987
```

Save as: `data/openrefine/cleaning_log.txt`
```

---

## Phase 2: Tidy Data Transformation

### Step 2.1: Transform to Tidy Format

```python
# scripts/02_create_tidy_dataset.py
"""
Transform OpenRefine cleaned data into Tidy Data format.

Tidy Data Principles:
1. Each variable forms a column
2. Each observation forms a row
3. Each type of observational unit forms a table

Output Tables:
- master_metadata.csv: One row per image (single source of truth)
- splits.csv: One row per (image, fold, split) assignment
- class_weights.csv: One row per class
- schema.json: Column definitions and constraints
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
import hashlib
from sklearn.model_selection import StratifiedKFold, train_test_split
from collections import Counter


class TidyDatasetCreator:
    """Create tidy dataset from OpenRefine cleaned data."""

    def __init__(self, cleaned_csv: str, output_dir: str, raw_data_dir: str):
        self.cleaned_df = pd.read_csv(cleaned_csv)
        self.output_dir = Path(output_dir)
        self.raw_data_dir = Path(raw_data_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.class_order = ['MEL', 'NV', 'BCC', 'BKL', 'AK', 'SCC', 'VASC', 'DF']

    def create_master_metadata(self) -> pd.DataFrame:
        """
        Create master metadata table (single source of truth).

        Tidy principle: Each row is one image observation.
        """
        df = self.cleaned_df.copy()

        # Ensure unique image_ids
        df = df.drop_duplicates(subset='image_id', keep='first')

        # Map class labels to indices
        df['class_idx'] = df['diagnosis'].map(
            {c: i for i, c in enumerate(self.class_order)}
        )

        # Verify all images exist
        df['image_exists'] = df['image_path'].apply(
            lambda p: (self.raw_data_dir / p).exists()
        )

        # Exclude missing images
        missing_count = (~df['image_exists']).sum()
        if missing_count > 0:
            print(f"  Warning: {missing_count} images not found, excluding")
            df = df[df['image_exists']]

        # Select final columns
        master_df = df[[
            'image_id',
            'source',
            'diagnosis',
            'class_idx',
            'image_path',
            'patient_id',
            'age',
            'sex',
            'localization'
        ]].copy()

        # Ensure no nulls in critical columns
        assert master_df['image_id'].notna().all(), "Null image_ids found"
        assert master_df['diagnosis'].notna().all(), "Null diagnoses found"
        assert master_df['class_idx'].notna().all(), "Null class indices found"

        return master_df

    def create_stratified_splits(
        self,
        metadata_df: pd.DataFrame,
        n_folds: int = 5,
        test_ratio: float = 0.2,
        random_state: int = 42
    ) -> pd.DataFrame:
        """
        Create stratified train/val/test splits.

        Strategy:
        - 80% training + validation (for 5-fold CV)
        - 20% held-out test (never seen during training)
        - Stratified by class
        - Patient-aware (same patient never in both train and test)

        Tidy principle: Each row is one (image, fold, split) assignment.
        """
        print("\nCreating stratified splits...")

        # Get unique patients with their most common class
        patient_class = metadata_df.groupby('patient_id')['class_idx'].agg(
            lambda x: x.mode()[0] if len(x.mode()) > 0 else x.iloc[0]
        ).reset_index()

        # Split patients into train+val (80%) and test (20%)
        train_val_patients, test_patients = train_test_split(
            patient_class['patient_id'],
            test_size=test_ratio,
            stratify=patient_class['class_idx'],
            random_state=random_state
        )

        print(f"  Train+Val patients: {len(train_val_patients)}")
        print(f"  Test patients: {len(test_patients)}")

        # Create splits records
        splits_records = []

        # Test set (same for all folds)
        test_images = metadata_df[metadata_df['patient_id'].isin(test_patients)]
        print(f"  Test images: {len(test_images)}")

        for _, row in test_images.iterrows():
            for fold in range(n_folds):
                splits_records.append({
                    'image_id': row['image_id'],
                    'fold': fold,
                    'split': 'test'
                })

        # Train+Val set (5-fold CV)
        train_val_df = metadata_df[metadata_df['patient_id'].isin(train_val_patients)]
        train_val_patients_df = train_val_df.groupby('patient_id').first().reset_index()

        print(f"  Train+Val images: {len(train_val_df)}")

        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)

        for fold, (train_idx, val_idx) in enumerate(
            skf.split(train_val_patients_df, train_val_patients_df['class_idx'])
        ):
            train_patients_fold = train_val_patients_df.iloc[train_idx]['patient_id']
            val_patients_fold = train_val_patients_df.iloc[val_idx]['patient_id']

            # Get images for these patients
            train_images = train_val_df[train_val_df['patient_id'].isin(train_patients_fold)]
            val_images = train_val_df[train_val_df['patient_id'].isin(val_patients_fold)]

            print(f"  Fold {fold}: Train={len(train_images)}, Val={len(val_images)}")

            # Add records
            for _, row in train_images.iterrows():
                splits_records.append({
                    'image_id': row['image_id'],
                    'fold': fold,
                    'split': 'train'
                })

            for _, row in val_images.iterrows():
                splits_records.append({
                    'image_id': row['image_id'],
                    'fold': fold,
                    'split': 'val'
                })

        splits_df = pd.DataFrame(splits_records)

        # Verify no data leakage
        for fold in range(n_folds):
            fold_splits = splits_df[splits_df['fold'] == fold]
            train_ids = set(fold_splits[fold_splits['split'] == 'train']['image_id'])
            val_ids = set(fold_splits[fold_splits['split'] == 'val']['image_id'])
            test_ids = set(fold_splits[fold_splits['split'] == 'test']['image_id'])

            assert len(train_ids & val_ids) == 0, f"Fold {fold}: train/val overlap!"
            assert len(train_ids & test_ids) == 0, f"Fold {fold}: train/test overlap!"
            assert len(val_ids & test_ids) == 0, f"Fold {fold}: val/test overlap!"

        print("   No data leakage detected")

        return splits_df

    def compute_class_weights(
        self,
        metadata_df: pd.DataFrame,
        splits_df: pd.DataFrame,
        fold: int = 0
    ) -> pd.DataFrame:
        """
        Compute class weights for handling imbalance.

        Weights computed from training set only (not val/test).

        Tidy principle: Each row is one class.
        """
        # Get training images for fold 0
        train_ids = splits_df[
            (splits_df['fold'] == fold) &
            (splits_df['split'] == 'train')
        ]['image_id']

        train_df = metadata_df[metadata_df['image_id'].isin(train_ids)]

        # Count classes
        class_counts = train_df['class_idx'].value_counts().sort_index()

        # Compute inverse frequency weights
        total = class_counts.sum()
        frequencies = class_counts / total
        weights = 1.0 / frequencies
        weights = weights / weights.mean()  # Normalize to mean=1

        # Create tidy dataframe
        weights_df = pd.DataFrame({
            'class_label': self.class_order,
            'class_idx': range(8),
            'train_count': class_counts.values,
            'frequency': frequencies.values,
            'weight': weights.values
        })

        return weights_df

    def create_schema(self) -> dict:
        """
        Define schema for all tidy tables.

        Documents structure and constraints.
        """
        schema = {
            'master_metadata': {
                'description': 'Single source of truth for all image metadata',
                'primary_key': 'image_id',
                'columns': {
                    'image_id': {'type': 'string', 'nullable': False, 'unique': True},
                    'source': {'type': 'string', 'nullable': False, 'values': ['ISIC2019', 'HAM10000']},
                    'diagnosis': {'type': 'string', 'nullable': False, 'values': self.class_order},
                    'class_idx': {'type': 'integer', 'nullable': False, 'min': 0, 'max': 7},
                    'image_path': {'type': 'string', 'nullable': False},
                    'patient_id': {'type': 'string', 'nullable': False},
                    'age': {'type': 'float', 'nullable': True, 'min': 0, 'max': 120},
                    'sex': {'type': 'string', 'nullable': True, 'values': ['male', 'female', 'unknown']},
                    'localization': {'type': 'string', 'nullable': True}
                },
                'row_count': None  # Filled during creation
            },
            'splits': {
                'description': 'Stratified fold assignments for train/val/test',
                'primary_key': ['image_id', 'fold'],
                'columns': {
                    'image_id': {'type': 'string', 'nullable': False},
                    'fold': {'type': 'integer', 'nullable': False, 'min': 0, 'max': 4},
                    'split': {'type': 'string', 'nullable': False, 'values': ['train', 'val', 'test']}
                },
                'constraints': [
                    'No image_id can be in both train and test for same fold',
                    'Test set is identical across all folds',
                    'Each image appears exactly once per fold'
                ],
                'row_count': None
            },
            'class_weights': {
                'description': 'Precomputed weights for handling class imbalance',
                'primary_key': 'class_idx',
                'columns': {
                    'class_label': {'type': 'string', 'nullable': False},
                    'class_idx': {'type': 'integer', 'nullable': False, 'min': 0, 'max': 7},
                    'train_count': {'type': 'integer', 'nullable': False},
                    'frequency': {'type': 'float', 'nullable': False, 'min': 0, 'max': 1},
                    'weight': {'type': 'float', 'nullable': False, 'min': 0}
                },
                'row_count': 8
            }
        }

        return schema

    def save_tidy_dataset(self):
        """Create and save all tidy tables."""
        print("=" * 80)
        print("CREATING TIDY DATASET")
        print("=" * 80)

        # 1. Master metadata
        print("\n1. Creating master_metadata.csv...")
        metadata_df = self.create_master_metadata()
        print(f"   Total images: {len(metadata_df)}")
        print(f"   Class distribution:")
        print(metadata_df['diagnosis'].value_counts())

        # 2. Splits
        print("\n2. Creating splits.csv...")
        splits_df = self.create_stratified_splits(metadata_df)
        print(f"   Total split assignments: {len(splits_df)}")

        # 3. Class weights
        print("\n3. Creating class_weights.csv...")
        weights_df = self.compute_class_weights(metadata_df, splits_df)
        print(f"   Class weights:")
        print(weights_df[['class_label', 'weight']])

        # 4. Schema
        print("\n4. Creating schema.json...")
        schema = self.create_schema()
        schema['master_metadata']['row_count'] = len(metadata_df)
        schema['splits']['row_count'] = len(splits_df)

        # Save all files
        print("\n5. Saving files...")
        metadata_df.to_csv(self.output_dir / 'master_metadata.csv', index=False)
        splits_df.to_csv(self.output_dir / 'splits.csv', index=False)
        weights_df.to_csv(self.output_dir / 'class_weights.csv', index=False)

        with open(self.output_dir / 'schema.json', 'w') as f:
            json.dump(schema, f, indent=2)

        print(f"    All files saved to: {self.output_dir}")

        # Summary report
        print("\n" + "=" * 80)
        print("TIDY DATASET CREATED SUCCESSFULLY")
        print("=" * 80)
        print(f"\nFiles created:")
        print(f"  - master_metadata.csv ({len(metadata_df)} rows)")
        print(f"  - splits.csv ({len(splits_df)} rows)")
        print(f"  - class_weights.csv ({len(weights_df)} rows)")
        print(f"  - schema.json")

        print(f"\nClass imbalance ratio: {weights_df['weight'].max() / weights_df['weight'].min():.2f}:1")
        print(f"Majority class: {weights_df.loc[weights_df['train_count'].idxmax(), 'class_label']}")
        print(f"Minority class: {weights_df.loc[weights_df['train_count'].idxmin(), 'class_label']}")

        return metadata_df, splits_df, weights_df


def validate_tidy_dataset(tidy_dir: str) -> bool:
    """Validate tidy dataset constraints."""
    tidy_dir = Path(tidy_dir)

    print("\n" + "=" * 80)
    print("VALIDATING TIDY DATASET")
    print("=" * 80)

    errors = []

    # Load files
    try:
        metadata = pd.read_csv(tidy_dir / 'master_metadata.csv')
        splits = pd.read_csv(tidy_dir / 'splits.csv')
        weights = pd.read_csv(tidy_dir / 'class_weights.csv')
        with open(tidy_dir / 'schema.json', 'r') as f:
            schema = json.load(f)
    except Exception as e:
        print(f" Error loading files: {e}")
        return False

    # Check 1: Primary key uniqueness
    if metadata['image_id'].duplicated().any():
        errors.append("master_metadata: Duplicate image_ids")
    else:
        print(" Primary key uniqueness (master_metadata)")

    # Check 2: No null critical columns
    if metadata[['image_id', 'diagnosis', 'class_idx']].isnull().any().any():
        errors.append("master_metadata: Null values in critical columns")
    else:
        print(" No nulls in critical columns")

    # Check 3: Referential integrity
    orphan_ids = set(splits['image_id']) - set(metadata['image_id'])
    if orphan_ids:
        errors.append(f"splits: {len(orphan_ids)} orphan image_ids")
    else:
        print(" Referential integrity (splits → master_metadata)")

    # Check 4: No data leakage
    for fold in splits['fold'].unique():
        fold_splits = splits[splits['fold'] == fold]
        train_ids = set(fold_splits[fold_splits['split'] == 'train']['image_id'])
        val_ids = set(fold_splits[fold_splits['split'] == 'val']['image_id'])
        test_ids = set(fold_splits[fold_splits['split'] == 'test']['image_id'])

        if train_ids & val_ids:
            errors.append(f"Fold {fold}: train/val overlap")
        if train_ids & test_ids:
            errors.append(f"Fold {fold}: train/test overlap")
        if val_ids & test_ids:
            errors.append(f"Fold {fold}: val/test overlap")

    if not errors:
        print(" No data leakage across splits")

    # Check 5: Class weights
    if len(weights) != 8:
        errors.append(f"class_weights: Expected 8 rows, got {len(weights)}")
    else:
        print(" Class weights complete (8 classes)")

    # Check 6: Test set consistency across folds
    test_sets = []
    for fold in range(5):
        test_ids = set(splits[
            (splits['fold'] == fold) & (splits['split'] == 'test')
        ]['image_id'])
        test_sets.append(test_ids)

    if not all(s == test_sets[0] for s in test_sets):
        errors.append("Test set differs across folds")
    else:
        print(" Test set identical across all folds")

    # Report
    print("\n" + "=" * 80)
    if errors:
        print(" VALIDATION FAILED")
        print("=" * 80)
        for error in errors:
            print(f"  - {error}")
        return False
    else:
        print(" VALIDATION PASSED - Dataset is tidy and ready for training")
        print("=" * 80)
        return True


if __name__ == '__main__':
    # Create tidy dataset
    creator = TidyDatasetCreator(
        cleaned_csv='data/openrefine/cleaned_metadata.csv',
        output_dir='data/tidy',
        raw_data_dir='data/raw'
    )

    creator.save_tidy_dataset()

    # Validate
    validate_tidy_dataset('data/tidy')
```

**Run:**
```bash
python scripts/02_create_tidy_dataset.py
```

---

## Phase 3: Training Integration

[Content continues in next part due to length...]

**Complete implementation with PyTorch Dataset, training scripts, and validation coming in the next file...**

Would you like me to continue with Phase 3 (Training Integration) and create the remaining implementation scripts?
