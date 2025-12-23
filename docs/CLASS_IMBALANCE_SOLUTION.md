# Class Imbalance Solution Strategy

## 🔴 Critical Finding: Extreme Class Imbalance

**Analysis Date:** 2025-12-23

---

## Current Situation

### Combined Dataset (ISIC2019 + HAM10000)

| Class | Count | Percentage | Imbalance vs. Smallest |
|-------|-------|------------|------------------------|
| **NV** (Majority) | 19,580 | 55.40% | **55.3× more** than DF |
| MEL | 5,635 | 15.94% | 15.9× more than DF |
| BCC | 3,837 | 10.86% | 10.8× more than DF |
| BKL | 3,723 | 10.53% | 10.5× more than DF |
| AK | 1,194 | 3.38% | 3.4× more than DF |
| SCC | 628 | 1.78% | 1.8× more than DF |
| VASC | 395 | 1.12% | 1.1× more than DF |
| **DF** (Minority) | 354 | 1.00% | **Baseline** |

**Imbalance Ratio:** 55.31:1 (NV:DF)
**Severity:** 🔴 **EXTREMELY IMBALANCED**

---

## Why This Matters

### Without Handling Class Imbalance:
❌ Model will predict **NV** for almost everything (easy 55% accuracy)
❌ Minority classes (DF, VASC, SCC) will be **ignored**
❌ **Dangerous** for medical diagnosis (missing rare cancers!)
❌ Performance metrics will be **misleading**

### Example Without Balancing:
```
Predicted Labels: All NV (19,580/35,346)
Accuracy: 55.4% ← Looks okay but USELESS!
DF Recall: 0% ← Missed ALL rare cases!
```

---

## ✅ Required Solutions (Must Use Multiple Techniques)

### 1. Weighted Cross-Entropy Loss (CRITICAL)

**What it does:** Penalizes misclassification of minority classes more heavily.

**Recommended Weights (from analysis):**
```python
class_weights = {
    'NV':   0.226,   # Majority class (least weight)
    'MEL':  0.784,
    'BCC':  1.151,
    'BKL':  1.187,
    'AK':   3.700,
    'SCC':  7.035,
    'VASC': 11.185,
    'DF':   12.481   # Minority class (highest weight)
}

# In PyTorch:
weights = torch.tensor([0.226, 0.784, 1.151, 1.187, 3.700, 7.035, 11.185, 12.481])
criterion = nn.CrossEntropyLoss(weight=weights)
```

**Expected Impact:** +10-15% on minority class recall

---

### 2. Focal Loss (HIGHLY RECOMMENDED)

**What it does:** Focuses learning on hard-to-classify examples (minority classes).

**Implementation:**
```python
class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0):
        super().__init__()
        self.alpha = alpha  # Class weights
        self.gamma = gamma  # Focusing parameter

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, weight=self.alpha, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        return focal_loss.mean()

# Usage:
criterion = FocalLoss(alpha=weights, gamma=2.0)
```

**Hyperparameters:**
- `gamma=2.0`: Standard value (increase to 3.0 for extreme imbalance)
- `alpha`: Use class weights from above

**Expected Impact:** +5-10% on minority class recall

---

### 3. Weighted Random Sampling (RECOMMENDED)

**What it does:** Oversamples minority classes during training so each class appears equally often per epoch.

**Implementation:**
```python
from torch.utils.data import WeightedRandomSampler

# Calculate sample weights
sample_weights = [class_weights[label] for label in train_labels]

# Create sampler
sampler = WeightedRandomSampler(
    weights=sample_weights,
    num_samples=len(train_labels),
    replacement=True  # Allow duplicates
)

# Use in DataLoader
train_loader = DataLoader(
    dataset,
    batch_size=64,
    sampler=sampler,  # Don't use shuffle with sampler!
    shuffle=False     # Must be False when using sampler
)
```

**Expected Impact:** +8-12% on minority class recall

---

### 4. Stratified Splitting (CRITICAL)

**What it does:** Ensures all classes appear in train/val/test sets proportionally.

**Implementation:**
```python
from sklearn.model_selection import train_test_split, StratifiedKFold

# 80-20 split (stratified)
train_idx, test_idx = train_test_split(
    range(len(labels)),
    test_size=0.2,
    stratify=labels,  # ← CRITICAL
    random_state=42
)

# 5-Fold CV (stratified)
skf = StratifiedKFold(n_folds=5, shuffle=True, random_state=42)
for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    # Each fold maintains class distribution
    pass
```

**Why Critical:** Without stratification, DF class might not appear in validation set!

---

### 5. Class-Balanced Batches (OPTIONAL)

**What it does:** Ensures each batch has representatives from all classes.

**Implementation:**
```python
from torch.utils.data import BatchSampler

class BalancedBatchSampler(BatchSampler):
    def __init__(self, dataset, batch_size):
        # Group indices by class
        self.class_indices = {}
        for idx, label in enumerate(dataset.labels):
            if label not in self.class_indices:
                self.class_indices[label] = []
            self.class_indices[label].append(idx)

        self.batch_size = batch_size
        self.num_classes = len(self.class_indices)

    def __iter__(self):
        # Sample equal number from each class per batch
        samples_per_class = self.batch_size // self.num_classes
        # ... implementation
```

**Expected Impact:** +3-5% on minority class recall

---

## 📊 Recommended Configuration (Combined Approach)

### For Your Thesis Training:

```python
# 1. Use Focal Loss with class weights
class_weights = torch.tensor([
    0.226, 0.784, 1.151, 1.187,
    3.700, 7.035, 11.185, 12.481
])

criterion = FocalLoss(alpha=class_weights, gamma=2.0)

# 2. Use Weighted Random Sampling
sample_weights = [class_weights[label] for label in train_labels]
sampler = WeightedRandomSampler(sample_weights, len(train_labels), replacement=True)

# 3. Create DataLoader
train_loader = DataLoader(
    train_dataset,
    batch_size=64,
    sampler=sampler,
    num_workers=8,
    pin_memory=True
)

# 4. Always use stratified splits
train_idx, test_idx = train_test_split(
    indices,
    test_size=0.2,
    stratify=labels,  # CRITICAL
    random_state=42
)
```

---

## Expected Performance Improvement

### Without Balancing:
```
Overall Accuracy: 55-60% (mostly predicting NV)
NV Recall: 95%
DF Recall: 0-5%  ← DISASTER
VASC Recall: 0-10%
```

### With Weighted Loss Only:
```
Overall Accuracy: 75-80%
NV Recall: 85%
DF Recall: 40-50%  ← Better but still poor
VASC Recall: 35-45%
```

### With Full Strategy (Recommended):
```
Overall Accuracy: 85-90%
NV Recall: 90%
DF Recall: 70-75%  ← Much better!
VASC Recall: 68-73%
```

---

## Implementation Checklist

### Phase 1: Data Preparation (DONE ✓)
- [x] Analyze class distribution
- [x] Calculate class weights
- [x] Generate imbalance report
- [x] Create visualizations

### Phase 2: Training Setup (TODO)
- [ ] Implement Focal Loss
- [ ] Implement Weighted Random Sampler
- [ ] Update training script with class weights
- [ ] Ensure stratified splits in all CV folds

### Phase 3: Validation (TODO)
- [ ] Monitor per-class metrics during training
- [ ] Track minority class recall specifically
- [ ] Compare balanced vs. unbalanced training
- [ ] Validate on external dataset (ISIC2020)

---

## Evaluation Metrics (Report These in Thesis)

### Standard Metrics (Overall):
- Accuracy
- Macro-averaged Precision/Recall/F1
- Micro-averaged Precision/Recall/F1
- AUC-ROC (macro/micro)

### Per-Class Metrics (CRITICAL):
- **Per-class Recall** (most important for medical!)
- Per-class Precision
- Per-class F1-Score
- Confusion Matrix

### Imbalance-Aware Metrics:
- **Balanced Accuracy** = Mean of per-class recalls
- **Cohen's Kappa** (accounts for imbalance)
- **Matthews Correlation Coefficient (MCC)**

**Report Format:**
```
Model: Swin Transformer

Overall Performance:
- Accuracy: 89.5%
- Balanced Accuracy: 84.2%  ← More meaningful!
- Macro F1: 82.8%

Per-Class Recall (Most Important):
- NV: 92.3%
- MEL: 87.5%
- BCC: 85.1%
- BKL: 83.6%
- AK: 76.4%
- SCC: 73.8%
- VASC: 71.2%
- DF: 69.5%  ← Acceptable for rare class!
```

---

## Common Mistakes to Avoid

### ❌ DON'T:
1. Use accuracy as primary metric (misleading!)
2. Ignore minority class performance
3. Forget to stratify splits
4. Use only one balancing technique
5. Skip per-class analysis

### ✅ DO:
1. Use balanced accuracy and per-class recall
2. Report minority class metrics prominently
3. Always stratify (80-20 split, 5-fold CV)
4. Combine multiple techniques (loss + sampling)
5. Analyze confusion matrix for each class

---

## Files Generated

```
results/class_balance/
├── isic2019_distribution.png         # Visualization
├── isic2019_report.md                # Detailed report
├── ham10000_distribution.png         # Visualization
├── ham10000_report.md                # Detailed report
├── combined_distribution.png         # Combined viz
└── combined_report.md                # Combined report
```

---

## Quick Start: Apply Solutions Now

### Step 1: Update Training Script
```bash
cd scripts/training
# Edit train_5fold_cv.py to add:
# - Focal Loss
# - Weighted Sampling
# - Stratified splits
```

### Step 2: Verify Class Weights
```python
# Use these exact weights in your loss function:
class_weights = torch.tensor([
    0.226,  # NV
    0.784,  # MEL
    1.151,  # BCC
    1.187,  # BKL
    3.700,  # AK
    7.035,  # SCC
    11.185, # VASC
    12.481  # DF
])
```

### Step 3: Monitor Training
```python
# Track per-class metrics every epoch:
for epoch in range(num_epochs):
    train_one_epoch()
    per_class_recall = evaluate_per_class(val_loader)

    # Log minority class performance
    print(f"DF Recall: {per_class_recall['DF']:.2f}%")
    print(f"VASC Recall: {per_class_recall['VASC']:.2f}%")
```

---

## Summary

**Question:** Are classes balanced?
**Answer:** 🔴 **NO - Extremely imbalanced (55:1 ratio)**

**Action Required:**
1. ✅ Use Focal Loss with class weights
2. ✅ Use Weighted Random Sampling
3. ✅ Ensure stratified splits
4. ✅ Report balanced accuracy and per-class metrics
5. ✅ Monitor minority class performance

**Expected Outcome:**
- Balanced accuracy: 84-87% (vs. 55% without balancing)
- DF recall: 70-75% (vs. 0-5% without balancing)
- All classes learned properly (not just majority!)

---

**This is CRITICAL for your thesis - medical AI must detect rare cancers!**
