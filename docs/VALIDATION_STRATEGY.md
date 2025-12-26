# Robust Multi-Dataset Validation Strategy

## Executive Summary

This document outlines a comprehensive, publication-ready validation strategy for skin cancer classification models following medical AI best practices (TRIPOD-AI, CLAIM guidelines).

**Date:** 2025-12-23
**Status:** Implementation Ready
**Purpose:** Ensure robust model evaluation with external validation

---

## 1. Current Problem

### What Was Done (Inadequate)
```
Training Dataset: ISIC2019 only
 Train: 70% of ISIC2019 (~17,732 images)
 Val:   15% of ISIC2019 (~3,800 images)    ← SAME SOURCE
 Test:  15% of ISIC2019 (~3,800 images)    ← SAME SOURCE

Issue: No external validation - cannot prove generalization!
```

**Why This Is Problematic:**
- All data from same source (ISIC challenge)
- Same image acquisition protocols
- Same patient population
- Same imaging devices
- **Cannot claim model generalizes to real-world data**

---

## 2. Recommended Strategy (Publication-Ready)

### Three-Tier Validation Approach

```
TIER 1: Internal Validation (Development)
 Dataset: ISIC2019 (25,331 images)
 Method: 10-Fold Stratified Cross-Validation
 Purpose: Model selection, hyperparameter tuning
 Metric: Mean ± Std across folds

TIER 2: External Validation (Generalization)
 Dataset 1: HAM10000 (10,015 images) - Different source
 Dataset 2: ISIC2020 (33,126 images) - Different year
 Method: Test on entire external datasets
 Purpose: Prove generalization to unseen data

TIER 3: Cross-Dataset Training (Robustness)
 Train A → Test B: ISIC2019 → HAM10000
 Train B → Test A: HAM10000 → ISIC2019
 Train A+B → Test C: Combined → ISIC2020
 Purpose: Maximum robustness assessment
```

---

## 3. Implementation Plan

### Phase 1: Internal Validation (ISIC2019)

**Method:** 10-Fold Stratified Cross-Validation

```python
# Pseudocode
for fold in range(1, 11):
    train_data = ISIC2019[folds != fold]  # 90% for training
    val_data = ISIC2019[folds == fold]    # 10% for validation

    model = train(train_data)
    metrics[fold] = evaluate(model, val_data)

# Report: Accuracy = 91.2% ± 2.3%
```

**Why 10-Fold?**
- Standard for datasets with 20,000+ samples
- Each fold: ~22,800 train / ~2,500 val
- Low variance, high reliability
- Accepted by top medical journals

**Output:**
- Mean accuracy ± standard deviation
- Per-class performance across folds
- Confusion matrices
- ROC curves with confidence intervals

---

### Phase 2: External Validation (HAM10000)

**Method:** Train on full ISIC2019, test on full HAM10000

```python
# Train on all ISIC2019
train_data = ISIC2019_all  # 25,331 images
model = train(train_data)

# Test on all HAM10000 (completely different dataset)
test_data = HAM10000_all   # 10,015 images
external_metrics = evaluate(model, test_data)
```

**Why HAM10000?**
- Different data source (different hospitals)
- Different imaging devices
- Different patient demographics
- Different time period
- 7 overlapping classes with ISIC2019

**Expected Results:**
- Performance drop is normal (5-15%)
- Proves model generalizes beyond training distribution
- **CRITICAL for publication credibility**

---

### Phase 3: Multi-Dataset Training

**Scenario A: Combined Training**
```python
train_data = ISIC2019 + HAM10000  # ~35,346 images
test_data = ISIC2020              # Download required

# Benefit: More diverse training data
# Expected: Better generalization
```

**Scenario B: Cross-Dataset Evaluation Matrix**
```
Training Dataset → Test Dataset

ISIC2019        → HAM10000   (Acc: ?)
HAM10000        → ISIC2019   (Acc: ?)
ISIC2019+HAM    → ISIC2020   (Acc: ?)
```

---

## 4. Datasets Overview

### ISIC2019 (Primary Training)
- **Images:** 25,331
- **Classes:** 8 (MEL, NV, BCC, AK, BKL, DF, VASC, SCC)
- **Source:** International Skin Imaging Collaboration
- **Year:** 2019
- **Status:**  Downloaded and validated

### HAM10000 (External Validation)
- **Images:** 10,015
- **Classes:** 7 (akiec, bcc, bkl, df, mel, nv, vasc)
- **Source:** Hospital in Austria
- **Year:** 2018
- **Status:**  Downloaded and available
- **Note:** Missing SCC class (only in ISIC2019)

### ISIC2020 (Optional - Future Validation)
- **Images:** 33,126
- **Classes:** 8 classes
- **Source:** ISIC Challenge 2020
- **Year:** 2020
- **Status:**  Not downloaded yet
- **Download:** Can be automated

---

## 5. Class Mapping (Important!)

### ISIC2019 Classes (8 total)
```
MEL  (Melanoma)
NV   (Melanocytic Nevus)
BCC  (Basal Cell Carcinoma)
AK   (Actinic Keratosis)
BKL  (Benign Keratosis)
DF   (Dermatofibroma)
VASC (Vascular Lesion)
SCC  (Squamous Cell Carcinoma)  ← Only in ISIC2019
```

### HAM10000 Classes (7 total)
```
mel   (Melanoma)                 → Maps to MEL
nv    (Melanocytic Nevus)        → Maps to NV
bcc   (Basal Cell Carcinoma)     → Maps to BCC
akiec (Actinic Keratoses)        → Maps to AK
bkl   (Benign Keratosis)         → Maps to BKL
df    (Dermatofibroma)           → Maps to DF
vasc  (Vascular Lesion)          → Maps to VASC
```

**Handling Strategy:**
- **Option 1:** Exclude SCC class when testing on HAM10000 (7-class evaluation)
- **Option 2:** Train 7-class model for fair comparison
- **Recommended:** Report both 8-class (ISIC2019) and 7-class (cross-dataset)

---

## 6. Evaluation Metrics (Comprehensive)

### Primary Metrics
1. **Accuracy** - Overall classification accuracy
2. **Balanced Accuracy** - Account for class imbalance
3. **AUC-ROC** - Area under ROC curve (per-class and macro/micro average)
4. **F1-Score** - Harmonic mean of precision and recall

### Secondary Metrics
5. **Sensitivity (Recall)** - Especially for malignant classes (MEL, BCC, SCC)
6. **Specificity** - True negative rate
7. **Precision** - Positive predictive value
8. **Cohen's Kappa** - Inter-rater agreement

### Clinical Metrics
9. **Malignant vs Benign** - Binary classification (most important clinically)
10. **Top-2 Accuracy** - Model's top 2 predictions include correct class
11. **Calibration** - Reliability of predicted probabilities

### Per-Class Analysis
- Confusion matrix
- Per-class precision, recall, F1
- Class-specific ROC curves

---

## 7. Expected Performance Ranges

### Internal Validation (10-Fold CV on ISIC2019)
Based on current training results:
```
Model              | Expected Accuracy (Mean ± Std)

Swin Transformer   | 90-92% ± 2-3%
ViT                | 87-89% ± 2-3%
DenseNet           | 86-88% ± 2-3%
ResNet50           | 85-87% ± 2-3%
EfficientNet       | 80-83% ± 2-3%
```

### External Validation (HAM10000)
Expected performance drop: **5-15%** (normal!)
```
Model              | Expected Accuracy on HAM10000

Swin Transformer   | 78-85%
ViT                | 75-82%
DenseNet           | 73-80%
ResNet50           | 72-79%
EfficientNet       | 68-75%
```

**Why Lower?**
- Different imaging devices
- Different patient population
- Dataset distribution shift
- This is **expected and proves we're testing properly!**

---

## 8. Reproducibility Requirements

### Data Split Versioning
```python
# Save exact train/val/test splits for reproducibility
splits = {
    'isic2019_fold_1_train_indices': [...],
    'isic2019_fold_1_val_indices': [...],
    'ham10000_test_indices': [...],
    'random_seed': 42,
    'split_date': '2025-12-23',
}
save_json(splits, 'data_splits_v1.json')
```

### Configuration Tracking
- Model architecture details
- Hyperparameters
- Training configuration
- Hardware used
- Software versions (PyTorch, CUDA, etc.)

### Version Control
- Git commit hash for code version
- Dataset versions
- Pre-trained weights source

---

## 9. Statistical Significance Testing

### Comparing Models
```python
# McNemar's test for paired classifiers
# Requires same test set for both models
p_value = mcnemar_test(predictions_modelA, predictions_modelB, ground_truth)

# Report: "Model A significantly outperforms Model B (p < 0.05)"
```

### Confidence Intervals
- Bootstrap confidence intervals (1000 iterations)
- Report: Accuracy = 91.2% (95% CI: 89.8-92.6%)

---

## 10. Documentation for Publication

### Required Reporting (TRIPOD-AI Guidelines)

#### Title & Abstract
- Clear statement of external validation
- Multiple datasets used

#### Methods Section
```markdown
Dataset Split Strategy:
- Internal validation: 10-fold stratified CV on ISIC2019 (n=25,331)
- External validation: HAM10000 (n=10,015)
- Stratification: Maintained class distribution in all folds
- Random seed: 42 (for reproducibility)
```

#### Results Section
```markdown
Model Performance:
- Internal (ISIC2019 CV): 91.2% ± 2.3%
- External (HAM10000): 82.4% (95% CI: 81.2-83.6%)
- Performance drop: 8.8% (expected for domain shift)
```

#### Discussion
- Acknowledge performance drop on external data
- Discuss domain shift and generalization
- Compare with literature (similar drops reported)

---

## 11. Implementation Timeline

### Week 1: Internal Validation
- [ ] Implement 10-fold stratified CV
- [ ] Retrain all 5 models with CV
- [ ] Generate per-fold metrics
- [ ] Statistical analysis

### Week 2: External Validation
- [ ] Prepare HAM10000 dataset
- [ ] Map classes correctly (7-class vs 8-class)
- [ ] Test all models on HAM10000
- [ ] Analyze performance drop

### Week 3: Multi-Dataset Training
- [ ] Train on ISIC2019 → Test on HAM10000
- [ ] Train on HAM10000 → Test on ISIC2019
- [ ] Combined training (optional)
- [ ] Cross-dataset comparison matrix

### Week 4: Documentation & Reporting
- [ ] Generate all figures and tables
- [ ] Statistical significance tests
- [ ] Write comprehensive results report
- [ ] Create publication-ready materials

---

## 12. Tools & Scripts to Create

### Data Management
- `prepare_validation_splits.py` - Create reproducible splits
- `verify_data_splits.py` - Validate no data leakage
- `class_mapper.py` - Handle ISIC2019 ↔ HAM10000 mapping

### Training
- `train_kfold_cv_robust.py` - 10-fold CV with proper tracking
- `train_external_validation.py` - Train on one, test on another
- `train_combined_datasets.py` - Multi-dataset training

### Evaluation
- `evaluate_external.py` - External validation metrics
- `statistical_tests.py` - Significance testing
- `calibration_analysis.py` - Probability calibration

### Reporting
- `generate_validation_report.py` - Automated reporting
- `create_publication_figures.py` - Publication-quality plots
- `export_results_table.py` - LaTeX/CSV tables

---

## 13. Success Criteria

### Minimum Requirements (For Publication)
 10-fold CV on primary dataset (ISIC2019)
 External validation on ≥1 dataset (HAM10000)
 Performance metrics with confidence intervals
 Statistical comparison between models
 Documented data splits (reproducible)
 Class-specific performance analysis

### Gold Standard (Top-Tier Publication)
 All minimum requirements
 External validation on ≥2 datasets (HAM10000 + ISIC2020)
 Cross-dataset training experiments
 Subgroup analysis (age, sex, lesion location)
 Calibration analysis
 Comparison with dermatologist performance
 Error analysis and failure cases

---

## 14. Key Takeaways

### What Makes This Robust?

1. **Multiple Validation Tiers**
   - Internal CV for model selection
   - External validation for generalization proof
   - Cross-dataset for maximum robustness

2. **Proper Statistical Analysis**
   - Confidence intervals
   - Significance testing
   - Per-fold variance reporting

3. **Reproducibility**
   - Fixed random seeds
   - Saved data splits
   - Version control

4. **Publication Standards**
   - Follows TRIPOD-AI guidelines
   - Follows CLAIM checklist
   - Comprehensive metrics

5. **Future Reference**
   - Clear documentation
   - Reusable scripts
   - Extensible to new datasets

### What This Enables

 **Publication in top journals** (external validation required)
 **Clinical credibility** (proven generalization)
 **Future work baseline** (reproducible benchmarks)
 **Model comparison** (fair evaluation)
 **Real-world deployment confidence** (tested on multiple sources)

---

## 15. Next Steps

**Immediate Action Items:**

1. **Confirm Strategy**
   - Review this document
   - Approve validation approach
   - Decide on additional datasets (ISIC2020?)

2. **Prepare Data**
   - Verify HAM10000 is complete
   - Create class mapping
   - Generate reproducible splits

3. **Implement Training**
   - Start with 10-fold CV on ISIC2019
   - Then external validation on HAM10000
   - Finally cross-dataset experiments

4. **Document Everything**
   - Track all experiments
   - Save all metrics
   - Generate reports

---

**Ready to implement? Which phase should we start with?**

1. Phase 1: 10-Fold CV on ISIC2019 (1 week)
2. Phase 2: External validation on HAM10000 (3 days)
3. Phase 3: Multi-dataset training (1 week)
4. All of the above (comprehensive - 3 weeks)

This framework will make your work **publication-ready and future-proof**!
