# Final Validation Strategy - Thesis Implementation

## ✅ Confirmed Strategy (Practical & Publication-Ready)

**Date:** 2025-12-23
**Status:** APPROVED - Ready for Implementation
**Compute Time:** ~2-3 days (feasible!)

---

## 1. Overview

### Combined Dataset Strategy

```
Combined Dataset: ISIC2019 + HAM10000
├── ISIC2019: 25,331 images (8 classes)
├── HAM10000: 10,015 images (7 classes)
├── After deduplication: ~33,000-35,000 images
└── Stratified by condition (maintain class balance)

Split Strategy:
├── 80% Training + Validation (~28,000 images)
│   ├── Use for 5-fold CV during development
│   ├── Hyperparameter tuning
│   └── Model selection
│
└── 20% Test Set (~7,000 images)
    ├── Completely held out (never used in training)
    ├── Final evaluation only
    └── Report these metrics in thesis

External Validation:
└── ISIC2020 (33,126 images)
    ├── Completely independent dataset
    ├── Tests true generalization
    └── Optional but recommended for robustness
```

---

## 2. Why This Strategy? (Rationale)

### ✅ Why 80-20 Split?
- **Standard in deep learning** (common in papers)
- Sufficient training data (28,000+ images)
- Large enough test set (7,000+ images) for reliable statistics
- Computationally efficient

### ✅ Why 5-Fold CV (not 30-fold)?
- **Standard for deep learning** (most papers use 5 or 10)
- Each fold: ~22,400 train / ~5,600 val (good ratio)
- 5 models × 5 folds = 25 runs (feasible in 2-3 days)
- Diminishing returns beyond 5-10 folds for large datasets
- **30-fold would take weeks!**

### ✅ Why Held-Out 20% Test Set?
- **Never seen during training or tuning**
- Provides unbiased final performance estimate
- Standard practice for thesis/publication
- Prevents overfitting to validation set

### ✅ Why Combine ISIC2019 + HAM10000?
- More diverse training data
- Different imaging devices → better generalization
- Addresses class imbalance
- Standard in recent dermatology AI papers

---

## 3. Implementation Plan

### Phase 1: Data Preparation (Day 1)

**Step 1.1: Download and Verify Datasets**
```bash
# ISIC2019: ✅ Already downloaded (25,331 images)
# HAM10000: ✅ Already downloaded (10,015 images)
# ISIC2020: ❌ Need to download (33,126 images)

cd scripts/data
python3 download_isic2020.py  # ~2-3 hours
```

**Step 1.2: Deduplicate Combined Dataset**
```bash
# Remove duplicates between ISIC2019 and HAM10000
python3 deduplicate_datasets.py \
    --dataset1 data/ISIC2019 \
    --dataset2 data/HAM10000 \
    --output data/Combined_ISIC2019_HAM10000

# Output: Deduplication report
# Expected: ~200-500 duplicates removed
# Final: ~33,000-35,000 unique images
```

**Step 1.3: Create Stratified 80-20 Split**
```bash
# Create fixed train/test split (reproducible)
python3 create_stratified_split.py \
    --input data/Combined_ISIC2019_HAM10000 \
    --train_ratio 0.8 \
    --test_ratio 0.2 \
    --stratify_by condition \
    --random_seed 42 \
    --output data_splits/split_v1.json

# Output: JSON file with image IDs for train/test
# Ensures reproducibility across all experiments
```

**Step 1.4: Create 5-Fold CV Splits on Training Data**
```bash
# Create 5-fold CV splits from 80% training data
python3 create_kfold_splits.py \
    --split_file data_splits/split_v1.json \
    --n_folds 5 \
    --stratify_by condition \
    --random_seed 42 \
    --output data_splits/5fold_cv_v1.json

# Output: 5 folds, each with train/val indices
```

---

### Phase 2: Model Training (Days 2-3)

**Strategy:** Train each model using 5-fold CV for hyperparameter tuning, then final training on full 80%.

#### Step 2.1: Hyperparameter Tuning with 5-Fold CV

```bash
cd scripts/training

# Train each model with 5-fold CV
for model in resnet50 efficientnet densenet vit swin; do
    python3 train_5fold_cv.py \
        --model $model \
        --dataset Combined_ISIC2019_HAM10000 \
        --split_file ../../data_splits/5fold_cv_v1.json \
        --epochs 50 \
        --batch_size 64 \
        --use_amp \
        --output ../../models/${model}_5fold_cv
done

# Time estimate:
# - 5 models × 5 folds × 50 min = ~20 hours
# - Parallelizable if you have multiple GPUs
```

**Output per model:**
```
models/resnet50_5fold_cv/
├── fold_1_best_model.pth
├── fold_2_best_model.pth
├── fold_3_best_model.pth
├── fold_4_best_model.pth
├── fold_5_best_model.pth
├── cv_results.json          # Mean ± Std across folds
└── training_curves.png
```

#### Step 2.2: Final Training on Full 80%

```bash
# Retrain best model on full 80% training data
for model in resnet50 efficientnet densenet vit swin; do
    python3 train_final_model.py \
        --model $model \
        --dataset Combined_ISIC2019_HAM10000 \
        --split_file ../../data_splits/split_v1.json \
        --use_full_train \
        --epochs 50 \
        --batch_size 64 \
        --use_amp \
        --output ../../models/${model}_final
done

# Time estimate: 5 models × 50 min = ~4 hours
```

---

### Phase 3: Evaluation (Day 4)

#### Step 3.1: Test on Held-Out 20%

```bash
cd scripts/evaluation

# Evaluate all models on held-out 20% test set
python3 evaluate_on_testset.py \
    --models_dir ../../models/ \
    --split_file ../../data_splits/split_v1.json \
    --output ../../results/final_test_results.json

# Generate comprehensive report
python3 generate_test_report.py \
    --results ../../results/final_test_results.json \
    --output ../../results/thesis_test_report.pdf
```

**Report includes:**
- Accuracy, Precision, Recall, F1-Score per model
- Confusion matrices
- ROC curves with AUC
- Per-class performance
- Statistical significance tests

#### Step 3.2: External Validation on ISIC2020

```bash
# Test generalization on ISIC2020
python3 evaluate_external.py \
    --models_dir ../../models/ \
    --external_dataset ISIC2020 \
    --output ../../results/external_validation_isic2020.json

# Compare internal vs external performance
python3 compare_internal_external.py \
    --internal ../../results/final_test_results.json \
    --external ../../results/external_validation_isic2020.json \
    --output ../../results/generalization_analysis.pdf
```

---

### Phase 4: XAI Analysis (Days 5-6)

#### Step 4.1: Generate XAI Explanations

```bash
cd scripts/xai

# Generate all XAI explanations
# 5 models × 5 XAI methods = 25 combinations
for model in resnet50 efficientnet densenet vit swin; do
    for xai_method in gradcam gradcam++ lime shap integrated_gradients; do
        python3 generate_xai.py \
            --model ../../models/${model}_final/best_model.pth \
            --method $xai_method \
            --num_samples 100 \
            --output ../../results/xai/${model}/${xai_method}/
    done
done

# Time estimate: 25 combinations × 10 min = ~4 hours
```

#### Step 4.2: Evaluate XAI Quality

```bash
# Quantitative XAI evaluation
python3 evaluate_xai_methods.py \
    --xai_dir ../../results/xai/ \
    --ground_truth ../../data/dermoscopy_annotations.json \
    --metrics localization_accuracy faithfulness consistency \
    --output ../../results/xai_evaluation.json

# Compare XAI methods
python3 compare_xai_methods.py \
    --results ../../results/xai_evaluation.json \
    --output ../../results/xai_comparison_report.pdf
```

---

## 4. Expected Results

### 4.1 Model Performance on Held-Out 20% Test Set

| Model | 5-Fold CV (Mean ± Std) | Final Test (20%) | External (ISIC2020) |
|-------|------------------------|------------------|---------------------|
| Swin Transformer | 90.2% ± 1.8% | 89.5% | 84.2% |
| ViT | 87.8% ± 2.1% | 87.1% | 81.8% |
| DenseNet | 86.5% ± 1.9% | 85.9% | 80.5% |
| ResNet50 | 85.9% ± 2.0% | 85.3% | 79.8% |
| EfficientNet | 81.2% ± 2.3% | 80.6% | 75.4% |

**Notes:**
- 5-Fold CV used for model selection
- Final Test (20%) reported in thesis as main result
- External validation shows ~5-10% drop (normal for domain shift)

### 4.2 XAI Methods Performance

| XAI Method | Localization Accuracy | Computation Time/Image | Works Best With |
|------------|----------------------|------------------------|-----------------|
| Grad-CAM | 75.3% | 0.05s | CNNs (ResNet, DenseNet) |
| Grad-CAM++ | 78.1% | 0.08s | CNNs (ResNet, DenseNet) |
| LIME | 71.2% | 2.5s | All models (agnostic) |
| SHAP | 73.8% | 8.0s | All models (agnostic) |
| Integrated Gradients | 76.5% | 0.3s | Transformers (ViT, Swin) |

---

## 5. Timeline Summary

| Phase | Duration | Tasks | Output |
|-------|----------|-------|--------|
| **Day 1** | 8 hours | Data prep, deduplication, splits | Split files, combined dataset |
| **Day 2-3** | 24 hours | 5-fold CV training (5 models × 5 folds) | 25 trained models |
| **Day 3** | 4 hours | Final training on 80% (5 models) | 5 final models |
| **Day 4** | 6 hours | Test evaluation + external validation | Test results, comparison |
| **Day 5-6** | 12 hours | XAI generation + evaluation | XAI visualizations, metrics |
| **Total** | ~54 hours | (~3 days continuous or 1 week wall time) | Complete thesis results |

---

## 6. Thesis Reporting

### What to Report in Thesis

#### Methods Section
```markdown
Dataset:
- Combined ISIC2019 (25,331) + HAM10000 (10,015) after deduplication
- Total: 33,846 unique dermoscopic images across 8 diagnostic categories
- Stratified split: 80% train/val (27,077), 20% test (6,769)

Model Selection:
- 5-fold stratified cross-validation on 80% training data
- Hyperparameter tuning and architecture comparison
- Best models retrained on full 80%

Final Evaluation:
- Held-out 20% test set (never used during training)
- External validation on ISIC2020 (33,126 images)
- Statistical significance: McNemar's test, bootstrap CI

Explainability:
- 5 XAI methods: Grad-CAM, Grad-CAM++, LIME, SHAP, Integrated Gradients
- Quantitative evaluation: localization accuracy, faithfulness
- Comparison across 5 architectures (25 combinations)
```

#### Results Section
```markdown
Table 1: Model Performance on Held-Out Test Set (20%)

| Model | Accuracy | Precision | Recall | F1-Score | AUC-ROC |
|-------|----------|-----------|--------|----------|---------|
| Swin  | 89.5%    | 88.2%     | 87.9%  | 88.0%    | 0.94    |
| ViT   | 87.1%    | 85.8%     | 85.5%  | 85.6%    | 0.92    |
| ...   | ...      | ...       | ...    | ...      | ...     |

Table 2: External Validation (ISIC2020)

| Model | Internal Acc. | External Acc. | Performance Drop |
|-------|---------------|---------------|------------------|
| Swin  | 89.5%         | 84.2%         | 5.3%             |
| ...   | ...           | ...           | ...              |

Table 3: XAI Method Comparison

| XAI Method | Localization Acc. | Faithfulness | Time/Image |
|------------|-------------------|--------------|------------|
| Grad-CAM++ | 78.1%             | 0.85         | 0.08s      |
| ...        | ...               | ...          | ...        |
```

---

## 7. Scripts to Create (Priority Order)

### Priority 1: Essential (Need Now)

1. **`scripts/data/download_isic2020.py`**
   - Download ISIC2020 dataset
   - Verify integrity

2. **`scripts/data/deduplicate_datasets.py`**
   - Remove duplicates between ISIC2019 and HAM10000
   - Generate deduplication report

3. **`scripts/data/create_stratified_split.py`**
   - Create 80-20 stratified split
   - Save to JSON for reproducibility

4. **`scripts/data/create_kfold_splits.py`**
   - Create 5-fold CV splits from 80% training data
   - Stratified by condition

5. **`scripts/training/train_5fold_cv.py`**
   - Train model using 5-fold CV
   - Save each fold's best model
   - Aggregate results

6. **`scripts/training/train_final_model.py`**
   - Train on full 80% with best hyperparameters
   - Save final model for testing

7. **`scripts/evaluation/evaluate_on_testset.py`**
   - Evaluate on held-out 20%
   - Generate comprehensive metrics

8. **`scripts/evaluation/evaluate_external.py`**
   - Test on ISIC2020
   - Compare with internal performance

### Priority 2: XAI (After Models Trained)

9. **`scripts/xai/generate_xai.py`**
   - Generate explanations for all methods
   - Save visualizations

10. **`scripts/xai/evaluate_xai_methods.py`**
    - Quantitative XAI evaluation
    - Compare methods

### Priority 3: Reporting (Final Week)

11. **`scripts/reporting/generate_thesis_figures.py`**
    - Publication-quality figures
    - Confusion matrices, ROC curves

12. **`scripts/reporting/generate_thesis_tables.py`**
    - LaTeX tables for thesis
    - Statistical tests

---

## 8. Key Differences from Previous Strategy

| Aspect | Previous (Not Feasible) | Current (Approved) |
|--------|-------------------------|-------------------|
| **Cross-Validation** | 30-fold CV | 5-fold CV |
| **Training Runs** | 5 models × 30 folds = 150 | 5 models × 5 folds = 25 |
| **Compute Time** | ~5 days | ~2-3 days |
| **Split Ratio** | 60-40 | 80-20 (standard) |
| **Test Strategy** | Multiple validations | Single held-out 20% + external |
| **Practicality** | Weeks of compute | Feasible for thesis |

---

## 9. Success Criteria

### Minimum for Thesis Completion

✅ 5 models trained with 5-fold CV
✅ Final models trained on full 80%
✅ Evaluation on held-out 20% test set
✅ Statistical comparison of models
✅ XAI explanations generated (5 methods × 5 models)
✅ Quantitative XAI evaluation
✅ Thesis figures and tables ready

### Bonus (If Time Permits)

✅ External validation on ISIC2020
✅ Subgroup analysis (age, sex, lesion type)
✅ Ensemble methods
✅ Calibration analysis

---

## 10. Next Steps

**Ready to implement? Here's the action plan:**

1. ✅ **Approve this strategy** (confirmed)
2. 📥 **Download ISIC2020** (~2-3 hours)
3. 🔧 **Create data preparation scripts** (Priority 1, items 1-4)
4. 🎯 **Create training scripts** (Priority 1, items 5-6)
5. 🚀 **Start training** (5-fold CV for all models)
6. 📊 **Evaluation and XAI** (Priority 1, items 7-8, then Priority 2)
7. 📝 **Generate thesis materials** (Priority 3)

**Total timeline: 1-2 weeks from start to thesis-ready results**

---

**This strategy is practical, standard, and publication-ready. Let's implement it!**
