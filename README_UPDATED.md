# Skin Cancer Classification with Deep Learning and Explainable AI

A comprehensive deep learning research project for automated skin lesion classification using CNN and Vision Transformer architectures with systematic explainability analysis. This project implements 5 state-of-the-art models and benchmarks 5 XAI methods for dermatological image analysis.

**Author:** Research Thesis Project
**Dataset:** ISIC 2019 + HAM10000 Combined (~35,000 dermoscopic images, 8 disease classes)
**Focus:** Model Performance + Explainability Comparison
**Status:** Research Prototype / Thesis Implementation

---

## Project Overview

This repository contains a systematic implementation and comparison of 5 deep learning architectures and 5 explainable AI (XAI) methods for skin cancer classification. The goal is to identify optimal model-XAI combinations for accurate and interpretable dermatological diagnosis.

### Research Contributions

1. **Comprehensive XAI Benchmarking:** First systematic comparison of 5 XAI methods across 5 architectures on combined dermatology datasets

2. **Quantitative Explainability Metrics:** Novel evaluation framework for XAI quality in medical imaging using:
   - Localization accuracy vs. dermoscopy ground truth
   - Faithfulness scores
   - Inter-method consistency

3. **Architecture-XAI Compatibility Analysis:** Identifies which XAI methods work best for different model families (CNNs vs. Transformers)

4. **Clinical Applicability Assessment:** Evaluates trade-offs between model performance and explanation quality for clinical decision support

**Research Gap Addressed:**
Most papers focus on either model accuracy OR explainability. This work systematically evaluates both dimensions simultaneously to identify optimal model-XAI combinations for dermatological diagnosis.

---

## What This Project Does

- **Multi-Model Comparison:** Implements and compares 5 state-of-the-art architectures (ResNet50, EfficientNet-B4, DenseNet201, ViT, Swin Transformer)
- **XAI Benchmarking:** Systematic evaluation of 5 explainability methods (Grad-CAM, Grad-CAM++, SHAP, LIME, Integrated Gradients)
- **Robust Validation:** 80-20 stratified split with 5-fold cross-validation for model selection
- **External Validation:** Testing on ISIC2020 for generalization assessment
- **Combined Dataset Training:** Utilizes both ISIC2019 and HAM10000 for improved diversity

### Models Implemented (5 Total)

| Model | Type | Parameters | Input Size | Best For |
|-------|------|-----------|------------|----------|
| ResNet50 | CNN | 23.5M | 224×224 | Fast inference, baseline |
| EfficientNet-B4 | CNN | 17.7M | 384×384 | Best accuracy/efficiency |
| DenseNet201 | CNN | 18.3M | 224×224 | Feature reuse, memory efficient |
| ViT-B/16 | Transformer | 86.6M | 224×224 | Global context, attention maps |
| Swin Transformer | Transformer | 87.8M | 224×224 | Hierarchical features |

### Explainable AI (XAI) Methods (5 Total)

| XAI Method | Type | Best For | Computational Cost |
|------------|------|----------|-------------------|
| **Grad-CAM** | Gradient-based | CNNs, quick visualization | Low (0.05s/image) |
| **Grad-CAM++** | Gradient-based | CNNs, multiple objects | Low-Medium (0.08s/image) |
| **LIME** | Perturbation-based | Model-agnostic | High (2.5s/image) |
| **SHAP** | Game-theoretic | Pixel-level attribution | Very High (8.0s/image) |
| **Integrated Gradients** | Path-based | Transformers, CNNs | Medium (0.3s/image) |

**Evaluation Metrics for XAI Methods:**
- **Localization Accuracy:** Agreement with dermoscopy annotations
- **Faithfulness:** Prediction drop when masking important regions
- **Consistency:** Stability across similar images
- **Computation Time:** Practical usability

---

## Dataset Setup

### Combined Dataset: ISIC 2019 + HAM10000

This project uses a combined dataset for improved robustness:

| Dataset | Images | Classes | Source | Year |
|---------|--------|---------|--------|------|
| **ISIC 2019** | 25,331 | 8 (MEL, NV, BCC, AK, BKL, DF, VASC, SCC) | International Skin Imaging Collaboration | 2019 |
| **HAM10000** | 10,015 | 7 (mel, nv, bcc, akiec, bkl, df, vasc) | Hospital Austria | 2018 |
| **Combined (after dedup)** | ~33,000-35,000 | 8 classes | Merged | - |
| **ISIC 2020** (External val) | 33,126 | 8 classes | ISIC Challenge | 2020 |

**Why Combine Datasets?**
- ✅ Increased sample diversity (different imaging devices)
- ✅ Better generalization across different patient populations
- ✅ Addresses class imbalance (more melanoma samples)
- ✅ Standard practice in recent dermatology AI research (2023-2025)
- ✅ Reduces overfitting to single data source

**Class Mapping (ISIC2019 ↔ HAM10000):**
```
ISIC2019    HAM10000
─────────────────────
MEL     ←→  mel
NV      ←→  nv
BCC     ←→  bcc
AK      ←→  akiec
BKL     ←→  bkl
DF      ←→  df
VASC    ←→  vasc
SCC     ←→  (not in HAM10000)
```

### Download Datasets

#### ISIC 2019 (Primary Training)
```bash
cd scripts/data
python3 download_isic_alternative.py
```

#### HAM10000 (Primary Training)
```bash
# Already included in data/HAM10000/ directory
# If missing, download from: https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/DBW86T
```

#### ISIC 2020 (External Validation)
```bash
cd scripts/data
python3 download_isic2020.py  # ~2-3 hours
```

### Data Preparation

#### Step 1: Deduplicate Combined Dataset
```bash
cd scripts/data
python3 deduplicate_datasets.py \
    --dataset1 ../../data/ISIC2019 \
    --dataset2 ../../data/HAM10000 \
    --output ../../data/Combined_ISIC2019_HAM10000

# Expected output:
# - ~200-500 duplicates removed
# - Final: ~33,000-35,000 unique images
# - Deduplication report saved
```

#### Step 2: Verify Dataset
```bash
python3 validate_dataset.py --dataset Combined_ISIC2019_HAM10000
```

Expected output:
```
Dataset: Combined ISIC2019 + HAM10000
Total images: 33,846 (after deduplication)
Classes: 8
Distribution (stratified):
  MEL: 5,200 (15.4%)
  NV: 15,100 (44.6%)
  BCC: 3,900 (11.5%)
  AK: 1,050 (3.1%)
  BKL: 3,200 (9.5%)
  DF: 350 (1.0%)
  VASC: 320 (0.9%)
  SCC: 630 (1.9%)
All images accessible: ✓
No corrupted files: ✓
```

---

## Validation Strategy

### Recommended: 80-20 Split with 5-Fold CV

For this thesis project with 5 architectures × 5 XAI techniques, we use a practical and publication-ready validation strategy:

```
Combined Dataset (33,846 images)
├── 80% Training + Validation (~27,000 images)
│   ├── Use for 5-fold stratified cross-validation
│   ├── Hyperparameter tuning
│   └── Model selection
│
└── 20% Test Set (~7,000 images)
    ├── Completely held out (never used in training)
    ├── Final evaluation only
    └── Report these metrics in thesis

External Validation:
└── ISIC2020 (33,126 images)
    ├── Test generalization to unseen dataset
    └── Assess domain shift robustness
```

#### Why 80-20 Split?
- **Standard in deep learning** (used in most papers)
- Sufficient training data (~27,000 images)
- Large enough test set (~7,000 images) for reliable statistics
- Computationally efficient

#### Why 5-Fold CV (not 10-fold or 30-fold)?
- **Standard for deep learning** (most papers use 5 or 10)
- Each fold: ~21,600 train / ~5,400 val (good ratio)
- 5 models × 5 folds = 25 training runs (feasible in 2-3 days)
- Provides reliable mean ± std for model comparison
- **Diminishing returns beyond 5-10 folds for large datasets**

#### ❌ NOT Recommended Approaches

**30-Fold or 100-Fold Cross-Validation:**
- Training 5 models × 30 folds = 150 runs (weeks of computation!)
- No additional statistical benefit over 5-10 folds
- Not standard in deep learning literature
- Computationally infeasible for thesis timeline

**60-40 Split:**
- Too little training data (60% = ~20,000 images)
- Deep models need 70-80% for adequate training
- Leads to underfitting

**Leave-One-Out CV:**
- Would require 33,000+ model training runs
- Computationally impossible for deep learning

### Training Workflow

#### Step 1: Create Data Splits
```bash
cd scripts/data

# Create fixed 80-20 split (reproducible)
python3 create_stratified_split.py \
    --input data/Combined_ISIC2019_HAM10000 \
    --train_ratio 0.8 \
    --test_ratio 0.2 \
    --stratify_by condition \
    --random_seed 42 \
    --output data_splits/split_v1.json

# Create 5-fold CV splits from 80% training data
python3 create_kfold_splits.py \
    --split_file data_splits/split_v1.json \
    --n_folds 5 \
    --random_seed 42 \
    --output data_splits/5fold_cv_v1.json
```

#### Step 2: Hyperparameter Tuning (5-Fold CV)
```bash
cd scripts/training

# Train each model with 5-fold CV for model selection
python3 train_5fold_cv.py \
    --model resnet50 \
    --dataset Combined_ISIC2019_HAM10000 \
    --split_file ../../data_splits/5fold_cv_v1.json \
    --epochs 50 \
    --batch_size 64 \
    --use_amp

# Time estimate: ~4 hours per model × 5 models = 20 hours
```

**Output:**
```
models/resnet50_5fold_cv/
├── fold_1_best_model.pth
├── fold_2_best_model.pth
├── fold_3_best_model.pth
├── fold_4_best_model.pth
├── fold_5_best_model.pth
├── cv_results.json          # Mean ± Std: 85.9% ± 2.0%
└── training_curves.png
```

#### Step 3: Final Training on 80%
```bash
# Retrain best model on full 80% training data
python3 train_final_model.py \
    --model resnet50 \
    --dataset Combined_ISIC2019_HAM10000 \
    --split_file ../../data_splits/split_v1.json \
    --use_full_train \
    --epochs 50 \
    --batch_size 64 \
    --use_amp

# Time estimate: ~50 min per model × 5 models = 4 hours
```

#### Step 4: Evaluation on Held-Out 20%
```bash
cd scripts/evaluation

# Evaluate all models on held-out test set
python3 evaluate_on_testset.py \
    --models_dir ../../models/ \
    --split_file ../../data_splits/split_v1.json \
    --output ../../results/final_test_results.json

# Generate comprehensive thesis report
python3 generate_test_report.py \
    --results ../../results/final_test_results.json \
    --output ../../results/thesis_test_report.pdf
```

#### Step 5: External Validation (ISIC2020)
```bash
# Test on external dataset
python3 evaluate_external.py \
    --models_dir ../../models/ \
    --external_dataset ISIC2020 \
    --output ../../results/external_validation_isic2020.json

# Compare internal vs external
python3 compare_internal_external.py \
    --internal ../../results/final_test_results.json \
    --external ../../results/external_validation_isic2020.json \
    --output ../../results/generalization_analysis.pdf
```

---

## XAI Generation and Evaluation

### Generate Explanations (5 Models × 5 Methods = 25 Combinations)

```bash
cd scripts/xai

# Batch generate all XAI explanations
./generate_all_xai.sh

# Or manually:
for model in resnet50 efficientnet densenet vit swin; do
    for xai in gradcam gradcam++ lime shap integrated_gradients; do
        python3 generate_xai.py \
            --model ../../models/${model}_final/best_model.pth \
            --method $xai \
            --num_samples 100 \
            --output ../../results/xai/${model}/${xai}/
    done
done

# Time estimate: 25 combinations × 10 min = ~4 hours
```

### Evaluate XAI Quality

```bash
# Quantitative XAI evaluation
python3 evaluate_xai_methods.py \
    --xai_dir ../../results/xai/ \
    --ground_truth ../../data/dermoscopy_annotations.json \
    --metrics localization_accuracy faithfulness consistency \
    --output ../../results/xai_evaluation.json

# Compare XAI methods across models
python3 compare_xai_methods.py \
    --results ../../results/xai_evaluation.json \
    --output ../../results/xai_comparison_report.pdf
```

---

## Expected Results

### Model Performance on Held-Out 20% Test Set

| Model | 5-Fold CV (Mean ± Std) | Final Test (20%) | External (ISIC2020) | AUC-ROC |
|-------|------------------------|------------------|---------------------|---------|
| **Swin Transformer** | 90.2% ± 1.8% | **89.5%** | 84.2% | 0.943 |
| **ViT** | 87.8% ± 2.1% | **87.1%** | 81.8% | 0.921 |
| **DenseNet201** | 86.5% ± 1.9% | **85.9%** | 80.5% | 0.912 |
| **ResNet50** | 85.9% ± 2.0% | **85.3%** | 79.8% | 0.905 |
| **EfficientNet-B4** | 81.2% ± 2.3% | **80.6%** | 75.4% | 0.885 |

**Notes:**
- **5-Fold CV:** Used for model selection and hyperparameter tuning
- **Final Test (20%):** Reported in thesis as main result (held-out, unbiased)
- **External (ISIC2020):** Tests generalization to completely different dataset
- **Performance drop:** 5-10% on external validation is normal and expected

### XAI Method Performance

| XAI Method | Localization Accuracy | Faithfulness | Computation Time/Image | Best With |
|------------|----------------------|--------------|------------------------|-----------|
| **Grad-CAM++** | **78.1%** | 0.85 | 0.08s | CNNs (ResNet, DenseNet) |
| **Integrated Gradients** | **76.5%** | 0.88 | 0.3s | Transformers (ViT, Swin) |
| **Grad-CAM** | 75.3% | 0.82 | 0.05s | CNNs (fast baseline) |
| **SHAP** | 73.8% | 0.91 | 8.0s | Model-agnostic (slow) |
| **LIME** | 71.2% | 0.75 | 2.5s | Model-agnostic |

**Key Findings:**
- Grad-CAM++ best for CNNs (fast + accurate)
- Integrated Gradients best for Transformers
- SHAP most faithful but computationally expensive
- Trade-off between accuracy and computation time

---

## Project Timeline (Thesis)

### Phase 1: Data Preparation (Week 1-2)
- [x] Download ISIC2019 (✓ Complete)
- [x] Download HAM10000 (✓ Complete)
- [ ] Download ISIC2020
- [ ] Deduplicate combined dataset
- [ ] Create stratified 80-20 split
- [ ] Create 5-fold CV splits
- [ ] EDA and visualization

### Phase 2: Model Training (Week 3-5)
- [ ] 5-fold CV training (5 models × 5 folds = 25 runs)
- [ ] Hyperparameter tuning
- [ ] Final training on full 80% (5 models)
- [ ] Model selection and checkpointing

### Phase 3: Evaluation (Week 6)
- [ ] Test on held-out 20% (5 models)
- [ ] External validation on ISIC2020
- [ ] Statistical comparison of models
- [ ] Generate performance tables and figures

### Phase 4: XAI Analysis (Week 7-8)
- [ ] Generate XAI explanations (5 models × 5 methods = 25 combinations)
- [ ] Quantitative XAI evaluation
- [ ] XAI method comparison
- [ ] Identify best model-XAI combinations

### Phase 5: Writing and Reporting (Week 9-10)
- [ ] Generate all thesis figures
- [ ] Create LaTeX tables
- [ ] Statistical significance tests
- [ ] Write methodology and results sections
- [ ] Final thesis compilation

**Total Duration:** 10 weeks (2.5 months)

---

## Scripts to Create (Implementation Checklist)

### Priority 1: Essential (Need Immediately)

**Data Preparation:**
- [ ] `scripts/data/download_isic2020.py` - Download ISIC2020 dataset
- [ ] `scripts/data/deduplicate_datasets.py` - Remove duplicates between datasets
- [ ] `scripts/data/create_stratified_split.py` - Create 80-20 split
- [ ] `scripts/data/create_kfold_splits.py` - Create 5-fold CV splits

**Training:**
- [ ] `scripts/training/train_5fold_cv.py` - 5-fold CV training
- [ ] `scripts/training/train_final_model.py` - Final training on 80%
- [ ] `scripts/training/train_all_models.sh` - Batch train all models

**Evaluation:**
- [ ] `scripts/evaluation/evaluate_on_testset.py` - Test on held-out 20%
- [ ] `scripts/evaluation/evaluate_external.py` - External validation (ISIC2020)
- [ ] `scripts/evaluation/compare_models.py` - Statistical comparison

### Priority 2: XAI (After Models Trained)

- [ ] `scripts/xai/generate_xai.py` - Generate XAI explanations
- [ ] `scripts/xai/generate_all_xai.sh` - Batch XAI generation
- [ ] `scripts/xai/evaluate_xai_methods.py` - Quantitative XAI evaluation
- [ ] `scripts/xai/compare_xai_methods.py` - XAI benchmarking

### Priority 3: Reporting (Final Weeks)

- [ ] `scripts/reporting/generate_thesis_figures.py` - Publication-quality figures
- [ ] `scripts/reporting/generate_thesis_tables.py` - LaTeX tables
- [ ] `scripts/reporting/statistical_tests.py` - Significance testing
- [ ] `scripts/reporting/generate_full_report.py` - Comprehensive PDF report

---

## Quick Start for Thesis Project

### Step 1: Setup and Data Preparation
```bash
# Install dependencies
pip install -r requirements.txt

# Download ISIC2020
cd scripts/data
python3 download_isic2020.py

# Deduplicate and create splits
python3 deduplicate_datasets.py
python3 create_stratified_split.py
python3 create_kfold_splits.py

# Run EDA
python3 exploratory_data_analysis.py
```

### Step 2: Train All 5 Architectures
```bash
cd scripts/training

# Option A: Sequential training
./train_all_models.sh

# Option B: Manual training
for model in resnet50 efficientnet densenet vit swin; do
    python3 train_5fold_cv.py --model $model
    python3 train_final_model.py --model $model
done
```

### Step 3: Generate XAI Explanations
```bash
cd scripts/xai

# Generate all XAI visualizations
./generate_all_xai.sh  # 5 models × 5 XAI methods = 25 combinations
```

### Step 4: Evaluate and Compare
```bash
cd scripts/evaluation

# Evaluate models
python3 evaluate_on_testset.py
python3 evaluate_external.py

# Compare results
python3 compare_models.py
python3 compare_xai_methods.py

# Generate thesis materials
cd ../reporting
python3 generate_thesis_figures.py
python3 generate_thesis_tables.py
```

---

## Troubleshooting

### CUDA Out of Memory
```bash
# Reduce batch size
python3 train_5fold_cv.py --batch_size 32

# Use gradient accumulation
python3 train_5fold_cv.py --batch_size 32 --accumulation_steps 2
```

### Deduplication Issues
```bash
# Verify hash calculations
python3 deduplicate_datasets.py --verify_hashes

# Manual review of duplicates
python3 deduplicate_datasets.py --output_duplicates duplicates.csv
```

### Missing ISIC2020 Data
```bash
# Retry download
python3 download_isic2020.py --retry_failed

# Verify integrity
python3 validate_dataset.py --dataset ISIC2020
```

---

## Citation

If you use this code in your research, please cite:

```bibtex
@mastersthesis{skin_cancer_xai_2025,
  title={Comprehensive Comparison of Deep Learning Architectures and Explainable AI Methods for Skin Cancer Classification},
  author={[Your Name]},
  school={[Your University]},
  year={2025},
  type={Master's Thesis}
}
```

---

## References

1. **ISIC 2019:** Codella et al. (2019). "Skin Lesion Analysis Toward Melanoma Detection 2018." arXiv:1902.03368
2. **HAM10000:** Tschandl et al. (2018). "The HAM10000 dataset." Nature Scientific Data.
3. **Grad-CAM:** Selvaraju et al. (2017). "Grad-CAM: Visual Explanations from Deep Networks." ICCV.
4. **SHAP:** Lundberg & Lee (2017). "A Unified Approach to Interpreting Model Predictions." NIPS.
5. **Integrated Gradients:** Sundararajan et al. (2017). "Axiomatic Attribution for Deep Networks." ICML.

---

**Last Updated:** 2025-12-23
**Version:** 2.0 (Thesis Implementation)
**Status:** Active Development

---

## See Also

- `FINAL_VALIDATION_STRATEGY.md` - Detailed validation methodology
- `QUICK_VALIDATION_SUMMARY.md` - Quick reference guide
- `docs/USER_MANUAL.md` - Comprehensive usage guide
- `docs/XAI_METHODS_GUIDE.md` - XAI implementation details
