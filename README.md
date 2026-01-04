# Skin Lesion Classification with Skin Tone Fairness: A Deep Learning Approach

A deep learning framework for **equitable skin cancer diagnosis** that addresses performance disparities across different skin tones. Based on the findings of Daneshjou et al. (2022) showing 10-20% worse AI performance on darker skin, this project implements **skin tone-aware training strategies** that not only eliminate but **reverse typical AI bias** patterns.

**Based on:** Daneshjou, R., et al. (2022). "Disparities in dermatology AI performance on a diverse, curated clinical image set." *Science Advances*, 8(32). [DOI](https://www.science.org/doi/10.1126/sciadv.abq6147)

**Tools/Frameworks:** PyTorch, ISIC Archive, timm, MILK10k

**Authors:** [Author Names with ORCID]
**Status:** Proof of Concept Complete
**Last Updated:** December 2025

---

## Key Results

| Metric | Swin Transformer | DenseNet201 | ResNet50 |
|--------|------------------|-------------|----------|
| **Balanced Accuracy** | **90.8%** | 89.8% | 90.3% |
| **Fairness Gap** | **6.7%** | 8.9% | 10.4% |
| **AUC** | **99.1%** | 98.9% | 98.9% |

**Key Finding:** Our models achieve **higher accuracy on darker skin tones** (95.4%) compared to lighter skin tones (88.8%), **reversing the typical bias pattern** observed in medical AI systems.

---

## Project Highlights

- **Fairness-Aware Training:** Skin-tone-aware stratified sampling ensures balanced representation
- **Combined Dataset:** 30,468 images (ISIC2019 + MILK10k with skin tone labels)
- **Statistical Validation:** Spearman correlation (p=0.037*) confirms bias reversal
- **Explainability:** Grad-CAM++ visualizations verify focus on lesion features
- **External Validation:** Tested on HAM10000 dataset

---

## Dataset

### Combined Dataset Statistics

| Metric | Value |
|--------|-------|
| **Total Samples** | 30,468 dermoscopic images |
| **ISIC2019** | 25,331 images |
| **MILK10k** | 5,137 images (with skin tone labels) |
| **Classes** | 8 diagnostic categories |
| **Imbalance Ratio** | 46.8:1 |

### Class Distribution

| Class | Full Name | Count | % |
|-------|-----------|-------|---|
| NV | Melanocytic Nevus | 13,621 | 44.7% |
| BCC | Basal Cell Carcinoma | 5,845 | 19.2% |
| MEL | Melanoma | 4,972 | 16.3% |
| BKL | Benign Keratosis | 3,168 | 10.4% |
| AK | Actinic Keratosis | 1,170 | 3.8% |
| SCC | Squamous Cell Carcinoma | 1,101 | 3.6% |
| VASC | Vascular Lesion | 300 | 1.0% |
| DF | Dermatofibroma | 291 | 1.0% |

### Skin Tone Distribution (Monk Scale 0-5)

| Tone | Description | Count | % |
|------|-------------|-------|---|
| 0 | Very Dark | 3 | 0.06% |
| 1 | Dark | 100 | 1.95% |
| 2 | Medium-Dark | 487 | 9.48% |
| 3 | Medium | 3,131 | 60.95% |
| 4 | Medium-Light | 1,050 | 20.44% |
| 5 | Light | 366 | 7.12% |

---

## Model Architectures

| Model | Parameters | Input Size | Batch Size | Training Time/Fold |
|-------|------------|------------|------------|-------------------|
| **Swin Transformer** | 88M | 384×384 | 16 | ~3 hours |
| DenseNet201 | 20M | 224×224 | 32 | ~1 hour |
| ResNet50 | 25M | 224×224 | 32 | ~35 min |

---

## Fairness Analysis

### Performance by Skin Tone

| Skin Tone | Swin | DenseNet | ResNet50 |
|-----------|------|----------|----------|
| Very Dark (0-1) | **95.4%** | 94.1% | 94.7% |
| Dark (2) | 93.7% | 94.9% | 95.1% |
| Medium (3) | 90.3% | 88.9% | 90.3% |
| Light (4) | 88.8% | 88.7% | 88.0% |
| Very Light (5) | 90.1% | 86.0% | 84.7% |

### Statistical Tests

| Test | Result | Significance |
|------|--------|--------------|
| Spearman Correlation | ρ = -0.90 (negative) | p = 0.037* |
| Mann-Whitney U (Dark vs Light) | Dark performs better | p = 0.006** |
| Cohen's d | 0.64 - 1.19 | Medium to Large effect |

**Interpretation:** All models show statistically significant better performance on darker skin tones, reversing typical AI bias.

---

## Project Structure

```
713_skin_lis_dermoscopic/
├── data/
│   ├── ISIC2019/                    # ISIC 2019 dataset
│   ├── MILK10k/                     # MILK10k with skin tone labels
│   ├── HAM10000/                    # External validation dataset
│   └── combined/
│       ├── master_metadata.csv      # 30,468 samples
│       ├── class_weights.csv        # Focal loss weights
│       └── dataset_summary.json     # Statistics
├── models/
│   ├── swin_fold5_*/                # Trained Swin models
│   ├── densenet_fold5_*/            # Trained DenseNet models
│   └── resnet50_fold5_*/            # Trained ResNet50 models
├── results/
│   ├── figures/                     # Publication figures (7 figures)
│   ├── fairness/                    # Fairness evaluation results
│   ├── xai/                         # Grad-CAM visualizations
│   ├── external_validation/         # HAM10000 results
│   └── statistical_tests/           # Statistical analysis
├── scripts/
│   ├── run_10fold_cv.sh             # 10-fold CV automation
│   ├── check_cv_progress.sh         # Progress monitoring
│   ├── aggregate_cv_results.py      # Results aggregation
│   └── evaluation/
│       └── evaluate_skin_tone_fairness.py
├── src/
│   ├── models.py                    # Model architectures
│   ├── skin_tone_aware_sampler.py   # Fairness-aware sampling
│   └── xai_methods.py               # Grad-CAM++
├── train_combined_optimized.py      # Main training script
├── PROOF_OF_CONCEPT.md              # PoC documentation
├── PROJECT_REPORT.md                # Full project report
└── 10_FOLD_CV_GUIDE.md              # CV training guide
```

---

## Quick Start

### 1. Train a Single Fold

```bash
python3 train_combined_optimized.py \
    --metadata_path data/combined/master_metadata.csv \
    --images_root data/ \
    --class_weights_path data/combined/class_weights.csv \
    --model swin \
    --fold 0 \
    --epochs 50 \
    --batch_size 16 \
    --image_size 384 \
    --lr 1e-4 \
    --use_amp \
    --use_focal_loss \
    --use_skin_tone_sampling \
    --output_dir models
```

### 2. Run 10-Fold Cross-Validation

```bash
# All models (~41 hours)
./scripts/run_10fold_cv.sh

# Single model
./scripts/run_10fold_cv.sh --model swin      # ~27 hours
./scripts/run_10fold_cv.sh --model densenet  # ~9 hours
./scripts/run_10fold_cv.sh --model resnet50  # ~5 hours
```

### 3. Monitor Progress

```bash
./scripts/check_cv_progress.sh
```

### 4. Evaluate Fairness

```bash
python3 scripts/evaluation/evaluate_skin_tone_fairness.py \
    --model_path models/swin_fold5_*/best_model.pth \
    --model_name swin \
    --output_dir results/fairness/swin_fold5
```

### 5. Generate Grad-CAM Visualizations

```bash
python3 scripts/generate_gradcam.py \
    --model_path models/swin_fold5_*/best_model.pth \
    --model_name swin \
    --output_dir results/xai/swin
```

---

## Training Configuration

### Fairness-Aware Strategy

1. **Two-Level Stratified Sampling:**
   - Level 1: Stratify by 8 diagnostic classes
   - Level 2: Stratify by 3 skin tone bins (Dark: 0-1, Medium: 2-3, Light: 4-5)

2. **Focal Loss with Class Weighting:**
   - Gamma (γ) = 2.0
   - Alpha weights computed from class frequencies

3. **Data Augmentation:**
   - Random horizontal/vertical flips
   - Random rotation (±15°)
   - Color jitter
   - Random affine transformations

---

## Results & Outputs

### Generated Figures

| Figure | Description | File |
|--------|-------------|------|
| Fig 1 | Dataset Overview | `results/figures/fig1_dataset_overview.png` |
| Fig 2 | Skin Tone Distribution | `results/figures/fig2_skin_tone_distribution.png` |
| Fig 3 | Methodology Flowchart | `results/figures/fig3_methodology_flowchart.png` |
| Fig 4 | Model Comparison | `results/figures/fig4_model_comparison.png` |
| Fig 5 | Fairness by Skin Tone | `results/figures/fig5_fairness_by_skin_tone.png` |
| Fig 6 | External Validation | `results/figures/fig6_external_validation.png` |
| Fig 7 | EDA Statistics | `results/figures/fig7_eda_statistics.png` |

### XAI Outputs

| Output | Description | File |
|--------|-------------|------|
| Grad-CAM Grid | Sample visualizations | `results/xai/swin/gradcam_grid.png` |
| By Class | Per-class attention | `results/xai/swin/gradcam_by_class.png` |
| By Skin Tone | Fairness verification | `results/xai/swin/gradcam_by_skin_tone.png` |

---

## Validation Summary

| Validation Type | Status | Details |
|-----------------|--------|---------|
| Cross-Validation | Partial | Fold 5 complete; 10-fold in progress |
| External Dataset | **Complete** | HAM10000 (10,015 images) |
| Fairness Testing | **Complete** | 6 skin tone groups evaluated |
| Explainability | **Complete** | Grad-CAM++ for all models |
| Statistical Tests | **Complete** | Spearman, Mann-Whitney, Cohen's d |

---

## System Requirements

- **OS:** Linux (Ubuntu 20.04+) / WSL2
- **Python:** 3.8+
- **GPU:** NVIDIA GPU with 8GB+ VRAM (RTX 3060 or better)
- **RAM:** 16GB minimum
- **Storage:** 50GB for datasets and models

### Dependencies

```bash
pip install torch torchvision timm pandas numpy scipy scikit-learn matplotlib seaborn tqdm albumentations
```

---

## Documentation

| Document | Description |
|----------|-------------|
| `PROOF_OF_CONCEPT.md` | 1-page PoC summary with results |
| `PROJECT_REPORT.md` | Full project report |
| `10_FOLD_CV_GUIDE.md` | Guide for completing 10-fold CV |
| `EVALUATION_RESULTS.md` | Detailed evaluation results |

---

## References

1. Daneshjou, R., et al. (2022). Disparities in dermatology AI. *Science Advances*.
2. Liu, Z., et al. (2021). Swin Transformer. *ICCV 2021*.
3. ISIC Archive: https://www.isic-archive.com/
4. MILK10k Dataset: Monk Skin Tone Scale annotations

---

## Citation

```bibtex
@software{fairness_skin_cancer_2025,
  title={Fairness-Aware Skin Cancer Classification with Deep Learning},
  author={[Shidhartha Chakrabarty Turzo]},
  year={2025},
  url={https://github.com/turzo891/713_skin_lis_dermoscopic}
}
```

---

## License

MIT License - See LICENSE file for details.

---

**Version:** 2.0
**Status:** Proof of Concept Complete
