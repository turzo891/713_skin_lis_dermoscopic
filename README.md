# Skin Cancer Classification with Explainable AI

A comprehensive deep learning project for skin cancer classification using CNN and Vision Transformer architectures with explainable AI (XAI) methods.

## Project Overview

This project implements and compares multiple deep learning models for skin lesion classification on the HAM10000 dataset, along with various XAI techniques to interpret model predictions.

### Models Implemented
- **ResNet50** - Classic residual network
- **EfficientNet-B4** - Efficient scaling architecture
- **DenseNet201** - Dense connections for feature reuse
- **ViT-B/16** - Vision Transformer
- **Swin Transformer** - Hierarchical vision transformer

### XAI Methods
- Grad-CAM / Grad-CAM++
- Integrated Gradients
- SHAP (SHapley Additive exPlanations)
- LIME (Local Interpretable Model-agnostic Explanations)
- Occlusion Sensitivity
- Attention Rollout (for Vision Transformers)

## Project Structure

```
adv_rec_skin
├── configs/              # Configuration files
│   └── config.yaml       # Main configuration
├── data/                 # Datasets
│   ├── HAM10000/        # Primary dataset
│   ├── ISIC2019/        # External validation
│   └── PH2/             # External validation
├── notebooks/            # Jupyter notebooks
├── src/                  # Source code
│   ├── data_loader.py   # Dataset and DataLoader
│   ├── models.py        # Model definitions
│   ├── train.py         # Training pipeline
│   ├── evaluate.py      # Evaluation metrics
│   ├── xai_methods.py   # XAI implementations
│   ├── visualize.py     # Visualization utilities
│   ├── run_experiments.py
│   └── generate_tables.py
├── models/               # Saved model weights
├── results/              # Output results
│   ├── eda/             # EDA figures
│   ├── xai/             # XAI visualizations
│   ├── metrics/         # Metrics files
│   └── figures/         # Publication figures
├── tables/               # Generated LaTeX tables
├── requirements.txt
└── README.md
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/turzo891/adv_rec_skin
cd adv_rec_skin
```

2. Create virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Dataset Setup

### HAM10000
Download the HAM10000 dataset from [Harvard Dataverse](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/DBW86T) or [Kaggle](https://www.kaggle.com/kmader/skin-cancer-mnist-ham10000).

Place the images and metadata CSV in `data/HAM10000/`.

### Class Distribution
| Class | Description | Count |
|-------|-------------|-------|
| akiec | Actinic Keratoses | 327 |
| bcc | Basal Cell Carcinoma | 514 |
| bkl | Benign Keratosis | 1099 |
| df | Dermatofibroma | 115 |
| mel | Melanoma | 1113 |
| nv | Melanocytic Nevi | 6705 |
| vasc | Vascular Lesions | 142 |

## Usage

### Training a Single Model

```bash
python src/train.py \
    --model efficientnet \
    --data_path data/HAM10000 \
    --csv_path data/HAM10000/HAM10000_metadata.csv \
    --epochs 50 \
    --batch_size 32 \
    --lr 0.0001
```

### Training with K-Fold Cross-Validation

```bash
python src/train.py \
    --model resnet50 \
    --data_path data/HAM10000 \
    --kfold 5 \
    --epochs 50
```

### Running All Experiments

```bash
python src/run_experiments.py \
    --config configs/config.yaml \
    --data_path data/HAM10000 \
    --experiment all
```

### Evaluating Models

```bash
python src/evaluate.py \
    --model_dir models/ \
    --data_path data/HAM10000 \
    --output results/metrics
```

### Generating XAI Explanations

```bash
python src/xai_methods.py \
    --model models/efficientnet/best_model.pth \
    --model_name efficientnet \
    --data_path data/HAM10000 \
    --output results/xai
```

### Generating Tables

```bash
python src/generate_tables.py \
    --results_dir results/ \
    --output tables/
```

## Configuration

Edit `configs/config.yaml` to customize:
- Model architecture and hyperparameters
- Training settings (epochs, batch size, learning rate)
- Data augmentation parameters
- XAI method settings
- Evaluation metrics

## Results

Results are saved in the `results/` directory:
- `metrics/` - JSON files with evaluation metrics
- `figures/` - Visualization plots
- `xai/` - XAI heatmap visualizations

LaTeX tables for papers are generated in `tables/`.

## Experiments

1. **Model Comparison**: Train and compare all 5 architectures
2. **XAI Comparison**: Compare XAI methods for each model
3. **External Validation**: Test on ISIC 2019 and PH2 datasets
4. **Ensemble**: Combine best CNN and ViT models

## Citation

If you use this code in your research, please cite:

```bibtex
@misc{skin-cancer-xai,
  title={Skin Cancer Classification with Explainable AI},
  author={Shidhartha Chakrabarty Turzo},
  year={2025}
}
```

## License

MIT License
