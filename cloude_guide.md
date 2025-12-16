# Building Your Skin Cancer XAI Project with Claude Code

## Complete Step-by-Step Guide

---

## Part 1: Installation and Setup

### Step 1: Install Claude Code

```bash
# Install Claude Code globally via npm
npm install -g @anthropic-ai/claude-code

# Verify installation
claude --version
```

### Step 2: Set Up Your API Key

```bash
# Option 1: Set environment variable (recommended)
export ANTHROPIC_API_KEY="your-api-key-here"

# Option 2: Add to your shell profile (~/.bashrc or ~/.zshrc)
echo 'export ANTHROPIC_API_KEY="your-api-key-here"' >> ~/.bashrc
source ~/.bashrc
```

**Note:** You need a Claude Pro ($20/month) or Max subscription, or pay-as-you-go API access.

### Step 3: Create Project Directory

```bash
# Create and navigate to your project folder
mkdir skin-cancer-xai-project
cd skin-cancer-xai-project

# Initialize the project
git init

# Start Claude Code
claude
```

---

## Part 2: Project Structure Generation

### Prompt 1: Create Project Structure

Once inside Claude Code, use this prompt:

```
Create a complete Python project structure for a skin cancer classification 
research project with the following requirements:

1. Directory structure:
   - data/ (for datasets)
   - src/ (source code)
   - notebooks/ (Jupyter notebooks for EDA)
   - models/ (saved model weights)
   - results/ (outputs, figures, metrics)
   - configs/ (configuration files)

2. Create these Python modules in src/:
   - data_loader.py (dataset loading and preprocessing)
   - models.py (CNN and ViT model definitions)
   - train.py (training pipeline)
   - evaluate.py (evaluation metrics)
   - xai_methods.py (Grad-CAM, SHAP, LIME, etc.)
   - utils.py (helper functions)
   - visualize.py (visualization utilities)

3. Create requirements.txt with all necessary packages for:
   - PyTorch, torchvision, timm
   - SHAP, LIME, captum (for XAI)
   - scikit-learn, pandas, numpy
   - matplotlib, seaborn, plotly
   - albumentations (augmentation)
   - wandb or tensorboard (logging)

4. Create a config.yaml file with hyperparameters

5. Create a README.md with project description

Please create all these files with proper boilerplate code.
```

---

## Part 3: Data Loading and EDA

### Prompt 2: Create Data Loader

```
Create a comprehensive data loading module (src/data_loader.py) that:

1. Downloads and extracts HAM10000 dataset from:
   - Harvard Dataverse or Kaggle
   
2. Implements a custom PyTorch Dataset class for skin lesions with:
   - Image loading and preprocessing
   - Label encoding for 7 classes (akiec, bcc, bkl, df, mel, nv, vasc)
   - Data augmentation using albumentations:
     * RandomRotate90, Flip, Transpose
     * ShiftScaleRotate
     * RandomBrightnessContrast
     * HueSaturationValue
     * GaussNoise, GaussianBlur
     * Normalize with ImageNet stats
   
3. Implements train/val/test split (70/15/15) with stratification

4. Handles class imbalance with:
   - Weighted sampler
   - Oversampling minority classes
   - Class weights for loss function

5. Creates DataLoaders with configurable batch size

Include docstrings and type hints for all functions.
```

### Prompt 3: Create EDA Notebook

```
Create a Jupyter notebook (notebooks/01_EDA.ipynb) for Exploratory Data Analysis:

1. Dataset Overview:
   - Total images per class
   - Class distribution bar chart
   - Calculate imbalance ratio

2. Image Analysis:
   - Display sample images from each class (3x7 grid)
   - Image dimension statistics
   - Color channel histograms per class

3. Statistical Analysis:
   - Mean and std of pixel values per class
   - t-SNE/UMAP visualization of image embeddings
   - Correlation analysis between classes

4. Data Quality:
   - Check for duplicates
   - Identify blurry or low-quality images
   - Hair artifact analysis

5. Summary Statistics Table:
   - Class name, count, percentage, mean size

Use matplotlib and seaborn for visualizations.
Save all figures to results/eda/ folder.
```

---

## Part 4: Model Definitions

### Prompt 4: Create CNN Models

```
Create src/models.py with the following CNN architectures:

1. ResNet50 (baseline):
   - Load pretrained ImageNet weights
   - Replace final FC layer for 7 classes
   - Add dropout before final layer

2. EfficientNet-B4:
   - Use timm library
   - Pretrained weights
   - Custom classifier head

3. DenseNet201:
   - Pretrained from torchvision
   - Modified classifier

For each model include:
- Function to create model: get_resnet50(), get_efficientnet(), get_densenet()
- Option to freeze/unfreeze backbone
- Function to get feature extractor (for XAI)
- Print model summary with parameter count

Also create a model factory function:
def get_model(name: str, num_classes: int = 7, pretrained: bool = True) -> nn.Module
```

### Prompt 5: Create Vision Transformer Models

```
Add Vision Transformer models to src/models.py:

1. ViT-B/16 (Vision Transformer Base):
   - Use timm library: 'vit_base_patch16_224'
   - Pretrained on ImageNet-21k
   - Custom classification head for 7 classes

2. Swin Transformer (Swin-B):
   - Use timm: 'swin_base_patch4_window7_224'
   - Pretrained weights
   - Modified head

For Vision Transformers, also include:
- Function to extract attention weights for visualization
- Function to get intermediate features
- Proper input size handling (224 for ViT, 224 for Swin)

Create unified interface so all models can be used interchangeably.
```

---

## Part 5: Training Pipeline

### Prompt 6: Create Training Script

```
Create src/train.py with a complete training pipeline:

1. Training function with:
   - Mixed precision training (AMP)
   - Gradient clipping
   - Learning rate scheduling (CosineAnnealingWarmRestarts)
   - Early stopping with patience
   - Model checkpointing (best and last)
   - Logging to WandB or TensorBoard

2. Validation function with:
   - Running accuracy, loss
   - Per-class metrics calculation

3. Main training loop with:
   - 5-fold cross-validation option
   - Single train/val split option
   - Resume from checkpoint
   - Multi-GPU support (DataParallel)

4. Optimizer setup:
   - AdamW with weight decay
   - Different LR for backbone vs classifier (discriminative LR)

5. Loss function:
   - CrossEntropyLoss with class weights
   - Optional: Focal Loss for imbalance

6. Command line arguments:
   - --model (resnet50, efficientnet, densenet, vit, swin)
   - --epochs, --batch_size, --lr
   - --data_path, --output_dir
   - --resume (checkpoint path)

Include progress bars with tqdm and proper logging.
```

---

## Part 6: Evaluation and Metrics

### Prompt 7: Create Evaluation Module

```
Create src/evaluate.py with comprehensive evaluation:

1. Classification Metrics:
   - Accuracy (overall and per-class)
   - Precision, Recall, F1-Score (macro, weighted, per-class)
   - AUC-ROC (One-vs-Rest for multiclass)
   - Cohen's Kappa
   - Confusion Matrix

2. Statistical Tests:
   - Paired t-test between models
   - Wilcoxon signed-rank test
   - McNemar's test for classification comparison
   - Confidence intervals (95%)

3. Visualization Functions:
   - Plot confusion matrix heatmap
   - Plot ROC curves (all classes)
   - Plot precision-recall curves
   - Plot per-class performance bar chart

4. Report Generation:
   - Generate LaTeX table with results
   - Generate markdown summary
   - Save metrics to JSON/CSV

5. External Dataset Evaluation:
   - Function to evaluate on ISIC 2019
   - Function to evaluate on PH2 dataset
   - Domain shift analysis (compare performance)

Include functions to compare multiple models side by side.
```

---

## Part 7: Explainable AI (XAI) Methods

### Prompt 8: Create XAI Module - Part 1 (Gradient-based)

```
Create src/xai_methods.py with gradient-based XAI methods:

1. Grad-CAM++ Implementation:
   - Works with any CNN (ResNet, EfficientNet, DenseNet)
   - Specify target layer automatically based on architecture
   - Generate heatmap overlay on original image
   - Support for specific class or predicted class

2. Integrated Gradients:
   - Use captum library
   - Baseline: black image or blurred image
   - Number of steps: 50-200
   - Visualization with positive/negative attributions

3. Layer-wise Relevance Propagation (LRP):
   - Implement for CNNs
   - Different rules for different layers

4. Guided Backpropagation:
   - Modify ReLU backward pass
   - Combine with Grad-CAM (Guided Grad-CAM)

5. Saliency Maps (Vanilla Gradient):
   - Simple gradient w.r.t input
   - Absolute value and smooth grad variants

For each method include:
- Function to generate attribution map
- Function to visualize overlay
- Function to save visualization

Use captum library where possible for consistency.
```

### Prompt 9: Create XAI Module - Part 2 (Perturbation and Model-specific)

```
Add perturbation-based and model-specific XAI methods to src/xai_methods.py:

1. LIME (Local Interpretable Model-agnostic Explanations):
   - Use lime library
   - Configure number of samples and features
   - Generate superpixel explanations
   - Visualize top positive/negative features

2. SHAP (SHapley Additive exPlanations):
   - Use shap library
   - DeepExplainer for deep learning models
   - GradientExplainer as alternative
   - Generate SHAP values visualization
   - Summary plot for feature importance

3. Occlusion Sensitivity:
   - Sliding window occlusion
   - Configurable patch size and stride
   - Generate sensitivity heatmap

4. Vision Transformer Attention:
   - Extract attention weights from ViT
   - Visualize attention maps per head
   - Average attention across heads
   - Attention rollout for global view
   - Compare with CNN Grad-CAM

5. Unified XAI Interface:
   - Create class XAIExplainer with method:
     def explain(model, image, method='gradcam', **kwargs)
   - Returns: attribution_map, visualization

Include proper handling for different model architectures.
```

### Prompt 10: Create XAI Quantitative Evaluation

```
Add quantitative XAI evaluation to src/xai_methods.py:

1. Confidence Increase (CI) Metric:
   - Mask image except highlighted region
   - Compare prediction confidence
   - Higher CI = better explanation

2. Faithfulness/Fidelity:
   - Correlation between attribution importance and prediction change
   - Most Important First (MIF) - mask top features
   - Least Important First (LIF) - mask bottom features
   - Plot degradation curves

3. Localization Accuracy (IoU):
   - Compare XAI heatmap with ground truth lesion mask
   - Use segmentation masks from dataset
   - Calculate Intersection over Union
   - Dice coefficient

4. Robustness:
   - Consistency under small input perturbations
   - Measure variance of explanations

5. Complexity:
   - Sparsity of explanations
   - Number of highlighted pixels

6. Create comparison function:
   def compare_xai_methods(model, images, methods, metrics):
       # Returns DataFrame with all metrics for all methods

7. Statistical Analysis:
   - Paired t-test between XAI methods
   - Generate comparison table

Save metrics to CSV and generate visualization plots.
```

---

## Part 8: Experiments and Analysis

### Prompt 11: Create Main Experiment Script

```
Create src/run_experiments.py that orchestrates all experiments:

1. Experiment 1: Model Comparison
   - Train all 5 models (ResNet50, EfficientNet, DenseNet, ViT, Swin)
   - Same hyperparameters for fair comparison
   - 5-fold cross-validation
   - Save all metrics

2. Experiment 2: XAI Comparison per Model
   - For each trained model:
     * Apply all 5 XAI methods
     * Calculate quantitative metrics
     * Generate visualizations for sample images
   - Create comparison tables

3. Experiment 3: External Validation
   - Load best model from each architecture
   - Evaluate on ISIC 2019 dataset
   - Evaluate on PH2 dataset
   - Analyze performance drop (domain shift)

4. Experiment 4: Ensemble
   - Create ensemble of best CNN + best ViT
   - Weighted average ensemble
   - Majority voting ensemble
   - Evaluate ensemble performance
   - Generate aggregated XAI explanation

5. Experiment 5: Ablation Study
   - Effect of data augmentation
   - Effect of class balancing
   - Effect of different learning rates

Create config files for each experiment.
Use command line to select which experiment to run.
```

### Prompt 12: Create Visualization and Results

```
Create src/visualize.py for generating publication-quality figures:

1. Model Comparison Plots:
   - Bar chart comparing accuracy across models
   - Grouped bar chart for precision/recall/F1
   - Box plots from cross-validation results

2. XAI Visualization Grid:
   - For a sample image, show all XAI methods side by side
   - 3 rows (sample images) x 6 columns (original + 5 XAI methods)
   - Save as high-resolution PNG and PDF

3. Confusion Matrix Heatmaps:
   - Side-by-side for all models
   - Normalized and raw counts

4. ROC Curves:
   - All models on same plot
   - One subplot per class (7 subplots)

5. XAI Metrics Comparison:
   - Heatmap of XAI method vs metric
   - Radar chart for multi-metric comparison

6. Attention vs Grad-CAM:
   - Side-by-side comparison
   - ViT attention heads visualization

7. Domain Shift Analysis:
   - Performance bar chart across datasets
   - Confusion matrices for each dataset

8. LaTeX Table Generation:
   - Model comparison table
   - XAI metrics table
   - Format for IEEE/Springer papers

All figures should be:
- Vector graphics (SVG/PDF) for scalability
- Publication quality (300+ DPI for rasters)
- Consistent color scheme
- Proper labels and legends
```

---

## Part 9: Paper Writing Support

### Prompt 13: Generate Results Tables

```
Create src/generate_tables.py to automatically generate LaTeX tables:

1. Table 1: Dataset Statistics
   - Class name, training samples, test samples, percentage

2. Table 2: Model Comparison Results
   - Model, Accuracy, Precision, Recall, F1, AUC, Parameters
   - Bold best results
   - Include std from cross-validation

3. Table 3: External Validation Results
   - Model vs Dataset (HAM10000, ISIC 2019, PH2)
   - Show performance drop percentage

4. Table 4: XAI Methods Comparison
   - Method vs Metrics (CI, IoU, Faithfulness, Complexity)
   - Separate tables for CNN and ViT

5. Table 5: Statistical Significance
   - Pairwise p-values between models
   - Mark significant differences

6. Table 6: Ensemble Results
   - Comparison of ensemble strategies
   - vs individual best models

Output tables in:
- LaTeX format (for paper)
- Markdown format (for README)
- CSV format (for further analysis)
```

---

## Part 10: Running the Complete Pipeline

### Full Workflow Commands in Claude Code

```bash
# Step 1: Setup environment
Create a virtual environment and install all dependencies from requirements.txt

# Step 2: Download datasets
Download HAM10000 dataset and organize in data/ folder. 
Also download ISIC 2019 and PH2 for external validation.

# Step 3: Run EDA
Execute the EDA notebook and save all visualizations

# Step 4: Train all models
python src/train.py --model resnet50 --epochs 50 --batch_size 32
python src/train.py --model efficientnet --epochs 50 --batch_size 32
python src/train.py --model densenet --epochs 50 --batch_size 32
python src/train.py --model vit --epochs 50 --batch_size 16
python src/train.py --model swin --epochs 50 --batch_size 16

# Step 5: Evaluate all models
python src/evaluate.py --model_dir models/ --output results/

# Step 6: Generate XAI explanations
python src/xai_methods.py --model models/best_resnet50.pth --output results/xai/

# Step 7: Run all experiments
python src/run_experiments.py --experiment all

# Step 8: Generate figures and tables
python src/visualize.py --results_dir results/ --output figures/
python src/generate_tables.py --results_dir results/ --output tables/
```

---

## Final Project Structure

```
skin-cancer-xai-project/
|-- .claude/
|   +-- commands/           # Custom Claude Code commands
|-- configs/
|   |-- config.yaml         # Main configuration
|   |-- experiment1.yaml    # Model comparison config
|   +-- experiment2.yaml    # XAI comparison config
|-- data/
|   |-- HAM10000/          # Primary dataset
|   |-- ISIC2019/          # External validation
|   +-- PH2/               # External validation
|-- notebooks/
|   |-- 01_EDA.ipynb       # Exploratory Data Analysis
|   |-- 02_Training.ipynb  # Training experiments
|   +-- 03_XAI_Analysis.ipynb  # XAI experiments
|-- src/
|   |-- __init__.py
|   |-- data_loader.py     # Dataset and DataLoader
|   |-- models.py          # CNN and ViT models
|   |-- train.py           # Training pipeline
|   |-- evaluate.py        # Evaluation metrics
|   |-- xai_methods.py     # XAI implementations
|   |-- visualize.py       # Visualization utilities
|   |-- utils.py           # Helper functions
|   |-- run_experiments.py # Main experiment runner
|   +-- generate_tables.py # LaTeX table generator
|-- models/                 # Saved model weights
|-- results/
|   |-- eda/               # EDA figures
|   |-- xai/               # XAI visualizations
|   |-- metrics/           # CSV/JSON metrics
|   +-- figures/           # Publication figures
|-- tables/                 # Generated LaTeX tables
|-- requirements.txt
|-- README.md
+-- paper/
    |-- main.tex           # Paper LaTeX source
    +-- figures/           # Paper figures
```

---

## Tips for Using Claude Code Effectively

### 1. Use Iterative Prompts
Do not ask for everything at once. Build incrementally:
```
First: "Create the data loader module"
Then: "Now add data augmentation to the data loader"
Then: "Add class balancing with weighted sampler"
```

### 2. Ask for Debugging
```
I'm getting this error when running train.py:
[paste error]
Please fix it.
```

### 3. Request Tests
```
Write unit tests for the XAI methods in src/xai_methods.py
```

### 4. Ask for Documentation
```
Add comprehensive docstrings to all functions in models.py
Generate a README for the project
```

### 5. Use Custom Commands
Create `.claude/commands/train-model.md`:
```markdown
Train a model with the specified architecture.
$ARGUMENTS should be the model name (resnet50, efficientnet, etc.)

Run: python src/train.py --model $ARGUMENTS --epochs 50
```

Then use: `/project:train-model resnet50`

---

## Quick Start Checklist

- [ ] Install Claude Code (npm install -g @anthropic-ai/claude-code)
- [ ] Set up API key
- [ ] Create project directory
- [ ] Generate project structure (Prompt 1)
- [ ] Create data loader (Prompt 2)
- [ ] Run EDA notebook (Prompt 3)
- [ ] Create models (Prompts 4-5)
- [ ] Create training pipeline (Prompt 6)
- [ ] Create evaluation module (Prompt 7)
- [ ] Create XAI methods (Prompts 8-10)
- [ ] Run experiments (Prompt 11)
- [ ] Generate visualizations (Prompt 12)
- [ ] Generate tables for paper (Prompt 13)
- [ ] Write paper

