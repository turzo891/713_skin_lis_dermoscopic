# Notebooks Directory

## Why Notebooks?

Jupyter notebooks serve specific purposes in this machine learning project that complement the main Python scripts.

### Purpose of Notebooks in This Project

#### 1. Interactive Data Exploration
Notebooks allow you to interactively explore the dataset without writing full scripts:
- View sample images with labels
- Check class distributions
- Inspect metadata patterns
- Test data augmentation effects in real-time
- Visualize intermediate processing steps

#### 2. Experimentation and Prototyping
Before implementing features in production code, you can:
- Test new model architectures quickly
- Experiment with different hyperparameters
- Try new augmentation strategies
- Prototype custom loss functions
- Validate ideas before committing code

#### 3. Visualization and Analysis
Notebooks excel at creating and displaying visualizations:
- Training curve analysis
- Confusion matrix exploration
- ROC curve generation
- Grad-CAM heatmap visualization
- Model comparison charts

#### 4. Documentation and Tutorials
Notebooks serve as executable documentation:
- Step-by-step tutorials for using the project
- Explanation of model architectures
- Demonstration of XAI methods
- Walkthrough of training pipeline

#### 5. Reproducible Research
Notebooks capture the entire workflow:
- Document experiments with results inline
- Share findings with collaborators
- Create reproducible analysis pipelines
- Generate reports with code + outputs

### When to Use Notebooks vs Scripts

**Use Notebooks when:**
- Exploring data for the first time
- Testing ideas and prototyping
- Creating visualizations
- Teaching or learning
- Sharing analysis with non-programmers
- Generating reports

**Use Scripts (Python files) when:**
- Training production models
- Running batch experiments
- Deploying to production
- Automating workflows
- Running on remote servers without GUI
- Version controlling code (notebooks have large diffs)

### Typical Notebooks in This Project

Here are the notebooks you might create for this project:

1. **01_data_exploration.ipynb**
   - Load and explore ISIC 2019 dataset
   - Visualize class distributions
   - Check for data quality issues
   - Understand metadata patterns

2. **02_augmentation_preview.ipynb**
   - Preview data augmentation effects
   - Compare different augmentation strategies
   - Visualize CutMix, MixUp, etc.

3. **03_model_training_demo.ipynb**
   - Train a small model on subset of data
   - Monitor training in real-time
   - Plot training curves
   - Test early stopping

4. **04_model_evaluation.ipynb**
   - Load trained models
   - Generate confusion matrices
   - Plot ROC curves
   - Analyze per-class performance

5. **05_xai_visualization.ipynb**
   - Generate Grad-CAM heatmaps
   - Compare XAI methods (SHAP, LIME, etc.)
   - Visualize attention maps for ViT
   - Create publication-quality figures

6. **06_error_analysis.ipynb**
   - Analyze misclassified examples
   - Identify failure patterns
   - Investigate edge cases
   - Suggest improvements

### Example: Creating Your First Notebook

Create a new notebook for data exploration:

```bash
cd notebooks
jupyter notebook
# Click "New" -> "Python 3"
```

**Sample notebook code:**

```python
# Cell 1: Imports
import sys
sys.path.insert(0, '../src')

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import numpy as np

# Cell 2: Load dataset
labels_df = pd.read_csv('../data/ISIC2019/ISIC_2019_Training_GroundTruth.csv')
metadata_df = pd.read_csv('../data/ISIC2019/ISIC_2019_Training_Metadata.csv')

print(f"Dataset size: {len(labels_df)}")
print(f"Classes: {labels_df.columns[1:9].tolist()}")

# Cell 3: Visualize class distribution
class_counts = labels_df.iloc[:, 1:9].sum()

plt.figure(figsize=(12, 6))
class_counts.plot(kind='bar', color='skyblue', edgecolor='black')
plt.title('Class Distribution in ISIC 2019 Dataset')
plt.xlabel('Disease Class')
plt.ylabel('Number of Samples')
plt.xticks(rotation=45)
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.show()

# Cell 4: Load and display sample images
import random

fig, axes = plt.subplots(2, 4, figsize=(16, 8))
axes = axes.flatten()

class_names = labels_df.columns[1:9].tolist()
img_dir = '../data/ISIC2019/ISIC_2019_Training_Input/ISIC_2019_Training_Input'

for idx, class_name in enumerate(class_names):
    # Find a sample image for this class
    class_samples = labels_df[labels_df[class_name] == 1.0]
    sample = class_samples.sample(1).iloc[0]
    img_name = sample['image']

    # Load and display image
    img_path = f"{img_dir}/{img_name}.jpg"
    img = Image.open(img_path)

    axes[idx].imshow(img)
    axes[idx].set_title(f"{class_name}")
    axes[idx].axis('off')

plt.tight_layout()
plt.show()

# Cell 5: Analyze metadata
print("Age distribution:")
print(metadata_df['age_approx'].describe())

print("\nSex distribution:")
print(metadata_df['sex'].value_counts())

print("\nAnatomical location distribution:")
print(metadata_df['anatom_site_general'].value_counts())
```

### Running Jupyter Notebooks

**Install Jupyter:**
```bash
pip install jupyter notebook
```

**Start Jupyter:**
```bash
cd notebooks
jupyter notebook
```

This will open a browser window where you can create and run notebooks.

**Alternative: JupyterLab (modern interface):**
```bash
pip install jupyterlab
jupyter lab
```

### Best Practices for Notebooks

1. **Keep notebooks focused:** One topic per notebook
2. **Clear naming:** Use descriptive names (01_data_exploration.ipynb)
3. **Add markdown cells:** Explain what each code cell does
4. **Restart kernel regularly:** Ensure cells run in order
5. **Don't commit output:** Clear output before committing to git
6. **Extract to scripts:** Move production code to .py files
7. **Use relative paths:** `../data/` not absolute paths

### Notebooks vs Scripts Summary

| Aspect | Notebooks | Scripts |
|--------|-----------|---------|
| **Interactivity** | High | Low |
| **Visualization** | Excellent | Poor |
| **Production Use** | No | Yes |
| **Version Control** | Difficult | Easy |
| **Automation** | Limited | Excellent |
| **Sharing** | Great for results | Great for code |
| **Learning** | Excellent | Good |
| **Debugging** | Easy | Moderate |

### Can You Skip Notebooks?

**Yes!** Notebooks are optional for this project. Everything can be done with Python scripts:

- Instead of notebooks for EDA, use: `python3 scripts/data/exploratory_data_analysis.py`
- Instead of notebooks for training, use: `python3 scripts/training/train_single_model.py`
- Instead of notebooks for visualization, use: `python3 scripts/data/advanced_visualizations.py`

**However,** notebooks are valuable for:
- Learning how the code works
- Experimenting with parameters
- Creating custom visualizations
- Teaching others about the project

### Conclusion

Notebooks are **tools for exploration and experimentation**, not for production training.

For this project:
- **Use scripts** for all actual training and evaluation
- **Use notebooks** for understanding, exploring, and experimenting
- **Keep notebooks in this directory** for organization
- **Don't rely on notebooks** for critical workflows

If you prefer not to use notebooks at all, you can safely ignore this directory. All functionality is available through Python scripts.
