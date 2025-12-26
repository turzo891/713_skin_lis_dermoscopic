# Skin Lesion Classification with Skin Tone Fairness: A Deep Learning Approach

**Project Plan Summary Report**

**Based on:** Daneshjou, R., et al. (2022). "Disparities in dermatology AI performance on a diverse, curated clinical image set." *Science Advances*, 8(32). https://www.science.org/doi/10.1126/sciadv.abq6147

**Tools/Frameworks:** PyTorch Deep Learning Framework (https://pytorch.org/), ISIC Archive Dataset (https://challenge.isic-archive.com/), timm - PyTorch Image Models (https://github.com/huggingface/pytorch-image-models)

---

## 1. Summary

### 1.1 Motivation

Current dermatological AI systems exhibit significant performance disparities across different skin tones, with models performing 10-20% worse on darker skin compared to lighter skin tones. The referenced paper by Daneshjou et al. (2022) demonstrated that even state-of-the-art dermatology AI models showed substantial performance degradation when evaluated on diverse skin tones, particularly on Fitzpatrick skin types V and VI. This bias poses serious ethical concerns for clinical deployment, as it could lead to misdiagnosis and delayed treatment for patients with darker skin. Our project addresses this critical gap by implementing a novel skin tone-aware sampling strategy combined with the latest ISIC 2025 (MILK10k) dataset, which includes explicit skin tone labels on the Fitzpatrick scale. The motivation stems from the urgent need for fair and equitable AI systems in healthcare that perform consistently across all demographic groups, ensuring that technological advances in medical diagnosis do not exacerbate existing healthcare disparities.

### 1.2 Contribution

This project makes three primary contributions to dermatological AI research. First, we introduce a novel two-level stratification sampling strategy that simultaneously balances both disease class distribution and skin tone representation during training. Unlike traditional approaches that only address class imbalance, our method creates stratification groups based on (diagnosis, skin_tone_bin) pairs and samples them with equal probability, ensuring minority skin tones receive adequate representation. Second, we integrate the latest ISIC 2025 (MILK10k) dataset with 5,137 skin tone-labeled images with the larger ISIC 2019 dataset, creating a combined training corpus of 30,468 samples that provides both scale and diversity. Third, we implement a comprehensive evaluation framework using 10-fold stratified cross-validation that explicitly measures performance across skin tone groups, providing transparency in model fairness. Our implementation uses PyTorch and the timm library for state-of-the-art model architectures (ResNet50, EfficientNet-B4, DenseNet201, ViT, Swin Transformer), combined with focal loss to handle the 46.8:1 class imbalance ratio present in the dataset.

### 1.3 Methodology

Our methodology consists of four key components: dataset preparation, model training, fairness-aware optimization, and comprehensive evaluation. For dataset preparation, we combine MILK10k (5,137 samples with skin tone labels ranging 0-5 on the Fitzpatrick scale) with ISIC 2019 (25,331 samples) to create a master dataset of 30,468 dermoscopic images across 8 diagnostic categories (MEL, NV, BCC, BKL, AK, SCC, VASC, DF). We perform stratified splitting with 80% for training (24,374 samples divided into 10 folds) and 20% for testing (6,094 samples). The training pipeline implements our skin tone-aware sampler that bins skin tones into three categories (dark: 0-1, medium: 2-3, light: 4-5) and creates 32 stratification groups combining diagnosis and skin tone. During each training epoch, we sample from these groups with equal probability, ensuring balanced representation. We employ focal loss with gamma=2.0 and class-specific weights to address severe class imbalance, combined with standard data augmentation (rotation, flipping, color jitter). The training uses mixed precision (AMP) for efficiency, AdamW optimizer with learning rate 1e-4, and cosine annealing scheduling. Model selection is based on balanced accuracy on the validation fold, with early stopping patience of 10 epochs. Evaluation includes standard metrics (accuracy, F1-score, AUC-ROC) computed separately for each skin tone group to quantify fairness.

### 1.4 Conclusion

We expect this project to demonstrate that explicit skin tone-aware training can reduce performance disparities by 50-70% compared to baseline approaches. Based on literature and preliminary analysis, we anticipate achieving 75-80% overall accuracy with EfficientNet-B4, while reducing the performance gap between light and dark skin tones from an expected 15-20% (baseline) to 5-10% (with our method). The 10-fold cross-validation will provide robust statistical evidence of model performance across skin tone groups with confidence intervals. This work will contribute both a practical implementation of fairness-aware dermatological AI and empirical evidence that targeted sampling strategies can mitigate demographic bias without sacrificing overall performance. The complete pipeline, including the skin tone-aware sampler, trained models, and evaluation scripts, will be documented and made reproducible for future research. Our findings will be particularly relevant for clinical AI deployment, demonstrating that fairness considerations must be integrated into model development from the outset rather than addressed as an afterthought.

## 2. Limitations

### 2.1 First Limitation: Skin Tone Distribution Imbalance

Despite our skin tone-aware sampling strategy, the underlying dataset still exhibits significant imbalance in skin tone representation, with 60.95% of MILK10k samples having medium skin tone (Fitzpatrick III) and only 1.95% having dark skin tone (Fitzpatrick V). This means that even with our balanced sampling approach, the model will see fewer unique examples of lesions on darker skin during training. While our sampling ensures equal representation per epoch, the limited diversity of dark skin examples may still restrict the model's ability to learn robust features for these underrepresented groups. Additionally, ISIC 2019 samples (83% of our dataset) lack skin tone labels entirely, so they are grouped as "no_tone" in our stratification, potentially diluting the fairness improvements. This limitation could be addressed in future work by actively collecting more diverse skin tone data or using synthetic data augmentation techniques specifically for underrepresented skin tones.

### 2.2 Second Limitation: Generalization to Clinical Practice

Our evaluation is limited to dermoscopic images from the ISIC Archive, which may not fully represent real-world clinical settings. Dermoscopic images are acquired under controlled conditions with specialized equipment, whereas clinical practice often involves smartphone cameras, varying lighting conditions, and different image qualities. Furthermore, the ISIC dataset predominantly contains lesions from populations in developed countries, potentially missing skin tone and genetic variations present in global populations. The Fitzpatrick scale itself has limitations, as it was originally designed for UV sensitivity and may not perfectly capture the visual diversity of skin tones relevant to image-based diagnosis. Our skin tone binning strategy (dark, medium, light) reduces six categories to three, which may oversimplify the continuous spectrum of skin tone variation. External validation on independent clinical datasets with diverse acquisition methods and truly representative global populations would be necessary to confirm that our fairness improvements generalize beyond research datasets to actual clinical deployment scenarios.

## 3. Synthesis

The intersection of AI bias and healthcare equity represents one of the most critical challenges in deploying machine learning systems in clinical settings. As demonstrated by Daneshjou et al. (2022), existing dermatology AI systems can perpetuate and amplify healthcare disparities if fairness is not explicitly addressed during development. Our project directly tackles this issue by implementing skin tone-aware training for skin cancer classification, a potentially life-saving application where diagnostic delays disproportionately affect minority populations. The implications extend beyond dermatology to any medical imaging task where demographic factors (skin tone, age, sex) may correlate with image appearance, including radiology, ophthalmology, and pathology. By demonstrating that fairness can be improved through thoughtful data sampling without sacrificing overall accuracy, we provide a practical template for responsible AI development in healthcare. The availability of skin tone-labeled datasets like MILK10k, combined with mature deep learning frameworks like PyTorch and timm, makes such fairness-aware approaches increasingly feasible for researchers and practitioners. However, this also highlights a crucial responsibility: as AI tools become more accessible and are integrated into clinical decision support systems, developers must proactively measure and mitigate demographic biases rather than assuming that large datasets and powerful models will naturally produce fair outcomes. Our work contributes to the growing body of evidence that fairness in medical AI requires intentional design choices, explicit measurement, and continuous validation across diverse patient populations. The long-term societal impact depends not only on technical solutions like ours but also on establishing regulatory frameworks, clinical validation standards, and transparent reporting requirements that ensure AI systems are tested for fairness before deployment in healthcare settings where they can directly affect patient outcomes and health equity.

---

**References:**

**Primary Paper:**
- Daneshjou, R., et al. (2022). Disparities in dermatology AI performance on a diverse, curated clinical image set. *Science Advances*, 8(32). https://doi.org/10.1126/sciadv.abq6147

**Related Work on Fairness in Dermatology AI:**
- Groh, M., et al. (2021). Evaluating Deep Neural Networks Trained on Clinical Images in Dermatology with the Fitzpatrick 17k Dataset. *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*, 1820-1828. https://doi.org/10.1109/CVPRW53098.2021.00201
- Kinyanjui, N. M., et al. (2020). Fairness of Classifiers Across Skin Tones in Dermatology. *Medical Image Computing and Computer Assisted Intervention – MICCAI 2020*, 320-329. https://doi.org/10.1007/978-3-030-59725-2_31

**Dataset Papers:**
- Combalia, M., et al. (2019). BCN20000: Dermoscopic Lesions in the Wild. *arXiv preprint arXiv:1908.02288*. https://arxiv.org/abs/1908.02288 (ISIC 2019 Dataset)
- Cassidy, B., et al. (2022). Analysis of the ISIC image datasets: Usage, benchmarks and recommendations. *Medical Image Analysis*, 75, 102305. https://doi.org/10.1016/j.media.2021.102305

**Tools/Frameworks:**
- PyTorch Deep Learning Framework: https://pytorch.org/
- ISIC Archive Challenge Dataset: https://challenge.isic-archive.com/
- timm - PyTorch Image Models: https://github.com/huggingface/pytorch-image-models
- Wightman, R. (2019). PyTorch Image Models. https://github.com/rwightman/pytorch-image-models
