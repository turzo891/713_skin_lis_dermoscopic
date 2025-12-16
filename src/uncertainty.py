"""
Uncertainty Quantification for Skin Cancer Classification.
Implements MC Dropout, Deep Ensembles, and calibration methods.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm
from scipy import stats
from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt


class MCDropout:
    """Monte Carlo Dropout for uncertainty estimation."""
    
    def __init__(self, model: nn.Module, n_samples: int = 30, dropout_rate: float = 0.5):
        """
        Args:
            model: PyTorch model with dropout layers
            n_samples: Number of forward passes for MC sampling
            dropout_rate: Dropout rate to use (if modifying)
        """
        self.model = model
        self.n_samples = n_samples
        self.dropout_rate = dropout_rate
    
    def enable_dropout(self):
        """Enable dropout during inference."""
        for module in self.model.modules():
            if isinstance(module, nn.Dropout):
                module.train()
    
    def predict_with_uncertainty(
        self, 
        x: torch.Tensor, 
        device: torch.device
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Make predictions with uncertainty estimates.
        
        Args:
            x: Input tensor (batch_size, C, H, W)
            device: Device to run on
            
        Returns:
            mean_probs: Mean predicted probabilities
            epistemic_uncertainty: Model uncertainty (reducible with more data)
            aleatoric_uncertainty: Data uncertainty (irreducible)
        """
        x = x.to(device)
        self.model.eval()
        self.enable_dropout()
        
        predictions = []
        
        with torch.no_grad():
            for _ in range(self.n_samples):
                outputs = self.model(x)
                probs = F.softmax(outputs, dim=1)
                predictions.append(probs.cpu().numpy())
        
        predictions = np.array(predictions)  # (n_samples, batch, n_classes)
        
        # Mean prediction
        mean_probs = predictions.mean(axis=0)
        
        # Epistemic uncertainty (variance across samples)
        epistemic = predictions.var(axis=0).mean(axis=1)  # Average variance across classes
        
        # Aleatoric uncertainty (expected entropy)
        entropy_per_sample = -np.sum(predictions * np.log(predictions + 1e-10), axis=2)
        aleatoric = entropy_per_sample.mean(axis=0)
        
        return mean_probs, epistemic, aleatoric
    
    def predict_batch(
        self,
        dataloader: DataLoader,
        device: torch.device
    ) -> Dict[str, np.ndarray]:
        """
        Predict on entire dataloader with uncertainty.
        
        Returns:
            Dictionary with predictions, labels, and uncertainties
        """
        all_probs = []
        all_epistemic = []
        all_aleatoric = []
        all_labels = []
        
        for images, labels in tqdm(dataloader, desc="MC Dropout"):
            probs, epistemic, aleatoric = self.predict_with_uncertainty(images, device)
            all_probs.append(probs)
            all_epistemic.append(epistemic)
            all_aleatoric.append(aleatoric)
            all_labels.append(labels.numpy())
        
        return {
            'probs': np.concatenate(all_probs),
            'epistemic': np.concatenate(all_epistemic),
            'aleatoric': np.concatenate(all_aleatoric),
            'labels': np.concatenate(all_labels)
        }


class DeepEnsemble:
    """Deep Ensemble for uncertainty estimation."""
    
    def __init__(self, models: List[nn.Module]):
        """
        Args:
            models: List of trained models
        """
        self.models = models
        self.n_models = len(models)
    
    def predict_with_uncertainty(
        self,
        x: torch.Tensor,
        device: torch.device
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Make predictions with uncertainty from ensemble disagreement.
        
        Returns:
            mean_probs: Mean predicted probabilities
            uncertainty: Ensemble disagreement (epistemic uncertainty)
        """
        x = x.to(device)
        predictions = []
        
        with torch.no_grad():
            for model in self.models:
                model.eval()
                outputs = model(x)
                probs = F.softmax(outputs, dim=1)
                predictions.append(probs.cpu().numpy())
        
        predictions = np.array(predictions)
        
        # Mean prediction
        mean_probs = predictions.mean(axis=0)
        
        # Uncertainty (disagreement)
        uncertainty = predictions.var(axis=0).mean(axis=1)
        
        return mean_probs, uncertainty
    
    def predict_batch(
        self,
        dataloader: DataLoader,
        device: torch.device
    ) -> Dict[str, np.ndarray]:
        """Predict on entire dataloader."""
        all_probs = []
        all_uncertainty = []
        all_labels = []
        
        for images, labels in tqdm(dataloader, desc="Ensemble"):
            probs, uncertainty = self.predict_with_uncertainty(images, device)
            all_probs.append(probs)
            all_uncertainty.append(uncertainty)
            all_labels.append(labels.numpy())
        
        return {
            'probs': np.concatenate(all_probs),
            'uncertainty': np.concatenate(all_uncertainty),
            'labels': np.concatenate(all_labels)
        }


class TemperatureScaling:
    """Temperature scaling for calibration."""
    
    def __init__(self, model: nn.Module):
        self.model = model
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)
    
    def calibrate(
        self,
        val_loader: DataLoader,
        device: torch.device,
        n_epochs: int = 100,
        lr: float = 0.01
    ):
        """Learn optimal temperature on validation set."""
        self.model.eval()
        
        # Collect logits and labels
        logits_list = []
        labels_list = []
        
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                logits = self.model(images)
                logits_list.append(logits.cpu())
                labels_list.append(labels)
        
        logits = torch.cat(logits_list)
        labels = torch.cat(labels_list)
        
        # Optimize temperature
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)
        optimizer = torch.optim.LBFGS([self.temperature], lr=lr, max_iter=n_epochs)
        
        def eval_fn():
            optimizer.zero_grad()
            scaled_logits = logits / self.temperature
            loss = F.cross_entropy(scaled_logits, labels)
            loss.backward()
            return loss
        
        optimizer.step(eval_fn)
        
        print(f"Optimal temperature: {self.temperature.item():.4f}")
        return self.temperature.item()
    
    def predict(self, x: torch.Tensor, device: torch.device) -> np.ndarray:
        """Make calibrated predictions."""
        x = x.to(device)
        self.model.eval()
        
        with torch.no_grad():
            logits = self.model(x)
            scaled_logits = logits / self.temperature.to(device)
            probs = F.softmax(scaled_logits, dim=1)
        
        return probs.cpu().numpy()


def expected_calibration_error(
    probs: np.ndarray,
    labels: np.ndarray,
    n_bins: int = 15
) -> float:
    """
    Calculate Expected Calibration Error (ECE).
    
    Args:
        probs: Predicted probabilities (N, C)
        labels: True labels (N,)
        n_bins: Number of bins for calibration
        
    Returns:
        ECE value
    """
    confidences = probs.max(axis=1)
    predictions = probs.argmax(axis=1)
    accuracies = (predictions == labels).astype(float)
    
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    
    for i in range(n_bins):
        in_bin = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i + 1])
        prop_in_bin = in_bin.mean()
        
        if prop_in_bin > 0:
            avg_confidence = confidences[in_bin].mean()
            avg_accuracy = accuracies[in_bin].mean()
            ece += prop_in_bin * abs(avg_accuracy - avg_confidence)
    
    return ece


def maximum_calibration_error(
    probs: np.ndarray,
    labels: np.ndarray,
    n_bins: int = 15
) -> float:
    """Calculate Maximum Calibration Error (MCE)."""
    confidences = probs.max(axis=1)
    predictions = probs.argmax(axis=1)
    accuracies = (predictions == labels).astype(float)
    
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    mce = 0.0
    
    for i in range(n_bins):
        in_bin = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i + 1])
        
        if in_bin.sum() > 0:
            avg_confidence = confidences[in_bin].mean()
            avg_accuracy = accuracies[in_bin].mean()
            mce = max(mce, abs(avg_accuracy - avg_confidence))
    
    return mce


def brier_score(probs: np.ndarray, labels: np.ndarray) -> float:
    """Calculate Brier Score."""
    one_hot = np.zeros_like(probs)
    one_hot[np.arange(len(labels)), labels] = 1
    return ((probs - one_hot) ** 2).sum(axis=1).mean()


def plot_reliability_diagram(
    probs: np.ndarray,
    labels: np.ndarray,
    n_bins: int = 10,
    title: str = "Reliability Diagram",
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot reliability diagram (calibration curve).
    """
    confidences = probs.max(axis=1)
    predictions = probs.argmax(axis=1)
    accuracies = (predictions == labels).astype(float)
    
    # Calculate calibration curve
    fraction_of_positives, mean_predicted_value = calibration_curve(
        accuracies, confidences, n_bins=n_bins, strategy='uniform'
    )
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Reliability diagram
    ax1.plot([0, 1], [0, 1], 'k--', label='Perfect calibration')
    ax1.plot(mean_predicted_value, fraction_of_positives, 's-', label='Model')
    ax1.fill_between(mean_predicted_value, fraction_of_positives, mean_predicted_value, alpha=0.2)
    ax1.set_xlabel('Mean Predicted Probability')
    ax1.set_ylabel('Fraction of Positives')
    ax1.set_title(title)
    ax1.legend()
    ax1.set_xlim([0, 1])
    ax1.set_ylim([0, 1])
    
    # Confidence histogram
    ax2.hist(confidences, bins=n_bins, range=(0, 1), edgecolor='black', alpha=0.7)
    ax2.set_xlabel('Confidence')
    ax2.set_ylabel('Count')
    ax2.set_title('Confidence Distribution')
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_uncertainty_vs_accuracy(
    probs: np.ndarray,
    labels: np.ndarray,
    uncertainty: np.ndarray,
    title: str = "Uncertainty vs Accuracy",
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot uncertainty vs prediction accuracy.
    Shows that uncertain predictions are more likely to be wrong.
    """
    predictions = probs.argmax(axis=1)
    correct = (predictions == labels).astype(float)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Scatter plot
    ax1.scatter(uncertainty[correct == 1], [1] * sum(correct == 1), alpha=0.3, label='Correct', c='green')
    ax1.scatter(uncertainty[correct == 0], [0] * sum(correct == 0), alpha=0.3, label='Incorrect', c='red')
    ax1.set_xlabel('Uncertainty')
    ax1.set_ylabel('Correct (1) / Incorrect (0)')
    ax1.set_title('Uncertainty by Prediction Correctness')
    ax1.legend()
    
    # Box plot
    ax2.boxplot([uncertainty[correct == 1], uncertainty[correct == 0]], labels=['Correct', 'Incorrect'])
    ax2.set_ylabel('Uncertainty')
    ax2.set_title('Uncertainty Distribution')
    
    # Statistical test
    stat, p_value = stats.mannwhitneyu(uncertainty[correct == 1], uncertainty[correct == 0])
    ax2.text(0.5, 0.95, f'Mann-Whitney U p-value: {p_value:.4f}', transform=ax2.transAxes, ha='center')
    
    plt.suptitle(title)
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def selective_prediction_curve(
    probs: np.ndarray,
    labels: np.ndarray,
    uncertainty: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate accuracy at different coverage levels based on uncertainty.
    
    Returns:
        coverage: Fraction of samples covered
        accuracy: Accuracy at each coverage level
    """
    # Sort by uncertainty (ascending)
    sorted_indices = np.argsort(uncertainty)
    sorted_probs = probs[sorted_indices]
    sorted_labels = labels[sorted_indices]
    
    predictions = sorted_probs.argmax(axis=1)
    correct = (predictions == sorted_labels).astype(float)
    
    n = len(labels)
    coverage = np.arange(1, n + 1) / n
    accuracy = np.cumsum(correct) / np.arange(1, n + 1)
    
    return coverage, accuracy


def compute_aurc(probs: np.ndarray, labels: np.ndarray, uncertainty: np.ndarray) -> float:
    """
    Compute Area Under Risk-Coverage curve (AURC).
    Lower is better.
    """
    coverage, accuracy = selective_prediction_curve(probs, labels, uncertainty)
    risk = 1 - accuracy
    aurc = np.trapz(risk, coverage)
    return aurc


def uncertainty_metrics_summary(
    probs: np.ndarray,
    labels: np.ndarray,
    uncertainty: np.ndarray
) -> Dict[str, float]:
    """
    Compute comprehensive uncertainty metrics.
    """
    predictions = probs.argmax(axis=1)
    correct = (predictions == labels)
    
    return {
        'ece': expected_calibration_error(probs, labels),
        'mce': maximum_calibration_error(probs, labels),
        'brier': brier_score(probs, labels),
        'aurc': compute_aurc(probs, labels, uncertainty),
        'mean_uncertainty_correct': uncertainty[correct].mean(),
        'mean_uncertainty_incorrect': uncertainty[~correct].mean(),
        'uncertainty_auroc': compute_uncertainty_auroc(correct, uncertainty)
    }


def compute_uncertainty_auroc(correct: np.ndarray, uncertainty: np.ndarray) -> float:
    """
    Compute AUROC for uncertainty-based error detection.
    Higher is better - uncertainty should be higher for incorrect predictions.
    """
    from sklearn.metrics import roc_auc_score
    # We want to detect errors, so incorrect = 1
    return roc_auc_score(~correct, uncertainty)
