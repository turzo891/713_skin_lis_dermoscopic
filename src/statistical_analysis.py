"""
Statistical Analysis Module for Model Comparison.
Implements various statistical tests for rigorous research comparisons.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from scipy import stats
from sklearn.metrics import accuracy_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns


def paired_ttest(scores1: np.ndarray, scores2: np.ndarray) -> Tuple[float, float]:
    """
    Paired t-test for comparing two models.
    
    Args:
        scores1: Scores from model 1 (e.g., per-fold accuracies)
        scores2: Scores from model 2
        
    Returns:
        t_statistic, p_value
    """
    t_stat, p_value = stats.ttest_rel(scores1, scores2)
    return t_stat, p_value


def wilcoxon_signed_rank(scores1: np.ndarray, scores2: np.ndarray) -> Tuple[float, float]:
    """
    Wilcoxon signed-rank test (non-parametric alternative to paired t-test).
    
    Args:
        scores1: Scores from model 1
        scores2: Scores from model 2
        
    Returns:
        statistic, p_value
    """
    try:
        stat, p_value = stats.wilcoxon(scores1, scores2, alternative='two-sided')
    except ValueError:  # If all differences are zero
        stat, p_value = 0.0, 1.0
    return stat, p_value


def mcnemar_test(y_true: np.ndarray, pred1: np.ndarray, pred2: np.ndarray) -> Tuple[float, float]:
    """
    McNemar's test for comparing two classifiers.
    
    Args:
        y_true: True labels
        pred1: Predictions from model 1
        pred2: Predictions from model 2
        
    Returns:
        chi2_statistic, p_value
    """
    # Build contingency table
    correct1 = (pred1 == y_true)
    correct2 = (pred2 == y_true)
    
    # b: Model 1 correct, Model 2 wrong
    # c: Model 1 wrong, Model 2 correct
    b = np.sum(correct1 & ~correct2)
    c = np.sum(~correct1 & correct2)
    
    if b + c == 0:
        return 0.0, 1.0
    
    # McNemar's test with continuity correction
    chi2 = (abs(b - c) - 1) ** 2 / (b + c)
    p_value = 1 - stats.chi2.cdf(chi2, df=1)
    
    return chi2, p_value


def cochrans_q_test(predictions_matrix: np.ndarray, y_true: np.ndarray) -> Tuple[float, float]:
    """
    Cochran's Q test for comparing multiple classifiers.
    
    Args:
        predictions_matrix: (n_samples, n_classifiers) binary correctness matrix
        y_true: True labels (not used directly, correctness assumed computed)
        
    Returns:
        Q_statistic, p_value
    """
    n, k = predictions_matrix.shape
    
    # Row and column sums
    row_sums = predictions_matrix.sum(axis=1)
    col_sums = predictions_matrix.sum(axis=0)
    
    total = predictions_matrix.sum()
    
    # Q statistic
    numerator = (k - 1) * (k * np.sum(col_sums ** 2) - total ** 2)
    denominator = k * total - np.sum(row_sums ** 2)
    
    if denominator == 0:
        return 0.0, 1.0
    
    Q = numerator / denominator
    p_value = 1 - stats.chi2.cdf(Q, df=k - 1)
    
    return Q, p_value


def friedman_test(scores_matrix: np.ndarray) -> Tuple[float, float]:
    """
    Friedman test for comparing multiple models across multiple datasets/folds.
    
    Args:
        scores_matrix: (n_datasets, n_models) matrix of scores
        
    Returns:
        statistic, p_value
    """
    stat, p_value = stats.friedmanchisquare(*scores_matrix.T)
    return stat, p_value


def nemenyi_post_hoc(scores_matrix: np.ndarray, alpha: float = 0.05) -> Tuple[np.ndarray, float]:
    """
    Nemenyi post-hoc test after Friedman test.
    
    Args:
        scores_matrix: (n_datasets, n_models) matrix of scores
        alpha: Significance level
        
    Returns:
        critical_difference, average_ranks
    """
    n_datasets, n_models = scores_matrix.shape
    
    # Calculate average ranks
    ranks = np.zeros_like(scores_matrix)
    for i in range(n_datasets):
        ranks[i] = stats.rankdata(-scores_matrix[i])  # Negative for descending
    
    avg_ranks = ranks.mean(axis=0)
    
    # Critical difference
    q_alpha = {
        (0.05, 2): 1.960, (0.05, 3): 2.343, (0.05, 4): 2.569,
        (0.05, 5): 2.728, (0.05, 6): 2.850, (0.05, 7): 2.949,
        (0.10, 2): 1.645, (0.10, 3): 2.052, (0.10, 4): 2.291,
        (0.10, 5): 2.459, (0.10, 6): 2.589, (0.10, 7): 2.693
    }
    
    q = q_alpha.get((alpha, n_models), 2.728)  # Default to k=5
    cd = q * np.sqrt(n_models * (n_models + 1) / (6 * n_datasets))
    
    return cd, avg_ranks


def bootstrap_ci(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric_fn,
    n_bootstrap: int = 1000,
    confidence: float = 0.95,
    seed: int = 42
) -> Tuple[float, float, float]:
    """
    Bootstrap confidence interval for any metric.
    
    Args:
        y_true: True labels
        y_pred: Predictions
        metric_fn: Function that takes (y_true, y_pred) and returns a score
        n_bootstrap: Number of bootstrap samples
        confidence: Confidence level (0.95 = 95% CI)
        seed: Random seed
        
    Returns:
        point_estimate, lower_ci, upper_ci
    """
    np.random.seed(seed)
    n = len(y_true)
    scores = []
    
    for _ in range(n_bootstrap):
        indices = np.random.choice(n, n, replace=True)
        score = metric_fn(y_true[indices], y_pred[indices])
        scores.append(score)
    
    point_estimate = metric_fn(y_true, y_pred)
    lower = np.percentile(scores, (1 - confidence) / 2 * 100)
    upper = np.percentile(scores, (1 + confidence) / 2 * 100)
    
    return point_estimate, lower, upper


def delong_test(y_true: np.ndarray, probs1: np.ndarray, probs2: np.ndarray) -> Tuple[float, float]:
    """
    DeLong test for comparing two AUC values.
    
    Args:
        y_true: True binary labels
        probs1: Predicted probabilities from model 1
        probs2: Predicted probabilities from model 2
        
    Returns:
        z_statistic, p_value
    """
    from sklearn.metrics import roc_auc_score
    
    n1 = np.sum(y_true == 1)
    n0 = np.sum(y_true == 0)
    
    auc1 = roc_auc_score(y_true, probs1)
    auc2 = roc_auc_score(y_true, probs2)
    
    # Simplified DeLong test (approximation)
    # Full implementation requires structural components
    diff = auc1 - auc2
    
    # Variance estimation (simplified)
    var1 = auc1 * (1 - auc1) / min(n0, n1)
    var2 = auc2 * (1 - auc2) / min(n0, n1)
    
    # Assume some correlation
    cov = 0.5 * np.sqrt(var1 * var2)
    
    se = np.sqrt(var1 + var2 - 2 * cov)
    
    if se == 0:
        return 0.0, 1.0
    
    z = diff / se
    p_value = 2 * (1 - stats.norm.cdf(abs(z)))
    
    return z, p_value


def effect_size_cohens_d(scores1: np.ndarray, scores2: np.ndarray) -> float:
    """
    Calculate Cohen's d effect size.
    
    Args:
        scores1: Scores from group 1
        scores2: Scores from group 2
        
    Returns:
        Cohen's d value
        - Small: 0.2
        - Medium: 0.5
        - Large: 0.8
    """
    n1, n2 = len(scores1), len(scores2)
    var1, var2 = scores1.var(), scores2.var()
    
    # Pooled standard deviation
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    
    if pooled_std == 0:
        return 0.0
    
    d = (scores1.mean() - scores2.mean()) / pooled_std
    return d


def compare_models_comprehensive(
    model_results: Dict[str, Dict],
    y_true: np.ndarray,
    model_predictions: Dict[str, np.ndarray]
) -> pd.DataFrame:
    """
    Comprehensive pairwise model comparison.
    
    Args:
        model_results: Dict of {model_name: {fold_scores: [...], ...}}
        y_true: True labels
        model_predictions: Dict of {model_name: predictions}
        
    Returns:
        DataFrame with pairwise comparisons
    """
    model_names = list(model_results.keys())
    comparisons = []
    
    for i, model1 in enumerate(model_names):
        for model2 in model_names[i+1:]:
            scores1 = np.array(model_results[model1].get('fold_scores', [0]))
            scores2 = np.array(model_results[model2].get('fold_scores', [0]))
            
            pred1 = model_predictions.get(model1, np.zeros_like(y_true))
            pred2 = model_predictions.get(model2, np.zeros_like(y_true))
            
            # Paired t-test
            t_stat, t_pval = paired_ttest(scores1, scores2) if len(scores1) > 1 else (0, 1)
            
            # Wilcoxon
            w_stat, w_pval = wilcoxon_signed_rank(scores1, scores2) if len(scores1) > 1 else (0, 1)
            
            # McNemar
            mc_stat, mc_pval = mcnemar_test(y_true, pred1, pred2)
            
            # Effect size
            cohen_d = effect_size_cohens_d(scores1, scores2) if len(scores1) > 1 else 0
            
            comparisons.append({
                'Model 1': model1,
                'Model 2': model2,
                'Mean Diff': scores1.mean() - scores2.mean() if len(scores1) > 0 else 0,
                't-test p': t_pval,
                'Wilcoxon p': w_pval,
                'McNemar p': mc_pval,
                "Cohen's d": cohen_d,
                'Significant (p<0.05)': t_pval < 0.05 or mc_pval < 0.05
            })
    
    return pd.DataFrame(comparisons)


def plot_critical_difference_diagram(
    avg_ranks: np.ndarray,
    model_names: List[str],
    cd: float,
    title: str = "Critical Difference Diagram",
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot critical difference diagram for model comparison.
    
    Args:
        avg_ranks: Average ranks for each model
        model_names: Names of models
        cd: Critical difference value
        title: Plot title
        save_path: Path to save figure
    """
    n_models = len(model_names)
    
    # Sort by rank
    sorted_indices = np.argsort(avg_ranks)
    sorted_ranks = avg_ranks[sorted_indices]
    sorted_names = [model_names[i] for i in sorted_indices]
    
    fig, ax = plt.subplots(figsize=(10, 4))
    
    # Draw axis
    ax.plot([1, n_models], [0, 0], 'k-', linewidth=2)
    
    # Draw ticks
    for i in range(1, n_models + 1):
        ax.plot([i, i], [-0.1, 0.1], 'k-', linewidth=1)
        ax.text(i, -0.2, str(i), ha='center', va='top')
    
    # Plot model positions
    for i, (rank, name) in enumerate(zip(sorted_ranks, sorted_names)):
        y_pos = 0.5 + i * 0.3
        ax.plot(rank, y_pos, 'ko', markersize=8)
        ax.text(rank, y_pos + 0.1, name, ha='center', va='bottom', fontsize=10)
        ax.plot([rank, rank], [0, y_pos], 'k--', alpha=0.3)
    
    # Draw CD bar
    ax.plot([1, 1 + cd], [0.3, 0.3], 'r-', linewidth=3)
    ax.text(1 + cd/2, 0.4, f'CD = {cd:.2f}', ha='center', color='red')
    
    ax.set_xlim(0.5, n_models + 0.5)
    ax.set_ylim(-0.5, 1.5 + n_models * 0.3)
    ax.axis('off')
    ax.set_title(title)
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def generate_statistical_report(
    comparison_df: pd.DataFrame,
    save_path: Optional[str] = None
) -> str:
    """
    Generate a formatted statistical analysis report.
    """
    report = "# Statistical Analysis Report\n\n"
    report += "## Pairwise Model Comparisons\n\n"
    report += comparison_df.to_markdown(index=False)
    report += "\n\n"
    
    # Summary
    significant = comparison_df[comparison_df['Significant (p<0.05)']]
    report += f"## Summary\n\n"
    report += f"- Total comparisons: {len(comparison_df)}\n"
    report += f"- Significant differences (p<0.05): {len(significant)}\n"
    
    if len(significant) > 0:
        report += "\n### Significant Differences:\n\n"
        for _, row in significant.iterrows():
            winner = row['Model 1'] if row['Mean Diff'] > 0 else row['Model 2']
            report += f"- {row['Model 1']} vs {row['Model 2']}: {winner} is better "
            report += f"(p={min(row['t-test p'], row['McNemar p']):.4f}, d={row[\"Cohen's d\"]:.3f})\n"
    
    if save_path:
        with open(save_path, 'w') as f:
            f.write(report)
    
    return report
