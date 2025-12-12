"""
Comprehensive evaluation metrics and statistical tests
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, f1_score,
    precision_score, recall_score, confusion_matrix,
    cohen_kappa_score, matthews_corrcoef
)
from scipy.stats import friedmanchisquare, wilcoxon, ttest_rel
from typing import Dict, List, Tuple


def compute_all_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Compute comprehensive classification metrics
    
    Returns:
        Dictionary with all metrics
    """
    return {
        'accuracy': accuracy_score(y_true, y_pred),
        'balanced_accuracy': balanced_accuracy_score(y_true, y_pred),
        'f1_weighted': f1_score(y_true, y_pred, average='weighted'),
        'f1_macro': f1_score(y_true, y_pred, average='macro'),
        'precision_weighted': precision_score(y_true, y_pred, average='weighted', zero_division=0),
        'recall_weighted': recall_score(y_true, y_pred, average='weighted', zero_division=0),
        'cohen_kappa': cohen_kappa_score(y_true, y_pred),
        'matthews_corrcoef': matthews_corrcoef(y_true, y_pred),
        'confusion_matrix': confusion_matrix(y_true, y_pred)
    }


def compute_per_class_metrics(y_true: np.ndarray, y_pred: np.ndarray, class_names: List[str] = None) -> Dict:
    """
    Compute per-class precision, recall, F1
    """
    if class_names is None:
        class_names = [f'Class {i}' for i in range(len(np.unique(y_true)))]
    
    precision = precision_score(y_true, y_pred, average=None, zero_division=0)
    recall = recall_score(y_true, y_pred, average=None, zero_division=0)
    f1 = f1_score(y_true, y_pred, average=None, zero_division=0)
    
    return {
        class_name: {
            'precision': float(p),
            'recall': float(r),
            'f1': float(f)
        }
        for class_name, p, r, f in zip(class_names, precision, recall, f1)
    }


def friedman_test(results: Dict[str, List[float]]) -> Tuple[float, float]:
    """
    Friedman test for comparing multiple methods
    
    Args:
        results: Dict mapping method_name -> list of scores per subject
        
    Returns:
        statistic, p_value
    """
    # Organize data as (n_subjects, n_methods)
    method_names = list(results.keys())
    n_subjects = len(results[method_names[0]])
    
    # Verify all methods have same number of subjects
    assert all(len(scores) == n_subjects for scores in results.values())
    
    # Create matrix
    data = np.array([results[method] for method in method_names]).T
    
    # Run Friedman test
    statistic, p_value = friedmanchisquare(*data.T)
    
    return statistic, p_value


def wilcoxon_pairwise(method1_scores: List[float], method2_scores: List[float]) -> Tuple[float, float]:
    """
    Wilcoxon signed-rank test for paired comparison
    
    Returns:
        statistic, p_value
    """
    statistic, p_value = wilcoxon(method1_scores, method2_scores, alternative='two-sided')
    return statistic, p_value


def mcnemar_test(y_true: np.ndarray, pred1: np.ndarray, pred2: np.ndarray) -> float:
    """
    McNemar's test for comparing two classifiers
    
    Returns:
        p_value
    """
    from scipy.stats import chi2
    
    # Create contingency table
    n01 = np.sum((pred1 != y_true) & (pred2 == y_true))  # Model 1 wrong, Model 2 correct
    n10 = np.sum((pred1 == y_true) & (pred2 != y_true))  # Model 1 correct, Model 2 wrong
    
    # McNemar statistic
    statistic = (abs(n01 - n10) - 1) ** 2 / (n01 + n10) if (n01 + n10) > 0 else 0
    
    # P-value from chi-square distribution with 1 df
    p_value = 1 - chi2.cdf(statistic, df=1)
    
    return p_value


def bootstrap_ci(scores: np.ndarray, n_bootstrap: int = 10000, confidence: float = 0.95) -> Tuple[float, float, float]:
    """
    Bootstrap confidence interval
    
    Returns:
        mean, lower_ci, upper_ci
    """
    bootstrap_means = []
    
    for _ in range(n_bootstrap):
        sample = np.random.choice(scores, size=len(scores), replace=True)
        bootstrap_means.append(np.mean(sample))
    
    bootstrap_means = np.array(bootstrap_means)
    
    alpha = 1 - confidence
    lower = np.percentile(bootstrap_means, 100 * alpha / 2)
    upper = np.percentile(bootstrap_means, 100 * (1 - alpha / 2))
    
    return np.mean(scores), lower, upper