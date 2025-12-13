"""
Normalization strategies for EEG covariance computation
All strategies tested systematically with robust SPD enforcement
"""

import numpy as np
from typing import Literal, Optional
from scipy.linalg import fractional_matrix_power


NormStrategy = Literal[
    'none',
    'channel_center',
    'trial_center', 
    'channel_zscore',
    'trial_zscore',
    'soft_norm'
]


def ensure_spd(matrix: np.ndarray, epsilon: float = 1e-4) -> np.ndarray:
    """
    Ensure matrix is strictly symmetric positive definite
    
    This is the PROPER way to handle numerical issues:
    1. Symmetrize (in case of numerical asymmetry)
    2. Eigenvalue decomposition
    3. Clip eigenvalues to epsilon
    4. Reconstruct
    
    Args:
        matrix: Input matrix
        epsilon: Minimum eigenvalue (should be 1e-4 for robustness)
        
    Returns:
        SPD matrix
    """
    # Ensure symmetry
    matrix_sym = (matrix + matrix.T) / 2
    
    # Eigenvalue decomposition
    eigenvalues, eigenvectors = np.linalg.eigh(matrix_sym)
    
    # Clip negative/small eigenvalues
    eigenvalues = np.maximum(eigenvalues, epsilon)
    
    # Reconstruct
    matrix_spd = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T
    
    return matrix_spd.astype(np.float32)


def compute_covariance(
    trial: np.ndarray,
    strategy: NormStrategy = 'channel_center',
    trace_norm: bool = False,
    epsilon: float = 1e-4
) -> np.ndarray:
    """
    Compute covariance matrix with specified normalization
    
    Args:
        trial: (n_channels, n_timepoints) - raw EEG trial
        strategy: Normalization strategy
        trace_norm: Whether to normalize by trace (post-covariance)
        epsilon: Regularization for SPD property (1e-4 is robust)
        
    Returns:
        cov: (n_channels, n_channels) - SPD covariance matrix
    """
    
    # Step 1: Apply normalization to raw signal
    trial_processed = apply_normalization(trial, strategy)
    
    # Step 2: Compute covariance
    cov = np.cov(trial_processed)
    
    # Step 3: Ensure SPD property (PROPER regularization)
    cov = ensure_spd(cov, epsilon=epsilon)
    
    # Step 4: Optional trace normalization
    if trace_norm:
        trace = np.trace(cov)
        if trace > 1e-8:
            cov = cov / trace
            # Re-ensure SPD after trace norm (important!)
            cov = ensure_spd(cov, epsilon=epsilon)
    
    return cov


def apply_normalization(
    trial: np.ndarray,
    strategy: NormStrategy
) -> np.ndarray:
    """
    Apply normalization to raw EEG trial
    
    Args:
        trial: (n_channels, n_timepoints)
        strategy: Normalization strategy
        
    Returns:
        normalized: (n_channels, n_timepoints)
    """
    
    if strategy == 'none':
        return trial
    
    elif strategy == 'channel_center':
        # Remove DC offset per channel
        return trial - trial.mean(axis=1, keepdims=True)
    
    elif strategy == 'trial_center':
        # Remove global mean
        return trial - trial.mean()
    
    elif strategy == 'channel_zscore':
        # Z-score normalize each channel
        mean = trial.mean(axis=1, keepdims=True)
        std = trial.std(axis=1, keepdims=True)
        # Prevent division by zero
        std = np.where(std < 1e-8, 1.0, std)
        return (trial - mean) / std
    
    elif strategy == 'trial_zscore':
        # Z-score normalize entire trial
        mean = trial.mean()
        std = trial.std()
        std = 1.0 if std < 1e-8 else std
        return (trial - mean) / std
    
    elif strategy == 'soft_norm':
        # Soft power normalization
        trial_centered = trial - trial.mean(axis=1, keepdims=True)
        channel_power = np.sqrt((trial_centered ** 2).mean(axis=1, keepdims=True))
        target_power = np.median(channel_power)
        scale_factor = np.clip(target_power / (channel_power + 1e-8), 0.5, 2.0)
        return trial_centered * scale_factor
    
    else:
        raise ValueError(f"Unknown normalization strategy: {strategy}")


def batch_compute_covariances(
    trials: np.ndarray,
    strategy: NormStrategy = 'channel_center',
    trace_norm: bool = False,
    epsilon: float = 1e-4,
    verbose: bool = True
) -> np.ndarray:
    """
    Compute covariances for multiple trials with robust SPD enforcement
    
    Args:
        trials: (n_trials, n_channels, n_timepoints)
        strategy: Normalization strategy
        trace_norm: Whether to normalize by trace
        epsilon: Regularization (1e-4 recommended)
        verbose: Show progress bar
        
    Returns:
        covariances: (n_trials, n_channels, n_channels)
    """
    n_trials = len(trials)
    n_channels = trials.shape[1]
    
    covariances = np.zeros((n_trials, n_channels, n_channels), dtype=np.float32)
    
    iterator = range(n_trials)
    if verbose:
        from tqdm import tqdm
        iterator = tqdm(iterator, desc=f"Computing covs ({strategy})")
    
    for i in iterator:
        covariances[i] = compute_covariance(
            trials[i],
            strategy=strategy,
            trace_norm=trace_norm,
            epsilon=epsilon
        )
    
    return covariances


# Verify SPD property of batch
def verify_spd_batch(covariances: np.ndarray, epsilon: float = 1e-4) -> dict:
    """
    Verify that all matrices in batch are SPD
    
    Returns statistics about eigenvalues
    """
    min_eigenvalues = []
    
    for cov in covariances:
        eigenvalues = np.linalg.eigvalsh(cov)
        min_eigenvalues.append(eigenvalues.min())
    
    return {
        'all_positive': all(e > 0 for e in min_eigenvalues),
        'min_eigenvalue': min(min_eigenvalues),
        'mean_min_eigenvalue': np.mean(min_eigenvalues),
        'below_epsilon': sum(e < epsilon for e in min_eigenvalues)
    }


# All strategies to test
ALL_STRATEGIES = [
    ('none', False),
    ('channel_center', False),
    ('channel_center', True),
    ('trial_center', False),
    ('trial_center', True),
    ('channel_zscore', False),
    ('channel_zscore', True),
    ('soft_norm', False),
]


def get_strategy_name(strategy: str, trace_norm: bool) -> str:
    """Human-readable name for strategy"""
    name = strategy.replace('_', ' ').title()
    if trace_norm:
        name += " + Trace"
    return name