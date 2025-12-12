"""
Normalization strategies for EEG covariance computation
All strategies tested systematically
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


def compute_covariance(
    trial: np.ndarray,
    strategy: NormStrategy = 'channel_center',
    trace_norm: bool = False,
    epsilon: float = 1e-6
) -> np.ndarray:
    """
    Compute covariance matrix with specified normalization
    
    Args:
        trial: (n_channels, n_timepoints) - raw EEG trial
        strategy: Normalization strategy
        trace_norm: Whether to normalize by trace (post-covariance)
        epsilon: Regularization for SPD property
        
    Returns:
        cov: (n_channels, n_channels) - SPD covariance matrix
    """
    
    # Step 1: Apply normalization to raw signal
    trial_processed = apply_normalization(trial, strategy)
    
    # Step 2: Compute covariance
    cov = np.cov(trial_processed)
    
    # Step 3: Ensure SPD property
    cov = cov + epsilon * np.eye(cov.shape[0])
    
    # Step 4: Optional trace normalization
    if trace_norm:
        trace = np.trace(cov)
        if trace > 1e-8:
            cov = cov / trace
    
    return cov.astype(np.float32)


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
        # Preserves: Full power information
        # Removes: DC drift
        return trial - trial.mean(axis=1, keepdims=True)
    
    elif strategy == 'trial_center':
        # Remove global mean across all channels and time
        # Preserves: Relative channel differences
        # Removes: Global baseline shift
        return trial - trial.mean()
    
    elif strategy == 'channel_zscore':
        # Z-score normalize each channel independently
        # Preserves: Correlation structure
        # Removes: Power differences between channels (DANGEROUS for fatigue!)
        mean = trial.mean(axis=1, keepdims=True)
        std = trial.std(axis=1, keepdims=True)
        std = np.where(std < 1e-8, 1.0, std)
        return (trial - mean) / std
    
    elif strategy == 'trial_zscore':
        # Z-score normalize entire trial
        # Preserves: Correlation structure
        # Removes: All power information (VERY DANGEROUS for fatigue!)
        mean = trial.mean()
        std = trial.std()
        std = 1.0 if std < 1e-8 else std
        return (trial - mean) / std
    
    elif strategy == 'soft_norm':
        # Soft power normalization: reduce extreme channel variance differences
        # Preserves: Relative power information (partially)
        # Removes: Extreme channel imbalances
        
        # Center first
        trial_centered = trial - trial.mean(axis=1, keepdims=True)
        
        # Compute per-channel power
        channel_power = np.sqrt((trial_centered ** 2).mean(axis=1, keepdims=True))
        
        # Soft scaling: bring to similar range but don't equalize completely
        target_power = np.median(channel_power)
        scale_factor = np.clip(target_power / channel_power, 0.5, 2.0)
        
        return trial_centered * scale_factor
    
    else:
        raise ValueError(f"Unknown normalization strategy: {strategy}")


def batch_compute_covariances(
    trials: np.ndarray,
    strategy: NormStrategy = 'channel_center',
    trace_norm: bool = False,
    epsilon: float = 1e-6,
    verbose: bool = True
) -> np.ndarray:
    """
    Compute covariances for multiple trials
    
    Args:
        trials: (n_trials, n_channels, n_timepoints)
        strategy: Normalization strategy
        trace_norm: Whether to normalize by trace
        epsilon: Regularization
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