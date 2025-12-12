"""
Alignment methods for cross-subject transfer
Implementations: RA, RPA-LEM
"""

import numpy as np
from scipy.linalg import logm, expm, fractional_matrix_power, orthogonal_procrustes
from typing import Tuple, Optional


class AlignmentMethod:
    """Base class for alignment methods"""
    
    def fit(self, source_covs: np.ndarray, target_covs: np.ndarray):
        """Learn alignment parameters"""
        raise NotImplementedError
    
    def transform(self, covs: np.ndarray) -> np.ndarray:
        """Apply alignment"""
        raise NotImplementedError
    
    def fit_transform(self, source_covs: np.ndarray, target_covs: np.ndarray) -> np.ndarray:
        """Fit and transform"""
        self.fit(source_covs, target_covs)
        return self.transform(source_covs)


class RiemannianAlignment(AlignmentMethod):
    """
    Basic Riemannian Alignment (RA)
    Re-centers source distribution to target mean
    
    Reference: Zanini et al. (2018)
    """
    
    def __init__(self):
        self.C_source = None
        self.C_target = None
        self.C_source_invsqrt = None
        self.C_target_sqrt = None
    
    def _compute_mean(self, covs: np.ndarray) -> np.ndarray:
        """Compute Riemannian mean using iterative algorithm"""
        from pyriemann.utils.mean import mean_covariance
        return mean_covariance(covs, metric='riemann')
    
    def fit(self, source_covs: np.ndarray, target_covs: np.ndarray):
        """
        Args:
            source_covs: (n_source, n_channels, n_channels)
            target_covs: (n_target, n_channels, n_channels)
        """
        # Compute Riemannian means
        self.C_source = self._compute_mean(source_covs)
        self.C_target = self._compute_mean(target_covs)
        
        # Precompute transformation matrices
        self.C_source_invsqrt = fractional_matrix_power(self.C_source, -0.5)
        self.C_target_sqrt = fractional_matrix_power(self.C_target, 0.5)
        
        return self
    
    def transform(self, covs: np.ndarray) -> np.ndarray:
        """
        Transform covariances: re-center from source to target
        
        Formula: C_aligned = C_target^(1/2) @ C_source^(-1/2) @ C @ C_source^(-1/2) @ C_target^(1/2)
        """
        if self.C_source is None:
            raise ValueError("Must call fit() before transform()")
        
        aligned = []
        for C in covs:
            # Whiten with source
            C_whitened = self.C_source_invsqrt @ C @ self.C_source_invsqrt
            
            # Re-color with target
            C_aligned = self.C_target_sqrt @ C_whitened @ self.C_target_sqrt
            
            aligned.append(C_aligned)
        
        return np.array(aligned)


class RPA_LEM(AlignmentMethod):
    """
    Riemannian Procrustes Analysis with Log-Euclidean Metric
    Fast alignment using log-domain operations
    
    Reference: Paper we reviewed (2024)
    Advantage: 10-100x faster than standard RPA while maintaining accuracy
    """
    
    def __init__(self):
        self.mean_source = None
        self.mean_target = None
        self.mean_source_invsqrt = None
        self.mean_target_sqrt = None
        self.dispersion_ratio = None
        self.rotation = None
    
    def _log_euclidean_mean(self, covs: np.ndarray) -> np.ndarray:
        """Compute mean in log-Euclidean space (much faster!)"""
        log_covs = np.array([logm(C) for C in covs])
        mean_log = log_covs.mean(axis=0)
        return expm(mean_log)
    
    def _compute_dispersion(self, covs: np.ndarray, mean_cov: np.ndarray) -> float:
        """Compute dispersion (spread) of covariances"""
        mean_invsqrt = fractional_matrix_power(mean_cov, -0.5)
        
        dispersions = []
        for C in covs:
            # Center at identity
            C_centered = mean_invsqrt @ C @ mean_invsqrt
            # Compute Frobenius norm in log space
            log_C_centered = logm(C_centered)
            disp = np.linalg.norm(log_C_centered, 'fro')
            dispersions.append(disp)
        
        return np.mean(dispersions)
    
    def fit(self, source_covs: np.ndarray, target_covs: np.ndarray):
        """
        Learn alignment: re-centering + re-scaling + rotation
        
        All operations in log-Euclidean space for speed!
        """
        # Step 1: Compute log-Euclidean means
        self.mean_source = self._log_euclidean_mean(source_covs)
        self.mean_target = self._log_euclidean_mean(target_covs)
        
        # Precompute for transformations
        self.mean_source_invsqrt = fractional_matrix_power(self.mean_source, -0.5)
        self.mean_target_sqrt = fractional_matrix_power(self.mean_target, 0.5)
        
        # Step 2: Compute dispersions
        disp_source = self._compute_dispersion(source_covs, self.mean_source)
        disp_target = self._compute_dispersion(target_covs, self.mean_target)
        
        self.dispersion_ratio = disp_target / disp_source if disp_source > 1e-8 else 1.0
        
        # Step 3: Compute optimal rotation (Procrustes in log space)
        # Center both at identity
        source_centered = []
        target_centered = []
        
        for C_s in source_covs[:min(100, len(source_covs))]:  # Use subset for speed
            C_s_white = self.mean_source_invsqrt @ C_s @ self.mean_source_invsqrt
            source_centered.append(logm(C_s_white))
        
        for C_t in target_covs[:min(100, len(target_covs))]:
            C_t_white = fractional_matrix_power(self.mean_target, -0.5) @ C_t @ fractional_matrix_power(self.mean_target, -0.5)
            target_centered.append(logm(C_t_white))
        
        # Find optimal rotation via Procrustes
        source_centered = np.array(source_centered).reshape(len(source_centered), -1)
        target_centered = np.array(target_centered).reshape(len(target_centered), -1)
        
        self.rotation, _ = orthogonal_procrustes(source_centered, target_centered)
        
        return self
    
    def transform(self, covs: np.ndarray) -> np.ndarray:
        """
        Apply learned alignment: re-center → re-scale → rotate
        """
        if self.mean_source is None:
            raise ValueError("Must call fit() before transform()")
        
        aligned = []
        n_channels = covs.shape[1]
        
        for C in covs:
            # Step 1: Whiten with source mean
            C_white = self.mean_source_invsqrt @ C @ self.mean_source_invsqrt
            
            # Step 2: Scale dispersion
            C_scaled = fractional_matrix_power(C_white, self.dispersion_ratio)
            
            # Step 3: Rotate in log space
            log_C_scaled = logm(C_scaled)
            log_C_rotated = (self.rotation @ log_C_scaled.reshape(-1)).reshape(n_channels, n_channels)
            C_rotated = expm(log_C_rotated)
            
            # Step 4: Re-color with target mean
            C_aligned = self.mean_target_sqrt @ C_rotated @ self.mean_target_sqrt
            
            aligned.append(C_aligned)
        
        return np.array(aligned)


def apply_alignment(
    train_covs: np.ndarray,
    test_covs: np.ndarray,
    method: Optional[str] = None,
    n_reference: int = 10
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply alignment method
    
    Args:
        train_covs: Training covariances
        test_covs: Test covariances
        method: 'ra', 'rpa_lem', or None
        n_reference: Number of test samples to use as reference
        
    Returns:
        train_aligned, test_covs (test unchanged)
    """
    if method is None or method == 'none':
        return train_covs, test_covs
    
    # Use small subset of test as reference for alignment
    test_reference = test_covs[:min(n_reference, len(test_covs))]
    
    if method == 'ra':
        aligner = RiemannianAlignment()
    elif method == 'rpa_lem':
        aligner = RPA_LEM()
    else:
        raise ValueError(f"Unknown alignment method: {method}")
    
    # Fit on train (source) and test reference (target)
    aligner.fit(train_covs, test_reference)
    
    # Transform only training data
    train_aligned = aligner.transform(train_covs)
    
    return train_aligned, test_covs