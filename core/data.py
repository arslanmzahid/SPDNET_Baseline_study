"""
Data loading and dataset creation
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, Optional
from sklearn.utils.class_weight import compute_class_weight


class CovarianceDataset(Dataset):
    """PyTorch dataset for covariance matrices"""
    
    def __init__(self, covariances: np.ndarray, labels: np.ndarray):
        self.covariances = torch.FloatTensor(covariances)
        self.labels = torch.LongTensor(labels)
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return self.covariances[idx], self.labels[idx]


def load_data(data_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load raw EEG data
    
    Returns:
        trials: (n_trials, n_channels, n_timepoints)
        labels: (n_trials,)
        subject_ids: (n_trials,)
    """
    data = np.load(data_path)
    
    trials = data['data']  # (2952, 30, 384)
    labels = data['labels']  # (2952,)
    subjects = data['subject_ids']  # (2952,)
    
    print(f"Loaded data:")
    print(f"  Trials: {trials.shape}")
    print(f"  Subjects: {np.unique(subjects)}")
    print(f"  Labels: Alert={np.sum(labels==0)}, Fatigue={np.sum(labels==1)}")
    
    return trials, labels, subjects


def get_class_weights(labels: np.ndarray, boost_factor: float = 1.0) -> torch.Tensor:
    """
    Compute class weights for imbalanced data
    
    Args:
        labels: Class labels
        boost_factor: Additional boost for minority class (1.0 = no boost)
        
    Returns:
        weights: Tensor of class weights
    """
    weights = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(labels),
        y=labels
    )
    
    # Apply boost to minority class
    if boost_factor > 1.0:
        n_counts = [np.sum(labels == c) for c in np.unique(labels)]
        minority_idx = np.argmin(n_counts)
        weights[minority_idx] *= boost_factor
    
    return torch.FloatTensor(weights)


def create_dataloaders(
    train_covs: np.ndarray,
    train_labels: np.ndarray,
    val_covs: np.ndarray,
    val_labels: np.ndarray,
    test_covs: np.ndarray,
    test_labels: np.ndarray,
    batch_size: int = 64,
    num_workers: int = 0
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create train/val/test dataloaders"""
    
    train_dataset = CovarianceDataset(train_covs, train_labels)
    val_dataset = CovarianceDataset(val_covs, val_labels)
    test_dataset = CovarianceDataset(test_covs, test_labels)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader