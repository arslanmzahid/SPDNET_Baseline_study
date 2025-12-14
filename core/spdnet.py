"""
SPDNet: Symmetric Positive Definite Network
Optimized for EEG fatigue detection
"""

import torch
import torch.nn as nn
from typing import Tuple


class BiMap(nn.Module):
    """Bilinear mapping on SPD manifold"""
    
    def __init__(self, input_dim: int, output_dim: int, epsilon: float = 1e-4):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.epsilon = epsilon  # ADD THIS
        self.W = nn.Parameter(torch.empty(output_dim, input_dim))
        nn.init.orthogonal_(self.W)
    
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """X: (batch, input_dim, input_dim) -> (batch, output_dim, output_dim)"""
        result = self.W @ X @ self.W.transpose(-2, -1)
        
        # ADD REGULARIZATION HERE (crucial for stability)
        result = (result + result.transpose(-2, -1)) / 2  # Ensure symmetry
        # Add small diagonal regularization
        identity = torch.eye(result.size(-1), device=result.device, dtype=result.dtype)
        result = result + self.epsilon * identity
        
        return result


class ReEig(nn.Module):
    """Eigenvalue rectification to ensure SPD property"""
    
    def __init__(self, epsilon: float = 1e-4):
        super().__init__()
        self.epsilon = epsilon
    
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        eigenvalues, eigenvectors = torch.linalg.eigh(X)
        eigenvalues = torch.clamp(eigenvalues, min=self.epsilon)
        return eigenvectors @ torch.diag_embed(eigenvalues) @ eigenvectors.transpose(-2, -1)


class LogEig(nn.Module):
    """Logarithmic eigenvalue mapping to tangent space"""
    
    def __init__(self, epsilon: float = 1e-4):
        super().__init__()
        self.epsilon = epsilon
    
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        eigenvalues, eigenvectors = torch.linalg.eigh(X)
        eigenvalues = torch.clamp(eigenvalues, min=self.epsilon)
        log_eigenvalues = torch.log(eigenvalues)
        return eigenvectors @ torch.diag_embed(log_eigenvalues) @ eigenvectors.transpose(-2, -1)


class SPDNet(nn.Module):
    """
    Complete SPDNet architecture
    Based on: Huang & Van Gool (2017) + Phase-SPDNet (2024)
    """
    
    def __init__(
        self,
        input_dim: int = 30,
        bimap1_out: int = 20,
        bimap2_out: int = 15,
        epsilon: float = 1e-4
    ):
        super().__init__()
        
        assert bimap2_out < bimap1_out, "bimap2_out must be < bimap1_out"
        
        self.input_dim = input_dim
        self.final_dim = bimap2_out
        
        # Layer 1: 30 -> 20
        self.bimap1 = BiMap(input_dim, bimap1_out, epsilon)  # ADD epsilon
        self.reeig1 = ReEig(epsilon)
        
        # Layer 2: 20 -> 15
        self.bimap2 = BiMap(bimap1_out, bimap2_out, epsilon)  # ADD epsilon
        self.reeig2 = ReEig(epsilon)
        
        # Mapping to tangent space
        self.logeig = LogEig(epsilon)
    
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        Args:
            X: (batch, n_channels, n_channels) - SPD matrices
        Returns:
            features: (batch, d) - Flattened upper triangular of log matrix
        """
        # Layer 1
        X = self.bimap1(X)
        X = self.reeig1(X)
        
        # Layer 2
        X = self.bimap2(X)
        X = self.reeig2(X)
        
        # Log mapping
        X = self.logeig(X)
        
        # Flatten upper triangular (symmetric matrix -> vector)
        batch_size = X.shape[0]
        indices = torch.triu_indices(self.final_dim, self.final_dim)
        features = X[:, indices[0], indices[1]]
        
        return features


class SPDNetClassifier(nn.Module):
    """SPDNet + Classification Head"""
    
    def __init__(
        self,
        n_channels: int = 30,
        bimap1_out: int = 20,
        bimap2_out: int = 15,
        hidden_dim: int = 256,
        num_classes: int = 2,
        dropout: float = 0.5,
        epsilon: float = 1e-4
    ):
        super().__init__()
        
        # SPDNet feature extractor
        self.spdnet = SPDNet(
            input_dim=n_channels,
            bimap1_out=bimap1_out,
            bimap2_out=bimap2_out,
            epsilon=epsilon
        )
        
        # Feature dimension after SPDNet
        spdnet_output_dim = bimap2_out * (bimap2_out + 1) // 2
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(spdnet_output_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )
    
    def forward(self, cov_matrices: torch.Tensor) -> torch.Tensor:
        features = self.spdnet(cov_matrices)
        logits = self.classifier(features)
        return logits
    
    def get_num_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)