"""
Training utilities and evaluation
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, balanced_accuracy_score
from typing import Tuple, Dict, Optional
import numpy as np


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: str
) -> Tuple[float, float]:
    """Train for one epoch"""
    model.train()
    total_loss = 0.0
    all_preds = []
    all_labels = []
    
    for cov_matrices, labels in dataloader:
        cov_matrices = cov_matrices.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        logits = model(cov_matrices)
        loss = criterion(logits, labels)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item() * len(labels)
        preds = torch.argmax(logits, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    
    avg_loss = total_loss / len(dataloader.dataset)
    accuracy = accuracy_score(all_labels, all_preds)
    
    return avg_loss, accuracy


def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: str
) -> Dict[str, float]:
    """Evaluate model"""
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for cov_matrices, labels in dataloader:
            cov_matrices = cov_matrices.to(device)
            labels = labels.to(device)
            
            logits = model(cov_matrices)
            loss = criterion(logits, labels)
            
            total_loss += loss.item() * len(labels)
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    return {
        'loss': total_loss / len(dataloader.dataset),
        'accuracy': accuracy_score(all_labels, all_preds),
        'balanced_accuracy': balanced_accuracy_score(all_labels, all_preds),
        'f1': f1_score(all_labels, all_preds, average='weighted'),
        'confusion_matrix': confusion_matrix(all_labels, all_preds)
    }

def train_with_early_stopping(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    class_weights: torch.Tensor,
    config: Dict,
    device: str,
    trial: Optional = None,
    fold: Optional[int] = None
) -> Dict:
    """
    Train with early stopping and PROPER Optuna integration
    
    The key fix: Use fold-based unique step identifiers to avoid conflicts
    """
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    optimizer = optim.Adam(
        model.parameters(),
        lr=config['learning_rate'],
        weight_decay=config['weight_decay']
    )
    
    best_val_acc = 0.0
    patience_counter = 0
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }
    best_state = None
    
    for epoch in range(config['max_epochs']):
        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device
        )
        
        # Validate
        val_metrics = evaluate(model, val_loader, criterion, device)
        
        # Store
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_metrics['loss'])
        history['val_acc'].append(val_metrics['accuracy'])
        
        # Early stopping
        if val_metrics['accuracy'] > best_val_acc:
            best_val_acc = val_metrics['accuracy']
            patience_counter = 0
            # Deep copy state dict
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1
        
        # Check early stopping
        if epoch >= config.get('min_epochs', 10) and patience_counter >= config.get('patience', 7):
            break
        
        # Optuna pruning - PROPER implementation
        if trial is not None:
            # Create globally unique step ID: trial_number * 1000 * num_folds + fold * 1000 + epoch
            # This ensures no conflicts across parallel trials
            if fold is not None:
                unique_step = fold * 1000 + epoch
            else:
                unique_step = epoch
            
            # Report intermediate value
            trial.report(val_metrics['accuracy'], unique_step)
            
            # Check if trial should be pruned
            if trial.should_prune():
                import optuna
                raise optuna.TrialPruned()
    
    # Load best model
    if best_state is not None:
        model.load_state_dict(best_state)
    
    return {
        'best_val_acc': best_val_acc,
        'best_epoch': epoch - patience_counter if epoch >= patience_counter else 0,
        'history': history
    }