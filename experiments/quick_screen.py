"""
Quick screen of all normalization strategies
Goal: Identify top 3 performers before full Optuna search
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import GroupKFold
import json
from datetime import datetime
from pathlib import Path
import pandas as pd

from core.data import load_data, get_class_weights, create_dataloaders
from core.normalization import batch_compute_covariances, ALL_STRATEGIES, get_strategy_name
from core.alignment import apply_alignment
from core.spdnet import SPDNetClassifier
from core.training import train_with_early_stopping, evaluate


def quick_screen(
    data_path: str,
    output_dir: str,
    n_folds: int = 3,
    max_epochs: int = 15,
    device: str = 'cuda'
):
    """
    Quick screen: test all normalization strategies with minimal epochs
    """
    print("="*80)
    print("QUICK NORMALIZATION SCREEN")
    print("="*80)
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Load data
    print("\n[1/5] Loading data...")
    trials, labels, subjects = load_data(data_path)
    
    # Configuration for quick testing
    config = {
        'bimap1_out': 20,
        'bimap2_out': 15,
        'hidden_dim': 256,
        'dropout': 0.5,
        'learning_rate': 1e-4,
        'weight_decay': 1e-3,
        'batch_size': 64,
        'max_epochs': max_epochs,
        'min_epochs': 5,
        'patience': 5,
        'boost_minority': 1.0
    }
    
    # Results storage
    all_results = []
    
    # Test each normalization strategy
    print(f"\n[2/5] Testing {len(ALL_STRATEGIES)} normalization strategies...")
    print(f"  Using {n_folds}-fold GroupKFold cross-validation")
    print(f"  Max {max_epochs} epochs per fold (early stopping enabled)")
    print()
    
    for strategy_idx, (norm_strategy, trace_norm) in enumerate(ALL_STRATEGIES, 1):
        strategy_name = get_strategy_name(norm_strategy, trace_norm)
        
        print(f"\n{'='*80}")
        print(f"[{strategy_idx}/{len(ALL_STRATEGIES)}] Testing: {strategy_name}")
        print(f"{'='*80}")
        
        # Compute covariances with this strategy
        print(f"  Computing covariances...")
        covariances = batch_compute_covariances(
            trials,
            strategy=norm_strategy,
            trace_norm=trace_norm,
            epsilon=1e-4,
            verbose=True
        )
        
        # Cross-validation
        fold_results = []
        gkf = GroupKFold(n_splits=n_folds)
        
        for fold, (train_idx, test_idx) in enumerate(gkf.split(covariances, labels, subjects), 1):
            print(f"\n  Fold {fold}/{n_folds}")
            
            # Split data
            train_covs, train_labels = covariances[train_idx], labels[train_idx]
            test_covs, test_labels = covariances[test_idx], labels[test_idx]
            
            # Further split train into train/val (80/20)
            n_train = int(0.8 * len(train_idx))
            val_idx = np.arange(n_train, len(train_idx))
            train_idx_inner = np.arange(n_train)
            
            val_covs = train_covs[val_idx]
            val_labels = train_labels[val_idx]
            train_covs = train_covs[train_idx_inner]
            train_labels = train_labels[train_idx_inner]
            
            # Create dataloaders
            train_loader, val_loader, test_loader = create_dataloaders(
                train_covs, train_labels,
                val_covs, val_labels,
                test_covs, test_labels,
                batch_size=config['batch_size']
            )
            
            # Create model
            model = SPDNetClassifier(
                n_channels=30,
                bimap1_out=config['bimap1_out'],
                bimap2_out=config['bimap2_out'],
                hidden_dim=config['hidden_dim'],
                num_classes=2,
                dropout=config['dropout']
            ).to(device)
            
            # Get class weights
            class_weights = get_class_weights(train_labels, config['boost_minority'])
            
            # Train
            train_result = train_with_early_stopping(
                model, train_loader, val_loader,
                class_weights, config, device
            )
            
            # Evaluate on test
            test_metrics = evaluate(
                model, test_loader,
                nn.CrossEntropyLoss(weight=class_weights.to(device)),
                device
            )
            
            fold_results.append({
                'fold': fold,
                'val_acc': train_result['best_val_acc'],
                'test_acc': test_metrics['accuracy'],
                'test_bacc': test_metrics['balanced_accuracy'],
                'test_f1': test_metrics['f1'],
                'best_epoch': train_result['best_epoch']
            })
            
            print(f"    Val Acc: {train_result['best_val_acc']:.4f}")
            print(f"    Test Acc: {test_metrics['accuracy']:.4f}")
            print(f"    Test Balanced Acc: {test_metrics['balanced_accuracy']:.4f}")
            print(f"    Test F1: {test_metrics['f1']:.4f}")
        
        # Aggregate results
        mean_val_acc = np.mean([r['val_acc'] for r in fold_results])
        mean_test_acc = np.mean([r['test_acc'] for r in fold_results])
        mean_test_bacc = np.mean([r['test_bacc'] for r in fold_results])
        mean_test_f1 = np.mean([r['test_f1'] for r in fold_results])
        
        std_test_acc = np.std([r['test_acc'] for r in fold_results])
        std_test_bacc = np.std([r['test_bacc'] for r in fold_results])
        
        strategy_result = {
            'strategy': norm_strategy,
            'trace_norm': trace_norm,
            'strategy_name': strategy_name,
            'mean_val_acc': mean_val_acc,
            'mean_test_acc': mean_test_acc,
            'mean_test_bacc': mean_test_bacc,
            'mean_test_f1': mean_test_f1,
            'std_test_acc': std_test_acc,
            'std_test_bacc': std_test_bacc,
            'fold_results': fold_results
        }
        
        all_results.append(strategy_result)
        
        print(f"\n  ✓ {strategy_name} Summary:")
        print(f"    Mean Val Acc: {mean_val_acc:.4f}")
        print(f"    Mean Test Acc: {mean_test_acc:.4f} ± {std_test_acc:.4f}")
        print(f"    Mean Test Balanced Acc: {mean_test_bacc:.4f} ± {std_test_bacc:.4f}")
        print(f"    Mean Test F1: {mean_test_f1:.4f}")
    
    # Save results
    print("\n[3/5] Saving results...")
    results_file = output_path / f"quick_screen_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"  Saved to: {results_file}")
    
    # Create summary DataFrame
    print("\n[4/5] Creating summary...")
    summary_df = pd.DataFrame([
        {
            'Strategy': r['strategy_name'],
            'Test Acc': f"{r['mean_test_acc']:.4f} ± {r['std_test_acc']:.4f}",
            'Test Bal Acc': f"{r['mean_test_bacc']:.4f} ± {r['std_test_bacc']:.4f}",
            'Test F1': f"{r['mean_test_f1']:.4f}",
            'Val Acc': f"{r['mean_val_acc']:.4f}"
        }
        for r in all_results
    ])
    
    # Sort by balanced accuracy
    summary_df = summary_df.sort_values('Test Bal Acc', ascending=False)
    
    print("\n" + "="*80)
    print("QUICK SCREEN RESULTS (sorted by Balanced Accuracy)")
    print("="*80)
    print(summary_df.to_string(index=False))
    
    # Save summary
    summary_file = output_path / f"quick_screen_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    summary_df.to_csv(summary_file, index=False)
    
    # Identify top 3
    print("\n[5/5] Top 3 strategies for Optuna search:")
    top_3 = sorted(all_results, key=lambda x: x['mean_test_bacc'], reverse=True)[:3]
    
    for i, result in enumerate(top_3, 1):
        print(f"  {i}. {result['strategy_name']}")
        print(f"     Test Balanced Acc: {result['mean_test_bacc']:.4f} ± {result['std_test_bacc']:.4f}")
        print(f"     Test F1: {result['mean_test_f1']:.4f}")
    
    # Save top 3 for next stage
    top3_file = output_path / "top3_strategies.json"
    top3_data = [
        {
            'strategy': r['strategy'],
            'trace_norm': r['trace_norm'],
            'strategy_name': r['strategy_name'],
            'mean_test_bacc': r['mean_test_bacc']
        }
        for r in top_3
    ]
    with open(top3_file, 'w') as f:
        json.dump(top3_data, f, indent=2)
    
    print(f"\n✓ Quick screen complete! Top 3 saved to: {top3_file}")
    print(f"✓ Ready for Optuna optimization with these strategies!")
    
    return all_results, top_3


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Quick normalization screen")
    parser.add_argument('--data_path', type=str, required=True, help='Path to data_pkl.npz')
    parser.add_argument('--output_dir', type=str, default='./results/quick_screen', help='Output directory')
    parser.add_argument('--n_folds', type=int, default=3, help='Number of CV folds')
    parser.add_argument('--max_epochs', type=int, default=15, help='Max epochs per fold')
    parser.add_argument('--device', type=str, default='cuda', help='Device')
    
    args = parser.parse_args()
    
    quick_screen(
        data_path=args.data_path,
        output_dir=args.output_dir,
        n_folds=args.n_folds,
        max_epochs=args.max_epochs,
        device=args.device
    )