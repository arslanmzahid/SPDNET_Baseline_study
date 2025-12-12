"""
Final LOSO (Leave-One-Subject-Out) validation
Uses best configuration from Optuna
Publication-ready evaluation with comprehensive metrics
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, f1_score, 
    precision_score, recall_score, confusion_matrix,
    classification_report
)
from scipy.stats import friedmanchisquare, wilcoxon
import json
from datetime import datetime
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from core.data import load_data, get_class_weights, create_dataloaders
from core.normalization import batch_compute_covariances
from core.alignment import apply_alignment
from core.spdnet import SPDNetClassifier
from core.training import train_with_early_stopping, evaluate


def run_final_loso(
    data_path: str,
    best_params_file: str,
    output_dir: str,
    max_epochs: int = 50,
    device: str = 'cuda'
):
    """
    Run final LOSO validation with best parameters
    """
    print("="*80)
    print("FINAL LOSO VALIDATION")
    print("="*80)
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Load data
    print("\n[1/6] Loading data...")
    trials, labels, subjects = load_data(data_path)
    unique_subjects = np.unique(subjects)
    n_subjects = len(unique_subjects)
    print(f"  Total subjects: {n_subjects}")
    
    # Load best parameters
    print("\n[2/6] Loading best parameters from Optuna...")
    with open(best_params_file, 'r') as f:
        best_config = json.load(f)
    
    params = best_config['params']
    print("  Best configuration:")
    for key, value in params.items():
        print(f"    {key}: {value}")
    
    # Compute covariances
    print("\n[3/6] Computing covariances with best normalization...")
    
    # Get normalization strategy
    norm_strategy = params['norm_strategy']
    trace_norm = params['trace_norm']
    
    covariances = batch_compute_covariances(
        trials,
        strategy=norm_strategy,
        trace_norm=trace_norm,
        epsilon=1e-6,
        verbose=True
    )
    
    # LOSO validation
    print(f"\n[4/6] Running LOSO validation ({n_subjects} folds)...")
    print(f"  Max epochs per fold: {max_epochs}")
    print(f"  Estimated time: ~{n_subjects * 1} hours")
    print()
    
    loso_results = []
    all_predictions = []
    all_true_labels = []
    all_subject_ids = []
    
    for subject_idx, test_subject in enumerate(unique_subjects, 1):
        print(f"\n{'='*80}")
        print(f"[Subject {subject_idx}/{n_subjects}] Testing on Subject {test_subject}")
        print(f"{'='*80}")
        
        # Split by subject
        train_mask = subjects != test_subject
        test_mask = subjects == test_subject
        
        train_covs = covariances[train_mask]
        train_labels = labels[train_mask]
        test_covs = covariances[test_mask]
        test_labels = labels[test_mask]
        
        print(f"  Train: {len(train_covs)} trials from {n_subjects-1} subjects")
        print(f"  Test: {len(test_covs)} trials from 1 subject")
        print(f"  Test label distribution: Alert={np.sum(test_labels==0)}, Fatigue={np.sum(test_labels==1)}")
        
        # Apply alignment if enabled
        if params.get('use_alignment', False):
            print(f"  Applying {params['alignment_method'].upper()} alignment...")
            train_covs, test_covs = apply_alignment(
                train_covs, test_covs,
                method=params['alignment_method'],
                n_reference=10
            )
        
        # Further split train into train/val (90/10 for LOSO)
        n_train = int(0.9 * len(train_covs))
        val_idx = np.arange(n_train, len(train_covs))
        train_idx_inner = np.arange(n_train)
        
        val_covs = train_covs[val_idx]
        val_labels = train_labels[val_idx]
        train_covs_inner = train_covs[train_idx_inner]
        train_labels_inner = train_labels[train_idx_inner]
        
        # Create dataloaders
        train_loader, val_loader, test_loader = create_dataloaders(
            train_covs_inner, train_labels_inner,
            val_covs, val_labels,
            test_covs, test_labels,
            batch_size=params['batch_size']
        )
        
        # Create model
        model = SPDNetClassifier(
            n_channels=30,
            bimap1_out=params['bimap1_out'],
            bimap2_out=params['bimap2_out'],
            hidden_dim=params['hidden_dim'],
            num_classes=2,
            dropout=params['dropout']
        ).to(device)
        
        print(f"  Model parameters: {model.get_num_params():,}")
        
        # Get class weights
        class_weights = get_class_weights(train_labels_inner, params.get('boost_minority', 1.0))
        
        # Training config
        train_config = {
            'learning_rate': params['learning_rate'],
            'weight_decay': params['weight_decay'],
            'batch_size': params['batch_size'],
            'max_epochs': max_epochs,
            'min_epochs': 15,
            'patience': 10,
            'boost_minority': params.get('boost_minority', 1.0)
        }
        
        # Train
        print("  Training...")
        train_result = train_with_early_stopping(
            model, train_loader, val_loader,
            class_weights, train_config, device
        )
        
        print(f"  Best val acc: {train_result['best_val_acc']:.4f} at epoch {train_result['best_epoch']}")
        
        # Evaluate on test
        print("  Evaluating on test set...")
        test_metrics = evaluate(
            model, test_loader,
            nn.CrossEntropyLoss(weight=class_weights.to(device)),
            device
        )
        
        # Get predictions
        model.eval()
        with torch.no_grad():
            test_preds = []
            for cov_matrices, _ in test_loader:
                cov_matrices = cov_matrices.to(device)
                logits = model(cov_matrices)
                preds = torch.argmax(logits, dim=1)
                test_preds.extend(preds.cpu().numpy())
        
        test_preds = np.array(test_preds)
        
        # Store results
        subject_result = {
            'subject': int(test_subject),
            'n_train': len(train_covs),
            'n_test': len(test_covs),
            'test_acc': test_metrics['accuracy'],
            'test_bacc': test_metrics['balanced_accuracy'],
            'test_f1': test_metrics['f1'],
            'test_precision': precision_score(test_labels, test_preds, average='weighted', zero_division=0),
            'test_recall': recall_score(test_labels, test_preds, average='weighted', zero_division=0),
            'confusion_matrix': test_metrics['confusion_matrix'].tolist(),
            'best_val_acc': train_result['best_val_acc'],
            'best_epoch': train_result['best_epoch']
        }
        
        loso_results.append(subject_result)
        all_predictions.extend(test_preds)
        all_true_labels.extend(test_labels)
        all_subject_ids.extend([test_subject] * len(test_labels))
        
        # Print subject results
        print(f"\n  ✓ Subject {test_subject} Results:")
        print(f"    Test Accuracy: {test_metrics['accuracy']:.4f}")
        print(f"    Test Balanced Accuracy: {test_metrics['balanced_accuracy']:.4f}")
        print(f"    Test F1: {test_metrics['f1']:.4f}")
        print(f"    Test Precision: {subject_result['test_precision']:.4f}")
        print(f"    Test Recall: {subject_result['test_recall']:.4f}")
        print(f"    Confusion Matrix:")
        print(f"      {test_metrics['confusion_matrix']}")
    
    # Aggregate results
    print("\n" + "="*80)
    print("[5/6] FINAL LOSO RESULTS")
    print("="*80)
    
    # Overall metrics
    all_predictions = np.array(all_predictions)
    all_true_labels = np.array(all_true_labels)
    
    overall_acc = accuracy_score(all_true_labels, all_predictions)
    overall_bacc = balanced_accuracy_score(all_true_labels, all_predictions)
    overall_f1 = f1_score(all_true_labels, all_predictions, average='weighted')
    overall_precision = precision_score(all_true_labels, all_predictions, average='weighted')
    overall_recall = recall_score(all_true_labels, all_predictions, average='weighted')
    overall_cm = confusion_matrix(all_true_labels, all_predictions)
    
    # Per-subject statistics
    subject_accs = [r['test_acc'] for r in loso_results]
    subject_baccs = [r['test_bacc'] for r in loso_results]
    subject_f1s = [r['test_f1'] for r in loso_results]
    
    mean_acc = np.mean(subject_accs)
    std_acc = np.std(subject_accs)
    mean_bacc = np.mean(subject_baccs)
    std_bacc = np.std(subject_baccs)
    mean_f1 = np.mean(subject_f1s)
    std_f1 = np.std(subject_f1s)
    
    print("\n📊 OVERALL PERFORMANCE (across all trials):")
    print(f"  Accuracy: {overall_acc:.4f}")
    print(f"  Balanced Accuracy: {overall_bacc:.4f}")
    print(f"  F1 Score: {overall_f1:.4f}")
    print(f"  Precision: {overall_precision:.4f}")
    print(f"  Recall: {overall_recall:.4f}")
    print(f"\n  Overall Confusion Matrix:")
    print(f"    {overall_cm}")
    
    print("\n📊 PER-SUBJECT STATISTICS:")
    print(f"  Mean Accuracy: {mean_acc:.4f} ± {std_acc:.4f}")
    print(f"  Mean Balanced Accuracy: {mean_bacc:.4f} ± {std_bacc:.4f}")
    print(f"  Mean F1 Score: {mean_f1:.4f} ± {std_f1:.4f}")
    
    # Save detailed results
    results_summary = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'best_params': params,
        'overall_metrics': {
            'accuracy': float(overall_acc),
            'balanced_accuracy': float(overall_bacc),
            'f1': float(overall_f1),
            'precision': float(overall_precision),
            'recall': float(overall_recall),
            'confusion_matrix': overall_cm.tolist()
        },
        'per_subject_statistics': {
            'mean_accuracy': float(mean_acc),
            'std_accuracy': float(std_acc),
            'mean_balanced_accuracy': float(mean_bacc),
            'std_balanced_accuracy': float(std_bacc),
            'mean_f1': float(mean_f1),
            'std_f1': float(std_f1)
        },
        'per_subject_results': loso_results
    }
    
    results_file = output_path / f"final_loso_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(results_file, 'w') as f:
        json.dump(results_summary, f, indent=2)
    print(f"\n  ✓ Results saved to: {results_file}")
    
    # Create summary table
    summary_df = pd.DataFrame(loso_results)
    summary_df = summary_df.sort_values('subject')
    
    print("\n" + "="*80)
    print("PER-SUBJECT RESULTS TABLE")
    print("="*80)
    print(summary_df[['subject', 'test_acc', 'test_bacc', 'test_f1', 'test_precision', 'test_recall']].to_string(index=False))
    
    summary_csv = output_path / "per_subject_results.csv"
    summary_df.to_csv(summary_csv, index=False)
    print(f"\n  ✓ Table saved to: {summary_csv}")
    
    # Statistical report
    print("\n" + "="*80)
    print("STATISTICAL ANALYSIS")
    print("="*80)
    
    print(f"\n  Classification Report:")
    print(classification_report(all_true_labels, all_predictions, target_names=['Alert', 'Fatigue'], digits=4))
    
    # Visualizations
    print("\n[6/6] Generating visualizations...")
    
    # 1. Per-subject performance
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    subjects_sorted = summary_df['subject'].values
    
    axes[0].bar(range(len(subjects_sorted)), summary_df['test_acc'].values, color='steelblue', alpha=0.7)
    axes[0].axhline(mean_acc, color='red', linestyle='--', label=f'Mean: {mean_acc:.3f}')
    axes[0].set_xlabel('Subject')
    axes[0].set_ylabel('Accuracy')
    axes[0].set_title('Per-Subject Accuracy')
    axes[0].set_xticks(range(len(subjects_sorted)))
    axes[0].set_xticklabels(subjects_sorted, rotation=45)
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    
    axes[1].bar(range(len(subjects_sorted)), summary_df['test_bacc'].values, color='coral', alpha=0.7)
    axes[1].axhline(mean_bacc, color='red', linestyle='--', label=f'Mean: {mean_bacc:.3f}')
    axes[1].set_xlabel('Subject')
    axes[1].set_ylabel('Balanced Accuracy')
    axes[1].set_title('Per-Subject Balanced Accuracy')
    axes[1].set_xticks(range(len(subjects_sorted)))
    axes[1].set_xticklabels(subjects_sorted, rotation=45)
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    
    axes[2].bar(range(len(subjects_sorted)), summary_df['test_f1'].values, color='mediumseagreen', alpha=0.7)
    axes[2].axhline(mean_f1, color='red', linestyle='--', label=f'Mean: {mean_f1:.3f}')
    axes[2].set_xlabel('Subject')
    axes[2].set_ylabel('F1 Score')
    axes[2].set_title('Per-Subject F1 Score')
    axes[2].set_xticks(range(len(subjects_sorted)))
    axes[2].set_xticklabels(subjects_sorted, rotation=45)
    axes[2].legend()
    axes[2].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path / "per_subject_performance.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    # 2. Overall confusion matrix
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(overall_cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Alert', 'Fatigue'],
                yticklabels=['Alert', 'Fatigue'],
                cbar_kws={'label': 'Count'})
    ax.set_xlabel('Predicted Label')
    ax.set_ylabel('True Label')
    ax.set_title(f'Overall Confusion Matrix\n(Accuracy: {overall_acc:.4f}, Balanced Acc: {overall_bacc:.4f})')
    plt.tight_layout()
    plt.savefig(output_path / "confusion_matrix.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    # 3. Per-subject confusion matrices (grid)
    n_rows = (n_subjects + 3) // 4
    fig, axes = plt.subplots(n_rows, 4, figsize=(16, 4*n_rows))
    axes = axes.flatten()
    
    for i, result in enumerate(loso_results):
        cm = np.array(result['confusion_matrix'])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[i],
                   xticklabels=['A', 'F'], yticklabels=['A', 'F'],
                   cbar=False)
        axes[i].set_title(f"Subject {result['subject']}\nAcc: {result['test_acc']:.3f}")
        axes[i].set_xlabel('Pred')
        axes[i].set_ylabel('True')
    
    # Hide extra subplots
    for i in range(len(loso_results), len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path / "per_subject_confusion_matrices.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    # 4. Distribution of metrics
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].hist(subject_accs, bins=10, color='steelblue', alpha=0.7, edgecolor='black')
    axes[0].axvline(mean_acc, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_acc:.3f}')
    axes[0].set_xlabel('Accuracy')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title('Distribution of Accuracy')
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    
    axes[1].hist(subject_baccs, bins=10, color='coral', alpha=0.7, edgecolor='black')
    axes[1].axvline(mean_bacc, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_bacc:.3f}')
    axes[1].set_xlabel('Balanced Accuracy')
    axes[1].set_ylabel('Frequency')
    axes[1].set_title('Distribution of Balanced Accuracy')
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    
    axes[2].hist(subject_f1s, bins=10, color='mediumseagreen', alpha=0.7, edgecolor='black')
    axes[2].axvline(mean_f1, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_f1:.3f}')
    axes[2].set_xlabel('F1 Score')
    axes[2].set_ylabel('Frequency')
    axes[2].set_title('Distribution of F1 Score')
    axes[2].legend()
    axes[2].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path / "metrics_distribution.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    print("  ✓ All visualizations saved")
    
    # Final summary
    print("\n" + "="*80)
    print("🎉 FINAL LOSO VALIDATION COMPLETE! 🎉")
    print("="*80)
    print(f"\n📈 BEST RESULTS:")
    print(f"  Overall Balanced Accuracy: {overall_bacc:.4f}")
    print(f"  Mean Subject Balanced Accuracy: {mean_bacc:.4f} ± {std_bacc:.4f}")
    print(f"  Overall F1 Score: {overall_f1:.4f}")
    print(f"\n📁 All results saved to: {output_path}")
    print(f"\n✅ SPDNet baseline is COMPLETE and ready for EEG-Deformer fusion!")
    
    return results_summary


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Final LOSO validation")
    parser.add_argument('--data_path', type=str, required=True, help='Path to data_pkl.npz')
    parser.add_argument('--best_params_file', type=str, required=True, help='Path to best_params.json from Optuna')
    parser.add_argument('--output_dir', type=str, default='./results/final_loso', help='Output directory')
    parser.add_argument('--max_epochs', type=int, default=50, help='Max epochs per fold')
    parser.add_argument('--device', type=str, default='cuda', help='Device')
    
    args = parser.parse_args()
    
    run_final_loso(
        data_path=args.data_path,
        best_params_file=args.best_params_file,
        output_dir=args.output_dir,
        max_epochs=args.max_epochs,
        device=args.device
    )