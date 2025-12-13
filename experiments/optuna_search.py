"""
Optuna hyperparameter optimization
Uses top 3 normalization strategies from quick screen
Parallel execution + pruning + comprehensive search
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import GroupKFold
import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
import json
from datetime import datetime
from pathlib import Path
import joblib

from core.data import load_data, get_class_weights, create_dataloaders
from core.normalization import batch_compute_covariances
from core.alignment import apply_alignment
from core.spdnet import SPDNetClassifier
from core.training import train_with_early_stopping, evaluate


class OptunaObjective:
    """Objective function for Optuna"""
    
    def __init__(
        self,
        trials_data: np.ndarray,
        labels: np.ndarray,
        subjects: np.ndarray,
        top_strategies: list,
        n_folds: int = 5,
        device: str = 'cuda'
    ):
        self.trials_data = trials_data
        self.labels = labels
        self.subjects = subjects
        self.top_strategies = top_strategies
        self.n_folds = n_folds
        self.device = device
        
        print(f"Initialized OptunaObjective:")
        print(f"  Data shape: {trials_data.shape}")
        print(f"  Subjects: {len(np.unique(subjects))}")
        print(f"  Top strategies: {len(top_strategies)}")
        print(f"  CV folds: {n_folds}")
    
    def __call__(self, trial: optuna.Trial) -> float:
        """Single trial execution"""
        
        # Sample hyperparameters
        params = self._sample_hyperparameters(trial)
        
        # Compute covariances with ROBUST epsilon
        covariances = batch_compute_covariances(
            self.trials_data,
            strategy=params['norm_strategy'],
            trace_norm=params['trace_norm'],
            epsilon=1e-4,  # ← PROPER epsilon (was 1e-6)
            verbose=False
        )
        
        # Cross-validation
        fold_scores = []
        gkf = GroupKFold(n_splits=self.n_folds)
        
        for fold, (train_idx, test_idx) in enumerate(gkf.split(covariances, self.labels, self.subjects), 1):
            
            # Split data
            train_covs, train_labels = covariances[train_idx], self.labels[train_idx]
            test_covs, test_labels = covariances[test_idx], self.labels[test_idx]
            
            # Apply alignment if enabled
            if params['use_alignment']:
                train_covs, test_covs = apply_alignment(
                    train_covs, test_covs,
                    method=params['alignment_method'],
                    n_reference=10,
                    epsilon = 1e-4
                )
            
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
            ).to(self.device)
            
            # Get class weights
            class_weights = get_class_weights(train_labels, params['boost_minority'])
            
            # Training config
            train_config = {
                'learning_rate': params['learning_rate'],
                'weight_decay': params['weight_decay'],
                'batch_size': params['batch_size'],
                'max_epochs': 50,
                'min_epochs': 10,
                'patience': 7,
                'boost_minority': params['boost_minority']
            }
            
            # Train with pruning
            try:
                train_result = train_with_early_stopping(
                    model, train_loader, val_loader,
                    class_weights, train_config, self.device,
                    trial=trial, fold=fold
                )
                
                # Evaluate on test
                test_metrics = evaluate(
                    model, test_loader,
                    nn.CrossEntropyLoss(weight=class_weights.to(self.device)),
                    self.device
                )
                
                fold_scores.append(test_metrics['balanced_accuracy'])
                
            except optuna.TrialPruned:
                raise
            except Exception as e:
                print(f"Error in fold {fold}: {e}")
                fold_scores.append(0.0)
        
        # Return mean balanced accuracy
        mean_score = np.mean(fold_scores)
        
        # Store fold scores in trial user attrs
        trial.set_user_attr('fold_scores', fold_scores)
        trial.set_user_attr('mean_score', mean_score)
        trial.set_user_attr('std_score', np.std(fold_scores))
        
        return mean_score
    
    def _sample_hyperparameters(self, trial: optuna.Trial) -> dict:
        """Sample hyperparameters for this trial"""
        
        # Select normalization strategy from top 3
        strategy_idx = trial.suggest_categorical('strategy_idx', list(range(len(self.top_strategies))))
        selected_strategy = self.top_strategies[strategy_idx]
        
        params = {
            'norm_strategy': selected_strategy['strategy'],
            'trace_norm': selected_strategy['trace_norm'],
            
            # Alignment - NOW ROBUST, keep enabled
            'use_alignment': trial.suggest_categorical('use_alignment', [True, False]),
            'alignment_method': trial.suggest_categorical('alignment_method', ['ra', 'rpa_lem']),
            
            # Architecture
            'bimap1_out': trial.suggest_int('bimap1_out', 15, 25),
            'bimap2_out': trial.suggest_int('bimap2_out', 10, 20),
            
            # Classifier
            'hidden_dim': trial.suggest_categorical('hidden_dim', [128, 256, 512]),
            'dropout': trial.suggest_float('dropout', 0.3, 0.7),
            
            # Optimization
            'learning_rate': trial.suggest_float('learning_rate', 1e-5, 1e-3, log=True),
            'weight_decay': trial.suggest_float('weight_decay', 1e-5, 1e-2, log=True),
            'batch_size': trial.suggest_categorical('batch_size', [32, 64, 128]),
            
            # Class imbalance
            'boost_minority': trial.suggest_float('boost_minority', 1.0, 2.0)
        }
        
        # Constraint: bimap2_out < bimap1_out
        if params['bimap2_out'] >= params['bimap1_out']:
            params['bimap2_out'] = params['bimap1_out'] - 5
        
        return params


def run_optuna_optimization(
    data_path: str,
    top3_file: str,
    output_dir: str,
    n_trials: int = 200,
    n_jobs: int = 8,
    n_folds: int = 5,
    device: str = 'cuda'
):
    """
    Run Optuna hyperparameter optimization
    """
    print("="*80)
    print("OPTUNA HYPERPARAMETER OPTIMIZATION")
    print("="*80)
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Load data
    print("\n[1/5] Loading data...")
    trials, labels, subjects = load_data(data_path)
    
    # Load top 3 strategies
    print("\n[2/5] Loading top 3 normalization strategies...")
    with open(top3_file, 'r') as f:
        top_strategies = json.load(f)
    
    print("  Top 3 strategies:")
    for i, s in enumerate(top_strategies, 1):
        print(f"    {i}. {s['strategy_name']} (bacc={s['mean_test_bacc']:.4f})")
    
    # Create study
    print(f"\n[3/5] Creating Optuna study...")
    print(f"  Trials: {n_trials}")
    print(f"  Parallel jobs: {n_jobs}")
    print(f"  CV folds: {n_folds}")
    
    study = optuna.create_study(
        direction='maximize',
        sampler=TPESampler(seed=42),
        pruner=MedianPruner(n_startup_trials=10, n_warmup_steps=5)
    )
    
    # Create objective
    objective = OptunaObjective(
        trials_data=trials,
        labels=labels,
        subjects=subjects,
        top_strategies=top_strategies,
        n_folds=n_folds,
        device=device
    )
    
    # Run optimization
    print(f"\n[4/5] Running optimization...")
    print(f"  This will take approximately {n_trials / n_jobs * 20 / 60:.1f} hours")
    print(f"  Starting at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    study.optimize(
        objective,
        n_trials=n_trials,
        n_jobs=n_jobs,
        show_progress_bar=True
    )
    
    # Save results
    print(f"\n[5/5] Saving results...")
    
    # Save study
    study_file = output_path / f"optuna_study_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
    joblib.dump(study, study_file)
    print(f"  Study saved to: {study_file}")
    
    # Best trial
    best_trial = study.best_trial
    print("\n" + "="*80)
    print("BEST TRIAL")
    print("="*80)
    print(f"  Trial number: {best_trial.number}")
    print(f"  Balanced Accuracy: {best_trial.value:.4f}")
    print(f"  Std: {best_trial.user_attrs.get('std_score', 0):.4f}")
    print("\n  Best Parameters:")
    for key, value in best_trial.params.items():
        print(f"    {key}: {value}")
    
    # Save best params
    best_params_file = output_path / "best_params.json"
    best_params = {
        'trial_number': best_trial.number,
        'balanced_accuracy': best_trial.value,
        'std': best_trial.user_attrs.get('std_score', 0),
        'fold_scores': best_trial.user_attrs.get('fold_scores', []),
        'params': best_trial.params
    }
    with open(best_params_file, 'w') as f:
        json.dump(best_params, f, indent=2)
    print(f"\n  Best params saved to: {best_params_file}")
    
    # Create visualizations
    print("\n  Generating visualizations...")
    
    try:
        import matplotlib.pyplot as plt
        
        # 1. Optimization history
        fig = optuna.visualization.matplotlib.plot_optimization_history(study)
        fig.savefig(output_path / "optimization_history.png", dpi=150, bbox_inches='tight')
        plt.close()
        
        # 2. Parameter importances
        fig = optuna.visualization.matplotlib.plot_param_importances(study)
        fig.savefig(output_path / "param_importances.png", dpi=150, bbox_inches='tight')
        plt.close()
        
        # 3. Parallel coordinate plot
        fig = optuna.visualization.matplotlib.plot_parallel_coordinate(study)
        fig.savefig(output_path / "parallel_coordinate.png", dpi=150, bbox_inches='tight')
        plt.close()
        
        print("  ✓ Visualizations saved")
        
    except Exception as e:
        print(f"  Warning: Could not create visualizations: {e}")
    
    print(f"\n✓ Optuna optimization complete!")
    print(f"✓ Best configuration ready for final LOSO validation!")
    
    return study, best_trial


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Optuna hyperparameter optimization")
    parser.add_argument('--data_path', type=str, required=True, help='Path to data_pkl.npz')
    parser.add_argument('--top3_file', type=str, required=True, help='Path to top3_strategies.json')
    parser.add_argument('--output_dir', type=str, default='./results/optuna', help='Output directory')
    parser.add_argument('--n_trials', type=int, default=200, help='Number of trials')
    parser.add_argument('--n_jobs', type=int, default=8, help='Parallel jobs')
    parser.add_argument('--n_folds', type=int, default=5, help='CV folds')
    parser.add_argument('--device', type=str, default='cuda', help='Device')
    
    args = parser.parse_args()
    
    run_optuna_optimization(
        data_path=args.data_path,
        top3_file=args.top3_file,
        output_dir=args.output_dir,
        n_trials=args.n_trials,
        n_jobs=args.n_jobs,
        n_folds=args.n_folds,
        device=args.device
    )