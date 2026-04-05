# SPDNet Baseline Study — EEG Fatigue Detection

A Riemannian geometry deep learning pipeline for EEG-based fatigue detection, using **SPDNet** (Symmetric Positive Definite Networks) evaluated under rigorous **Leave-One-Subject-Out (LOSO)** cross-validation on the FATIG dataset.

This repository is the SPDNet baseline component of an MSc thesis on dual-branch EEG fatigue detection (SPDNet + EEG-Deformer) at the **University of Doha for Science and Technology (UDST)**.

---

## What This Does

EEG signals from fatigued vs. alert subjects are converted into covariance matrices — which naturally live on the **SPD (Symmetric Positive Definite) manifold**. SPDNet learns directly on this manifold using Riemannian geometry, avoiding the distortions that come from flattening matrices into vectors naively.

The pipeline handles everything from raw EEG trials to publication-ready LOSO results:

```
Raw EEG trials  →  Normalization  →  Covariance Matrices (SPD manifold)
                                              ↓
                                         SPDNet
                                   (BiMap → ReEig → LogEig)
                                              ↓
                                    Classification Head
                                              ↓
                               LOSO Evaluation + Metrics
```

---

## Repository Structure

```
SPDNET_Baseline_study/
│
├── core/
│   ├── spdnet.py          # SPDNet layers: BiMap, ReEig, LogEig + classifier head
│   ├── data.py            # Data loading, CovarianceDataset, class weight computation
│   ├── training.py        # Training loop, early stopping, Optuna integration
│   ├── normalization.py   # 6 normalization strategies + SPD enforcement
│   └── alignment.py       # Cross-subject alignment: Riemannian Alignment (RA), RPA-LEM
│
├── experiments/
│   ├── quick_screen.py    # Fast normalization strategy comparison (~2–3 hrs)
│   ├── optuna_search.py   # Hyperparameter optimisation with Optuna TPE (~8–12 hrs)
│   └── final_loso.py      # Full LOSO validation with best config (~10–12 hrs)
│
├── utils/
│   ├── metrics.py         # Balanced accuracy, F1, per-subject reporting
│   └── visualization.py   # Confusion matrices, training curves, result plots
│
├── run_quick_screen.sh    # Run Stage 1
├── run_optuna.sh          # Run Stage 2
├── run_final_loso.sh      # Run Stage 3
├── run_all.sh             # Run full pipeline end-to-end (~20–27 hrs)
└── requirements.txt
```

---

## Model Architecture

### SPDNet (`core/spdnet.py`)

Three custom layers operate entirely on the SPD manifold:

| Layer | What it does |
|-------|-------------|
| **BiMap** | Projects covariance matrices to a lower-dimensional SPD manifold via learned orthogonal weights (Stiefel manifold). Includes symmetry enforcement and diagonal regularisation for numerical stability. |
| **ReEig** | Rectifies eigenvalues to ensure all matrices remain strictly positive definite (eigenvalues clipped to ε = 1e-4). |
| **LogEig** | Maps SPD matrices to the tangent space (Euclidean) via matrix logarithm — enables use of standard classifiers. |

Default architecture: `input (30×30) → BiMap (20×20) → ReEig → BiMap (15×15) → ReEig → LogEig → flatten upper-triangular → FC classifier`

---

## Normalization Strategies

Six strategies are systematically compared in Stage 1:

- `none` — raw covariances
- `channel_center` — per-channel mean subtraction
- `trial_center` — per-trial mean subtraction
- `channel_zscore` — per-channel z-score
- `trial_zscore` — per-trial z-score
- `soft_norm` — soft normalisation

Covariance estimation uses **Ledoit-Wolf (LWF) shrinkage** by default, with optional trace normalisation. All matrices go through SPD enforcement (eigenvalue clipping + symmetrisation) before being passed to the network.

---

## Cross-Subject Alignment

`core/alignment.py` implements two alignment methods for handling inter-subject variability:

- **Riemannian Alignment (RA)** — aligns source covariances to a common Riemannian mean (Zanini et al., 2018)
- **RPA-LEM** — Riemannian Procrustes Analysis with Log-Euclidean metric

---

## Getting Started

### 1. Install dependencies

```bash
git clone https://github.com/arslanmzahid/SPDNET_Baseline_study.git
cd SPDNET_Baseline_study
pip install -r requirements.txt
```

**Key dependencies:** PyTorch ≥ 2.0, pyriemann ≥ 0.5, Optuna ≥ 3.0, scikit-learn, NumPy, SciPy, pandas, matplotlib, seaborn

### 2. Prepare data

Place the FATIG dataset (`.npz` format) under a `data/` directory:

```
data/
└── fatig_preprocessed.npz   # keys: 'data' (2952, 30, 384), 'labels' (2952,), 'subject_ids' (2952,)
```

> The dataset is not included in this repository. Please obtain it from the original source.

### 3. Run the full pipeline

```bash
# Run all three stages sequentially (~20–27 hours total)
bash run_all.sh

# Or run each stage individually:
bash run_quick_screen.sh     # Stage 1: normalization screen   (~2–3 hrs)
bash run_optuna.sh           # Stage 2: hyperparameter search  (~8–12 hrs)
bash run_final_loso.sh       # Stage 3: final LOSO validation  (~10–12 hrs)
```

Results are saved to `./results/`.

---

## Evaluation Protocol

All results use **Leave-One-Subject-Out (LOSO)** cross-validation:
- 11 subjects → 11 folds
- Each fold: 1 subject held out as test, remaining 10 used for training + validation
- Primary metric: **Balanced Accuracy** (accounts for class imbalance)
- Also reported: Accuracy, F1, Precision, Recall, Confusion Matrix

Class imbalance is handled via weighted cross-entropy loss (computed per fold using `sklearn.utils.class_weight`).

Training uses **Adam optimiser** with early stopping (patience = 7, min epochs = 10, max epochs = 50) and gradient clipping (max norm = 1.0).

---

## Hyperparameter Optimisation

Stage 2 uses **Optuna with TPE sampler** to search over:

- Normalization strategy and trace normalisation
- BiMap output dimensions
- Learning rate and weight decay
- Batch size, dropout, hidden layer size
- Alignment method

The best configuration is saved as JSON and loaded automatically by Stage 3.

---

## Citation

If you use this code, please cite:

```bibtex
@misc{zahid2025spdnet,
  author = {Arslan Muhammad Zahid},
  title  = {SPDNet Baseline Study: EEG-Based Fatigue Detection
            Using Riemannian Geometry},
  note   = {MSc Artificial Intelligence project,
            Istanbul Bilgi Univeristy - Türkiye},
  year   = {2026},
  url    = {https://github.com/arslanmzahid/SPDNET_Baseline_study}
}
```

---

## Related Work

- Huang & Van Gool (2017) — *A Riemannian Network for SPD Matrix Learning*, AAAI 2017
- Zanini et al. (2018) — *Transfer Learning: A Riemannian Geometry Framework*, IEEE Trans. Signal Process.
- This baseline feeds into a dual-branch fusion with **EEG-Deformer** (transformer-based temporal features) as part of the full thesis architecture.

---

## License

MIT License — see `LICENSE` for details.

---

*MSc Artificial Intelligence · University of Doha for Science and Technology (UDST)*
