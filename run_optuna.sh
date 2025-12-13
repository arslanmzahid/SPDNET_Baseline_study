#!/bin/bash

# Optuna hyperparameter optimization (Day 3-4)
# Estimated time: 8-12 hours with 8 parallel jobs

echo "================================================"
echo "OPTUNA HYPERPARAMETER OPTIMIZATION"
echo "================================================"

DATA_PATH="/teamspace/studios/this_studio/SPDNET_Baseline_study/data_pkl.npz"
TOP3_FILE="./results/quick_screen/top3_strategies.json"
OUTPUT_DIR="./results/optuna"

# Check if top3 file exists
if [ ! -f "$TOP3_FILE" ]; then
    echo "ERROR: Top 3 strategies file not found!"
    echo "Please run quick_screen.py first"
    exit 1
fi

python experiments/optuna_search.py \
    --data_path $DATA_PATH \
    --top3_file $TOP3_FILE \
    --output_dir $OUTPUT_DIR \
    --n_trials 200 \
    --n_jobs 8 \
    --n_folds 5 \
    --device cuda

echo ""
echo "✓ Optuna optimization complete!"
echo "✓ Check results in: $OUTPUT_DIR"
echo "✓ Next step: Run final LOSO validation with best parameters"