#!/bin/bash

# Quick normalization screen (Day 1-2)
# Estimated time: 2-3 hours

echo "================================================"
echo "QUICK NORMALIZATION SCREEN"
echo "================================================"

DATA_PATH="/Users/arslanzahid/Downloads/New_Journal_pipeline/data_pkl.npz"
OUTPUT_DIR="./results/quick_screen"

python experiments/quick_screen.py \
    --data_path $DATA_PATH \
    --output_dir $OUTPUT_DIR \
    --n_folds 3 \
    --max_epochs 15 \
    --device cuda

echo ""
echo "✓ Quick screen complete!"
echo "✓ Check results in: $OUTPUT_DIR"
echo "✓ Next step: Run Optuna optimization with top 3 strategies"