#!/bin/bash

# Final LOSO validation (Day 5)
# Estimated time: 10-12 hours

echo "================================================"
echo "FINAL LOSO VALIDATION"
echo "================================================"

DATA_PATH="/teamspace/studios/this_studio/SPDNET_Baseline_study/data_pkl.npz"
BEST_PARAMS_FILE="./results/optuna/best_params.json"
OUTPUT_DIR="./results/final_loso"

# Check if best params file exists
if [ ! -f "$BEST_PARAMS_FILE" ]; then
    echo "ERROR: Best parameters file not found!"
    echo "Please run optuna_search.py first"
    exit 1
fi

python experiments/final_loso.py \
    --data_path $DATA_PATH \
    --best_params_file $BEST_PARAMS_FILE \
    --output_dir $OUTPUT_DIR \
    --max_epochs 50 \
    --device cuda

echo ""
echo "✓ Final LOSO validation complete!"
echo "✓ Check results in: $OUTPUT_DIR"
echo "✓ SPDNet baseline is COMPLETE!"
echo "✓ Ready for EEG-Deformer fusion!"