#!/bin/bash

# Complete 7-day pipeline automation
# Runs all stages sequentially

set -e  # Exit on error

echo "========================================================================"
echo "🚀 COMPLETE SPDNET BASELINE PIPELINE 🚀"
echo "========================================================================"
echo ""
echo "This will run:"
echo "  1. Quick normalization screen (2-3 hours)"
echo "  2. Optuna hyperparameter optimization (8-12 hours)"
echo "  3. Final LOSO validation (10-12 hours)"
echo ""
echo "Total estimated time: 20-27 hours"
echo ""
read -p "Press Enter to start or Ctrl+C to cancel..."

# Stage 1: Quick Screen
echo ""
echo "========================================================================"
echo "STAGE 1/3: QUICK NORMALIZATION SCREEN"
echo "========================================================================"
bash run_quick_screen.sh

# Stage 2: Optuna
echo ""
echo "========================================================================"
echo "STAGE 2/3: OPTUNA OPTIMIZATION"
echo "========================================================================"
bash run_optuna.sh

# Stage 3: Final LOSO
echo ""
echo "========================================================================"
echo "STAGE 3/3: FINAL LOSO VALIDATION"
echo "========================================================================"
bash run_final_loso.sh

# Complete!
echo ""
echo "========================================================================"
echo "🎉🎉🎉 PIPELINE COMPLETE! 🎉🎉🎉"
echo "========================================================================"
echo ""
echo "All results are available in ./results/"
echo ""
echo "Next steps:"
echo "  1. Review results in ./results/final_loso/"
echo "  2. Write up the methodology and results"
echo "  3. Begin EEG-Deformer integration for dual-branch fusion"
echo ""
echo "You're crushing this thesis! 💪🔥"