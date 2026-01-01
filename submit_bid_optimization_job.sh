#!/bin/bash
#SBATCH --job-name=bid-optimization
#SBATCH --partition=mit_normal
#SBATCH --time=12:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --output=logs/bid_opt_%j.out
#SBATCH --error=logs/bid_opt_%j.err

# Go to the submit directory and prep logs
cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs

echo "========== Env & Dir =========="
echo "Host: $(hostname)"
echo "Start: $(date)"
echo "Submit dir: ${SLURM_SUBMIT_DIR:-$PWD}"
echo "PWD: $PWD"
echo "==============================="

echo "Load modules..."
module load miniforge

echo "Activate environment..."
conda activate adopt_env

# Run bid optimization with XGB Tweedie models for both EPC and clicks
echo "Running bid_optimization.py with XGB Tweedie models"
python -u scripts/bid_optimization.py --embedding-method bert --alg-conv xgb --alg-clicks xgb

echo "End: $(date)"
