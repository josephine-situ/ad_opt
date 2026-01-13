#!/bin/bash
#SBATCH --job-name=bid-optimization
#SBATCH --partition=mit_normal
#SBATCH --time=12:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --output=logs/bid_opt_%A_%a.out
#SBATCH --error=logs/bid_opt_%A_%a.err

# Lambda sweep (parallel): -1, 0, 1, 2, 4
#SBATCH --array=0-4

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

# Run bid optimization with GLM for EPC and XGB MSE models for clicks
echo "Running bid_optimization.py with GLM for EPC and XGB MSE models for clicks"

LAMBDAS=(-1 0 1 2 4)
LAMBDA=${LAMBDAS[$SLURM_ARRAY_TASK_ID]}

echo "SLURM_ARRAY_TASK_ID: ${SLURM_ARRAY_TASK_ID} -> lambda=${LAMBDA}"

# Use a concrete date string (YYYY-MM-DD) for consistent cache keys and file naming
TARGET_DAY=$(date +%F)

# Cache key controls reuse of base formulation + artifacts across lambdas
CACHE_KEY="bert_ridge_xgb_${TARGET_DAY}_full"

python -u scripts/bid_optimization.py \
	--embedding-method bert \
	--alg-conv ridge \
	--alg-clicks xgb \
	--target-day ${TARGET_DAY} \
	--exploration-lambda ${LAMBDA} \
	--cache-key ${CACHE_KEY} \
	--cache-mode auto \
	--skip-validation

echo "End: $(date)"
