#!/bin/bash
#SBATCH --job-name=regen-preprocessors
#SBATCH --partition=mit_normal
#SBATCH --time=1:00:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=2
#SBATCH --output=logs/regen_preprocessors_%j.out
#SBATCH --error=logs/regen_preprocessors_%j.err

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

# Regenerate only XGB preprocessors with current sklearn version (for speed)
echo "Regenerating XGB preprocessors with current sklearn version..."
python -u scripts/prediction_modeling_tweedie.py --target epc --embedding-method bert --models xgb
python -u scripts/prediction_modeling_tweedie.py --target clicks --embedding-method bert --models xgb

echo "End: $(date)"
