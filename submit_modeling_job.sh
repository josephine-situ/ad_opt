#!/bin/bash
#SBATCH --job-name=ad-opt-modeling
#SBATCH --partition=mit_normal
#SBATCH --time=12:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --output=logs/ad_opt_model_%j.out
#SBATCH --error=logs/ad_opt_model_%j.err

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

# Run the non-IAI Tweedie modeling script. -u prints output as it runs.
echo "Running prediction_modeling_tweedie.py - clicks prediction"
python -u scripts/prediction_modeling_tweedie.py --target clicks --embedding-method tfidf
python -u scripts/prediction_modeling_tweedie.py --target clicks --embedding-method bert

echo "Running prediction_modeling_tweedie.py - conversion value per click prediction"
python -u scripts/prediction_modeling_tweedie.py --target epc --embedding-method tfidf
python -u scripts/prediction_modeling_tweedie.py --target epc --embedding-method bert

echo "End: $(date)"