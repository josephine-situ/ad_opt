#!/bin/bash
#SBATCH --job-name=ad-opt-mirrored-ort
#SBATCH --partition=mit_normal
#SBATCH --time=12:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --output=logs/ad_opt_mort_%j.out
#SBATCH --error=logs/ad_opt_mort_%j.err

set -euo pipefail

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

# Distill the tweedie XGBoost models into mirrored ORT-H models.
# Use -u to flush output during long CV runs.

echo "Running mirrored_ORT.py - clicks distillation"
python -u scripts/mirrored_ORT.py \
  --split-type axis \
  --target clicks \
  --embedding-method tfidf \
  --xgb-model models/xgb_tweedie_tfidf_clicks.json \
  --xgb-source xgboost_booster

python -u scripts/mirrored_ORT.py \
  --split-type axis \
  --target clicks \
  --embedding-method bert \
  --xgb-model models/xgb_tweedie_bert_clicks.json \
  --xgb-source xgboost_booster

echo "Running mirrored_ORT.py - epc distillation"
python -u scripts/mirrored_ORT.py \
  --split-type axis \
  --target epc \
  --embedding-method tfidf \
  --xgb-model models/xgb_tweedie_tfidf_epc.json \
  --xgb-source xgboost_booster

python -u scripts/mirrored_ORT.py \
  --split-type axis \
  --target epc \
  --embedding-method bert \
  --xgb-model models/xgb_tweedie_bert_epc.json \
  --xgb-source xgboost_booster

echo "End: $(date)"
