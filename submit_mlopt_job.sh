#!/bin/bash
#SBATCH --job-name=mlopt-modeling
#SBATCH --partition=mit_normal
#SBATCH --time=12:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --output=logs/mlopt_model_%j.out
#SBATCH --error=logs/mlopt_model_%j.err

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
conda activate mlopt_env

# --- CRITICAL OPTIMIZATION FOR IAI ---
# This ensures Julia uses all 8 cores we requested above.
export JULIA_NUM_THREADS=$SLURM_CPUS_PER_TASK
echo "Running with JULIA_NUM_THREADS=$JULIA_NUM_THREADS"

# Run the script (single job, no array). -u prints output as it runs.
echo "Running prediction_modeling.py"
LD_LIBRARY_PATH=/orcd/software/community/001/pkg/julia/1.10.4/lib/julia/:"${LD_LIBRARY_PATH}" python -u scripts/prediction_modeling.py --target clicks --embedding-method tfidf
LD_LIBRARY_PATH=/orcd/software/community/001/pkg/julia/1.10.4/lib/julia/:"${LD_LIBRARY_PATH}" python -u scripts/prediction_modeling.py --target clicks --embedding-method bert

echo "End: $(date)"