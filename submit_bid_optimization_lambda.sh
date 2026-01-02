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

# Run bid optimization with different lambda values.
#
# Recommended parallel usage:
#   1) Build the cached formulation once:
#      python -u scripts/bid_optimization.py --embedding-method bert --alg-conv glm --alg-clicks xgb \
#        --formulation-lp opt_results/formulations/cached_glm_xgb.lp --write-formulation-only \
#        --warm-start on --warm-start-bid 0.0
#
#      This writes BOTH:
#        - cached_glm_xgb.lp  (formulation)
#        - cached_glm_xgb.mst (MIP start / warm start)
#
#   2) Submit as a job array (one lambda per task):
#      sbatch --array=0-3 submit_bid_optimization_lambda.sh
#
# Set the cached formulation path (must exist when using job arrays)
FORMULATION_LP=${FORMULATION_LP:-opt_results/formulations/cached_glm_xgb.lp}

# Lambda list (edit as needed). SLURM_ARRAY_TASK_ID selects which lambda to run.
LAMBDAS=(0 0.1 1 5)
IDX=${SLURM_ARRAY_TASK_ID:-0}
LAMBDA=${LAMBDAS[$IDX]}

echo "Solving cached formulation with lambda=$LAMBDA"
python -u scripts/bid_optimization.py --reuse-formulation --formulation-lp "$FORMULATION_LP" --explore-lambda "$LAMBDA"
echo "End: $(date)"
