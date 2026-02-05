#!/bin/bash
#SBATCH --job-name=llm_keyword_scoring
#SBATCH --partition=pi_dbertsim
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gpus=1
#SBATCH --output=logs/llm_scoring_%j.out
#SBATCH --error=logs/llm_scoring_%j.err

# =============================================================================
# SLURM Batch Script for LLM Keyword Scoring
# =============================================================================
# This script runs the tidy_get_data.py pipeline with LLM-based keyword scoring
# using the Prometheus model for relevance evaluation.
#
# Usage:
#   sbatch submit_llm_scoring.sh [course]
#
# Arguments:
#   course  - Course name (default: gen_ai)
#             Options: gen_ai, ml, sys_eng
#
# Example:
#   sbatch submit_llm_scoring.sh gen_ai
#   sbatch submit_llm_scoring.sh ml
# =============================================================================

# Default course
COURSE=${1:-gen_ai}

echo "=============================================="
echo "LLM Keyword Scoring Job"
echo "=============================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Course: $COURSE"
echo "Start time: $(date)"
echo "=============================================="

# Create logs directory if it doesn't exist
mkdir -p logs

# Load required modules (adjust based on your cluster's module system)
# Common module names - uncomment and adjust as needed for your cluster:
# module purge
# module load Python/3.10.8-GCCcore-12.2.0
# module load CUDA/12.1.1
# module load cuDNN/8.9.2.26-CUDA-12.1.1

# Activate virtual environment (adjust path as needed)
# Option 1: Using conda
# source $(conda info --base)/etc/profile.d/conda.sh
# conda activate ad_opt

# Option 2: Using venv
# source venv/bin/activate

# Print Python and GPU info
echo ""
echo "Environment Info:"
echo "-----------------"
python --version
which python
echo ""

# Check for GPU availability
python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU count: {torch.cuda.device_count()}'); print(f'GPU name: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')" 2>/dev/null || echo "PyTorch not available or GPU check failed"

echo ""
echo "Starting LLM keyword scoring pipeline..."
echo "=============================================="

# Run the data pipeline with LLM embedding method
python scripts/tidy_get_data.py \
    --embedding-method llm \
    --course "$COURSE" \
    --force-reload \
    --log-file "logs/tidy_get_data_${COURSE}_llm_${SLURM_JOB_ID}.log"

EXIT_CODE=$?

echo ""
echo "=============================================="
echo "Job completed with exit code: $EXIT_CODE"
echo "End time: $(date)"
echo "=============================================="

exit $EXIT_CODE
