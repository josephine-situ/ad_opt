#!/bin/bash
#SBATCH --job-name=adopt-backtest
#SBATCH --partition=mit_normal
#SBATCH --time=12:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --array=0-9%2
#SBATCH --output=logs/backtest_%A_%a.out
#SBATCH --error=logs/backtest_%A_%a.err

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

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export NUMEXPR_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}

START_DAY="2025-12-01"
TASK_ID=${SLURM_ARRAY_TASK_ID:-0}
DAY=$(python - <<PY
import datetime as dt
start = dt.date.fromisoformat("$START_DAY")
print((start + dt.timedelta(days=int("$TASK_ID"))).isoformat())
PY
)

echo "Running backtest_daily.py --day $DAY"
python -u scripts/backtest_daily.py --day "$DAY" --x-max None 30 60 90 --alpha 0 0.25 0.5 --masked

echo "End: $(date)"
