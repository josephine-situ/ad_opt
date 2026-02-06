#!/bin/bash
#SBATCH --job-name=adopt-backtest
#SBATCH --mem=128G
#SBATCH --cpus-per-task=16
#SBATCH --output=logs/backtest_%A_%a.out
#SBATCH --error=logs/backtest_%A_%a.err

# NOTE: Partition, Time, GPU, and Array settings are removed here
# so they can be set via command line arguments.

# Go to the submit directory and prep logs
cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs

echo "========== Env & Dir =========="
echo "Host: $(hostname)"
echo "Start: $(date)"
echo "Job ID: $SLURM_JOB_ID, Array ID: $SLURM_ARRAY_TASK_ID"
echo "==============================="

module load miniforge
conda activate adopt_env

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export NUMEXPR_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}

START_DAY="2025-12-01"
TASK_ID=${SLURM_ARRAY_TASK_ID:-0}

# Calculate the specific day
DAY=$(python - <<PY
import datetime as dt
start = dt.date.fromisoformat("$START_DAY")
print((start + dt.timedelta(days=int("$TASK_ID"))).isoformat())
PY
)

EXP_NAME="${1:-backtests}"
echo "Running backtest_daily.py --day $DAY --exp-name $EXP_NAME"

# Run the python script
python -u scripts/backtest_daily.py --day "$DAY" --order-budget --exp-name "$EXP_NAME" --max-purch --course sys_eng --embedding-method llm

echo "End: $(date)"