#!/bin/bash
#SBATCH --job-name=compare-solvers
#SBATCH --partition=mit_normal
#SBATCH --time=12:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --output=logs/compare_solvers_%A.out
#SBATCH --error=logs/compare_solvers_%A.err

# ---------------------------------------------------------------
# Compare Gurobi vs. SCIP solvers on the bid-optimization MIP.
# Usage:
#   sbatch submit_compare_solvers.sh                       # default: gen_ai
#   sbatch submit_compare_solvers.sh ml                    # specify course
#   sbatch submit_compare_solvers.sh sys_eng 847.46        # course + budget
#   sbatch submit_compare_solvers.sh gen_ai "" 7200        # course + custom time limit
# ---------------------------------------------------------------

COURSE="${1:-ml}"
BUDGET="${2:-}"
TIME_LIMIT="${3:-40000}"

cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs

echo "========== Env & Dir =========="
echo "Host: $(hostname)"
echo "Start: $(date)"
echo "Submit dir: ${SLURM_SUBMIT_DIR:-$PWD}"
echo "PWD: $PWD"
echo "COURSE: $COURSE"
echo "BUDGET: ${BUDGET:-<from config>}"
echo "TIME_LIMIT: ${TIME_LIMIT}s per solver"
echo "Mem requested: $SLURM_MEM_PER_NODE MB"
echo "CPUs: $SLURM_CPUS_PER_TASK"
echo "==============================="

echo "Load modules..."
module load miniforge

echo "Activate environment..."
conda activate adopt_env

# Build command
CMD="python -u scripts/compare_solvers.py --course $COURSE --time-limit $TIME_LIMIT"
if [ -n "$BUDGET" ]; then
    CMD="$CMD --budget $BUDGET"
fi
CMD="$CMD --output opt_results/${COURSE}/solver_comparison.csv"

echo "Running: $CMD"
eval "$CMD"

echo "End: $(date)"
