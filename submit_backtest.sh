#!/bin/bash
#SBATCH --job-name=adopt-backtest
#SBATCH --partition=mit_normal
#SBATCH --time=12:00:00
#SBATCH --mem=128G
#SBATCH --cpus-per-task=16
#SBATCH --array=0-30%2
#SBATCH --output=logs/backtest_%A_%a.out
#SBATCH --error=logs/backtest_%A_%a.err

# ── Configurable parameters ──────────────────────────────────────────
# Override via environment variables or edit defaults below.
EXP_NAME="${EXP_NAME:-min_spend}"
COURSES="${COURSES:-gen_ai}"                  # space-separated course names
START_DAY="${START_DAY:-2025-12-01}"
EMBEDDING_METHOD="${EMBEDDING_METHOD:-bert}"
K_POLICY="${K_POLICY:-0}"                     # 0 = full BERT, no SVD
EXTRA_ARGS="${EXTRA_ARGS:---order-budget --max-purch --min-spend 1}"

# ── Environment setup ────────────────────────────────────────────────
cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs

echo "========== Env & Dir =========="
echo "Host: $(hostname)"
echo "Start: $(date)"
echo "Submit dir: ${SLURM_SUBMIT_DIR:-$PWD}"
echo "PWD: $PWD"
echo "Courses: $COURSES"
echo "==============================="

echo "Load modules..."
module load miniforge

echo "Activate environment..."
conda activate adopt_env

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export NUMEXPR_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}

# ── Compute the day for this array task ──────────────────────────────
TASK_ID=${SLURM_ARRAY_TASK_ID:-0}
DAY=$(python - <<PY
import datetime as dt
start = dt.date.fromisoformat("$START_DAY")
print((start + dt.timedelta(days=int("$TASK_ID"))).isoformat())
PY
)

# ── Save configuration to the backtest folder (once, from task 0) ────
save_config() {
    local course="$1"
    local cfg_dir="opt_results/${course}/backtests/${EXP_NAME}"
    mkdir -p "$cfg_dir"
    cat > "$cfg_dir/backtest_config.json" <<EOF
{
    "exp_name": "$EXP_NAME",
    "course": "$course",
    "courses_run": "$COURSES",
    "start_day": "$START_DAY",
    "embedding_method": "$EMBEDDING_METHOD",
    "k_policy": "$K_POLICY",
    "extra_args": "$EXTRA_ARGS",
    "slurm_job_id": "${SLURM_ARRAY_JOB_ID:-local}",
    "submitted_at": "$(date -Iseconds)",
    "submitted_by": "$(whoami)",
    "host": "$(hostname)"
}
EOF
}

# ── Loop over courses ────────────────────────────────────────────────
for COURSE in $COURSES; do
    echo ""
    echo "############################################################"
    echo "  Course: $COURSE | Day: $DAY"
    echo "############################################################"

    # Save config once from task 0
    if [ "$TASK_ID" -eq 0 ]; then
        save_config "$COURSE"
    fi

    # ── Run the daily backtest ───────────────────────────────────────
    echo "Running backtest_daily.py --day $DAY --exp-name $EXP_NAME --course $COURSE --embedding-method $EMBEDDING_METHOD --k-policy $K_POLICY $EXTRA_ARGS"
    python -u scripts/backtest_daily.py \
        --day "$DAY" \
        --exp-name "$EXP_NAME" \
        --course "$COURSE" \
        --embedding-method "$EMBEDDING_METHOD" \
        --k-policy $K_POLICY \
        $EXTRA_ARGS

    echo "Backtest day $DAY complete for $COURSE: $(date)"
done

# ── Post-processing: run eval & analysis after the last array task ───
MAX_TASK_ID=$(echo "${SLURM_ARRAY_TASK_MAX:-$TASK_ID}")
END_DAY=$(python - <<PY
import datetime as dt
start = dt.date.fromisoformat("$START_DAY")
print((start + dt.timedelta(days=int("$MAX_TASK_ID"))).isoformat())
PY
)

if [ "$TASK_ID" -eq "$MAX_TASK_ID" ]; then
    echo ""
    echo "========================================"
    echo "  Last array task — running post-processing"
    echo "========================================"

    for COURSE in $COURSES; do
        echo ""
        echo "---- Post-processing: $COURSE ----"

        # 1. Evaluate all k-policy values in one call
        echo "Running backtest_eval.py for $COURSE ..."
        python -u scripts/backtest_eval.py \
            --course "$COURSE" \
            --exp-name "$EXP_NAME" \
            --start "$START_DAY" \
            --end "$END_DAY" \
            --embedding-method "$EMBEDDING_METHOD" \
            --k-policy $K_POLICY

        # 2. Analyze each k-policy subdirectory
        for k in $K_POLICY; do
            if [ "$k" -eq 0 ]; then
                SUB_EXP="${EXP_NAME}/k_full"
            else
                SUB_EXP="${EXP_NAME}/k${k}"
            fi
            echo "Running analyze_backtest_results.py for $COURSE k=${k} ..."
            python -u scripts/analyze_backtest_results.py \
                --course "$COURSE" \
                --exp-name "$SUB_EXP"
        done
    done

    echo "Post-processing complete: $(date)"
fi

echo "End: $(date)"
