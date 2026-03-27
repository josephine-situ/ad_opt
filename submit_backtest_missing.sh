#!/bin/bash
#SBATCH --job-name=adopt-backtest-missing
#SBATCH --partition=mit_normal
#SBATCH --time=12:00:00
#SBATCH --mem=128G
#SBATCH --cpus-per-task=16
#SBATCH --output=logs/backtest_missing_%j.out
#SBATCH --error=logs/backtest_missing_%j.err

# Rerun missing days from a dynamic daily budget backtest and then re-run post-processing.
# By default, missing dates are inferred from the dynamic budget bids directory.

EXP_NAME="${EXP_NAME:-dynamic_daily_budget}"
COURSES="${COURSES:-ml sys_think}"
START_DAY="${START_DAY:-2025-10-01}"
END_DAY="${END_DAY:-2026-02-10}"
EMBEDDING_METHOD="${EMBEDDING_METHOD:-bert}"
K_POLICY="${K_POLICY:-0}"
EXTRA_ARGS="${EXTRA_ARGS:---order-budget --max-purch --min-spend 1}"
MISSING_DATES="${MISSING_DATES:-}"

cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs

echo "========== Env & Dir =========="
echo "Host: $(hostname)"
echo "Start: $(date)"
echo "Submit dir: ${SLURM_SUBMIT_DIR:-$PWD}"
echo "PWD: $PWD"
echo "Courses: $COURSES"
echo "Missing dates: ${MISSING_DATES:-<auto-detect>}"
echo "==============================="

load_dynamic_config() {
    local course="$1"
    python - <<PY
from pathlib import Path
import json

course = "$course"
cfg_path = Path(f"opt_results/{course}/backtests/{EXP_NAME}/backtest_config.json")
if not cfg_path.exists():
    raise SystemExit(f"Missing config: {cfg_path}")

cfg = json.loads(cfg_path.read_text())
print(cfg.get("campaign_budget", ""))
PY
}

get_missing_dates() {
    local course="$1"
    python - <<PY
from pathlib import Path
import json
import pandas as pd

course = "$course"
exp_name = "$EXP_NAME"

cfg_path = Path(f"opt_results/{course}/backtests/{exp_name}/backtest_config.json")
if not cfg_path.exists():
    raise SystemExit(f"Missing config: {cfg_path}")

cfg = json.loads(cfg_path.read_text())
start_day = pd.to_datetime(cfg["start_day"])
end_day = pd.to_datetime(cfg["end_day"])
campaign_budget = int(float(cfg["campaign_budget"]))

bids_dir = Path(f"opt_results/{course}/backtests/{exp_name}/budget_{campaign_budget}/bids")
if not bids_dir.exists():
    for candidate in sorted(Path(f"opt_results/{course}/backtests/{exp_name}").glob("budget_*/bids")):
        if list(candidate.glob("optimized_costs_*.csv")):
            bids_dir = candidate
            break

if not bids_dir.exists():
    raise SystemExit(f"No bids directory found for {course}")

observed = set()
for file_path in bids_dir.glob("optimized_costs_*.csv"):
    date_part = file_path.stem.replace("optimized_costs_", "")
    try:
        observed.add(pd.to_datetime(date_part).date())
    except Exception:
        continue

missing = [
    day.date().isoformat()
    for day in pd.date_range(start=start_day, end=end_day, freq="D")
    if day.date() not in observed
]

print(" ".join(missing))
PY
}

echo "Load modules..."
module load miniforge

echo "Activate environment..."
conda activate adopt_env

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export NUMEXPR_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}

for COURSE in $COURSES; do
    echo ""
    echo "############################################################"
    echo "  Course: $COURSE"
    echo "############################################################"

    CAMPAIGN_BUDGET=$(load_dynamic_config "$COURSE")
    if [ -z "$CAMPAIGN_BUDGET" ]; then
        echo "Could not load campaign budget for $COURSE"
        exit 1
    fi

    if [ -n "$MISSING_DATES" ]; then
        DATE_RANGE="$MISSING_DATES"
    else
        DATE_RANGE=$(get_missing_dates "$COURSE")
    fi

    if [ -z "$DATE_RANGE" ]; then
        echo "No missing dates detected for $COURSE; skipping rerun."
        continue
    fi

    echo "Missing dates detected for $COURSE: $DATE_RANGE"

    for DAY in $DATE_RANGE; do
        echo "Running day $DAY for $COURSE ..."
        python -u scripts/backtest_daily.py \
            --day "$DAY" \
            --exp-name "$EXP_NAME" \
            --course "$COURSE" \
            --start "$START_DAY" \
            --end "$END_DAY" \
            --embedding-method "$EMBEDDING_METHOD" \
            --k-policy $K_POLICY \
            $EXTRA_ARGS \
            --dynamic-budget \
            --campaign-budget "$CAMPAIGN_BUDGET"
    done

    echo "Running post-processing for $COURSE ..."
    python -u scripts/backtest_eval.py \
        --course "$COURSE" \
        --exp-name "$EXP_NAME" \
        --start "$START_DAY" \
        --end "$END_DAY" \
        --embedding-method "$EMBEDDING_METHOD" \
        --k-policy $K_POLICY

    for k in $K_POLICY; do
        if [ "$k" -eq 0 ]; then
            SUB_EXP="${EXP_NAME}/k_full"
        else
            SUB_EXP="${EXP_NAME}/k${k}"
        fi
        python -u scripts/analyze_backtest_results.py \
            --course "$COURSE" \
            --exp-name "$SUB_EXP"
    done
done

echo "Missing-date rerun complete: $(date)"