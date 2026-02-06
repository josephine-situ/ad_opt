#!/bin/bash

EXP_NAME="${1:-backtests}"
echo "Launching pipeline for experiment: $EXP_NAME"

# 1. Submit Data Prep (pi_dbertsim)
# This runs first. We capture its ID to tell the next job to wait.
JID_PREP=$(sbatch --parsable --partition=pi_dbertsim --gpus=1 run_prep.sh)
echo "Submitted Data Prep: $JID_PREP"

# 2. Submit GPU Backtest (Index 0)
# dependency=afterok:$JID_PREP -> Waits for Prep to finish successfully
JID_GPU=$(sbatch --parsable --partition=mit_normal_gpu --gpus=1 --time=6:00:00 --array=0 --dependency=afterok:$JID_PREP submit_backtest_gpu.sh "$EXP_NAME")
echo "Submitted GPU Job:   $JID_GPU (Waiting for Prep)"

# 3. Submit CPU Backtests (Indices 1-30)
# dependency=afterok:$JID_GPU -> Waits for GPU Backtest (Index 0) to finish
JID_CPU=$(sbatch --parsable --partition=mit_normal --time=12:00:00 --array=1-30%2 --dependency=afterok:$JID_GPU submit_backtest_gpu.sh "$EXP_NAME")
echo "Submitted CPU Jobs:  $JID_CPU (Waiting for GPU Job)"