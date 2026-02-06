#!/bin/bash
#SBATCH --job-name=adopt-prep
#SBATCH --output=logs/prep_%j.out
#SBATCH --error=logs/prep_%j.err
#SBATCH --time=4:00:00

# Setup Environment
module load miniforge
conda activate adopt_env

echo "Running data prep..."
python -u scripts/tidy_get_data.py --course sys_eng --embedding-method llm
echo "Data prep complete."