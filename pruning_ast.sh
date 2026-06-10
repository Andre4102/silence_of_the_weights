#!/bin/bash
#SBATCH --job-name=astslpph_v         # Name of your job
#SBATCH --output=pruning_logs/%x_%j.out            # Output file (%x for job name, %j for job ID)
#SBATCH --error=pruning_logs/%x_%j.err             # Error file
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --time=24:00:00
#SBATCH --partition=L40S,A40         

# Print job details
echo "Starting job on node: $(hostname)"
echo "Job started at: $(date)"

# Activate the environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate myenv
# ✅ Create log dir
mkdir -p pruning_logs

export CUBLAS_WORKSPACE_CONFIG=:4096:8

python pruning_ast.py