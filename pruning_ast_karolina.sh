#!/bin/bash
#SBATCH --job-name=astphgf
#SBATCH --account eu-25-53
#SBATCH --output=ast_logs/%x.out
#SBATCH --error=ast_logs/%x.err
#SBATCH --mail-user=andrea.diecidue@polimi.it
#SBATCH --mail-type=ALL
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --partition=qgpu     

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