#!/bin/bash
# =====================================================================
# Whisper attention-block structural pruning — SLURM skeleton.
#
# Submit:  sbatch pruning_whisper.sh
# Or override any variable inline:
#   PRUNING_STRATEGY=entire_head THRESHOLD_STRATEGY=local sbatch pruning_whisper.sh
# =====================================================================
#SBATCH --job-name=whisper_prune
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --time=24:00:00
# #SBATCH --account=<your_account>
# #SBATCH --partition=<your_partition>

set -euo pipefail

# ---- Inputs (override via environment) -------------------------------
PRUNING_STRATEGY="${PRUNING_STRATEGY:-per_head}"        # per_head | entire_head
THRESHOLD_STRATEGY="${THRESHOLD_STRATEGY:-global}"      # global | local
IMPORTANCE_STRATEGY="${IMPORTANCE_STRATEGY:-fisher_information}"  # magnitude | fisher_information
EPOCHS="${EPOCHS:-1}"                                   # fine-tune epochs per pruning iteration
LR="${LR:-1e-4}"
OPTIM="${OPTIM:-sgd}"
RESULTS_ROOT="${RESULTS_ROOT:-./results/whisper}"

CONDA_ENV="${CONDA_ENV:-myenv}"
# ----------------------------------------------------------------------

mkdir -p logs "${RESULTS_ROOT}"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate "${CONDA_ENV}"

export HF_DATASETS_DISABLE_TORCHCODEC=1
export HF_HUB_OFFLINE=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8

python pruning_whisper.py \
  --pruning_strategy   "${PRUNING_STRATEGY}" \
  --threshold_strategy "${THRESHOLD_STRATEGY}" \
  --importance_strategy "${IMPORTANCE_STRATEGY}" \
  --epochs             "${EPOCHS}" \
  --lr                 "${LR}" \
  --optim              "${OPTIM}" \
  --results_root       "${RESULTS_ROOT}"
