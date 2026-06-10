#!/bin/bash
# =====================================================================
# AST attention-block structural pruning — SLURM skeleton.
#
# Submit:  sbatch pruning_ast.sh
# Or override any variable inline:
#   PRUNING_STRATEGY=entire_head IMPORTANCE_STRATEGY=magnitude \
#     DATASET=speechcommands sbatch pruning_ast.sh
# =====================================================================
#SBATCH --job-name=ast_prune
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=40G
#SBATCH --time=24:00:00
# #SBATCH --account=<your_account>
# #SBATCH --partition=<your_partition>

set -euo pipefail

# ---- Inputs (override via environment) -------------------------------
PRUNING_STRATEGY="${PRUNING_STRATEGY:-per_head}"        # same_channel | per_head | entire_head
THRESHOLD_STRATEGY="${THRESHOLD_STRATEGY:-global}"      # global | local
IMPORTANCE_STRATEGY="${IMPORTANCE_STRATEGY:-fisher_information}"  # magnitude | fisher_information
DATASET="${DATASET:-audioset}"                          # audioset | speechcommands
NUM_EPOCHS="${NUM_EPOCHS:-1}"                           # fine-tune epochs per pruning iteration
LR="${LR:-1e-4}"
OPTIM="${OPTIM:-adam}"
RESULT_DIR="${RESULT_DIR:-./results/ast}"

CONDA_ENV="${CONDA_ENV:-myenv}"
# ----------------------------------------------------------------------

mkdir -p logs "${RESULT_DIR}"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate "${CONDA_ENV}"

export CUBLAS_WORKSPACE_CONFIG=:4096:8

python pruning_ast.py \
  --pruning_strategy   "${PRUNING_STRATEGY}" \
  --threshold_strategy "${THRESHOLD_STRATEGY}" \
  --importance_strategy "${IMPORTANCE_STRATEGY}" \
  --dataset            "${DATASET}" \
  --num_epochs         "${NUM_EPOCHS}" \
  --lr                 "${LR}" \
  --optim              "${OPTIM}" \
  --result_dir         "${RESULT_DIR}"
