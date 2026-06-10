#!/bin/bash
# =====================================================================
# Whisper evaluation (single task) — SLURM skeleton.
#
# Evaluates one pruned Whisper checkpoint on one task (WER for ASR,
# BLEU for translation), logging to TensorBoard.
#
# Submit:  MODEL_PATH=/path/to/model_1234M_params TASK=librispeech_en \
#            sbatch whisper_eval.sh
# =====================================================================
#SBATCH --job-name=whisper_eval
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=12:00:00
# #SBATCH --account=<your_account>
# #SBATCH --partition=<your_partition>

set -euo pipefail

# ---- Inputs (override via environment) -------------------------------
MODEL_PATH="${MODEL_PATH:?set MODEL_PATH to a model_*M_params folder}"
TASK="${TASK:-librispeech_en}"   # librispeech_en | commonvoice_<lang> | mls_<lang> | fleurs_<lang> | covost_<src>_<tgt>
TB_LOGDIR="${TB_LOGDIR:-./results/whisper/tensorboard_logs}"
ITERATION="${ITERATION:-1}"      # pruning iteration index (for logging)
N_PARAMS="${N_PARAMS:-0}"        # model size in millions (for logging)
BATCH_SIZE="${BATCH_SIZE:-48}"
NUM_WORKERS="${NUM_WORKERS:-4}"

CONDA_ENV="${CONDA_ENV:-myenv}"
# ----------------------------------------------------------------------

mkdir -p logs "${TB_LOGDIR}"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate "${CONDA_ENV}"

export HF_DATASETS_DISABLE_TORCHCODEC=1
export HF_HUB_OFFLINE=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8

python whisper_eval_single_task.py \
  --model_path  "${MODEL_PATH}" \
  --task        "${TASK}" \
  --tb_logdir   "${TB_LOGDIR}" \
  --iteration   "${ITERATION}" \
  --n_params    "${N_PARAMS}" \
  --batch_size  "${BATCH_SIZE}" \
  --num_workers "${NUM_WORKERS}"
