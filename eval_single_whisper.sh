#!/bin/bash
# -----------------------------------------------------------------------
# eval_model.sh  —  run on the LOGIN NODE:
#
#   bash eval_model.sh <path>
#
# <path> can be either:
#
#   A) A single model folder:
#      bash eval_model.sh .../basemodel
#      bash eval_model.sh .../global_per_head_fisher_information/model_1501M_params
#      → submits one job per task for that model (iteration=1, n_params from folder
#        name or 0 if not a model_*M_params folder e.g. basemodel)
#
#   B) A results folder containing model_*M_params subfolders:
#      bash eval_model.sh .../global_per_head_fisher_information
#      bash eval_model.sh .../basemodel  (if it contains submodels)
#      → submits one job per (model × task), iteration assigned by param count
#
# Optional env var overrides:
#   TASKS="librispeech_en commonvoice_de" bash eval_model.sh <path>
# -----------------------------------------------------------------------

set -euo pipefail

# ── Configuration ────────────────────────────────────────────────────────
EVAL_SCRIPT="whisper_eval_single_task.py"
CONDA_ENV="myenv"
ACCOUNT="INA24_C6B06"     
PARTITION="boost_usr_prod"
JOB_TIME="12:00:00"
BATCH_SIZE=96
NUM_WORKERS=4
LOGDIR='whisper_eval_logs'

ALL_TASKS=(
    librispeech_en
    commonvoice_fr
    # commonvoice_de
    # commonvoice_es
    commonvoice_it
    # commonvoice_zh
    # mls_pl
    # commonvoice_ru
    # commonvoice_ar
    # fleurs_hi
    covost_de_en
    # covost_zh-CN_en
    # covost_ar_en
)

if [[ -n "${TASKS:-}" ]]; then
    read -ra SELECTED_TASKS <<< "$TASKS"
else
    SELECTED_TASKS=("${ALL_TASKS[@]}")
fi

# ── Parse argument ────────────────────────────────────────────────────────
if [[ $# -lt 1 ]]; then
    echo "Usage: bash eval_model.sh <model_or_results_folder>"
    exit 1
fi

TARGET="$(realpath "$1")"

if [[ ! -d "$TARGET" ]]; then
    echo "ERROR: $TARGET is not a directory"
    exit 1
fi

# ── Helper: extract param count from folder name ──────────────────────────
extract_params() {
    basename "$1" | grep -oP '\d+(?=M_params)' || echo "0"
}

# ── Determine mode: single model or results folder ────────────────────────
mapfile -t SUBMODEL_DIRS < <(
    find "$TARGET" -maxdepth 1 -mindepth 1 -type d -name "model_*M_params" 2>/dev/null || true
)

if [[ ${#SUBMODEL_DIRS[@]} -gt 0 ]]; then
    # ── Mode B: results folder with multiple model_*M_params subdirs ──────
    echo "Mode: results folder  →  ${#SUBMODEL_DIRS[@]} models found"

    TB_LOGDIR="${TARGET}/tensorboard_logs"
    mkdir -p "${TB_LOGDIR}"
    mkdir -p "${LOGDIR}/slurm_logs"

    # Sort by param count descending (largest = iteration 1)
    mapfile -t MODEL_DIRS < <(
        for d in "${SUBMODEL_DIRS[@]}"; do
            echo "$(extract_params "$d") $d"
        done | sort -rn | awk '{print $2}'
    )

    echo ""
    for i in "${!MODEL_DIRS[@]}"; do
        params=$(extract_params "${MODEL_DIRS[$i]}")
        echo "  iter $((i+1))  |  ${params}M params  |  $(basename "${MODEL_DIRS[$i]}")"
    done
    echo ""

    N_SUBMITTED=0
    for i in "${!MODEL_DIRS[@]}"; do
        MODEL_PATH="${MODEL_DIRS[$i]}"
        ITERATION=$((i + 1))
        N_PARAMS=$(extract_params "$MODEL_PATH")

        for TASK in "${SELECTED_TASKS[@]}"; do
            LOG_FILE="${LOGDIR}/slurm_logs/${TASK}_iter${ITERATION}_%j.out"
            ERR_FILE="${LOGDIR}/slurm_logs/${TASK}_iter${ITERATION}_%j.err"

            JOB_ID=$(sbatch --parsable \
                --job-name="ev_${TASK:0:10}_i${ITERATION}" \
                --partition="${PARTITION}" \
                --account="${ACCOUNT}" \
                --time="${JOB_TIME}" \
                --nodes=1 --ntasks=1 \
                --cpus-per-task="${NUM_WORKERS}" \
                --gres=gpu:1 --mem=32G \
                --output="${LOG_FILE}" \
                --error="${ERR_FILE}" \
                --wrap="
source ~/.bashrc
conda activate ${CONDA_ENV}
python ${EVAL_SCRIPT} \
    --model_path '${MODEL_PATH}' \
    --task '${TASK}' \
    --tb_logdir '${TB_LOGDIR}' \
    --iteration ${ITERATION} \
    --n_params ${N_PARAMS} \
    --batch_size ${BATCH_SIZE} \
    --num_workers ${NUM_WORKERS}
"
            )
            echo "  [${JOB_ID}]  iter${ITERATION} (${N_PARAMS}M)  ${TASK}"
            N_SUBMITTED=$((N_SUBMITTED + 1))
        done
    done

else
    # ── Mode A: single model folder ───────────────────────────────────────
    echo "Mode: single model  →  $(basename "$TARGET")"

    # TensorBoard logs go in the parent folder (the results dir)
    TB_LOGDIR="$(dirname "$TARGET")/tensorboard_logs"
    mkdir -p "${TB_LOGDIR}/slurm_logs"

    ITERATION=0
    N_PARAMS=$(extract_params "$TARGET")

    echo "  iteration ${ITERATION}  |  ${N_PARAMS}M params  |  $(basename "$TARGET")"
    echo ""

    N_SUBMITTED=0
    for TASK in "${SELECTED_TASKS[@]}"; do
        LOG_FILE="${LOGDIR}/slurm_logs/${TASK}_iter${ITERATION}_%j.out"
        ERR_FILE="${LOGDIR}/slurm_logs/${TASK}_iter${ITERATION}_%j.err"

        JOB_ID=$(sbatch --parsable \
            --job-name="ev_${TASK:0:10}_i${ITERATION}" \
            --partition="${PARTITION}" \
            --account="${ACCOUNT}" \
            --time="${JOB_TIME}" \
            --nodes=1 --ntasks=1 \
            --cpus-per-task="${NUM_WORKERS}" \
            --gres=gpu:1 --mem=32G \
            --output="${LOG_FILE}" \
            --error="${ERR_FILE}" \
            --wrap="
source ~/.bashrc
conda activate ${CONDA_ENV}
python ${EVAL_SCRIPT} \
    --model_path '${TARGET}' \
    --task '${TASK}' \
    --tb_logdir '${TB_LOGDIR}' \
    --iteration ${ITERATION} \
    --n_params ${N_PARAMS} \
    --batch_size ${BATCH_SIZE} \
    --num_workers ${NUM_WORKERS}
"
        )
        echo "  [${JOB_ID}]  ${TASK}"
        N_SUBMITTED=$((N_SUBMITTED + 1))
    done
fi

echo ""
echo "════════════════════════════════════════════════════════"
echo "  ${N_SUBMITTED} jobs submitted"
echo "  Monitor:     squeue -u \$USER"
echo "  TensorBoard: tensorboard --logdir ${TB_LOGDIR}"
echo "════════════════════════════════════════════════════════"