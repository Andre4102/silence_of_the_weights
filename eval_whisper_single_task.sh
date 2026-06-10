#!/bin/bash
# -----------------------------------------------------------------------
# eval_launcher.sh  —  run this directly on the LOGIN NODE:
#
#   bash eval_launcher.sh
#
# It discovers all experiment folders and submits one SLURM job per
# (experiment × model × task). Each job gets 1 GPU.
#
# If the job limit (MAX_JOBS) is reached, remaining combinations are
# written to eval_launcher_remaining.sh for a follow-up submission.
#
# Directory structure expected:
#   MODELS_ROOT/
#     whisper_lr1e4_optadam_ep1/
#       tensorboard_logs/          ← per-experiment TB logs go here
#       model_1501M_params/
#       model_1463M_params/
#       ...
#     whisper_lr1e5_optadam_ep1/
#       ...
#
# Env var overrides:
#   MODELS_ROOT=/other/path bash eval_launcher.sh
#   TASKS="librispeech_en commonvoice_de" bash eval_launcher.sh
#   EXPERIMENTS="whisper_lr1e4_optadam_ep1 whisper_lr1e5_optadam_ep1" bash eval_launcher.sh
#   MAX_JOBS=100 bash eval_launcher.sh
# -----------------------------------------------------------------------

set -euo pipefail

# ── Configuration ────────────────────────────────────────────────────────
MODELS_ROOT="${MODELS_ROOT:-/leonardo_scratch/large/userexternal/adiecidu/pruning/results/whisper}"
EVAL_SCRIPT="whisper_eval_single_task.py"
CONDA_ENV="myenv"
ACCOUNT="INA24_C6B06"     
PARTITION="boost_usr_prod"
JOB_TIME="12:00:00"
BATCH_SIZE=96
NUM_WORKERS=4
LOGDIR="${MODELS_ROOT}/whisper_eval_logs"
MAX_JOBS=888 #"${MAX_JOBS:-200}"   # cap on jobs submitted in a single run
REMAINING_SCRIPT="$(dirname "$0")/eval_launcher_remaining.sh"
mkdir -p "${LOGDIR}/slurm_logs"

# Folder names to skip at the experiment level — exact matches
SKIP_NAMES=("basemodel" "large2" "tensorboard_logs" "whisper_eval_logs")

# Skip any folder whose name contains one of these substrings (e.g. "ep2")
SKIP_SUBSTRINGS=("ep2")
# Examples:
#   SKIP_SUBSTRINGS=("ep2" "ep3")
# Or override at runtime:
#   SKIP_SUBSTRINGS="ep2 ep3" bash eval_launcher.sh
if [[ -n "${SKIP_SUBSTRINGS_ENV:-}" ]]; then
    read -ra SKIP_SUBSTRINGS <<< "$SKIP_SUBSTRINGS_ENV"
fi

# All available tasks — must match eval_task.py
ALL_TASKS=(
    librispeech_en
    commonvoice_fr
    commonvoice_de
    commonvoice_es
    commonvoice_it
    commonvoice_zh
    mls_pl
    commonvoice_ru
    commonvoice_ar
    fleurs_hi
    covost_en_de
    covost_de_en
    covost_en_fr
    covost_fr_en
    covost_zh-CN_en
    covost_ar_en
)

# Override task list via env var
if [[ -n "${TASKS:-}" ]]; then
    read -ra SELECTED_TASKS <<< "$TASKS"
else
    SELECTED_TASKS=("${ALL_TASKS[@]}")
fi

# ── Helpers ──────────────────────────────────────────────────────────────
extract_params() {
    basename "$1" | grep -oP '\d+(?=M_params)'
}

should_skip() {
    local name="$1"
    for skip in "${SKIP_NAMES[@]}"; do
        [[ "$name" == "$skip" ]] && return 0
    done
    for substr in "${SKIP_SUBSTRINGS[@]}"; do
        [[ "$name" == *"$substr"* ]] && return 0
    done
    return 1
}

# Append one pending combination to the remaining-script buffer.
# Args: MODEL_PATH TASK TB_LOGDIR ITERATION N_PARAMS EXP_NAME
queue_remaining() {
    REMAINING_JOBS+=("$1 $2 $3 $4 $5 $6")
}

# ── Discover experiment folders ──────────────────────────────────────────
if [[ -n "${EXPERIMENTS:-}" ]]; then
    read -ra EXP_NAMES <<< "$EXPERIMENTS"
    EXPERIMENT_DIRS=()
    for e in "${EXP_NAMES[@]}"; do
        EXPERIMENT_DIRS+=("${MODELS_ROOT}/${e}")
    done
else
    mapfile -t EXPERIMENT_DIRS < <(
        find "$MODELS_ROOT" -maxdepth 1 -mindepth 1 -type d | sort
    )
fi

N_SUBMITTED=0
REMAINING_JOBS=()   # accumulates "MODEL_PATH TASK TB_LOGDIR ITERATION N_PARAMS EXP_NAME"
HIT_LIMIT=0

for EXP_DIR in "${EXPERIMENT_DIRS[@]}"; do
    EXP_NAME="$(basename "$EXP_DIR")"

    if should_skip "$EXP_NAME"; then
        continue
    fi

    if [[ ! -d "$EXP_DIR" ]]; then
        echo "WARNING: $EXP_DIR does not exist, skipping"
        continue
    fi

    RESULTS_DIR="${EXP_DIR}/global_per_head_fisher_information"
    if [[ -d "$RESULTS_DIR" ]]; then
        SEARCH_ROOT="$RESULTS_DIR"
    else
        SEARCH_ROOT="$EXP_DIR"
    fi

    TB_LOGDIR="${SEARCH_ROOT}/tensorboard_logs"

    mapfile -t MODEL_DIRS_UNSORTED < <(
        find "$SEARCH_ROOT" -maxdepth 1 -mindepth 1 -type d -name "model_*M_params" \
            ! -name "basemodel" ! -name "large2"
    )

    if [[ ${#MODEL_DIRS_UNSORTED[@]} -eq 0 ]]; then
        echo "[$EXP_NAME] no model_*M_params dirs found, skipping"
        continue
    fi

    mapfile -t MODEL_DIRS < <(
        for d in "${MODEL_DIRS_UNSORTED[@]}"; do
            echo "$(extract_params "$d") $d"
        done | sort -rn | awk '{print $2}'
    )

    echo ""
    echo "════════════════════════════════════════════════════════"
    echo "  Experiment : $EXP_NAME"
    echo "  Models     : ${#MODEL_DIRS[@]}"
    echo "  TB logdir  : $TB_LOGDIR"
    echo "════════════════════════════════════════════════════════"
    for i in "${!MODEL_DIRS[@]}"; do
        params=$(extract_params "${MODEL_DIRS[$i]}")
        echo "    iter $((i+1))  |  ${params}M params  |  $(basename "${MODEL_DIRS[$i]}")"
    done
    echo ""

    for i in "${!MODEL_DIRS[@]}"; do
        MODEL_PATH="${MODEL_DIRS[$i]}"
        ITERATION=$((i + 1))
        N_PARAMS=$(extract_params "$MODEL_PATH")

        for TASK in "${SELECTED_TASKS[@]}"; do

            # ── Job limit check ──────────────────────────────────────────
            if [[ $N_SUBMITTED -ge $MAX_JOBS ]]; then
                HIT_LIMIT=1
                queue_remaining "$MODEL_PATH" "$TASK" "$TB_LOGDIR" \
                                "$ITERATION" "$N_PARAMS" "$EXP_NAME"
                continue
            fi

            LOG_FILE="${LOGDIR}/slurm_logs/${TASK}_iter${ITERATION}_%j.out"
            ERR_FILE="${LOGDIR}/slurm_logs/${TASK}_iter${ITERATION}_%j.err"
            SHORT_TASK="${TASK:0:3}_${TASK: -2}"

            JOB_ID=$(sbatch --parsable \
                --job-name="${EXP_NAME}_${SHORT_TASK}_i${ITERATION}" \
                --partition="${PARTITION}" \
                --account="${ACCOUNT}" \
                --time="${JOB_TIME}" \
                --nodes=1 \
                --ntasks=1 \
                --cpus-per-task="${NUM_WORKERS}" \
                --gres=gpu:1 \
                --mem=64G \
                --output="${LOG_FILE}" \
                --error="${ERR_FILE}" \
                --wrap="
source ~/miniconda3/etc/profile.d/conda.sh
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

            echo "  [${JOB_ID}]  ${EXP_NAME}  iter${ITERATION} (${N_PARAMS}M)  ${TASK}"
            N_SUBMITTED=$((N_SUBMITTED + 1))
        done
    done
done

echo ""
echo "════════════════════════════════════════════════════════"
echo "  ${N_SUBMITTED} jobs submitted"
echo "  Monitor:     squeue -u \$USER"
echo "  TensorBoard: tensorboard --logdir <exp_dir>/tensorboard_logs"
echo "════════════════════════════════════════════════════════"

# ── Write remaining-jobs script if needed ────────────────────────────────
if [[ $HIT_LIMIT -eq 1 ]]; then
    N_REMAINING=${#REMAINING_JOBS[@]}
    echo ""
    echo "  ⚠  Job limit (MAX_JOBS=${MAX_JOBS}) reached."
    echo "     ${N_REMAINING} combinations were NOT submitted."
    echo "     Writing follow-up launcher → ${REMAINING_SCRIPT}"

    cat > "$REMAINING_SCRIPT" << HEADER
#!/bin/bash
# -----------------------------------------------------------------------
# eval_launcher_remaining.sh  —  AUTO-GENERATED by eval_launcher.sh
#
# Run this after your current jobs finish (or when quota frees up):
#   bash eval_launcher_remaining.sh
#
# Contains ${N_REMAINING} combinations that hit the MAX_JOBS=${MAX_JOBS} cap.
# Override the cap again with:  MAX_JOBS=300 bash eval_launcher_remaining.sh
# -----------------------------------------------------------------------

set -euo pipefail

EVAL_SCRIPT="${EVAL_SCRIPT}"
CONDA_ENV="${CONDA_ENV}"
ACCOUNT="${ACCOUNT}"
PARTITION="${PARTITION}"
JOB_TIME="${JOB_TIME}"
BATCH_SIZE=${BATCH_SIZE}
NUM_WORKERS=${NUM_WORKERS}
LOGDIR="${LOGDIR}"
MAX_JOBS="\${MAX_JOBS:-${MAX_JOBS}}"
REMAINING_SCRIPT="\$(dirname "\$0")/eval_launcher_remaining.sh"

mkdir -p "\${LOGDIR}/slurm_logs"

# Pending combinations (MODEL_PATH TASK TB_LOGDIR ITERATION N_PARAMS EXP_NAME)
PENDING=(
HEADER

    for entry in "${REMAINING_JOBS[@]}"; do
        echo "  $(printf '%q ' $entry)" >> "$REMAINING_SCRIPT"
    done

    cat >> "$REMAINING_SCRIPT" << 'FOOTER'
)

N_SUBMITTED=0
NEXT_REMAINING=()

for entry in "${PENDING[@]}"; do
    read -r MODEL_PATH TASK TB_LOGDIR ITERATION N_PARAMS EXP_NAME <<< "$entry"

    if [[ $N_SUBMITTED -ge $MAX_JOBS ]]; then
        NEXT_REMAINING+=("$MODEL_PATH $TASK $TB_LOGDIR $ITERATION $N_PARAMS $EXP_NAME")
        continue
    fi

    LOG_FILE="${LOGDIR}/slurm_logs/${TASK}_iter${ITERATION}_%j.out"
    ERR_FILE="${LOGDIR}/slurm_logs/${TASK}_iter${ITERATION}_%j.err"
    SHORT_TASK="${TASK:0:3}_${TASK: -2}"

    JOB_ID=$(sbatch --parsable \
        --job-name="${EXP_NAME}_${SHORT_TASK}_i${ITERATION}" \
        --partition="${PARTITION}" \
        --account="${ACCOUNT}" \
        --time="${JOB_TIME}" \
        --nodes=1 \
        --ntasks=1 \
        --cpus-per-task="${NUM_WORKERS}" \
        --gres=gpu:1 \
        --mem=64G \
        --output="${LOG_FILE}" \
        --error="${ERR_FILE}" \
        --wrap="
source ~/miniconda3/etc/profile.d/conda.sh
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

    echo "  [${JOB_ID}]  ${EXP_NAME}  iter${ITERATION} (${N_PARAMS}M)  ${TASK}"
    N_SUBMITTED=$((N_SUBMITTED + 1))
done

echo ""
echo "════════════════════════════════════════════════════════"
echo "  ${N_SUBMITTED} jobs submitted from remaining queue"
echo "  Monitor:  squeue -u \$USER"
echo "════════════════════════════════════════════════════════"

if [[ ${#NEXT_REMAINING[@]} -gt 0 ]]; then
    echo ""
    echo "  ⚠  Still ${#NEXT_REMAINING[@]} combinations left — rewriting ${REMAINING_SCRIPT}"

    # Rewrite this script with the new leftovers (self-updating)
    TMP_PENDING=""
    for e in "${NEXT_REMAINING[@]}"; do
        TMP_PENDING+="  $(printf '%q ' $e)"$'\n'
    done

    sed -i "/^PENDING=(/,/^)/{ /^PENDING=(/{ n; d }; /^)/!d }" "$REMAINING_SCRIPT"
    # Simpler approach: just warn and dump to a plain list file
    LEFTOVER_FILE="$(dirname "$REMAINING_SCRIPT")/eval_leftover_combos.txt"
    printf '%s\n' "${NEXT_REMAINING[@]}" > "$LEFTOVER_FILE"
    echo "  Leftover combinations written to: ${LEFTOVER_FILE}"
    echo "  Re-run ${REMAINING_SCRIPT} after updating PENDING from that file."
fi
FOOTER

    chmod +x "$REMAINING_SCRIPT"
    echo ""
    echo "  Run when ready:  bash ${REMAINING_SCRIPT}"
fi