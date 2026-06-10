#!/bin/bash
set -euo pipefail

pruning_strategies=("per_head")
threshold_strategies=("global")
importance_strategies=("fisher_information")

LEARNING_RATES=(1e-5 1e-4 1e-3)
OPTIMIZERS=("sgd" "adam")
EPOCHS=(1 2 3)
DATASETS=("audioset" "speechcommands")

# -------------------------
# Iterate over all combinations and submit jobs via Slurm
# -------------------------
for pruning in "${pruning_strategies[@]}"; do
  for threshold in "${threshold_strategies[@]}"; do
    for importance in "${importance_strategies[@]}"; do
      for lr in "${LEARNING_RATES[@]}"; do
        for optim in "${OPTIMIZERS[@]}"; do
          for ep in "${EPOCHS[@]}"; do
            for ds in "${DATASETS[@]}"; do

              # Format learning rate string (e.g., 1e-05 → 1e5)
              lr_str=$(printf "%.0e" "$lr" | sed 's/e-0/e/; s/e-/e/')

              # Job name
              job_name="ast_${ds}_lr${lr_str}_opt${optim}_ep${ep}"

              echo "Submitting job: $job_name"

              sbatch <<EOT
#!/bin/bash
#SBATCH --job-name=${job_name}
#SBATCH --account INA24_C6B05
#SBATCH --output=ast_logs/%x.out
#SBATCH --error=ast_logs/%x.err
#SBATCH --mail-user=andrea.diecidue@polimi.it
#SBATCH --mail-type=ALL
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --partition=boost_usr_prod

echo "Starting job on node: \$(hostname)"
echo "Job started at: \$(date)"

# Activate conda
source ~/miniconda3/etc/profile.d/conda.sh
conda activate myenv

# Ensure log directory exists
mkdir -p pruning_clip_logs

export CUBLAS_WORKSPACE_CONFIG=:4096:8

RESULTS_ROOT="/leonardo_scratch/large/userexternal/adiecidu/pruning/results/ast/${job_name}"

python pruning_ast.py \
  --pruning_strategy "$pruning" \
  --threshold_strategy "$threshold" \
  --importance_strategy "$importance" \
  --lr "$lr" \
  --optim "$optim" \
  --num_epochs "$ep"\
  --dataset "$ds" \
  --result_dir "\$RESULTS_ROOT"

EOT
            done
          done
        done
      done
    done
  done
done