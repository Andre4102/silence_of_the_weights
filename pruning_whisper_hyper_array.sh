#!/bin/bash
set -euo pipefail

pruning_strategies=("per_head")
threshold_strategies=("global")
importance_strategies=("fisher_information")
LEARNING_RATES=(5e-4 1e-4 5e-5 1e-5)
OPTIMIZERS=("sgd" "adam")
EPOCHS=(1 2)

# -------------------------
# Iterate over all combinations and submit jobs via Slurm
# -------------------------
for pruning in "${pruning_strategies[@]}"; do
  for threshold in "${threshold_strategies[@]}"; do
    for importance in "${importance_strategies[@]}"; do
      for lr in "${LEARNING_RATES[@]}"; do
        for optim in "${OPTIMIZERS[@]}"; do
          for ep in "${EPOCHS[@]}"; do
          # Format learning rate string (e.g., 1e-05 → 1e5)
            lr_str=$(printf "%.0e" "$lr" | sed 's/e-0/e/; s/e-/e/')

            # Job name
            job_name="whisper_lr${lr_str}_opt${optim}_ep${ep}"
            RESULTS_ROOT="/leonardo_scratch/large/userexternal/adiecidu/pruning/results/whisper/${job_name}"

            echo "Submitting job: $job_name"
            # Submit a Slurm job for each combination
            sbatch <<EOT
#!/bin/bash
#SBATCH --job-name=${job_name}
#SBATCH --account=INA24_C6B05
#SBATCH --output=whisper_logs/%x.out
#SBATCH --error=whisper_logs/%x.err
#SBATCH --mail-user=andrea.diecidue@polimi.it
#SBATCH --mail-type=ALL
#SBATCH --time=24:00:00                     
#SBATCH --nodes=1                            
#SBATCH --ntasks-per-node=1              
#SBATCH --cpus-per-task=8                 
#SBATCH --gres=gpu:1                        
#SBATCH --partition=boost_usr_prod           
#SBATCH --qos=normal
#SBATCH --mem=128G


echo "Starting job on node: \$(hostname)"
echo "Job started at: \$(date)"

# Activate conda
source ~/miniconda3/etc/profile.d/conda.sh
conda activate myenv

# Ensure log directory exists
mkdir -p whisper_logs

export HF_DATASETS_DISABLE_TORCHCODEC=1
export HF_HUB_OFFLINE=1 
export CUBLAS_WORKSPACE_CONFIG=:4096:8

python pruning_whisper.py \
  --pruning_strategy "$pruning" \
  --threshold_strategy "$threshold" \
  --importance_strategy "$importance"\
  --epochs "$ep"\
  --lr "$lr"\
  --optim "$optim"\
   --results_root ${RESULTS_ROOT}

EOT
          done
        done
      done
    done
  done
done