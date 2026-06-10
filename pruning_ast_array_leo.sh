#!/bin/bash
set -euo pipefail

pruning_strategies=("per_head" "entire_head")
threshold_strategies=("global" "local")
importance_strategies=("magnitude" "fisher_information")
dataset=("audioset" "speechcommands")
lr=1e-4
optim="adam"
epochs=3

# -------------------------
# Iterate over all combinations and submit jobs via Slurm
# -------------------------
for pruning in "${pruning_strategies[@]}"; do
  for threshold in "${threshold_strategies[@]}"; do
    for importance in "${importance_strategies[@]}"; do
      for ds in "${dataset[@]}"; do
        echo "Submitting job: pruning=$pruning, threshold=$threshold, importance=$importance, dataset=$ds"
        # Abbreviations
        short_pruning=$(echo "$pruning" | awk -F'_' '{print substr($1,1,1) (NF>1 ? substr($2,1,1) : "")}')
        short_threshold=${threshold:0:1}
        short_importance=$(echo "$importance" | awk -F'_' '{print substr($1,1,1) (NF>1 ? substr($2,1,1) : "")}')


      # Submit a Slurm job for each combination
      sbatch <<EOT
#!/bin/bash
#SBATCH --job-name=ast${short_pruning}${short_threshold}${short_importance}${ds}
#SBATCH --account=INA24_C6B05
#SBATCH --output=ast_logs/%x.out
#SBATCH --error=ast_logs/%x.err
#SBATCH --mail-user=andrea.diecidue@polimi.it
#SBATCH --mail-type=ALL
#SBATCH --time=24:00:00                     
#SBATCH --nodes=1                            
#SBATCH --ntasks-per-node=1              
#SBATCH --cpus-per-task=8                 
#SBATCH --gres=gpu:1                        
#SBATCH --partition=boost_usr_prod           
#SBATCH --qos=normal
#SBATCH --mem=32G


echo "Starting job on node: \$(hostname)"
echo "Job started at: \$(date)"

# Activate conda
source ~/miniconda3/etc/profile.d/conda.sh
conda activate myenv

# Ensure log directory exists
mkdir -p pruning_ast_logs

export CUBLAS_WORKSPACE_CONFIG=:4096:8
export TORCHAUDIO_USE_BACKEND_DISPATCHER=0

python pruning_ast.py \
  --pruning_strategy "$pruning" \
  --threshold_strategy "$threshold" \
  --importance_strategy "$importance"\
  --dataset "$ds"\
  --lr $lr\
  --num_epochs $epochs\
  --optim $optim
EOT
      done
    done
  done
done