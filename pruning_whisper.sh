#!/bin/bash
#SBATCH --job-name=wspphgm
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

export HF_DATASETS_DISABLE_TORCHCODEC=1
export HF_HUB_OFFLINE=1 
export CUBLAS_WORKSPACE_CONFIG=:4096:8

source ~/miniconda3/etc/profile.d/conda.sh
conda activate myenv

mkdir -p whisper_logs


python pruning_whisper.py