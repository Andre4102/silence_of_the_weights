import torch
device = "cuda" if torch.cuda.is_available() else "cpu"
SPLIT = "test-clean"
MAX_SAMPLES = None
USE_LORA = False

from structured_pruning_utils_fisher_rope import PruningStrategy, ThresholdStrategy, ImportanceStrategy
VERBOSE = True

LIBRISPEECH_ROOT = "/leonardo_scratch/large/userexternal/adiecidu/pruning/data/audio/LibriSpeech"
FLEURS_ROOT = "/leonardo_scratch/large/userexternal/adiecidu/pruning/data/audio/fleurs/data"
MEANWHILE_ROOT = "/leonardo_scratch/large/userexternal/adiecidu/pruning/data/audio/meanwhile"
COVOST_ROOT = "/leonardo_scratch/large/userexternal/adiecidu/pruning/data/audio/covost/data/"
COMMON_VOICE_ROOT = "/leonardo_scratch/large/userexternal/adiecidu/pruning/data/audio/commonvoice/cv-corpus-24.0-2025-12-05"
VOXPOPULI_ROOT = "/leonardo_scratch/large/userexternal/adiecidu/pruning/data/audio/voxpopuli"
MLS_ROOT = "/leonardo_scratch/large/userexternal/adiecidu/pruning/data/audio/mls/data"

MODEL_NAME = "/leonardo_scratch/large/userexternal/adiecidu/pruning/results/whisper/basemodel"
teacher_path = "/leonardo_scratch/large/userexternal/adiecidu/pruning/results/whisper/large2"

results_root = '/leonardo_scratch/large/userexternal/adiecidu/pruning/results/whisper/'

seed = 42
batch_size = 4
num_workers=6
prune_amount_per_iteration = 0.1 
num_iterations = 10
num_epochs = 1
order=1
gradient_accumulation_steps = 4
learning_rate = 1e-4
weight_decay = 0.01
adam_epsilon = 1e-8
max_grad_norm = 1.0
warmup_steps = .05
lr = 1e-5

# LoRA parameters
LORA_RANK = 16
LORA_ALPHA = 16.0
LORA_LR = 3e-5 

#Fisher parameters
fisher_num_samples=1000 
fisher_damping=1e-6
fisher_use_diagonal=True

pruning_strategy = PruningStrategy.MULTI_HEAD_PER_HEAD
"""possible values 
    MULTI_HEAD_SAME_CHANNEL  # Same rows across all heads
    MULTI_HEAD_PER_HEAD # Global per-head pruning (each head is pruned by the same amount, but different channels are possible in each head)
    MULTI_HEAD_ENTIRE_HEAD # prune entire heads
"""

threshold_strategy = ThresholdStrategy.GLOBAL
"""
    GLOBAL  # Global threshold across all layers  
                Layers compete against each other - some layers might be heavily pruned while others barely touched
    LOCAL   # Layer-specific thresholds 
                Every layer loses the same percentage of channels, more uniform distribution
"""


importance_strategy = ImportanceStrategy.FISHER_INFORMATION
"""
    MAGNITUDE  # standard magnitude pruning
    FISHER_INFORMATION   # uses fisher information to compute channel importance
"""