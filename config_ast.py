# config.py
WEIGHTED_SOFTMAX = True
PRUNE_V = True
USE_LORA = True
STRUCTURAL_PRUNING = True

from structured_pruning_utils_fisher_rope import PruningStrategy, ThresholdStrategy, ImportanceStrategy
VERBOSE = True

dataset = "speechcommands"

"""
One of:
    -speechcommands
    -audioset
"""

model_name = 'ast'

"""
- ast
"""


dataset_config={
    'speechcommands':{
        'dataset_name': 'speechcommands',
        'data_root' : f'/leonardo_scratch/large/userexternal/adiecidu/pruning/data/speechcommands/',
        'model_path' : '/leonardo_scratch/large/userexternal/adiecidu/pruning/results/ast/speechcommands/models/best_model.pth',
        'imagenet_pretrain':False,
        'audioset_pretrain':False,
        'bal':None,
        'lr':1e-4,
        'n_epochs':3,
        'freqm':48,
        'timem':48,
        'mixup':0.6,
        
        'train_batch_size':196,
        'test_batch_size' : 64,
        'n_print_steps' : 50,
        
        'fstride':10,
        'tstride':10,

        'dataset_mean':-6.845978,
        'dataset_std':5.5654526,
        'audio_length':128,
        'noise':True,

        'metrics' : 'acc',
        'loss':'BCE',
        'warmup':1,
        'lrscheduler_start':5,
        'lrscheduler_step':1,
        'lrscheduler_decay':0.85,
    
        'n_class' : 35,
        'tr_data':'datafiles/speechcommand_train_data.json',
        'val_data':'datafiles/speechcommand_valid_data.json',
        'eval_data':'datafiles/speechcommand_eval_data.json',
        'label_csv' : 'speechcommands_class_labels_indices.csv'
    },
    'audioset':{
        'dataset_name':'audioset',
        'data_root' : f'/leonardo_scratch/large/userexternal/adiecidu/pruning/data/ast_format',
        'model_path' : '/leonardo_scratch/large/userexternal/adiecidu/pruning/results/ast/audioset/models/best_model.pth',
        'imagenet_pretrain':False,
        'audioset_pretrain':False,
        'freqm':48,
        'timem':192,
        'mixup':0.5,
        'lr':5e-5,
        'n_epochs':4,
        # corresponding to overlap of 6 for 16*16 patches
        'fstride':10,
        'tstride':10,
        
        'train_batch_size':16,
        'test_batch_size' : 16,
        'n_print_steps' : 50,

        'dataset_mean':-4.2677393,
        'dataset_std':4.5689974,
        'audio_length':1024,
        'noise':False,

        'metrics':'mAP',
        'loss':'bce',
        'warmup':1,
        'wa':True,
        
        'lrscheduler_start':5,
        'lrscheduler_step':1,
        'lrscheduler_decay':0.85,
        
        'n_class' : 527,
        'tr_data':'balanced_train_data.json',
        'tf_ub_data':'unbalanced_train_data.json',
        'eval_data':'eval_data.json',
        'label_csv' : 'class_labels_indices.csv'
    }
}
num_workers = 6
    
output_dir = f'/leonardo_scratch/large/userexternal/adiecidu/pruning/results/ast/'

prune_amount_per_iteration = 0.1  # Smaller for iterative effect
num_iterations = 10
order=1
grad_accumulation_steps = 1
warmup = 0.05

seed=42

# LoRA parameters
LORA_RANK = 16
LORA_ALPHA = 16.0
LORA_DROPOUT = 0.1
LORA_LR = 1e-4

# LoRA target modules (which layers to apply LoRA to)
LORA_TARGET_MODULES = ['attention', 'mlp']

#Fisher parameters
fisher_num_samples=1000 
fisher_damping=1e-6
fisher_use_diagonal=True

"""possible values 
    MULTI_HEAD_SAME_CHANNEL  # Same rows across all heads
    MULTI_HEAD_INDEPENDENT #NOT IN USE # Different rows per head, same amount of rows for each head
    MULTI_HEAD_PER_HEAD # Global per-head pruning (each head is pruned by the same amount, but different channels are possible in each head)
    MULTI_HEAD_ENTIRE_HEAD # prune entire heads
"""
pruning_strategy = PruningStrategy.MULTI_HEAD_PER_HEAD

"""
    GLOBAL  # Global threshold across all layers  
                Layers compete against each other - some layers might be heavily pruned while others barely touched
    LOCAL   # Layer-specific thresholds 
                Every layer loses the same percentage of channels, more uniform distribution
"""
threshold_strategy = ThresholdStrategy.GLOBAL

"""
    MAGNITUDE  # standard magnitude pruning
    FISHER_INFORMATION   # uses fisher information to compute channel importance
"""

importance_strategy = ImportanceStrategy.FISHER_INFORMATION