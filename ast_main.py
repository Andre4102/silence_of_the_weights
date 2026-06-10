import torch
import os
import json
import numpy as np
from utils import set_seed
from ast_utils import AudiosetDataset, ASTModel, validate
import config_ast as config
from torch.utils.tensorboard import SummaryWriter
if config.STRUCTURAL_PRUNING:
    from structured_pruning_utils_fisher import PruningChanges, FisherConfig, ImportanceStrategy, get_model_layers, get_model_sparsity_stats, structured_prune_model as prune_model
else:
    from pruning_utils2 import get_model_layers, get_model_sparsity_stats, prune_model
from custom_attention import Attention, Block
from utils import log_sparsity, log_heatmap, set_seed, SystemMonitor, log_system_metrics_to_tensorboard
from ast_utils import train, validate, ASTModel, AudiosetDataset

def convert_state_dict(state_dict):
    new_state_dict = {}

    for k, v in state_dict.items():
        new_k = k

        # Add "v." prefix for most keys
        if not k.startswith("mlp_head"):
            new_k = "v." + new_k

        # Map fc_norm -> norm
        if "fc_norm" in new_k:
            new_k = new_k.replace("fc_norm", "norm")

        # Map head -> head (already fine), but add dist head placeholders if needed
        if new_k == "v.head.weight":
            new_state_dict["v.head_dist.weight"] = torch.zeros_like(v)
        if new_k == "v.head.bias":
            new_state_dict["v.head_dist.bias"] = torch.zeros_like(v)

        new_state_dict[new_k] = v

    return new_state_dict

def main():
    """Main execution function using modular training functions"""
    
    freqm=48
    timem=192
    mixup=0.5
    lr=2.5e-4
    audio_length = 1024
    n_epochs=5
    # corresponding to overlap of 6 for 16*16 patches
    fstride=10
    tstride=10
    
    set_seed(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    weights = "/home/ids/diecidue/results/ast/audioset/audioset_10_10_0.4593.pth"
    num_labels = 527
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    audio_conf = {'num_mel_bins': 128, 'target_length': config.audio_length, 
                      'freqm': config.freqm, 'timem': config.timem, 'mixup': config.mixup, 
                      'dataset': config.dataset_name, 'mode':'train', 
                      'mean':config.dataset_mean, 'std':config.dataset_std, 'noise':config.noise}
    val_audio_conf = {'num_mel_bins': 128, 'target_length': config.audio_length, 
                        'freqm': 0, 'timem': 0, 'mixup': 0, 
                        'dataset': config.dataset_name, 'mode':'evaluation', 
                        'mean':config.dataset_mean, 'std':config.dataset_std, 'noise':False}
    
    
    if config.dataset_name == 'speechcommands':
    # Load Speech Commands data

        train_dataloader = torch.utils.data.DataLoader(AudiosetDataset(config.tr_data, audio_conf, config.label_csv), batch_size=config.train_batch_size) 
        val_dataloader = torch.utils.data.DataLoader(AudiosetDataset(config.val_data, val_audio_conf, config.label_csv), batch_size=config.test_batch_size)
        test_dataloader = torch.utils.data.DataLoader(AudiosetDataset(config.eval_data, val_audio_conf, config.label_csv), batch_size=config.test_batch_size)
    else:
        
        train_dataloader = torch.utils.data.DataLoader(AudiosetDataset(config.tr_data, audio_conf, config.label_csv), batch_size=config.train_batch_size) 
        val_dataloader = None
        test_dataloader = torch.utils.data.DataLoader(AudiosetDataset(config.eval_data, val_audio_conf, config.label_csv), batch_size=config.test_batch_size)
    
    if config.dataset_name == 'audioset':
        main_metrics = 'mAP'
        loss_fn = torch.nn.BCEWithLogitsLoss()
    elif config.dataset_name == 'speechcommands':
        main_metrics = 'acc'
        loss_fn = torch.nn.BCEWithLogitsLoss()
        
    # Train AST model
    model = ASTModel(label_dim=num_labels, fstride=fstride, tstride=tstride, input_fdim=128,
                     input_tdim=audio_length, imagenet_pretrain=True,
                     audioset_pretrain=False, model_size='base384')
    
    state_dict = torch.load(weights, weights_only=True)
    print('Loaded from', weights)
    #state_dict = convert_state_dict(state_dict)
        
    model.load_state_dict(state_dict)
    
    model.to(device)
    
    for n, p in model.named_parameters():
        p.requires_grad = False
    optimizer, scheduler = create_optimizer_and_scheduler(
        model, train_dataloader, config.learning_rate, 
        config.weight_decay, config.num_epochs
    )
    
    for epoch in config.num_epochs:
        treain_loss = train_one_epoch(model, train_dataloader, optimizer, scheduler, device, epoch)
        
        
        if val_dataloader is not None:
            avg_val_loss, val_metrics, _, _= validate_model(model, val_dataloader, device, epoch)
    
    
    
    # Evaluate AST model
    print("\n--- Evaluating AST Model ---")
    model.eval()
    stats, loss, time = validate(model, test_dataloader, loss_fn)
    
    eval_acc = stats[0]['acc']
    eval_mAUC = np.mean([stat['auc'] for stat in stats])
    eval_map = np.mean([stat['AP'] for stat in stats])
    print('Evaluation accuracy', eval_acc)
    print('Evaluation mAUC', eval_mAUC)
    print('Evaluation mAP', eval_map)
    
    torch.save(model.state_dict(), "/home/ids/diecidue/results/audio_output/ast/audioset/best_model.pth")
    
    print("\nTraining completed!")

if __name__ == "__main__":
    main()