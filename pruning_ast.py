import torch
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
import os
import torch.optim as optim
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import LambdaLR, ReduceLROnPlateau
from lora import LoRAWrapper, merge_lora_weights
import timm
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset
import argparse
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit
from fvcore.nn import FlopCountAnalysis
from copy import deepcopy
import time

import config_ast as config
from structured_pruning_utils_fisher_rope import (get_enum, 
                                                  PruningChanges, 
                                                  PruningStrategy, 
                                                  ThresholdStrategy, 
                                                  FisherConfig, 
                                                  ImportanceStrategy,  
                                                  get_model_sparsity_stats, 
                                                  structured_prune_model as prune_model)

from custom_attention import Attention, Block

from utils import (
    get_layers, get_layer_weights, 
    log_sparsity, log_heatmap, 
    set_seed,
    SystemMonitor, 
    log_system_metrics_to_tensorboard, 
    update_config_from_model, 
    capture_encoder_layer_names, 
    count_parameters,
    compute_flops_with_handlers,
    build_warmup_scheduler)

from ast_utils import train, validate, ASTModel, AudiosetDataset
timm.models.vision_transformer.Attention = Attention
timm.models.vision_transformer.Block = Block

def prune_attention(pruning_strategy, threshold_strategy, importance_strategy, dataset, num_epochs, lr, opti, result_dir, num_iterations=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset_config = config.dataset_config[dataset]
    # output_dir = os.path.join(config.output_dir, dataset_config['dataset_name'])
    output_path = os.path.join(result_dir, dataset_config['dataset_name'], f"pruning_{pruning_strategy.value}_{threshold_strategy.value}_{importance_strategy.value}")
    model_path = dataset_config['model_path']
    
    os.makedirs(output_path, exist_ok=True)
    print("Saving file in directory:", output_path)
    writer = SummaryWriter(os.path.join(output_path, 'tensorboard_logs'))
    
    seed = config.seed
    num_labels = dataset_config['n_class']
    #set seed for experiments
    set_seed(seed)
    
    prune_amount_per_iteration = config.prune_amount_per_iteration  # Smaller for iterative effect
    num_iterations = config.num_iterations if num_iterations is None else num_iterations
    
    audio_conf = {'num_mel_bins': 128, 'target_length': dataset_config['audio_length'], 
                      'freqm': dataset_config['freqm'], 'timem': dataset_config['timem'], 'mixup': dataset_config['mixup'], 
                      'dataset': dataset_config['dataset_name'], 'mode':'train', 
                      'mean':dataset_config['dataset_mean'], 'std':dataset_config['dataset_std'], 'noise':dataset_config['noise']}
    val_audio_conf = {'num_mel_bins': 128, 'target_length': dataset_config['audio_length'], 
                        'freqm': 0, 'timem': 0, 'mixup': 0, 
                        'dataset': dataset_config['dataset_name'], 'mode':'evaluation', 
                        'mean':dataset_config['dataset_mean'], 'std':dataset_config['dataset_std'], 'noise':dataset_config['noise']}
    
    
    
    if dataset_config['dataset_name'] == 'speechcommands':
    # Load Speech Commands data
        tr_data = os.path.join(dataset_config['data_root'], dataset_config['tr_data'])
        val_data = os.path.join(dataset_config['data_root'], dataset_config['val_data'])
        eval_data = os.path.join(dataset_config['data_root'], dataset_config['eval_data'])
        label_csv = os.path.join(dataset_config['data_root'], dataset_config['label_csv'])
        train_dataloader = DataLoader(AudiosetDataset(tr_data, audio_conf, label_csv), batch_size=dataset_config['train_batch_size']) 
        val_dataloader = DataLoader(AudiosetDataset(val_data, val_audio_conf, label_csv), batch_size=dataset_config['test_batch_size'])
        test_dataloader = DataLoader(AudiosetDataset(eval_data, val_audio_conf, label_csv), batch_size=dataset_config['test_batch_size'])
        main_metrics = 'acc'
        loss_fn = torch.nn.BCEWithLogitsLoss()
        
    else:
        
        tr_data = os.path.join(dataset_config['data_root'], dataset_config['tr_data'])
        eval_data = os.path.join(dataset_config['data_root'], dataset_config['eval_data'])
        label_csv = os.path.join(dataset_config['data_root'], dataset_config['label_csv'])
        
        dataset = AudiosetDataset(tr_data, audio_conf, label_csv)

        X = np.arange(len(dataset))               # just indices
        y = dataset.get_all_labels()              # multilabel binary matrix

        msss = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

        train_idx, val_idx = next(msss.split(X, y))
        
        train_dataset = Subset(AudiosetDataset(tr_data, audio_conf, label_csv), train_idx)
        val_dataset   = Subset(AudiosetDataset(tr_data, val_audio_conf, label_csv), val_idx)
        
        train_dataloader = DataLoader(train_dataset, batch_size=dataset_config['train_batch_size'], shuffle=True) 
        val_dataloader = DataLoader(val_dataset, batch_size=dataset_config['test_batch_size'], shuffle=False)
        test_dataloader = DataLoader(AudiosetDataset(eval_data, val_audio_conf, label_csv), batch_size=dataset_config['test_batch_size'], shuffle=False)
        
        loss_fn = torch.nn.BCEWithLogitsLoss()
        main_metrics = 'mAP'
        
        
    for data, label in train_dataloader:
        input_shape = data.shape
        print(input_shape)  
        break
    
    input_shape = list(input_shape)
    input_shape[0] = 1
    print(input_shape) 
    
    flop_inputs = torch.randn(input_shape).to(device)

    # Train AST model
    model = ASTModel(label_dim=num_labels, fstride=dataset_config['fstride'], tstride=dataset_config['tstride'], input_fdim=128,
                     input_tdim=dataset_config['audio_length'], imagenet_pretrain=dataset_config['imagenet_pretrain'],
                     audioset_pretrain=dataset_config['audioset_pretrain'], model_size='base384')
    
    print('Loaded from', model_path)
    state_dict = torch.load(model_path, weights_only=False)
    
        
    model.load_state_dict(state_dict)
    if not hasattr(model, "_encoder_layer_names"):
        model._encoder_layer_names = capture_encoder_layer_names(model)
    model.to(device)
    
    for n, p in model.named_parameters():
        p.requires_grad = False
    
    # Evaluate AST model
    print("\n--- Evaluating AST Model ---")
    model.eval()
    
    eval_monitor = SystemMonitor(interval=0.1)
    eval_monitor.start_monitoring()
    stats, _, inf_time = validate(model, test_dataloader, loss_fn)
    eval_metrics = eval_monitor.stop_monitoring()
    
    log_system_metrics_to_tensorboard(writer, eval_metrics, 0, "System Metrics/")
    
    eval_acc = stats[0]['acc']
    eval_mAUC = np.mean([stat['auc'] for stat in stats])
    eval_map = np.mean([stat['AP'] for stat in stats])
    print('Evaluation accuracy', eval_acc)
    print('Evaluation mAUC', eval_mAUC)
    print('Evaluation mAP', eval_map)
    flops = compute_flops_with_handlers(model, flop_inputs)

    with torch.no_grad():
        times = []
        for i in range(110):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            model(flop_inputs)
            torch.cuda.synchronize()
            times.append(time.perf_counter() - t0)
        times = times[10:]
    avg_ms = (sum(times) / len(times)) * 1000
    print(f"FLOPs: {flops / 1e9:.2f} GFLOPs")
    print(f"Average forward pass speed {avg_ms:.4f} ms")
    writer.add_scalar('flops (GFLOPS)', flops / 1e9, 0)
    writer.add_scalar('Avg speed (ms)', avg_ms, 0)
    
    
    # Compute metrics
    writer.add_scalar('Eval Acc', eval_acc, 0)
    writer.add_scalar('Evaluation mAP', eval_map, 0)
    writer.add_scalar('Evaluation mAUC', eval_mAUC, 0)
    writer.add_scalar('Eval Acc (before fine tuning)', eval_acc, 0)
    writer.add_scalar('Evaluation mAP (before fine tuning)', eval_map, 0)
    writer.add_scalar('Evaluation mAUC (before fine tuning)', eval_mAUC, 0)
    writer.add_scalar('Inference time (s)', inf_time, 0)
    
    attention_layers = get_layers(model)
    
    if config.STRUCTURAL_PRUNING:
        changes = PruningChanges()
    else:
        changes = None
    
    if importance_strategy == ImportanceStrategy.FISHER_INFORMATION:
        fisher_config = FisherConfig(config.fisher_num_samples, config.fisher_damping, config.fisher_use_diagonal)
    else:
        fisher_config = None
    
    # Log heatmaps before pruning
    for i, sa in enumerate(attention_layers):
        #log sparsity of initial model
        writer.add_scalar(f"Sparsity/Layer_{i}_in_proj_weight", 0, 0)
        #log the attention heatmaps of the initial model
        tag = f'Attention_Heatmaps/Layer_{i}_in_proj_weight'
        log_heatmap(writer, sa, tag=tag, iteration=0, order = config.order, strategy_name=pruning_strategy.value)
        _, _, _, _, embed_dim, num_heads, head_dims= get_layer_weights(sa)
        
        num_heads_q, num_heads_k, num_heads_v, num_heads_o = num_heads
        head_dim_q, head_dim_k, head_dim_v, head_dim_o = head_dims
        #log model stats (only needed if structural pruning)
        if config.STRUCTURAL_PRUNING:
            changes.original_config[i] = {
                'embed_dim': embed_dim,
                'num_heads_q': num_heads_q,
                'num_heads_k': num_heads_k,
                'num_heads_v': num_heads_v,
                'num_heads_o': num_heads_o,
                'head_dim_q': head_dim_q,
                'head_dim_k': head_dim_k,
                'head_dim_v': head_dim_v,
                'head_dim_o': head_dim_o
            }

    for iteration in range(num_iterations):
        model = prune_model(model, prune_amount=prune_amount_per_iteration, 
                            pruning_strategy=pruning_strategy, threshold_strategy=threshold_strategy,
                            importance_strategy=importance_strategy, fisher_data_loader= val_dataloader, 
                            fisher_criterion= loss_fn, fisher_config=fisher_config,
                            order=config.order, device=device)
        param_count = count_parameters(model)
        ckpt_dir = os.path.join(output_path, f"model_{param_count//1_000_000}M_params")
        os.makedirs(ckpt_dir, exist_ok=True)
        
        # Collect all self-attention layers
        attention_layers = get_layers(model)
        # Log heatmaps before pruning
        for i, sa in enumerate(attention_layers):
            tag = f'Attention_Heatmaps/Layer_{i}_in_proj_weight'
            log_heatmap(writer, sa, tag=tag, iteration=iteration+1, order = config.order, strategy_name=pruning_strategy.value)
        
        final_stats = get_model_sparsity_stats(model, changes, detailed=True)
        sparsity = final_stats['overall']['sparsity']
        log_sparsity(writer, final_stats['layers'], sparsity*100)
        model.to(device)
        stats, _, _ = validate(model, test_dataloader, loss_fn)
        
        eval_acc = stats[0]['acc']
        eval_mAUC = np.mean([stat['auc'] for stat in stats])
        eval_map = np.mean([stat['AP'] for stat in stats])
        print('Evaluation accuracy', eval_acc)
        print('Evaluation mAUC', eval_mAUC)
        print('Evaluation mAP', eval_map)
        
        # Compute metrics
        writer.add_scalar('Eval Acc (before fine tuning)', eval_acc, sparsity*100)
        writer.add_scalar('Evaluation mAP (before fine tuning)', eval_map, sparsity*100)
        writer.add_scalar('Evaluation mAUC (before fine tuning)', eval_mAUC, sparsity*100)
        flops = compute_flops_with_handlers(model, flop_inputs)
        with torch.no_grad():
            times = []
            for i in range(110):
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                model(flop_inputs)
                torch.cuda.synchronize()
                times.append(time.perf_counter() - t0)
            times = times[10:]
        avg_ms = (sum(times) / len(times)) * 1000
        print(f"FLOPs: {flops / 1e9:.2f} GFLOPs")
        print(f"Average forward pass speed {avg_ms:.4f} ms")
        writer.add_scalar('flops (GFLOPS)', flops / 1e9, sparsity*100)
        writer.add_scalar('Avg speed (ms)', avg_ms, sparsity*100)
        
        #apply LoRA for fine tuning only
        if config.USE_LORA:
            model = LoRAWrapper(model, rank=getattr(config, "LORA_RANK", 16), alpha=getattr(config, "LORA_ALPHA", 16.0)).to(device)
            
            param_list = [p for _, p in model.named_parameters() if p.requires_grad]
            base_params = [p for _, p in model.named_parameters() if not p.requires_grad]

            # Optional: safety assert
            assert all(not bp.requires_grad for bp in base_params), "Base weights must be frozen for LoRA FT"
        else:
            # standard full-model training
            for p in model.parameters():
                p.requires_grad = True
            param_list = [p for _, p in model.named_parameters() if p.requires_grad]
        
        if opti == 'sgd':
            optimizer = optim.SGD(param_list, lr=lr, momentum=0.9, weight_decay=1e-4)
        else:
            optimizer = optim.AdamW(param_list, lr=lr, weight_decay=0.0)

        # warmup
        warmup_scheduler = build_warmup_scheduler(optimizer, num_epochs, len(train_dataloader), config.grad_accumulation_steps, config.warmup)
        
        best_loss = float('inf')
        
        #train and validate the model
        for epoch in range(num_epochs):
            train(model, train_dataloader, loss_fn, optimizer, warmup_scheduler, dataset_config, epoch)
            
            stats, val_loss, inf_time = validate(model, val_dataloader, loss_fn)
            
            eval_acc = stats[0]['acc']
            eval_mAUC = np.mean([stat['auc'] for stat in stats])
            eval_map = np.mean([stat['AP'] for stat in stats])
            
            print(f"Iteration: {iteration}, epoch: {epoch}, accuracy:{eval_acc}, mAUC: {eval_mAUC}, mAP: {eval_map}")
            
            if val_loss < best_loss:
                best_loss = val_loss
                if config.USE_LORA:
                    #Only save the original model weights 
                    original_model = deepcopy(model) 
                    original_model = merge_lora_weights(original_model)
                    torch.save(original_model.state_dict(), os.path.join(ckpt_dir, f'model.pth'))
                    del original_model
                else:
                    torch.save(model.state_dict(), os.path.join(ckpt_dir, f'model.pth'))
            else:
                break
            
        if isinstance(model, LoRAWrapper):
        #Remove LoRA, evaluate and start again
            model = merge_lora_weights(model)
        #load the best weights from the original model
        model.load_state_dict(torch.load(os.path.join(ckpt_dir, f'model.pth'), weights_only=True))

        # Evaluation on test set
        eval_monitor = SystemMonitor(interval=0.1)
        eval_monitor.start_monitoring()
        stats, _, inf_time = validate(model, test_dataloader, loss_fn)
        eval_metrics = eval_monitor.stop_monitoring()
        
        log_system_metrics_to_tensorboard(writer, eval_metrics, sparsity*100, "System Metrics/")

        eval_acc = stats[0]['acc']
        eval_mAUC = np.mean([stat['auc'] for stat in stats])
        eval_map = np.mean([stat['AP'] for stat in stats])
        print('Evaluation accuracy', eval_acc)
        print('Evaluation mAUC', eval_mAUC)
        print('Evaluation mAP', eval_map)
        
        # Compute metrics
        writer.add_scalar('Eval Acc', eval_acc, sparsity*100)
        writer.add_scalar('Evaluation mAP', eval_map, sparsity*100)
        writer.add_scalar('Evaluation mAUC', eval_mAUC, sparsity*100)
        writer.add_scalar('Inference time (s)', inf_time, sparsity*100)

    torch.save(model.state_dict(), os.path.join(ckpt_dir, f'model.pth'))
    writer.close()

        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--pruning_strategy", default=config.pruning_strategy)
    parser.add_argument("--threshold_strategy", default=config.threshold_strategy)
    parser.add_argument("--importance_strategy", default=config.importance_strategy)
    parser.add_argument("--dataset", default=config.dataset)
    parser.add_argument("--num_epochs", default=1, type=int)
    parser.add_argument("--lr", default=1e-4, type=float)
    parser.add_argument("--optim", default='adam')
    parser.add_argument("--result_dir", default=config.output_dir)
    parser.add_argument("--num_iterations", default=None, type=int,
                        help="Override config.num_iterations (e.g. 1 for a quick smoke test)")

    args = parser.parse_args()
    dataset = args.dataset


    # Convert strings to Enum objects
    pruning_strategy = get_enum(PruningStrategy, args.pruning_strategy) if isinstance(args.pruning_strategy, str) else args.pruning_strategy 
    threshold_strategy = get_enum(ThresholdStrategy, args.threshold_strategy) if isinstance(args.threshold_strategy, str) else args.threshold_strategy
    importance_strategy = get_enum(ImportanceStrategy, args.importance_strategy) if isinstance(args.importance_strategy, str) else args.importance_strategy
    
    prune_attention(pruning_strategy, threshold_strategy, importance_strategy, dataset, args.num_epochs, args.lr, args.optim, args.result_dir, args.num_iterations)