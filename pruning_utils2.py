import torch
import torch.nn as nn
import numpy as np
from enum import Enum
from collections import defaultdict
from typing import Dict, List, Tuple, Optional
from custom_attention import MultiheadAttention
import tf_locoformer 

class ImportanceStrategy(Enum):
    MAGNITUDE = "magnitude"
    FISHER_INFORMATION = "fisher_information"
    FISHER_DIAGONAL = "fisher_diagonal"

class ThresholdStrategy(Enum):
    GLOBAL = "global"
    LOCAL = "local"


class PruningStrategy(Enum):
    MULTI_HEAD_SAME_CHANNEL = "multi_head_same_channel"
    MULTI_HEAD_PER_HEAD = "multi_head_per_head"  
    MULTI_HEAD_ENTIRE_HEAD = "multi_head_entire_head"


def make_hook(mask_tensor):
    """Creates a gradient masking hook."""
    def hook(grad):
        return grad * mask_tensor
    return hook

def get_model_layers(model):
    """Extract encoder layers from model."""
    layers = []
    if isinstance(model, torch.nn.DataParallel):
        base_model = model.module
    else:
        base_model = model
        
    for _, module in base_model.named_modules():
        if isinstance(module, (nn.MultiheadAttention, MultiheadAttention, tf_locoformer.MultiHeadSelfAttention)):
            layers.append(module)
    
    return layers
    
def get_model_sparsity_stats(model, changes = None, detailed: bool = True):
    """
    Compute sparsity statistics for attention matrices in the model.
    
    Args:
        model: Vision Transformer model
        detailed: If True, returns per-layer and per-head statistics
        
    Returns:
        Dictionary containing sparsity statistics
    """
    # Get layers
    layers = get_model_layers(model)
    
    total_params = 0
    total_zero_params = 0
    layer_stats = []
    
    for layer_idx, sa in enumerate(layers):
        embed_dim = sa.embed_dim
        num_heads = sa.num_heads
        head_dim = sa.head_dim
        
        # Get weights and biases
        weight = sa.in_proj_weight  # Shape: (3*embed_dim, embed_dim)
        bias = sa.in_proj_bias      # Shape: (3*embed_dim,)
        
        # Split into Q, K, V
        q_weight = weight[:embed_dim, :]
        k_weight = weight[embed_dim:2*embed_dim, :]
        v_weight = weight[2*embed_dim:, :]
        
        q_bias = bias[:embed_dim]
        k_bias = bias[embed_dim:2*embed_dim]
        v_bias = bias[2*embed_dim:]
        
        # Count parameters and zeros
        layer_total = weight.numel() + bias.numel()
        layer_zeros = (weight == 0).sum().item() + (bias == 0).sum().item()
        
        total_params += layer_total
        total_zero_params += layer_zeros
        
        layer_sparsity = layer_zeros / layer_total if layer_total > 0 else 0.0
        
        layer_info = {
            'layer_idx': layer_idx,
            'total_params': layer_total,
            'zero_params': layer_zeros,
            'sparsity': layer_sparsity,
            'q_sparsity': (q_weight == 0).sum().item() / q_weight.numel(),
            'k_sparsity': (k_weight == 0).sum().item() / k_weight.numel(),
            'v_sparsity': (v_weight == 0).sum().item() / v_weight.numel(),
            'q_bias_sparsity': (q_bias == 0).sum().item() / q_bias.numel(),
            'k_bias_sparsity': (k_bias == 0).sum().item() / k_bias.numel(),
            'v_bias_sparsity': (v_bias == 0).sum().item() / v_bias.numel(),
        }
        
        # Add per-head statistics if detailed
        if detailed:
            head_stats = []
            
            # Reshape weights to analyze per head
            q_heads = q_weight.reshape(num_heads, head_dim, embed_dim)
            k_heads = k_weight.reshape(num_heads, head_dim, embed_dim)
            v_heads = v_weight.reshape(num_heads, head_dim, embed_dim)
            
            q_bias_heads = q_bias.reshape(num_heads, head_dim)
            k_bias_heads = k_bias.reshape(num_heads, head_dim)
            v_bias_heads = v_bias.reshape(num_heads, head_dim)
            
            for head_idx in range(num_heads):
                q_head = q_heads[head_idx]
                k_head = k_heads[head_idx]
                v_head = v_heads[head_idx]
                
                q_bias_head = q_bias_heads[head_idx]
                k_bias_head = k_bias_heads[head_idx]
                v_bias_head = v_bias_heads[head_idx]
                
                head_total = q_head.numel() + k_head.numel() + v_head.numel() + \
                           q_bias_head.numel() + k_bias_head.numel() + v_bias_head.numel()
                           
                head_zeros = (q_head == 0).sum().item() + (k_head == 0).sum().item() + \
                           (v_head == 0).sum().item() + (q_bias_head == 0).sum().item() + \
                           (k_bias_head == 0).sum().item() + (v_bias_head == 0).sum().item()
                
                # Check if entire head is pruned
                entire_head_pruned = (q_head == 0).all() and (k_head == 0).all() and (v_head == 0).all()
                
                # Count pruned rows (channels)
                q_pruned_rows = (q_head.abs().sum(dim=1) == 0).sum().item()
                k_pruned_rows = (k_head.abs().sum(dim=1) == 0).sum().item() 
                v_pruned_rows = (v_head.abs().sum(dim=1) == 0).sum().item()
                
                head_info = {
                    'head_idx': head_idx,
                    'total_params': head_total,
                    'zero_params': head_zeros,
                    'sparsity': head_zeros / head_total if head_total > 0 else 0.0,
                    'entire_head_pruned': entire_head_pruned,
                    'q_pruned_rows': q_pruned_rows,
                    'k_pruned_rows': k_pruned_rows,
                    'v_pruned_rows': v_pruned_rows,
                    'q_row_sparsity': q_pruned_rows / head_dim,
                    'k_row_sparsity': k_pruned_rows / head_dim,
                    'v_row_sparsity': v_pruned_rows / head_dim,
                }
                head_stats.append(head_info)
            
            layer_info['heads'] = head_stats
        
        layer_stats.append(layer_info)
    
    # Compute overall statistics
    overall_sparsity = total_zero_params / total_params if total_params > 0 else 0.0
    
    # Compute matrix-wise statistics across all layers
    all_q_params = all_k_params = all_v_params = 0
    all_q_zeros = all_k_zeros = all_v_zeros = 0
    all_q_bias_params = all_k_bias_params = all_v_bias_params = 0
    all_q_bias_zeros = all_k_bias_zeros = all_v_bias_zeros = 0
    
    for layer_info in layer_stats:
        layer_idx = layer_info['layer_idx']
        layer = layers[layer_idx]
        sa = layer.self_attention
        embed_dim = sa.embed_dim
        
        weight = sa.in_proj_weight
        bias = sa.in_proj_bias
        
        q_w = weight[:embed_dim, :]
        k_w = weight[embed_dim:2*embed_dim, :]
        v_w = weight[2*embed_dim:, :]
        
        q_b = bias[:embed_dim]
        k_b = bias[embed_dim:2*embed_dim]
        v_b = bias[2*embed_dim:]
        
        all_q_params += q_w.numel()
        all_k_params += k_w.numel()
        all_v_params += v_w.numel()
        
        all_q_zeros += (q_w == 0).sum().item()
        all_k_zeros += (k_w == 0).sum().item()
        all_v_zeros += (v_w == 0).sum().item()
        
        all_q_bias_params += q_b.numel()
        all_k_bias_params += k_b.numel()
        all_v_bias_params += v_b.numel()
        
        all_q_bias_zeros += (q_b == 0).sum().item()
        all_k_bias_zeros += (k_b == 0).sum().item()
        all_v_bias_zeros += (v_b == 0).sum().item()
    
    # Count entirely pruned heads across all layers
    entirely_pruned_heads = 0
    if detailed:
        for layer_info in layer_stats:
            if 'heads' in layer_info:
                entirely_pruned_heads += sum(1 for head in layer_info['heads'] if head['entire_head_pruned'])
    
    stats = {
        'overall': {
            'total_params': total_params,
            'zero_params': total_zero_params,
            'sparsity': overall_sparsity,
            'num_layers': len(layers),
            'entirely_pruned_heads': entirely_pruned_heads if detailed else None,
        },
        'by_matrix': {
            'q_weight_sparsity': all_q_zeros / all_q_params if all_q_params > 0 else 0.0,
            'k_weight_sparsity': all_k_zeros / all_k_params if all_k_params > 0 else 0.0,
            'v_weight_sparsity': all_v_zeros / all_v_params if all_v_params > 0 else 0.0,
            'q_bias_sparsity': all_q_bias_zeros / all_q_bias_params if all_q_bias_params > 0 else 0.0,
            'k_bias_sparsity': all_k_bias_zeros / all_k_bias_params if all_k_bias_params > 0 else 0.0,
            'v_bias_sparsity': all_v_bias_zeros / all_v_bias_params if all_v_bias_params > 0 else 0.0,
        },
        'layers': layer_stats if detailed else None,
    }
    
    return stats

def prune_model(model, prune_amount: float, 
                threshold_strategy: ThresholdStrategy = ThresholdStrategy.GLOBAL,
                pruning_strategy: PruningStrategy = PruningStrategy.MULTI_HEAD_SAME_CHANNEL,
                order: int = 2, prune_v: bool = False):
    """
    Main function to prune a model with specified strategies.
    
    Args:
        model: Vision Transformer model
        prune_amount: Fraction of parameters to prune (0.0 to 1.0)
        threshold_strategy: GLOBAL or LOCAL threshold computation
        pruning_strategy: Type of pruning to apply
        order: Norm order (1 for L1, 2 for L2)
        prune_v: Whether to also prune V matrices
        writer: Optional tensorboard writer
    
    Returns:
        Pruned model
    """
    # Get layers
    layers = get_model_layers(model)
    
    # Apply pruning based on strategies
    if threshold_strategy == ThresholdStrategy.GLOBAL:
        _global_prune(layers, prune_amount, pruning_strategy, order, prune_v)
    elif threshold_strategy == ThresholdStrategy.LOCAL:
        _local_prune(layers, prune_amount, pruning_strategy, order, prune_v)
    
    return model


def _global_prune(layers, prune_amount: float, pruning_strategy: PruningStrategy, 
                  order: int, prune_v: bool):
    """Apply global pruning strategy."""
    if pruning_strategy == PruningStrategy.MULTI_HEAD_SAME_CHANNEL:
        threshold = _compute_global_same_channel_threshold(layers, prune_amount, order)
        for layer in layers:
            _apply_same_channel_pruning(layer.self_attention, threshold, order, prune_v)
    
    elif pruning_strategy == PruningStrategy.MULTI_HEAD_PER_HEAD:
        norms = _collect_per_head_norms(layers, order)
        plan = _build_global_per_head_plan(norms, prune_amount)
        for idx, layer in enumerate(layers):
            _apply_per_head_pruning(layer.self_attention, plan, idx, prune_v)
    
    elif pruning_strategy == PruningStrategy.MULTI_HEAD_ENTIRE_HEAD:
        plan = _build_global_entire_head_plan(layers, prune_amount, order)
        for idx, layer in enumerate(layers):
            _apply_entire_head_pruning(layer.self_attention, plan, idx, prune_v)


def _local_prune(layers, prune_amount: float, pruning_strategy: PruningStrategy,
                 order: int, prune_v: bool):
    """Apply local pruning strategy (same budget per layer)."""
    for idx, layer in enumerate(layers):
        if pruning_strategy == PruningStrategy.MULTI_HEAD_SAME_CHANNEL:
            threshold = _compute_local_same_channel_threshold(layer.self_attention, prune_amount, order)
            _apply_same_channel_pruning(layer.self_attention, threshold, order, prune_v)
        
        elif pruning_strategy == PruningStrategy.MULTI_HEAD_PER_HEAD:
            _apply_local_per_head_pruning(layer.self_attention, prune_amount, order, prune_v)
        
        elif pruning_strategy == PruningStrategy.MULTI_HEAD_ENTIRE_HEAD:
            _apply_local_entire_head_pruning(layer.self_attention, prune_amount, order, prune_v)


# Same Channel Pruning Functions
def _compute_global_same_channel_threshold(layers, prune_amount: float, order: int) -> float:
    """Compute global threshold for same channel pruning."""
    all_norms = []
    
    for layer in layers:
        sa = layer.self_attention
        embed_dim = sa.embed_dim
        num_heads = sa.num_heads
        head_dim = sa.head_dim

        q = sa.in_proj_weight[:embed_dim, :].detach().cpu().numpy()
        k = sa.in_proj_weight[embed_dim:2*embed_dim, :].detach().cpu().numpy()
        
        # Reshape to (num_heads, head_dim, embed_dim)
        q = q.reshape(num_heads, head_dim, embed_dim)
        k = k.reshape(num_heads, head_dim, embed_dim)
        
        # Concatenate Q and K
        qk = np.concatenate((q, k), axis=2)
        
        # Compute row norms per head and aggregate
        row_norms_per_head = np.linalg.norm(qk, ord=order, axis=2)
        row_norms = row_norms_per_head.mean(axis=0)  # Average across heads
        
        all_norms.extend(row_norms)

    all_norms = np.array([n for n in all_norms if n != 0])
    k = min(int(len(all_norms) * prune_amount), len(all_norms) - 1)
    return np.partition(all_norms, k)[k]


def _compute_local_same_channel_threshold(sa, prune_amount: float, order: int) -> float:
    """Compute local threshold for same channel pruning."""
    embed_dim = sa.embed_dim
    num_heads = sa.num_heads
    head_dim = sa.head_dim

    q = sa.in_proj_weight[:embed_dim, :].detach().cpu().numpy()
    k = sa.in_proj_weight[embed_dim:2*embed_dim, :].detach().cpu().numpy()
    
    # Reshape to (num_heads, head_dim, embed_dim)
    q = q.reshape(num_heads, head_dim, embed_dim)
    k = k.reshape(num_heads, head_dim, embed_dim)
    
    # Concatenate Q and K
    qk = np.concatenate((q, k), axis=2)
    
    # Compute row norms per head and aggregate
    row_norms_per_head = np.linalg.norm(qk, ord=order, axis=2)
    row_norms = row_norms_per_head.mean(axis=0)  # Average across heads
    
    row_norms = np.array([n for n in row_norms if n != 0])
    k = min(int(len(row_norms) * prune_amount), len(row_norms) - 1)
    return np.partition(row_norms, k)[k]


def _apply_same_channel_pruning(sa, threshold: float, order: int, prune_v: bool):
    """Apply same channel pruning to attention layer."""
    weight, bias = sa.in_proj_weight, sa.in_proj_bias
    embed_dim = sa.embed_dim
    num_heads = sa.num_heads
    head_dim = sa.head_dim

    # Split weights and biases
    q, k, v = weight[:embed_dim], weight[embed_dim:2*embed_dim], weight[2*embed_dim:]
    q_bias, k_bias, v_bias = bias[:embed_dim], bias[embed_dim:2*embed_dim], bias[2*embed_dim:]

    # Reshape to (num_heads, head_dim, embed_dim)
    q = q.reshape(num_heads, head_dim, embed_dim)
    k = k.reshape(num_heads, head_dim, embed_dim)
    v = v.reshape(num_heads, head_dim, embed_dim)

    # Compute mask based on aggregated row norms
    qk = torch.cat((q, k), dim=2)
    row_norms_per_head = torch.norm(qk, p=order, dim=2)
    row_norms = row_norms_per_head.mean(dim=0)  # Average across heads
    
    mask = row_norms >= threshold  # shape: (head_dim,)
    mask_tensor = mask.to(dtype=weight.dtype, device=weight.device)

    # Apply mask to all heads (same channels pruned)
    mask_expanded = mask_tensor[None, :, None]  # (1, head_dim, 1)
    
    # Zero out pruned parameters
    q.data *= mask_expanded
    k.data *= mask_expanded
    if prune_v:
        v.data *= mask_expanded

    # Handle biases
    q_bias = q_bias.reshape(num_heads, head_dim)
    k_bias = k_bias.reshape(num_heads, head_dim)
    v_bias = v_bias.reshape(num_heads, head_dim)

    q_bias.data *= mask_tensor[None, :]
    k_bias.data *= mask_tensor[None, :]
    if prune_v:
        v_bias.data *= mask_tensor[None, :]

    # Reshape back and update parameters
    new_weight = torch.cat([
        q.reshape(embed_dim, embed_dim),
        k.reshape(embed_dim, embed_dim), 
        v.reshape(embed_dim, embed_dim)
    ], dim=0)
    
    new_bias = torch.cat([
        q_bias.reshape(embed_dim),
        k_bias.reshape(embed_dim),
        v_bias.reshape(embed_dim)
    ], dim=0)

    sa.in_proj_weight.data = new_weight
    sa.in_proj_bias.data = new_bias

    # Apply gradient masking
    _apply_same_channel_gradient_mask(sa, mask_tensor, embed_dim, num_heads, prune_v)


def _apply_same_channel_gradient_mask(sa, mask_tensor, embed_dim, num_heads, prune_v):
    """Apply gradient masking for same channel pruning."""
    broadcast_mask = mask_tensor.repeat(num_heads)  # (embed_dim,)
    full_mask = broadcast_mask[:, None].expand(-1, embed_dim)

    qk_mask_full = torch.cat((full_mask, full_mask), dim=0)
    v_mask_full = full_mask if prune_v else torch.ones_like(sa.in_proj_weight[2*embed_dim:, :])

    weight_mask = torch.cat((qk_mask_full, v_mask_full), dim=0)
    bias_mask = torch.cat((
        broadcast_mask,
        broadcast_mask,
        broadcast_mask if prune_v else torch.ones_like(sa.in_proj_bias[2*embed_dim:])
    ))

    sa.in_proj_weight.register_hook(make_hook(weight_mask))
    sa.in_proj_bias.register_hook(make_hook(bias_mask))


# Per Head Pruning Functions
def _collect_per_head_norms(layers, order: int) -> Dict:
    """Collect QK row norms per head from all layers."""
    norms = {}
    
    for layer_idx, layer in enumerate(layers):
        sa = layer.self_attention
        embed_dim = sa.embed_dim
        num_heads = sa.num_heads
        head_dim = sa.head_dim

        q = sa.in_proj_weight[:embed_dim, :].detach()
        k = sa.in_proj_weight[embed_dim:2*embed_dim, :].detach()

        q = q.reshape(num_heads, head_dim, embed_dim)
        k = k.reshape(num_heads, head_dim, embed_dim)

        qk = torch.cat((q, k), dim=2)
        row_norms = torch.norm(qk, p=order, dim=2)  # (num_heads, head_dim)

        norms[layer_idx] = {head_idx: row_norms[head_idx] for head_idx in range(num_heads)}

    return norms


def _build_global_per_head_plan(norms_dict: Dict, prune_amount: float) -> Dict:
    """Build global pruning plan for per-head strategy."""
    # Count total rows and allocate budget proportionally
    total_rows = 0
    layer_row_counts = defaultdict(int)
    layer_head_counts = defaultdict(int)
    
    for layer_idx, heads in norms_dict.items():
        for head_idx, scores in heads.items():
            num_rows = len(scores)
            total_rows += num_rows
            layer_row_counts[layer_idx] += num_rows
            layer_head_counts[layer_idx] += 1

    total_to_prune = int(prune_amount * total_rows)

    # Allocate budget per layer proportionally
    layer_prune_counts = {
        layer_idx: int(total_to_prune * (layer_row_counts[layer_idx] / total_rows))
        for layer_idx in norms_dict
    }

    # Build pruning plan
    pruning_plan = {}
    
    for layer_idx, heads in norms_dict.items():
        pruning_plan[layer_idx] = {}
        num_heads = layer_head_counts[layer_idx]
        num_to_prune = layer_prune_counts[layer_idx]
        
        rows_per_head = len(next(iter(heads.values())))
        rows_to_prune_per_head = min(num_to_prune // num_heads, rows_per_head)

        for head_idx, scores in heads.items():
            if rows_to_prune_per_head == 0:
                prune_indices = []
            else:
                prune_indices = torch.topk(scores, rows_to_prune_per_head, largest=False).indices.tolist()
            pruning_plan[layer_idx][head_idx] = prune_indices

    return pruning_plan


def _apply_per_head_pruning(sa, pruning_plan: Dict, layer_idx: int, prune_v: bool):
    """Apply per-head pruning based on plan."""
    if layer_idx not in pruning_plan:
        return

    weight, bias = sa.in_proj_weight, sa.in_proj_bias
    embed_dim = sa.embed_dim
    num_heads = sa.num_heads
    head_dim = sa.head_dim

    q, k, v = weight[:embed_dim], weight[embed_dim:2*embed_dim], weight[2*embed_dim:]
    q_bias, k_bias, v_bias = bias[:embed_dim], bias[embed_dim:2*embed_dim], bias[2*embed_dim:]

    q = q.reshape(num_heads, head_dim, embed_dim)
    k = k.reshape(num_heads, head_dim, embed_dim)
    v = v.reshape(num_heads, head_dim, embed_dim)

    # Create mask
    mask = torch.ones((num_heads, head_dim), dtype=weight.dtype, device=weight.device)

    for head_idx, rows in pruning_plan[layer_idx].items():
        mask[head_idx, rows] = 0

    mask_exp = mask[:, :, None]  # (num_heads, head_dim, 1)

    # Zero out pruned parameters
    q.data *= mask_exp
    k.data *= mask_exp
    if prune_v:
        v.data *= mask_exp

    # Handle biases
    q_bias = q_bias.reshape(num_heads, head_dim)
    k_bias = k_bias.reshape(num_heads, head_dim)
    v_bias = v_bias.reshape(num_heads, head_dim)

    q_bias.data *= mask
    k_bias.data *= mask
    if prune_v:
        v_bias.data *= mask

    # Update parameters
    sa.in_proj_weight.data = torch.cat([
        q.reshape(embed_dim, embed_dim),
        k.reshape(embed_dim, embed_dim),
        v.reshape(embed_dim, embed_dim)
    ], dim=0)
    
    sa.in_proj_bias.data = torch.cat([
        q_bias.reshape(embed_dim),
        k_bias.reshape(embed_dim), 
        v_bias.reshape(embed_dim)
    ], dim=0)

    # Apply gradient masking
    _apply_per_head_gradient_mask(sa, mask, embed_dim, prune_v)


def _apply_local_per_head_pruning(sa, prune_amount: float, order: int, prune_v: bool):
    """Apply local per-head pruning."""
    embed_dim = sa.embed_dim
    num_heads = sa.num_heads
    head_dim = sa.head_dim
    
    q = sa.in_proj_weight[:embed_dim, :].reshape(num_heads, head_dim, embed_dim)
    k = sa.in_proj_weight[embed_dim:2*embed_dim, :].reshape(num_heads, head_dim, embed_dim)
    
    qk = torch.cat((q, k), dim=2)
    row_norms = torch.norm(qk, p=order, dim=2)  # (num_heads, head_dim)
    
    # Create local plan
    plan = {0: {}}  # Layer 0
    rows_to_prune_per_head = max(1, int(head_dim * prune_amount))
    
    for head_idx in range(num_heads):
        prune_indices = torch.topk(row_norms[head_idx], rows_to_prune_per_head, largest=False).indices.tolist()
        plan[0][head_idx] = prune_indices
    
    _apply_per_head_pruning(sa, plan, 0, prune_v)


def _apply_per_head_gradient_mask(sa, mask, embed_dim, prune_v):
    """Apply gradient masking for per-head pruning."""
    broadcast_mask = mask.reshape(-1)  # (embed_dim,)
    full_mask = broadcast_mask[:, None].expand(-1, embed_dim)

    qk_mask_full = torch.cat((full_mask, full_mask), dim=0)
    v_mask_full = full_mask if prune_v else torch.ones_like(sa.in_proj_weight[2*embed_dim:, :])

    weight_mask = torch.cat((qk_mask_full, v_mask_full), dim=0)
    bias_mask = torch.cat((
        broadcast_mask,
        broadcast_mask,
        broadcast_mask if prune_v else torch.ones_like(sa.in_proj_bias[2*embed_dim:])
    ))

    sa.in_proj_weight.register_hook(make_hook(weight_mask))
    sa.in_proj_bias.register_hook(make_hook(bias_mask))


# Entire Head Pruning Functions
def _build_global_entire_head_plan(layers, prune_amount: float, order: int) -> Dict:
    """Build global plan for entire head pruning."""
    all_head_scores = []
    head_map = []
    
    for layer_idx, layer in enumerate(layers):
        sa = layer.self_attention
        embed_dim = sa.embed_dim
        num_heads = sa.num_heads
        head_dim = sa.head_dim
        
        q = sa.in_proj_weight[:embed_dim, :].reshape(num_heads, head_dim, embed_dim)
        k = sa.in_proj_weight[embed_dim:2*embed_dim, :].reshape(num_heads, head_dim, embed_dim)
        
        qk = torch.cat((q, k), dim=2)
        
        # Score entire head by average row norm
        head_scores = torch.norm(qk, p=order, dim=2).mean(dim=1)  # (num_heads,)
        
        for head_idx, score in enumerate(head_scores):
            all_head_scores.append(score.item())
            head_map.append((layer_idx, head_idx))
    
    # Select heads to prune globally
    total_heads = len(all_head_scores)
    heads_to_prune = int(total_heads * prune_amount)
    
    all_head_scores = torch.tensor(all_head_scores)
    _, prune_indices = torch.topk(all_head_scores, heads_to_prune, largest=False)
    
    # Build plan
    plan = defaultdict(list)
    for idx in prune_indices:
        layer_idx, head_idx = head_map[idx]
        plan[layer_idx].append(head_idx)
        
    return dict(plan)


def _apply_entire_head_pruning(sa, plan: Dict, layer_idx: int, prune_v: bool):
    """Apply entire head pruning."""
    if layer_idx not in plan:
        return
        
    weight, bias = sa.in_proj_weight, sa.in_proj_bias
    embed_dim = sa.embed_dim
    num_heads = sa.num_heads
    head_dim = sa.head_dim
    
    heads_to_prune = plan[layer_idx]
    
    # Create head mask
    head_mask = torch.ones(num_heads, dtype=weight.dtype, device=weight.device)
    head_mask[heads_to_prune] = 0
    
    # Apply to Q, K, V weights
    for start_idx in [0, embed_dim, 2*embed_dim]:  # Q, K, V
        if start_idx == 2*embed_dim and not prune_v:
            continue
        weight_section = weight[start_idx:start_idx+embed_dim].reshape(num_heads, head_dim, embed_dim)
        weight_section.data *= head_mask[:, None, None]
        weight[start_idx:start_idx+embed_dim] = weight_section.reshape(embed_dim, embed_dim)
    
    # Apply to biases
    for start_idx in [0, embed_dim, 2*embed_dim]:  # Q, K, V
        if start_idx == 2*embed_dim and not prune_v:
            continue
        bias_section = bias[start_idx:start_idx+embed_dim].reshape(num_heads, head_dim)
        bias_section.data *= head_mask[:, None]
        bias[start_idx:start_idx+embed_dim] = bias_section.reshape(embed_dim)
    
    # Apply gradient masking
    _apply_entire_head_gradient_mask(sa, head_mask, embed_dim, head_dim, prune_v)


def _apply_local_entire_head_pruning(sa, prune_amount: float, order: int, prune_v: bool):
    """Apply local entire head pruning."""
    embed_dim = sa.embed_dim
    num_heads = sa.num_heads
    head_dim = sa.head_dim
    
    q = sa.in_proj_weight[:embed_dim, :].reshape(num_heads, head_dim, embed_dim)
    k = sa.in_proj_weight[embed_dim:2*embed_dim, :].reshape(num_heads, head_dim, embed_dim)
    
    qk = torch.cat((q, k), dim=2)
    head_scores = torch.norm(qk, p=order, dim=2).mean(dim=1)  # (num_heads,)
    
    heads_to_prune = max(1, int(num_heads * prune_amount))
    _, prune_indices = torch.topk(head_scores, heads_to_prune, largest=False)
    
    plan = {0: prune_indices.tolist()}
    _apply_entire_head_pruning(sa, plan, 0, prune_v)


def _apply_entire_head_gradient_mask(sa, head_mask, embed_dim, head_dim, prune_v):
    """Apply gradient masking for entire head pruning."""
    # Expand head mask to full dimensions
    expanded_mask = head_mask.repeat_interleave(head_dim)  # (embed_dim,)
    full_mask = expanded_mask[:, None].expand(-1, embed_dim)
    
    # Create masks for Q, K, V
    q_mask = full_mask
    k_mask = full_mask  
    v_mask = full_mask if prune_v else torch.ones_like(sa.in_proj_weight[2*embed_dim:, :])
    
    weight_mask = torch.cat((q_mask, k_mask, v_mask), dim=0)
    bias_mask = torch.cat((
        expanded_mask,  # Q bias
        expanded_mask,  # K bias  
        expanded_mask if prune_v else torch.ones_like(sa.in_proj_bias[2*embed_dim:])  # V bias
    ))
    
    sa.in_proj_weight.register_hook(make_hook(weight_mask))
    sa.in_proj_bias.register_hook(make_hook(bias_mask))