import torch
import torch.nn as nn
from enum import Enum
from collections import defaultdict
from typing import Dict, List, Optional
from dataclasses import dataclass
import tf_locoformer
from custom_attention import Attention, LlamaAttention, CustomViTAttention
from transformers.models.bert.modeling_bert import BertAttention
from utils import get_model_layers, get_layer_weights, get_layer_weights_fisher
from torch.nn import CrossEntropyLoss
from tqdm import tqdm
import math 

NORM = False

@dataclass
class FisherConfig:
    """Configuration for Fisher Information computation."""
    num_samples: int = 1000  # Number of samples to use for Fisher computation
    damping: float = 1e-6    # Damping factor for numerical stability
    use_diagonal: bool = True  # Use diagonal approximation (much faster)

class ThresholdStrategy(Enum):
    GLOBAL = "global"
    LOCAL = "local"


class PruningStrategy(Enum):
    MULTI_HEAD_SAME_CHANNEL = "same_channel"
    MULTI_HEAD_PER_HEAD = "per_head"  
    MULTI_HEAD_ENTIRE_HEAD = "entire_head"

# Add new importance strategy enum
class ImportanceStrategy(Enum):
    MAGNITUDE = "magnitude"
    FISHER_INFORMATION = "fisher_information"
    
def get_enum(enum_class, value):
    for e in enum_class:
        if e.value == value:
            return e
    raise ValueError(f"{value} is not a valid value for {enum_class}")

class PruningChanges:
    """Track changes made during structured pruning."""
    
    def __init__(self, model=None):
        self.removed_layers = []  # List of layer indices that were completely removed
        self.layer_changes = {}   # Dict mapping layer_idx to LayerChanges
        self.original_config = {}  # Original model configuration
        self.final_config = {}     # Final model configuration after pruning
        
        if model is not None:
            layers = get_model_layers(model)
    
            # Store original configuration
            for i, sa in enumerate(layers):
                self.original_config[i] = {
                    'embed_dim': sa.embed_dim,
                    'num_heads_q': sa.num_heads_q,
                    'num_heads_k': sa.num_heads_k,
                    'num_heads_v': sa.num_heads_v,
                    'num_heads_o': sa.num_heads_out,
                    'head_dim_q': sa.head_dim_q,
                    'head_dim_k': sa.head_dim_k,
                    'head_dim_v': sa.head_dim_v,
                    'head_dim_o': sa.head_dim_out
                }
    
    def add_layer_change(self, layer_idx: int, removed_heads: List[int], removed_channels_qk: List[int], removed_channels_vo: List[int]):
        """Add changes for a specific layer."""
        if layer_idx not in self.layer_changes:
            self.layer_changes[layer_idx] = {
                'removed_heads_q': set(),
                'removed_heads_k': set(), 
                'removed_heads_v': set(),
                'removed_heads_o': set(),
                'removed_heads_out': set(),
                'removed_channels_q': set(),
                'removed_channels_k': set(),
                'removed_channels_v': set(),
                'removed_channels_o': set(),
                'heads_q': 0,
                'heads_k': 0,
                'heads_v': 0,
                'heads_out': 0,
                'head_dim_q': 0,
                'head_dim_k': 0,
                'embed_dim': 0,
            }
        
        # For simplicity, assume all matrix types have same removed heads/channels
        # In practice, you might want to track them separately
        self.layer_changes[layer_idx]['removed_heads_q'].update(removed_heads)
        self.layer_changes[layer_idx]['removed_heads_k'].update(removed_heads)
        self.layer_changes[layer_idx]['removed_heads_v'].update(removed_heads)
        self.layer_changes[layer_idx]['removed_heads_out'].update(removed_heads)
        
        self.layer_changes[layer_idx]['removed_channels_q'].update(removed_channels_qk)
        self.layer_changes[layer_idx]['removed_channels_k'].update(removed_channels_qk)
        self.layer_changes[layer_idx]['removed_channels_v'].update(removed_channels_vo)
        self.layer_changes[layer_idx]['removed_channels_o'].update(removed_channels_vo)
    
    def mark_layer_removed(self, layer_idx: int):
        """Mark an entire layer as removed."""
        self.removed_layers.append(layer_idx)
    
    def get_summary(self):
        """Get a summary of all changes."""
        summary = {
            'removed_layers': self.removed_layers,
            'modified_layers': len(self.layer_changes),
            'total_removed_heads': 0,
            'total_removed_channels': 0
        }
        
        for layer_idx, changes in self.layer_changes.items():
            summary['total_removed_heads'] += len(changes['removed_heads_q'])
            summary['total_removed_channels'] += len(changes['removed_channels_q'])
        
        return summary


def extract_structured_weights(original_attn, kept_heads_q: List[int], kept_heads_k: List[int],
                              kept_heads_v: List[int], kept_heads_o: List[int], kept_channels_q: List[int], 
                              kept_channels_k: List[int], kept_channels_v: List[int], kept_channels_o: List[int]):
    """Extract weights for kept heads and channels."""
    
    # Get original weights
    weight, o_weight, bias, out_bias, embed_dim, num_heads, head_dims = get_layer_weights(original_attn)
    
    num_heads_q, num_heads_k, num_heads_v, num_heads_o = num_heads
    head_dim_q, head_dim_k, head_dim_v, head_dim_o = head_dims
    
    q_out_dim = num_heads_q * head_dim_q
    k_out_dim = num_heads_k * head_dim_k
    v_out_dim = num_heads_v * head_dim_v
    
    # Split original weights
    q_weight = weight[:q_out_dim, :]
    k_weight = weight[q_out_dim:q_out_dim+k_out_dim, :]
    v_weight = weight[q_out_dim+k_out_dim:, :]
    
    if bias is not None:
        q_bias = bias[:q_out_dim]
        k_bias = bias[q_out_dim:q_out_dim+k_out_dim] 
        v_bias = bias[q_out_dim+k_out_dim:]
    
    # Reshape to head structure
    q_weight = q_weight.reshape(num_heads_q, head_dim_q, -1)
    k_weight = k_weight.reshape(num_heads_k, head_dim_k, -1)
    v_weight = v_weight.reshape(num_heads_v, head_dim_v, -1)
    o_weight = o_weight.reshape(embed_dim, num_heads_o, head_dim_o)
    
    if bias is not None:
        q_bias = q_bias.reshape(num_heads_q, head_dim_q)
        k_bias = k_bias.reshape(num_heads_k, head_dim_k)
        v_bias = v_bias.reshape(num_heads_v, head_dim_v)
    
    # Extract kept portions
    # For Q matrix: keep specified heads and channels
    new_q_weight = q_weight[kept_heads_q][:, kept_channels_q, :]
    # For K matrix: keep specified heads and channels  
    new_k_weight = k_weight[kept_heads_k][:, kept_channels_k, :]
    # For V matrix: keep specified heads and channels
    new_v_weight = v_weight[kept_heads_v][:, kept_channels_v, :]
    # For O matrix: keep specified heads and channels
    new_o_weight = o_weight[:, kept_heads_o, :]  # First select heads
    new_o_weight = new_o_weight[:, :, kept_channels_o]
    
    if bias is not None:
        new_q_bias = q_bias[kept_heads_q][:, kept_channels_q]
        new_k_bias = k_bias[kept_heads_k][:, kept_channels_k] 
        new_v_bias = v_bias[kept_heads_v][:, kept_channels_v]
    
    # Reshape back to linear layer format
    new_q_weight = new_q_weight.reshape(-1, new_q_weight.shape[-1]) #num_head_q * head_dim_q, embed_size
    new_k_weight = new_k_weight.reshape(-1, new_k_weight.shape[-1]) #num_head_k * head_dim_k, embed_size
    new_v_weight = new_v_weight.reshape(-1, new_v_weight.shape[-1]) #num_head_v * head_dim_v, embed_size
    new_o_weight = new_o_weight.reshape(new_o_weight.shape[0], -1) #embed_size, num_head_o * head_dim_o
    
    new_weight = torch.cat([new_q_weight, new_k_weight, new_v_weight], dim=0)
    
    if bias is not None:
        new_q_bias = new_q_bias.reshape(-1)
        new_k_bias = new_k_bias.reshape(-1)
        new_v_bias = new_v_bias.reshape(-1)
        new_bias = torch.cat([new_q_bias, new_k_bias, new_v_bias], dim=0)
    else:
        new_bias = None
    
    return new_weight, new_bias, new_o_weight

def _safe_norm(x, p, dim):
    n = torch.norm(x, p=p, dim=dim)
    # Avoid div-by-zero in normalization
    maxv = torch.max(n)
    if maxv > 0 and NORM:
        n = n / maxv
    return n

def _gather_qk_vo_scores(sa, order,
                         num_heads_q, head_dim_q,
                         num_heads_k, head_dim_k,
                         num_heads_v, head_dim_v,
                         num_heads_o, head_dim_o,
                         q_out_dim, k_out_dim, embed_dim):
    # Weights
    weight, o_weight, _, _, _, _, _ = get_layer_weights(sa)
    q_weight = weight[:q_out_dim, :].view(num_heads_q, head_dim_q, -1)
    k_weight = weight[q_out_dim:q_out_dim+k_out_dim, :].view(num_heads_k, head_dim_k, -1)
    v_weight = weight[q_out_dim+k_out_dim:,:].view(num_heads_v, head_dim_v, -1)
    o_weight = o_weight.view(embed_dim, num_heads_o, head_dim_o)

    # Per-channel norms
    q_norms = _safe_norm(q_weight, p=order, dim=2)  # (Hq, Dq)
    k_norms = _safe_norm(k_weight, p=order, dim=2)  # (Hk, Dk)
    v_norms = _safe_norm(v_weight, p=order, dim=2)  # (Hv, Dv)
    o_norms = _safe_norm(o_weight, p=order, dim=0)  # (Ho, Do)
    
    q_head_scores = _safe_norm(q_weight, p=order, dim=(1, 2)) 
    k_head_scores = _safe_norm(k_weight, p=order, dim=(1, 2))
    v_head_scores = _safe_norm(v_weight, p=order, dim=(1, 2))
    o_head_scores = _safe_norm(o_weight, p=order, dim=(2, 0))

    # Align shapes (assumes Hq==Hk and Dq==Dk; Hv==Ho and Dv==Do)
    qk_norms = (q_norms + k_norms) / 2.0     # (Hq, Dq)
    vo_norms = (v_norms + o_norms) / 2.0     # (Hv, Dv)
    
    head_scores = (q_head_scores + k_head_scores + v_head_scores + o_head_scores) / 4.0

    return qk_norms, vo_norms, head_scores

def _layer_marginals_per_head(sorted_vals_per_head):
    """
    sorted_vals_per_head: list of tensors [head]->(D,) sorted ascending
    Returns:
      deltas: (max_j,) where deltas[j-1] = sum over heads of j-th smallest value
      max_j: maximum pruneable per-head count (D-1)
    """
    H = len(sorted_vals_per_head)
    D = sorted_vals_per_head[0].numel()
    max_j = max(D - 1, 0)  # keep at least 1 channel per head
    if max_j == 0:
        return torch.empty(0), 0

    # Stack as (H, D); take rows’ j-th entries and sum across heads
    S = torch.stack(sorted_vals_per_head, dim=0)  # (H, D)
    deltas = torch.sum(S[:, :max_j], dim=0)       # (max_j,)
    return deltas, max_j

def _build_equal_per_head_plan(scores_by_layer, prune_amount):
    """
    scores_by_layer: list over layers of tensors (H, D) with per-channel scores (lower = less important)
    Returns:
      per_layer_m: list of integers; how many channels to prune per head in each layer
      per_layer_indices: list over layers of list over heads: pruned channel indices per head
    """
    L = len(scores_by_layer)
    Hs = [scores_by_layer[l].shape[0] for l in range(L)]
    Ds = [scores_by_layer[l].shape[1] for l in range(L)]

    total_channels = sum(Hs[l] * Ds[l] for l in range(L))
    target_budget = int(total_channels * prune_amount)

    # Precompute, for each layer:
    #   - per-head sorted values and indices
    #   - marginal “bundle” costs deltas[j] = cost to add 1 more pruned channel per head
    per_head_sorted_vals = []
    per_head_sorted_idx  = []
    layer_deltas = []      # list of tensors (max_j,)
    layer_maxj   = []

    for l in range(L):
        H, D = Hs[l], Ds[l]
        vals_l = []
        idx_l  = []
        for h in range(H):
            v = scores_by_layer[l][h]  # (D,)
            sv, si = torch.sort(v, dim=0, descending=False)  # ascending (low first)
            vals_l.append(sv)
            idx_l.append(si)
        per_head_sorted_vals.append(vals_l)
        per_head_sorted_idx.append(idx_l)

        deltas_l, max_j = _layer_marginals_per_head(vals_l)
        layer_deltas.append(deltas_l)     # (max_j,)
        layer_maxj.append(max_j)

    # Greedy allocate bundles with smallest marginal cost
    # Each pick on layer l adds Hs[l] to budget use and increments m_l by 1 (up to layer_maxj[l]).
    per_layer_m = [0 for _ in range(L)]
    used = 0

    # Build a pointer for each layer’s next marginal j (1..max_j)
    next_j = [1 if layer_maxj[l] > 0 else 0 for l in range(L)]

    # We’ll repeatedly choose the layer with the smallest available delta for its next_j
    while used < target_budget:
        best_l = -1
        best_cost = None
        for l in range(L):
            if 1 <= next_j[l] <= layer_maxj[l]:
                # deltas indexed by (j-1)
                cost = layer_deltas[l][next_j[l]-1].item()
                # Can we fit this bundle in the remaining budget?
                bundle = Hs[l]
                if used + bundle <= target_budget:
                    if (best_cost is None) or (cost < best_cost):
                        best_cost = cost
                        best_l = l
        if best_l == -1:
            # No bundle fits the remaining budget; stop (budget becomes feasible ≤ target)
            break
        per_layer_m[best_l] += 1
        used += Hs[best_l]
        next_j[best_l] += 1

    # Build indices to prune: per layer, per head take the first m entries (low scores)
    per_layer_indices = []
    for l in range(L):
        m = per_layer_m[l]
        if m == 0:
            per_layer_indices.append([[] for _ in range(Hs[l])])
            continue
        idxs_l = []
        for h in range(Hs[l]):
            idxs_l.append(per_head_sorted_idx[l][h][:m].tolist())
        per_layer_indices.append(idxs_l)

    return per_layer_m, per_layer_indices, used, target_budget

def apply_structured_pruning(model, pruning_plan: Dict, remove_empty_layers: bool = True):
    """Apply structured pruning by actually removing parameters."""
    
    changes = PruningChanges()
    layers = get_model_layers(model)
    
    for layer_idx, layer in enumerate(layers):
            
        sa = layer
        plan = pruning_plan[layer_idx]
        weight, o_weight, bias, out_bias, embed_dim, num_heads, head_dims = get_layer_weights(sa)
            
        num_heads_q, num_heads_k, num_heads_v, num_heads_o = num_heads
        head_dim_q, head_dim_k, head_dim_v, head_dim_o = head_dims
        
        changes.original_config[layer_idx] = {
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
        
        all_heads_q = list(range(num_heads_q))
        all_heads_k = list(range(num_heads_k))
        all_heads_v = list(range(num_heads_v))
        all_heads_o = list(range(num_heads_o))
        all_channels_q = list(range(head_dim_q))
        all_channels_k = list(range(head_dim_k))
        all_channels_v = list(range(head_dim_v))
        all_channels_o = list(range(head_dim_o))
        
        # Check if this is MULTI_HEAD_PER_HEAD strategy (nested dict structure)
        is_per_head = not isinstance(plan.get(list(plan.keys())[0] if plan else 0, {}), list)
        
        if is_per_head: 
            
            q_out_dim = num_heads_q * head_dim_q
            k_out_dim = num_heads_k * head_dim_k
            v_out_dim = num_heads_v * head_dim_v
            
            # Split original weights
            q_weight = weight[:q_out_dim, :].reshape(num_heads_q, head_dim_q, -1)
            k_weight = weight[q_out_dim:q_out_dim+k_out_dim, :].reshape(num_heads_k, head_dim_k, -1)
            v_weight = weight[q_out_dim+k_out_dim:, :].reshape(num_heads_v, head_dim_v, -1)
            o_weight = o_weight.reshape(embed_dim, num_heads_o, head_dim_o)
            
            if bias is not None:
                q_bias = bias[:q_out_dim].reshape(num_heads_q, head_dim_q)
                k_bias = bias[q_out_dim:q_out_dim+k_out_dim].reshape(num_heads_k, head_dim_k)
                v_bias = bias[q_out_dim+k_out_dim:].reshape(num_heads_v, head_dim_v)
            
            # Process each head individually
            new_q_weights = []
            new_k_weights = []
            new_v_weights = []
            new_o_weights = []
            
            if bias is not None:
                new_q_biases = []
                new_k_biases = []
                new_v_biases = []
            
            # Track changes for each head
            removed_heads = []
            removed_channels_qk = set()
            removed_channels_vo = set()
            
            new_head_dim_q = None
            new_head_dim_v = None
            
            for head_idx in all_heads_q:
                head_plan = plan.get(head_idx, {})
                
                # Get channels to remove for this head
                remove_channels_qk = head_plan.get('remove_channels_qk', [])
                remove_channels_vo = head_plan.get('remove_channels_vo', [])
                
                # Debug: ensure these are flat lists of integers
                if remove_channels_qk and not isinstance(remove_channels_qk[0], int):
                    # Flatten if nested
                    remove_channels_qk = [item for sublist in remove_channels_qk for item in sublist]
                if remove_channels_vo and not isinstance(remove_channels_vo[0], int):
                    # Flatten if nested
                    remove_channels_vo = [item for sublist in remove_channels_vo for item in sublist]
                
                # Determine kept channels for this head
                kept_channels_q = [c for c in all_channels_q if c not in remove_channels_qk]
                kept_channels_k = [c for c in all_channels_k if c not in remove_channels_qk]
                kept_heads_q = all_heads_q
                kept_heads_k = all_heads_k
                kept_heads_v = all_heads_v
                kept_heads_o = all_heads_o
                
                # For V/O, only process if head exists in V dimension
                if head_idx < num_heads_v:
                    kept_channels_v = [c for c in all_channels_v if c not in remove_channels_vo]
                    kept_channels_o = [c for c in all_channels_o if c not in remove_channels_vo]
                else:
                    kept_channels_v = all_channels_v
                    kept_channels_o = all_channels_o
                
                # Ensure we don't remove all channels
                if not kept_channels_q:
                    kept_channels_q = [0]
                if not kept_channels_k:
                    kept_channels_k = [0]
                if head_idx < num_heads_v and not kept_channels_v:
                    kept_channels_v = [0]
                if head_idx < num_heads_o and not kept_channels_o:
                    kept_channels_o = [0]
                
                # Extract weights for this head
                head_q_weight = q_weight[head_idx][kept_channels_q, :]
                head_k_weight = k_weight[head_idx][kept_channels_k, :]
                
                if head_idx < num_heads_v:
                    head_v_weight = v_weight[head_idx][kept_channels_v, :]
                else:
                    head_v_weight = v_weight[min(head_idx, num_heads_v-1)][kept_channels_v, :]
                
                if head_idx < num_heads_o:
                    head_o_weight = o_weight[:, head_idx, :][:, kept_channels_o]
                else:
                    head_o_weight = o_weight[:, min(head_idx, num_heads_o-1), :][:, kept_channels_o]
                
                new_q_weights.append(head_q_weight)
                new_k_weights.append(head_k_weight)
                new_v_weights.append(head_v_weight)
                new_o_weights.append(head_o_weight)
                
                if bias is not None:
                    head_q_bias = q_bias[head_idx][kept_channels_q]
                    head_k_bias = k_bias[head_idx][kept_channels_k]
                    
                    if head_idx < num_heads_v:
                        head_v_bias = v_bias[head_idx][kept_channels_v]
                    else:
                        head_v_bias = v_bias[min(head_idx, num_heads_v-1)][kept_channels_v]
                    
                    new_q_biases.append(head_q_bias)
                    new_k_biases.append(head_k_bias)
                    new_v_biases.append(head_v_bias)
                
                # Track dimensions (should be same for all heads after pruning)
                if new_head_dim_q is None:
                    new_head_dim_q = len(kept_channels_q)
                    new_head_dim_v = len(kept_channels_v) if head_idx < num_heads_v else head_dim_v
                
                # Record changes - add individual channel indices
                removed_channels_qk.update(remove_channels_qk)
                if head_idx < num_heads_v:
                    removed_channels_vo.update(remove_channels_vo)
            
            # Reconstruct the weight matrices
            new_q_weight = torch.stack(new_q_weights, dim=0)
            new_k_weight = torch.stack(new_k_weights, dim=0)
            new_v_weight = torch.stack(new_v_weights, dim=0)
            new_o_weight = torch.stack(new_o_weights, dim=1)
            
            new_q_weight = new_q_weight.reshape(-1, embed_dim)
            new_k_weight = new_k_weight.reshape(-1, embed_dim)
            new_v_weight = new_v_weight.reshape(-1, embed_dim)
            new_o_weight = new_o_weight.reshape(embed_dim, -1)
            
            # Record changes
            changes.add_layer_change(layer_idx, [], list(removed_channels_qk), list(removed_channels_vo))
            new_weight = torch.cat([new_q_weight, new_k_weight, new_v_weight], dim=0)
        
            if bias is not None:
                new_q_bias = torch.stack(new_q_biases, dim=0).reshape(-1)
                new_k_bias = torch.stack(new_k_biases, dim=0).reshape(-1)
                new_v_bias = torch.stack(new_v_biases, dim=0).reshape(-1)
                new_bias = torch.cat([new_q_bias, new_k_bias, new_v_bias], dim=0)
            else:
                new_bias = None
            
        else:
            # Handle existing strategies (MULTI_HEAD_SAME_CHANNEL and MULTI_HEAD_ENTIRE_HEAD)
            # Determine what to keep
            
            
            kept_heads_q = [h for h in all_heads_q if h not in plan.get('remove_heads', [])]
            kept_heads_k = [h for h in all_heads_k if h not in plan.get('remove_heads', [])]
            kept_heads_v = [h for h in all_heads_v if h not in plan.get('remove_heads', [])]
            kept_heads_o = [h for h in all_heads_o if h not in plan.get('remove_heads', [])] 
            
            kept_channels_q = [c for c in all_channels_q if c not in plan.get('remove_channels_qk', [])]
            kept_channels_k = [c for c in all_channels_k if c not in plan.get('remove_channels_qk', [])]
            kept_channels_v = [c for c in all_channels_v if c not in plan.get('remove_channels_vo', [])]
            kept_channels_o = [c for c in all_channels_o if c not in plan.get('remove_channels_vo', [])]
            
            # Ensure we don't remove everything
            if not kept_heads_q:
                kept_heads_q = [0]  
            if not kept_channels_q:
                kept_channels_q = [0]  
            if not kept_heads_k:
                kept_heads_k = [0]
            if not kept_heads_v: 
                kept_heads_v = [0]
            if not kept_channels_k:
                kept_channels_k = [0]
            if not kept_channels_v:
                kept_channels_v = [0]
            if not kept_heads_o:
                kept_heads_o = [0]
            if not kept_channels_o:
                kept_channels_o = [0]

            # Record changes
            changes.add_layer_change(layer_idx, plan.get('remove_heads', []), 
                                   plan.get('remove_channels_qk', []), plan.get('remove_channels_vo', []))
            
            # Extract and assign weights
            new_weight, new_bias, new_o_weight = extract_structured_weights(
                sa, kept_heads_q, kept_heads_k, kept_heads_v, kept_heads_o,
                kept_channels_q, kept_channels_k, kept_channels_v, kept_channels_o
            )
    
        # Update dimensions
        if hasattr(sa, 'num_heads_q'):
            sa.num_heads_q = len(kept_heads_q)
            sa.num_heads_k = len(kept_heads_k)
            sa.num_heads_v = len(kept_heads_v)
            sa.num_heads_out = len(kept_heads_o) 
            
            sa.head_dim_q = len(kept_channels_q)
            sa.head_dim_k = len(kept_channels_k)
            sa.head_dim_v = len(kept_channels_v)
            sa.head_dim_out = len(kept_channels_o) 
            
            sa.q_out_dim = sa.num_heads_q * sa.head_dim_q
            sa.k_out_dim = sa.num_heads_k * sa.head_dim_k
            sa.v_out_dim = sa.num_heads_v * sa.head_dim_v
        else:
            #transformer's bert version of attention
            sa.self.num_attention_heads_q = len(kept_heads_q)
            sa.self.num_attention_heads_k = len(kept_heads_k)
            sa.self.num_attention_heads_v = len(kept_heads_o)
            
            sa.self.head_dim_q = len(kept_channels_q)
            sa.self.head_dim_k = len(kept_channels_k)
            sa.self.head_dim_v = len(kept_channels_v)
            sa.self.all_head_size_q = sa.self.num_attention_heads_q * sa.self.head_dim_q
            sa.self.all_head_size_k = sa.self.num_attention_heads_k * sa.self.head_dim_k
            sa.self.all_head_size_v = sa.self.num_attention_heads_v * sa.self.head_dim_v
        
        #Apply the new weights and reinstantiate the layers if needed
        if isinstance(sa, (nn.MultiheadAttention)):
        # Update the module
            sa.in_proj_weight.data = new_weight
            sa.out_proj.weight.data = new_o_weight
            if new_bias is not None:
                sa.in_proj_bias.data = new_bias
        elif isinstance(sa, (tf_locoformer.MultiHeadSelfAttention)):
            with torch.no_grad():
                device = next(sa.parameters()).device
                sa.qkv = nn.Linear(in_features = sa.embed_dim, out_features=sa.q_out_dim + sa.k_out_dim + sa.v_out_dim, bias=False, device=device)
                sa.qkv.weight.copy_(new_weight.contiguous())
                sa.aggregate_heads[0] = nn.Linear(in_features = sa.v_out_dim, out_features = sa.embed_dim, bias=False, device=device)
                sa.aggregate_heads[0].weight.copy_(new_o_weight.contiguous())
                if new_bias is not None:
                    sa.qkv.bias.copy_(new_bias.contiguous())
                #set the gradient computation to false
                sa.qkv.requires_grad_(False)
                sa.aggregate_heads[0].requires_grad_(False)
            if sa.rope is not None:
                # get old freqs (parameter or buffer)
                old_freqs = sa.rope.freqs.data if isinstance(sa.rope.freqs, torch.nn.Parameter) else sa.rope.freqs
                
                # reinitialize with new head_dim_q
                new_dim = sa.head_dim_q
                if new_dim > 2:
                    new_rope = tf_locoformer.RotaryEmbeddingOdd(new_dim, custom_freqs=None)  # fresh init
                    min_dim = min(new_rope.freqs.shape[0], old_freqs.shape[0])
                    # try to copy as much of the old freqs as possible
                    new_rope.freqs.data[:min_dim] = old_freqs[:min_dim].to(new_rope.freqs.device)
                    sa.rope = new_rope.to(sa.rope.device)
                else:
                    sa.rope = None
        elif isinstance(sa, (Attention)):
            with torch.no_grad():
                device = next(sa.parameters()).device
                sa.qkv = nn.Linear(in_features = sa.embed_dim, out_features=sa.q_out_dim + sa.k_out_dim + sa.v_out_dim, bias=False if new_bias is None else True, device=device)
                sa.qkv.weight.copy_(new_weight.contiguous())
                old_out_bias = sa.proj.bias.detach().clone()
                sa.proj = nn.Linear(in_features = sa.v_out_dim, out_features = sa.embed_dim, device=device)
                sa.proj.weight.copy_(new_o_weight.contiguous())
                sa.proj.bias.copy_(old_out_bias.contiguous())
                if new_bias is not None:
                    sa.qkv.bias.copy_(new_bias)
                #set the gradient computation to false
                sa.qkv.requires_grad_(False)
                sa.proj.requires_grad_(False)
        elif isinstance(sa, (BertAttention)):
            with torch.no_grad():
                device = next(sa.parameters()).device
                #query
                q_w = new_weight[:sa.self.all_head_size_q]
                q_b = new_bias[:sa.self.all_head_size_q]
                sa.self.query = nn.Linear(embed_dim, sa.self.all_head_size_q, device=device)
                sa.self.query.weight.copy_(q_w.contiguous())
                sa.self.query.bias.copy_(q_b.contiguous())
                #key
                k_w = new_weight[sa.self.all_head_size_q:sa.self.all_head_size_q+sa.self.all_head_size_k]
                k_b = new_bias[sa.self.all_head_size_q:sa.self.all_head_size_q+sa.self.all_head_size_k]
                sa.self.key = nn.Linear(embed_dim, sa.self.all_head_size_k, device=device)
                sa.self.key.weight.copy_(k_w.contiguous())
                sa.self.key.bias.copy_(k_b.contiguous())
                #value
                v_w = new_weight[sa.self.all_head_size_q+sa.self.all_head_size_k:]
                v_b = new_bias[sa.self.all_head_size_q+sa.self.all_head_size_k:]
                sa.self.value = nn.Linear(embed_dim, sa.self.all_head_size_v, device=device)
                sa.self.value.weight.copy_(v_w.contiguous())
                sa.self.value.bias.copy_(v_b.contiguous())
                #output proj
                o_b = sa.output.dense.bias.detach().clone()
                sa.output.dense = nn.Linear(sa.self.all_head_size_v, embed_dim, device=device)
                sa.output.dense.weight.copy_(new_o_weight.contiguous())
                sa.output.dense.bias.copy_(o_b.contiguous())
                #this is not used in Bert, but in other implementation with relative positional embedding
                #must be pruned in case it exists, but it's complicated to do
                if hasattr(sa.self, 'distance_embedding'):
                    sa.self.distance_embedding = torch.nn.Embedding(2 * sa.self.max_position_embeddings - 1, sa.self.attention_head_size_q)  
        elif isinstance(sa, LlamaAttention):
            with torch.no_grad():
                device = next(sa.parameters()).device
                #query
                q_w = new_weight[:sa.q_out_dim]
                q_b = new_bias[:sa.q_out_dim]
                sa.q_proj = nn.Linear(embed_dim, sa.q_out_dim, device=device, bias=False)
                sa.q_proj.weight.copy_(q_w.contiguous())
                sa.q_proj.bias.copy_(q_b.contiguous())
                
                #key
                k_w = new_weight[sa.q_out_dim:sa.q_out_dim+sa.k_out_dim]
                k_b = new_bias[sa.q_out_dim:sa.q_out_dim+sa.k_out_dim]
                sa.k_proj = nn.Linear(embed_dim, sa.k_out_dim, device=device)
                sa.k_proj.weight.copy_(k_w.contiguous())
                sa.k_proj.bias.copy_(k_b.contiguous())
                
                #value
                v_w = new_weight[sa.q_out_dim+sa.k_out_dim:]
                v_b = new_bias[sa.q_out_dim+sa.k_out_dim:]
                sa.v_proj = nn.Linear(embed_dim, sa.v_out_dim, device=device)
                sa.v_proj.weight.copy_(v_w.contiguous())
                sa.v_proj.bias.copy_(v_b.contiguous())
                
                #output proj
                o_b = sa.o_proj.bias.detach().clone()
                sa.o_proj = nn.Linear(sa.v_out_dim, embed_dim, device=device)
                sa.o_proj.weight.copy_(new_o_weight.contiguous())
                sa.o_proj.bias.copy_(o_b.contiguous())
                
        # Store final configuration
        changes.final_config[layer_idx] = {
            'embed_dim': embed_dim,
            'num_heads_q': kept_heads_q,
            'num_heads_k': kept_heads_k,
            'num_heads_v': kept_heads_v,
            'num_heads_o': kept_heads_o,
            'head_dim_q': kept_channels_q,
            'head_dim_k': kept_channels_k,
            'head_dim_v': kept_channels_v,
            'head_dim_o': kept_channels_o
        }
    
    return model

def compute_fisher_information(model, data_loader, criterion=None, config: FisherConfig = None, device='cuda'):
    """
    Compute Fisher Information incrementally - one sample at a time.
    Only stores running statistics, not individual gradients.
    Memory: O(num_params) instead of O(num_samples * num_params)
    """
    if config is None:
        config = FisherConfig()
        
    model.eval()
    
    # Initialize running statistics
    # fisher_sum: accumulated sum of squared gradients
    # sample_count: number of samples processed
    fisher_sum = {}
    sample_count = 0
    
    # Disable gradients for non-attention layers
    for p in model.parameters():
        p.requires_grad = False
    
    layers = get_model_layers(model)
    for layer in layers:
        for p in layer.parameters():
            p.requires_grad = True
    
    print(f"Computing Fisher Information incrementally using {config.num_samples} samples...")
    max_steps = config.num_samples // data_loader.batch_size
    
    try:
        for i, batch in enumerate(tqdm(data_loader, desc="Computing Fisher", total=max_steps)):
            if i >= max_steps:
                break
            
            try:
                # Zero gradients
                model.zero_grad()
                
                # Forward pass
                if criterion is not None:
                    inputs, targets = batch
                    inputs = inputs.to(device)
                    targets = targets.to(device)
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                else:
                    batch = {k: v.to(device) for k, v in batch.items()}
                    outputs = model(**batch)
                    loss = outputs.loss
                
                # Backward pass
                loss.backward()
                
                # Compute squared gradients and accumulate directly
                for layer_idx, layer in enumerate(layers):
                    weight, o_weight = get_layer_weights_fisher(layer)
                    
                    # Process QKV weights
                    if isinstance(weight, list):
                        q, k, v = weight
                        if q.grad is not None:
                            # Compute squared gradients on GPU, accumulate on CPU
                            qkv_grad_sq = torch.cat([
                                q.grad.detach() ** 2,
                                k.grad.detach() ** 2,
                                v.grad.detach() ** 2
                            ], dim=0).cpu()
                            
                            key = (layer_idx, 'qkv_weight')
                            if key not in fisher_sum:
                                fisher_sum[key] = qkv_grad_sq
                            else:
                                fisher_sum[key].add_(qkv_grad_sq)  # In-place addition
                            
                            del qkv_grad_sq
                            
                    elif weight is not None and weight.grad is not None:
                        grad_sq = (weight.grad.detach() ** 2).cpu()
                        
                        key = (layer_idx, 'qkv_weight')
                        if key not in fisher_sum:
                            fisher_sum[key] = grad_sq
                        else:
                            fisher_sum[key].add_(grad_sq)
                        
                        del grad_sq
                    
                    # Process output weights
                    if o_weight is not None and o_weight.grad is not None:
                        grad_sq = (o_weight.grad.detach() ** 2).cpu()
                        
                        key = (layer_idx, 'out_weight')
                        if key not in fisher_sum:
                            fisher_sum[key] = grad_sq
                        else:
                            fisher_sum[key].add_(grad_sq)
                        
                        del grad_sq
                
                sample_count += data_loader.batch_size
                
                # Aggressive cleanup
                del loss, outputs, batch
                model.zero_grad()
                
                # Periodic GPU cleanup
                if i % 10 == 0:
                    torch.cuda.empty_cache()
                
                # Periodic CPU cleanup
                if i % 50 == 0:
                    import gc
                    gc.collect()
                    
            except RuntimeError as e:
                if "out of memory" in str(e):
                    print(f"\nOOM at batch {i}, using partial Fisher information")
                    torch.cuda.empty_cache()
                    break
                else:
                    raise e
        
        # Convert accumulated sums to means and add damping
        print(f"\nProcessed {sample_count} samples")
        print("Finalizing Fisher Information...")
        
        fisher_info = defaultdict(dict)
        for (layer_idx, param_name), fisher_val in fisher_sum.items():
            # Compute mean: sum / count
            fisher_mean = fisher_val / sample_count
            # Add damping for numerical stability
            fisher_mean.add_(config.damping)
            fisher_info[layer_idx][param_name] = fisher_mean
        
        # Clear the accumulation dict
        fisher_sum.clear()
        
        print("Fisher Information computation completed")
        
    finally:
        # Cleanup
        for p in model.parameters():
            p.requires_grad = False
        
        model.zero_grad()
        torch.cuda.empty_cache()
        import gc
        gc.collect()
    
    return fisher_info

def compute_channel_head_importance(sa, layer_idx, importance_strategy: ImportanceStrategy = ImportanceStrategy.MAGNITUDE,
                                   fisher_info: Optional[Dict] = None, order: int = 2):
    """
    Compute channel and head importance scores using either magnitude or Fisher Information.
    
    Args:
        sa: Attention layer
        layer_idx: Index of the layer
        importance_strategy: Strategy for computing importance
        fisher_info: Precomputed Fisher Information (required for FISHER_INFORMATION strategy)
        order: Norm order for magnitude-based computation
        
    Returns:
        Tuple of (qk_scores, vo_scores, head_scores)
        - qk_scores: (num_heads_q, head_dim_q) importance scores
        - vo_scores: (num_heads_v, head_dim_v) importance scores  
        - head_scores: (num_heads,) head-level importance scores
    """
    
    _, _, _, _, embed_dim, num_heads, head_dims= get_layer_weights(sa)
    
    num_heads_q, num_heads_k, num_heads_v, num_heads_o = num_heads
    head_dim_q, head_dim_k, head_dim_v, head_dim_o = head_dims
    
    q_out_dim = num_heads_q * head_dim_q
    k_out_dim = num_heads_k * head_dim_k
    v_out_dim = num_heads_v * head_dim_v
    
    if importance_strategy == ImportanceStrategy.FISHER_INFORMATION:
        if fisher_info is None or layer_idx not in fisher_info:
            print(f"Warning: Fisher info not available for layer {layer_idx}, falling back to magnitude")
            importance_strategy = ImportanceStrategy.MAGNITUDE
        else:
            return _compute_fisher_importance_scores(sa, fisher_info[layer_idx], 
                                                   num_heads_q, num_heads_k, num_heads_v, num_heads_o,
                                                   head_dim_q, head_dim_k, head_dim_v, head_dim_o,
                                                   q_out_dim, k_out_dim, v_out_dim, embed_dim)
    
    # Magnitude-based importance (original method)
    if importance_strategy == ImportanceStrategy.MAGNITUDE:
        return _gather_qk_vo_scores(sa, order, num_heads_q, head_dim_q, num_heads_k, head_dim_k,
                                  num_heads_v, head_dim_v, num_heads_o, head_dim_o,
                                  q_out_dim, k_out_dim, embed_dim)

def _compute_fisher_importance_scores(sa, layer_fisher_info, num_heads_q, num_heads_k, num_heads_v, num_heads_o,
                                    head_dim_q, head_dim_k, head_dim_v, head_dim_o,
                                    q_out_dim, k_out_dim, v_out_dim, embed_dim):
    """
    Compute importance scores using Fisher Information.
    """
    
    # Extract Fisher information for QKV weights
    if 'qkv_weight' not in layer_fisher_info:
        raise ValueError("Fisher information missing 'qkv_weight' for layer")
    
    qkv_fisher = layer_fisher_info['qkv_weight']  # Shape matches weight matrix
    
    # Split Fisher information for Q, K, V
    q_fisher = qkv_fisher[:q_out_dim, :].reshape(num_heads_q, head_dim_q, -1)
    k_fisher = qkv_fisher[q_out_dim:q_out_dim+k_out_dim, :].reshape(num_heads_k, head_dim_k, -1)
    v_fisher = qkv_fisher[q_out_dim+k_out_dim:, :].reshape(num_heads_v, head_dim_v, -1)
    
    # Get output Fisher information
    if 'out_weight' in layer_fisher_info:
        o_fisher = layer_fisher_info['out_weight'].reshape(embed_dim, num_heads_o, head_dim_o)
    else:
        o_fisher = torch.zeros(embed_dim, num_heads_o, head_dim_o, device=qkv_fisher.device)
    
    # Compute per-channel Fisher scores by summing across input/output dimensions
    q_channel_fisher = torch.sum(q_fisher, dim=2)  # (num_heads_q, head_dim_q)
    k_channel_fisher = torch.sum(k_fisher, dim=2)  # (num_heads_k, head_dim_k)
    v_channel_fisher = torch.sum(v_fisher, dim=2)  # (num_heads_v, head_dim_v)
    o_channel_fisher = torch.sum(o_fisher, dim=0)  # (num_heads_o, head_dim_o)
    
    # Compute head-level scores (sum across channels within each head)
    q_head_scores = torch.sum(q_channel_fisher, dim=1)
    k_head_scores = torch.sum(k_channel_fisher, dim=1) 
    v_head_scores = torch.sum(v_channel_fisher, dim=1)
    o_head_scores = torch.sum(o_channel_fisher, dim=1)
    
    #Normalization should perform worse
    # Normalize Fisher scores (avoid division by zero)
    q_channel_fisher = q_channel_fisher / ((torch.max(q_channel_fisher) + 1e-8) if NORM else 1.)
    k_channel_fisher = k_channel_fisher / ((torch.max(k_channel_fisher) + 1e-8) if NORM else 1.)
    v_channel_fisher = v_channel_fisher / ((torch.max(v_channel_fisher) + 1e-8) if NORM else 1.)
    o_channel_fisher = o_channel_fisher / ((torch.max(o_channel_fisher) + 1e-8) if NORM else 1.)
    
    # Normalize Fisher scores (avoid division by zero)
    q_head_scores = q_head_scores / ((torch.max(q_head_scores) + 1e-8) if NORM else 1.)
    k_head_scores = k_head_scores / ((torch.max(k_head_scores) + 1e-8) if NORM else 1.)
    v_head_scores = v_head_scores / ((torch.max(v_head_scores) + 1e-8) if NORM else 1.)
    o_head_scores = o_head_scores / ((torch.max(o_head_scores) + 1e-8) if NORM else 1.)
    
    # Combine Q/K and V/O scores
    qk_fisher_scores = (q_channel_fisher + k_channel_fisher) 
    vo_fisher_scores = (v_channel_fisher + o_channel_fisher)
    
    # Average head scores across Q, K, V, O
    min_heads = min(len(q_head_scores), len(k_head_scores), len(v_head_scores), len(o_head_scores))
    head_fisher_scores = (q_head_scores[:min_heads] + k_head_scores[:min_heads] + 
                         v_head_scores[:min_heads] + o_head_scores[:min_heads])
    
    return qk_fisher_scores, vo_fisher_scores, head_fisher_scores

def determine_pruning_plan(layers, prune_amount: float, pruning_strategy: PruningStrategy,
                                         threshold_strategy: ThresholdStrategy, 
                                         importance_strategy: ImportanceStrategy = ImportanceStrategy.MAGNITUDE,
                                         fisher_info: Optional[Dict] = None, order: int = 2):
    """
    Extended version of determine_pruning_plan that supports both magnitude and Fisher Information.
    
    Args:
        layers: List of attention layers
        prune_amount: Fraction to prune (0.0 to 1.0)
        pruning_strategy: How to prune (same channel, per head, entire head)
        threshold_strategy: Global or local pruning
        importance_strategy: Magnitude or Fisher Information
        fisher_info: Precomputed Fisher Information (required for Fisher strategy)
        order: Norm order for magnitude-based computation
        
    Returns:
        Pruning plan dictionary
    """
    
    if pruning_strategy == PruningStrategy.MULTI_HEAD_PER_HEAD and threshold_strategy == ThresholdStrategy.GLOBAL:
        return _determine_optimal_per_head_assignment(layers, prune_amount, importance_strategy, fisher_info, order)
    else:
        # Use existing logic for other strategies
        if threshold_strategy == ThresholdStrategy.GLOBAL:
            return _determine_global_pruning_plan(layers, prune_amount, pruning_strategy, importance_strategy, fisher_info, order)
        else:
            return _determine_local_pruning_plan(layers, prune_amount, pruning_strategy, importance_strategy, fisher_info, order)

def _determine_global_pruning_plan(layers, prune_amount: float, pruning_strategy: PruningStrategy, importance_strategy: ImportanceStrategy, fisher_info: Optional[Dict], order: int):
    """Global pruning plan with configurable importance strategy."""
    plan = {}
    if pruning_strategy == PruningStrategy.MULTI_HEAD_SAME_CHANNEL:
        # Collect all channel importance scores globally
        all_qk_scores = []
        all_vo_scores = []
        qk_channel_map = []  # (layer_idx, channel_idx)
        vo_channel_map = []
        
        for layer_idx, sa in enumerate(layers):
            qk_scores, vo_scores, _ = compute_channel_head_importance(sa, layer_idx, importance_strategy, fisher_info, order)
            
            # Average scores across heads for same-channel pruning
            if importance_strategy == ImportanceStrategy.FISHER_INFORMATION:
                qk_channel_scores = torch.sum(qk_scores, dim=0)  # (head_dim_q,)
                vo_channel_scores = torch.sum(vo_scores, dim=0)  # (head_dim_v,)                
            else:
                qk_channel_scores = torch.mean(qk_scores, dim=0)  # (head_dim_q,)
                vo_channel_scores = torch.mean(vo_scores, dim=0)  # (head_dim_v,)
            
            # Map channels to layers
            for ch_idx, score in enumerate(qk_channel_scores):
                all_qk_scores.append(score.item())
                qk_channel_map.append((layer_idx, ch_idx))
                
            for ch_idx, score in enumerate(vo_channel_scores):
                all_vo_scores.append(score.item())
                vo_channel_map.append((layer_idx, ch_idx))
        
        # Determine channels to prune globally (lowest importance scores)
        total_channels_qk = len(all_qk_scores)
        qk_channels_to_prune = math.ceil(total_channels_qk * prune_amount)
        all_qk_scores = torch.tensor(all_qk_scores)
        _, qk_prune_indices = torch.topk(all_qk_scores, qk_channels_to_prune, largest=False)
        total_channels_vo = len(all_vo_scores)
        vo_channels_to_prune = math.ceil(total_channels_vo * prune_amount)
        
        all_vo_scores = torch.tensor(all_vo_scores)
        _, vo_prune_indices = torch.topk(all_vo_scores, vo_channels_to_prune, largest=False)
        
        # Build plan
        for layer_idx, layer in enumerate(layers):
            plan[layer_idx] = {
                'remove_heads': [],
                'remove_channels_qk': [],
                'remove_channels_vo': []
            }
        
        for idx in qk_prune_indices:
            layer_idx, ch_idx = qk_channel_map[idx]
            plan[layer_idx]['remove_channels_qk'].append(ch_idx)
            
        for idx in vo_prune_indices:
            layer_idx, ch_idx = vo_channel_map[idx]
            plan[layer_idx]['remove_channels_vo'].append(ch_idx)
    
    elif pruning_strategy == PruningStrategy.MULTI_HEAD_ENTIRE_HEAD:
        # Collect all head importance scores globally
        all_head_scores = []
        head_map = []  # (layer_idx, head_idx)
        
        for layer_idx, sa in enumerate(layers):
            _, _, head_scores = compute_channel_head_importance(sa, layer_idx, importance_strategy, fisher_info, order)
            
            for head_idx, score in enumerate(head_scores):
                all_head_scores.append(score.item())
                head_map.append((layer_idx, head_idx))
        
        # Determine heads to prune globally (lowest importance scores)
        total_heads = len(all_head_scores)
        heads_to_prune = math.ceil(total_heads * prune_amount)
        
        all_head_scores = torch.tensor(all_head_scores)
        _, prune_indices = torch.topk(all_head_scores, heads_to_prune, largest=False)
        
        # Build plan
        for layer_idx, layer in enumerate(layers):
            plan[layer_idx] = {
                'remove_heads': [],
                'remove_channels_qk': [],
                'remove_channels_vo': []
            }
        
        for idx in prune_indices:
            layer_idx, head_idx = head_map[idx]
            plan[layer_idx]['remove_heads'].append(head_idx)
    
    return plan

def _determine_local_pruning_plan(layers, prune_amount: float, pruning_strategy: PruningStrategy,
                                               importance_strategy: ImportanceStrategy, fisher_info: Optional[Dict], order: int):
    """Local pruning plan with configurable importance strategy."""
    plan = {}
    
    for layer_idx, sa in enumerate(layers):
        qk_scores, vo_scores, head_scores = compute_channel_head_importance(sa, layer_idx, importance_strategy, fisher_info, order)
        
        if pruning_strategy == PruningStrategy.MULTI_HEAD_PER_HEAD:
            plan[layer_idx] = {}
            for head in range(sa.num_heads_q):
                plan[layer_idx][head] = {'remove_heads': [], 
                                       'remove_channels_qk': [],
                                       'remove_channels_vo': []}
        else:
            plan[layer_idx] = {'remove_heads': [], 
                              'remove_channels_qk': [],
                              'remove_channels_vo': []}
        
        if pruning_strategy == PruningStrategy.MULTI_HEAD_SAME_CHANNEL:
            # Prune channels with lowest importance scores across all heads
            # Average scores across heads for same-channel pruning
            if importance_strategy == ImportanceStrategy.FISHER_INFORMATION:
                qk_channel_scores = torch.sum(qk_scores, dim=0)  # (head_dim_q,)
                vo_channel_scores = torch.sum(vo_scores, dim=0)  # (head_dim_v,)                
            else:
                qk_channel_scores = torch.mean(qk_scores, dim=0)  # (head_dim_q,)
                vo_channel_scores = torch.mean(vo_scores, dim=0)  # (head_dim_v,)
            
            channels_to_prune_qk = math.ceil(sa.head_dim_q * prune_amount)
            channels_to_prune_qk = min(channels_to_prune_qk, sa.head_dim_q - 1)
            
            _, prune_indices_qk = torch.topk(qk_channel_scores, channels_to_prune_qk, largest=False)
            
            channels_to_prune_vo = math.ceil(sa.head_dim_v * prune_amount)
            channels_to_prune_vo = min(channels_to_prune_vo, sa.head_dim_v - 1)
            
            _, prune_indices_vo = torch.topk(vo_channel_scores, channels_to_prune_vo, largest=False)
            
            plan[layer_idx]['remove_channels_qk'] = prune_indices_qk.tolist()
            plan[layer_idx]['remove_channels_vo'] = prune_indices_vo.tolist()
            
        elif pruning_strategy == PruningStrategy.MULTI_HEAD_ENTIRE_HEAD:
            # Prune heads with lowest importance scores
            heads_to_prune = math.ceil(sa.num_heads_q * prune_amount)
            heads_to_prune = min(heads_to_prune, sa.num_heads_q - 1)
            
            _, prune_indices = torch.topk(head_scores, heads_to_prune, largest=False)
            plan[layer_idx]['remove_heads'] = prune_indices.tolist()
            
        elif pruning_strategy == PruningStrategy.MULTI_HEAD_PER_HEAD:
            # Prune channels within each head based on importance scores
            for head in range(sa.num_heads_q):
                channels_to_prune_qk = math.ceil(sa.head_dim_q * prune_amount)
                channels_to_prune_qk = min(channels_to_prune_qk, sa.head_dim_q - 1)
                
                head_qk_scores = qk_scores[head]  # (head_dim_q,)
                _, prune_indices_qk = torch.topk(head_qk_scores, channels_to_prune_qk, largest=False)
                plan[layer_idx][head]['remove_channels_qk'] = prune_indices_qk.tolist()
                
            for head in range(sa.num_heads_v):
                channels_to_prune_vo = math.ceil(sa.head_dim_v * prune_amount)
                channels_to_prune_vo = min(channels_to_prune_vo, sa.head_dim_v - 1)
                
                head_vo_scores = vo_scores[head]  # (head_dim_v,)
                _, prune_indices_vo = torch.topk(head_vo_scores, channels_to_prune_vo, largest=False)
                plan[layer_idx][head]['remove_channels_vo'] = prune_indices_vo.tolist()
    
    return plan

def _determine_per_head_pruning_plan_old(layers, prune_amount: float, 
                                           importance_strategy: ImportanceStrategy, 
                                           fisher_info: Optional[Dict], order: int):
    """
    Improved per-head pruning that:
    1. Ranks all channels across all heads globally
    2. Ensures equal number of channels pruned per head within each layer
    3. Allows different channels to be pruned in different heads
    """
    plan = defaultdict(lambda: defaultdict(dict))
    
    for layer_idx, sa in enumerate(layers):
        qk_scores, vo_scores, _ = compute_channel_head_importance(sa, layer_idx, importance_strategy, fisher_info, order)
        
        _, _, _, _, _, num_heads, head_dims = get_layer_weights(sa)
        num_heads_q, _, num_heads_v, _ = num_heads
        head_dim_q, _, head_dim_v, _ = head_dims
        
        # Calculate how many channels to prune per head
        qk_channels_to_prune_per_head = max(1, int(head_dim_q * prune_amount))
        qk_channels_to_prune_per_head = min(qk_channels_to_prune_per_head, head_dim_q - 1)
        
        vo_channels_to_prune_per_head = max(1, int(head_dim_v * prune_amount))
        vo_channels_to_prune_per_head = min(vo_channels_to_prune_per_head, head_dim_v - 1)
        
        # Process QK channels
        qk_pruning_plan = _assign_channels_to_heads(
            qk_scores,  # (num_heads_q, head_dim_q)
            qk_channels_to_prune_per_head,
            num_heads_q,
            head_dim_q
        )
        
        # Process VO channels  
        vo_pruning_plan = _assign_channels_to_heads(
            vo_scores,  # (num_heads_v, head_dim_v)
            vo_channels_to_prune_per_head,
            num_heads_v,
            head_dim_v
        )
        
        # Build the plan for this layer
        max_heads = max(num_heads_q, num_heads_v)
        for head_idx in range(max_heads):
            plan[layer_idx][head_idx] = {
                'remove_channels_qk': qk_pruning_plan.get(head_idx, []),
                'remove_channels_vo': vo_pruning_plan.get(head_idx, []),
                'remove_head': []
            }
    
    return plan

def _assign_channels_to_heads(scores, channels_per_head, num_heads, head_dim):
    """
    Assign channels to be pruned to heads using global ranking but equal distribution.
    
    Args:
        scores: (num_heads, head_dim) importance scores
        channels_per_head: number of channels to prune per head
        num_heads: number of heads
        head_dim: dimension of each head
        
    Returns:
        Dict mapping head_idx -> list of channel indices to prune
    """
    
    # Create global ranking of all (head, channel) pairs
    channel_importance = []
    for head_idx in range(num_heads):
        for channel_idx in range(head_dim):
            importance = scores[head_idx, channel_idx].item()
            channel_importance.append((importance, head_idx, channel_idx))
    
    # Sort by importance (lowest first for pruning)
    channel_importance.sort(key=lambda x: x[0])
    
    # Assign channels to heads ensuring equal distribution
    head_assignments = {h: [] for h in range(num_heads)}
    head_counts = {h: 0 for h in range(num_heads)}
    
    # Strategy: Iterate through sorted channels and assign to heads that need more channels
    for importance, head_idx, channel_idx in channel_importance:
        # Only assign if this head still needs more channels
        if head_counts[head_idx] < channels_per_head:
            head_assignments[head_idx].append(channel_idx)
            head_counts[head_idx] += 1
            
            # Stop when all heads have enough channels
            if all(count >= channels_per_head for count in head_counts.values()):
                break
    
    # Handle case where some heads didn't get enough channels due to ranking
    # This can happen if a head has all its channels ranked very low
    for head_idx in range(num_heads):
        while head_counts[head_idx] < channels_per_head:
            # Find remaining channels for this head
            assigned_channels = set(head_assignments[head_idx])
            available_channels = [ch for ch in range(head_dim) if ch not in assigned_channels]
            
            if available_channels:
                # Add the least important remaining channel
                remaining_scores = [(scores[head_idx, ch].item(), ch) for ch in available_channels]
                remaining_scores.sort()
                head_assignments[head_idx].append(remaining_scores[0][1])
                head_counts[head_idx] += 1
            else:
                # This shouldn't happen if channels_per_head < head_dim
                break
    
    return head_assignments

def _determine_global_per_head_pruning_plan_improved_old(layers, prune_amount: float, 
                                                   importance_strategy: ImportanceStrategy,
                                                   fisher_info: Optional[Dict], order: int):
    """
    Global version of improved per-head pruning.
    Ranks channels globally across all layers and heads, then distributes pruning.
    """
    
    # Step 1: Collect all channel importance scores globally
    all_qk_channels = []  # [(importance, layer_idx, head_idx, channel_idx), ...]
    all_vo_channels = []
    
    layer_info = []  # Store layer dimensions for later use
    
    for layer_idx, sa in enumerate(layers):
        qk_scores, vo_scores, _ = compute_channel_head_importance(sa, layer_idx, importance_strategy, fisher_info, order)
        
        _, _, _, _, _, num_heads, head_dims = get_layer_weights(sa)
        num_heads_q, _, num_heads_v, _ = num_heads
        head_dim_q, _, head_dim_v, _ = head_dims
        
        layer_info.append({
            'num_heads_q': num_heads_q,
            'num_heads_v': num_heads_v,
            'head_dim_q': head_dim_q,
            'head_dim_v': head_dim_v
        })
        
        # Collect QK channel scores
        for head_idx in range(num_heads_q):
            for channel_idx in range(head_dim_q):
                importance = qk_scores[head_idx, channel_idx].item()
                all_qk_channels.append((importance, layer_idx, head_idx, channel_idx))
        
        # Collect VO channel scores
        for head_idx in range(num_heads_v):
            for channel_idx in range(head_dim_v):
                importance = vo_scores[head_idx, channel_idx].item()
                all_vo_channels.append((importance, layer_idx, head_idx, channel_idx))
    
    # Step 2: Sort globally by importance
    all_qk_channels.sort(key=lambda x: x[0])  # Sort by importance (lowest first)
    all_vo_channels.sort(key=lambda x: x[0])
    
    # Step 3: Calculate total channels to prune
    total_qk_channels = len(all_qk_channels)
    total_vo_channels = len(all_vo_channels)
    
    qk_channels_to_prune = int(total_qk_channels * prune_amount)
    vo_channels_to_prune = int(total_vo_channels * prune_amount)
    
    # Step 4: Select channels to prune globally
    selected_qk = all_qk_channels[:qk_channels_to_prune]
    selected_vo = all_vo_channels[:vo_channels_to_prune]
    
    # Step 5: Group selections by layer and distribute evenly across heads
    plan = defaultdict(lambda: defaultdict(dict))
    
    # Process QK selections
    qk_by_layer = defaultdict(list)
    for _, layer_idx, head_idx, channel_idx in selected_qk:
        qk_by_layer[layer_idx].append((head_idx, channel_idx))
    
    # Process VO selections
    vo_by_layer = defaultdict(list)
    for _, layer_idx, head_idx, channel_idx in selected_vo:
        vo_by_layer[layer_idx].append((head_idx, channel_idx))
    
    # Step 6: Redistribute to ensure equal channels per head within each layer
    for layer_idx, layer_data in enumerate(layer_info):
        num_heads_q = layer_data['num_heads_q']
        num_heads_v = layer_data['num_heads_v']
        head_dim_q = layer_data['head_dim_q']
        head_dim_v = layer_data['head_dim_v']
        
        # Get selected channels for this layer
        layer_qk_selections = qk_by_layer.get(layer_idx, [])
        layer_vo_selections = vo_by_layer.get(layer_idx, [])
        
        # Redistribute QK channels
        qk_redistribution = _redistribute_channels_equally(
            layer_qk_selections, num_heads_q, head_dim_q
        )
        
        # Redistribute VO channels
        vo_redistribution = _redistribute_channels_equally(
            layer_vo_selections, num_heads_v, head_dim_v
        )
        
        # Build plan for this layer
        max_heads = max(num_heads_q, num_heads_v)
        for head_idx in range(max_heads):
            plan[layer_idx][head_idx] = {
                'remove_channels_qk': qk_redistribution.get(head_idx, []),
                'remove_channels_vo': vo_redistribution.get(head_idx, []),
                'remove_head': []
            }
    
    return plan

def _redistribute_channels_equally(selected_channels, num_heads, head_dim):
    """
    Redistribute selected channels to ensure equal number per head.
    
    Args:
        selected_channels: List of (head_idx, channel_idx) tuples
        num_heads: Number of heads
        head_dim: Dimension of each head
        
    Returns:
        Dict mapping head_idx -> list of channel indices
    """
    if not selected_channels:
        return {}
    
    total_to_prune = len(selected_channels)
    channels_per_head = total_to_prune // num_heads
    remainder = total_to_prune % num_heads
    
    # Create sets of available channels per head
    available_by_head = {h: set(range(head_dim)) for h in range(num_heads)}
    
    # Remove channels that were already selected (to avoid conflicts)
    for head_idx, channel_idx in selected_channels:
        if head_idx < num_heads:
            available_by_head[head_idx].discard(channel_idx)
    
    # Distribute channels equally
    result = {h: [] for h in range(num_heads)}
    
    # First, try to honor original selections as much as possible
    head_counts = {h: 0 for h in range(num_heads)}
    used_selections = set()
    
    for head_idx, channel_idx in selected_channels:
        if head_idx < num_heads:
            target_count = channels_per_head + (1 if head_idx < remainder else 0)
            if head_counts[head_idx] < target_count:
                result[head_idx].append(channel_idx)
                head_counts[head_idx] += 1
                used_selections.add((head_idx, channel_idx))
    
    # Fill remaining slots with available channels
    unused_selections = [(h, c) for h, c in selected_channels if (h, c) not in used_selections]
    
    for head_idx in range(num_heads):
        target_count = channels_per_head + (1 if head_idx < remainder else 0)
        while head_counts[head_idx] < target_count and available_by_head[head_idx]:
            # Try to use an unused selection first
            found = False
            for h, c in unused_selections:
                if c in available_by_head[head_idx]:
                    result[head_idx].append(c)
                    head_counts[head_idx] += 1
                    available_by_head[head_idx].discard(c)
                    unused_selections.remove((h, c))
                    found = True
                    break
            
            if not found:
                # Pick any available channel
                channel_idx = available_by_head[head_idx].pop()
                result[head_idx].append(channel_idx)
                head_counts[head_idx] += 1
    
    return result

def _determine_per_head_pruning_plan_improved(layers, prune_amount: float, 
                                             importance_strategy: ImportanceStrategy, 
                                             fisher_info: Optional[Dict], order: int):
    """
    Corrected per-head pruning that STRICTLY enforces:
    1. All heads in a layer prune exactly the same number of channels
    2. But can prune different channel indices based on per-head importance
    """
    plan = defaultdict(lambda: defaultdict(dict))
    
    for layer_idx, sa in enumerate(layers):
        qk_scores, vo_scores, _ = compute_channel_head_importance(sa, layer_idx, importance_strategy, fisher_info, order)
        
        _, _, _, _, _, num_heads, head_dims = get_layer_weights(sa)
        num_heads_q, _, num_heads_v, _ = num_heads
        head_dim_q, _, head_dim_v, _ = head_dims
        
        # Calculate how many channels to prune per head (SAME FOR ALL HEADS)
        qk_channels_to_prune_per_head = max(1, int(head_dim_q * prune_amount))
        qk_channels_to_prune_per_head = min(qk_channels_to_prune_per_head, head_dim_q - 1)
        
        vo_channels_to_prune_per_head = max(1, int(head_dim_v * prune_amount))
        vo_channels_to_prune_per_head = min(vo_channels_to_prune_per_head, head_dim_v - 1)
        
        # For each head, prune its least important channels
        for head_idx in range(num_heads_q):
            # Get this head's channel scores
            head_qk_scores = qk_scores[head_idx]  # (head_dim_q,)
            
            # Find the least important channels for THIS head
            _, prune_indices_qk = torch.topk(head_qk_scores, qk_channels_to_prune_per_head, largest=False)
            
            plan[layer_idx][head_idx]['remove_channels_qk'] = prune_indices_qk.tolist()
            plan[layer_idx][head_idx]['remove_head'] = []
        
        for head_idx in range(num_heads_v):
            # Get this head's channel scores  
            head_vo_scores = vo_scores[head_idx]  # (head_dim_v,)
            
            # Find the least important channels for THIS head
            _, prune_indices_vo = torch.topk(head_vo_scores, vo_channels_to_prune_per_head, largest=False)
            
            if head_idx not in plan[layer_idx]:
                plan[layer_idx][head_idx] = {'remove_channels_qk': [], 'remove_head': []}
            plan[layer_idx][head_idx]['remove_channels_vo'] = prune_indices_vo.tolist()
    
    return plan

def _determine_global_per_head_pruning_plan_improved(layers, prune_amount: float, 
                                                     importance_strategy: ImportanceStrategy,
                                                     fisher_info: Optional[Dict], order: int):
    """
    Global per-head pruning that enforces equal channels per head within each layer.
    Uses global ranking to influence which channels get pruned, but maintains the constraint.
    """
    plan = defaultdict(lambda: defaultdict(dict))
    
    # Step 1: Collect global channel importance across all layers
    global_qk_channel_scores = defaultdict(float)  # channel_idx -> total_importance
    global_vo_channel_scores = defaultdict(float)
    
    for layer_idx, sa in enumerate(layers):
        qk_scores, vo_scores, _ = compute_channel_head_importance(sa, layer_idx, importance_strategy, fisher_info, order)
        
        # Sum importance across all heads and add to global scores
        layer_qk_totals = torch.sum(qk_scores, dim=0)  # (head_dim_q,)
        layer_vo_totals = torch.sum(vo_scores, dim=0)  # (head_dim_v,)
        
        for ch_idx, score in enumerate(layer_qk_totals):
            global_qk_channel_scores[ch_idx] += score.item()
            
        for ch_idx, score in enumerate(layer_vo_totals):
            global_vo_channel_scores[ch_idx] += score.item()
    
    # Step 2: Create global channel ranking (least important first)
    sorted_qk_channels = sorted(global_qk_channel_scores.keys(), 
                               key=lambda x: global_qk_channel_scores[x])
    sorted_vo_channels = sorted(global_vo_channel_scores.keys(), 
                               key=lambda x: global_vo_channel_scores[x])
    
    # Step 3: For each layer, apply pruning with global guidance but per-head flexibility
    for layer_idx, sa in enumerate(layers):
        qk_scores, vo_scores, _ = compute_channel_head_importance(sa, layer_idx, importance_strategy, fisher_info, order)
        
        _, _, _, _, _, num_heads, head_dims = get_layer_weights(sa)
        num_heads_q, _, num_heads_v, _ = num_heads
        head_dim_q, _, head_dim_v, _ = head_dims
        
        # Calculate channels to prune per head (SAME FOR ALL HEADS IN THIS LAYER)
        qk_channels_to_prune_per_head = max(1, int(head_dim_q * prune_amount))
        qk_channels_to_prune_per_head = min(qk_channels_to_prune_per_head, head_dim_q - 1)
        
        vo_channels_to_prune_per_head = max(1, int(head_dim_v * prune_amount))
        vo_channels_to_prune_per_head = min(vo_channels_to_prune_per_head, head_dim_v - 1)
        
        # For each head, combine global ranking with local head importance
        for head_idx in range(num_heads_q):
            head_qk_scores = qk_scores[head_idx]  # (head_dim_q,)
            
            # Create combined scores: global ranking + local head importance
            combined_scores = []
            for ch_idx in range(head_dim_q):
                global_rank = sorted_qk_channels.index(ch_idx) if ch_idx in sorted_qk_channels else len(sorted_qk_channels)
                local_importance = head_qk_scores[ch_idx].item()
                
                # Combine: lower is worse (global_rank is already ordered least->most important)
                # Weight global ranking more heavily for consistency across heads
                combined_score = 0.7 * global_rank + 0.3 * (1.0 / (local_importance + 1e-8))
                combined_scores.append((combined_score, ch_idx))
            
            # Sort by combined score and take the worst channels
            combined_scores.sort(key=lambda x: x[0], reverse=True)  # Highest combined score = worst
            prune_channels_qk = [ch_idx for _, ch_idx in combined_scores[:qk_channels_to_prune_per_head]]
            
            plan[layer_idx][head_idx]['remove_channels_qk'] = prune_channels_qk
            plan[layer_idx][head_idx]['remove_head'] = []
        
        # Same for VO channels
        for head_idx in range(num_heads_v):
            head_vo_scores = vo_scores[head_idx]  # (head_dim_v,)
            
            combined_scores = []
            for ch_idx in range(head_dim_v):
                global_rank = sorted_vo_channels.index(ch_idx) if ch_idx in sorted_vo_channels else len(sorted_vo_channels)
                local_importance = head_vo_scores[ch_idx].item()
                
                combined_score = 0.7 * global_rank + 0.3 * (1.0 / (local_importance + 1e-8))
                combined_scores.append((combined_score, ch_idx))
            
            combined_scores.sort(key=lambda x: x[0], reverse=True)
            prune_channels_vo = [ch_idx for _, ch_idx in combined_scores[:vo_channels_to_prune_per_head]]
            
            if head_idx not in plan[layer_idx]:
                plan[layer_idx][head_idx] = {'remove_channels_qk': [], 'remove_head': []}
            plan[layer_idx][head_idx]['remove_channels_vo'] = prune_channels_vo
    
    return plan

def _determine_optimal_per_head_assignment(layers, prune_amount: float, 
                                          importance_strategy: ImportanceStrategy,
                                          fisher_info: Optional[Dict], order: int):
    """
    Find optimal assignment of channels to prune across all layers and heads,
    subject to equal pruning per head within each layer.
    
    This is a constrained optimization problem:
    - Minimize: sum of importance scores of pruned channels
    - Subject to: equal channels pruned per head within each layer
    """
    
    # Step 1: Collect all possible pruning "bundles" 
    # A bundle = (layer_idx, channels_per_head, total_cost)
    layer_bundles = []
    layer_info = []
    
    for layer_idx, sa in enumerate(layers):
        qk_scores, vo_scores, _ = compute_channel_head_importance(sa, layer_idx, importance_strategy, fisher_info, order)
        
        _, _, _, _, _, num_heads, head_dims = get_layer_weights(sa)
        num_heads_q, _, num_heads_v, _ = num_heads
        head_dim_q, _, head_dim_v, _ = head_dims
        
        layer_info.append({
            'num_heads_q': num_heads_q,
            'num_heads_v': num_heads_v, 
            'head_dim_q': head_dim_q,
            'head_dim_v': head_dim_v,
            'qk_scores': qk_scores,
            'vo_scores': vo_scores
        })
        
        # For this layer, compute cost of pruning k channels per head for k=1,2,...,max_k
        max_qk_per_head = head_dim_q - 1  # Keep at least 1 channel
        max_vo_per_head = head_dim_v - 1
        
        qk_bundles = []
        vo_bundles = []
        
        # QK bundles
        for k in range(1, max_qk_per_head + 1):
            total_cost = 0
            for head_idx in range(num_heads_q):
                head_scores = qk_scores[head_idx]
                # Cost = sum of k smallest importance scores for this head
                k_smallest_costs, _ = torch.topk(head_scores, k, largest=False)
                total_cost += k_smallest_costs.sum().item()
            
            qk_bundles.append({
                'channels_per_head': k,
                'total_channels': k * num_heads_q,
                'total_cost': total_cost,
                'cost_per_channel': total_cost / (k * num_heads_q)
            })
        
        # VO bundles
        for k in range(1, max_vo_per_head + 1):
            total_cost = 0
            for head_idx in range(num_heads_v):
                head_scores = vo_scores[head_idx]
                k_smallest_costs, _ = torch.topk(head_scores, k, largest=False)
                total_cost += k_smallest_costs.sum().item()
            
            vo_bundles.append({
                'channels_per_head': k,
                'total_channels': k * num_heads_v,
                'total_cost': total_cost,
                'cost_per_channel': total_cost / (k * num_heads_v)
            })
        
        layer_bundles.append({
            'layer_idx': layer_idx,
            'qk_bundles': qk_bundles,
            'vo_bundles': vo_bundles
        })
    
    # Step 2: Calculate total budget
    total_qk_channels = sum(info['num_heads_q'] * info['head_dim_q'] for info in layer_info)
    total_vo_channels = sum(info['num_heads_v'] * info['head_dim_v'] for info in layer_info)
    
    qk_budget = int(total_qk_channels * prune_amount)
    vo_budget = int(total_vo_channels * prune_amount)
    
    # Step 3: Solve optimization problem using greedy approach
    # (For exact solution, you'd use dynamic programming or integer programming)
    qk_assignment = _solve_channel_assignment(layer_bundles, qk_budget, 'qk')
    vo_assignment = _solve_channel_assignment(layer_bundles, vo_budget, 'vo')
    
    # Step 4: Convert assignment to pruning plan
    plan = defaultdict(lambda: defaultdict(dict))
    
    for layer_idx, info in enumerate(layer_info):
        qk_channels_per_head = qk_assignment.get(layer_idx, 0)
        vo_channels_per_head = vo_assignment.get(layer_idx, 0)
        
        num_heads_q = info['num_heads_q']
        num_heads_v = info['num_heads_v']
        qk_scores = info['qk_scores']
        vo_scores = info['vo_scores']
        
        # For each head, find the specific channels to prune
        max_heads = max(num_heads_q, num_heads_v)
        
        for head_idx in range(max_heads):
            plan[layer_idx][head_idx]['remove_head'] = []
            
            # QK channels
            if head_idx < num_heads_q and qk_channels_per_head > 0:
                head_qk_scores = qk_scores[head_idx]
                _, prune_indices = torch.topk(head_qk_scores, qk_channels_per_head, largest=False)
                plan[layer_idx][head_idx]['remove_channels_qk'] = prune_indices.tolist()
            else:
                plan[layer_idx][head_idx]['remove_channels_qk'] = []
            
            # VO channels
            if head_idx < num_heads_v and vo_channels_per_head > 0:
                head_vo_scores = vo_scores[head_idx]
                _, prune_indices = torch.topk(head_vo_scores, vo_channels_per_head, largest=False)
                plan[layer_idx][head_idx]['remove_channels_vo'] = prune_indices.tolist()
            else:
                plan[layer_idx][head_idx]['remove_channels_vo'] = []
    
    return plan

def _solve_channel_assignment(layer_bundles, budget, bundle_type):
    """
    Solve the channel assignment optimization problem using greedy approach.
    OBJECTIVE: Minimize total importance loss while staying within budget.
    
    Args:
        layer_bundles: List of bundles per layer
        budget: Total channels to prune
        bundle_type: 'qk' or 'vo'
        
    Returns:
        Dict mapping layer_idx -> channels_per_head
    """
    
    # Create list of all possible "moves": (layer_idx, marginal_loss, marginal_channels)
    # A move represents increasing pruning in a layer by 1 channel per head
    moves = []
    current_assignment = {}  # layer_idx -> current_channels_per_head
    
    # Initialize current assignment to 0 for all layers
    for layer_data in layer_bundles:
        layer_idx = layer_data['layer_idx']
        current_assignment[layer_idx] = 0
    
    # Populate initial moves (going from 0 to 1 channel per head)
    for layer_data in layer_bundles:
        layer_idx = layer_data['layer_idx']
        bundles = layer_data[f'{bundle_type}_bundles']
        
        if bundles:  # Make sure bundles exist
            first_bundle = bundles[0]  # k=1 channels per head
            moves.append({
                'layer_idx': layer_idx,
                'marginal_loss': first_bundle['total_cost'],  # Loss from this move
                'marginal_channels': first_bundle['total_channels'],
                'new_channels_per_head': 1,
                'loss_per_channel': first_bundle['cost_per_channel']
            })
    
    # Greedy selection: always pick the move with LOWEST loss per channel
    # This minimizes the total importance loss for our pruning budget
    used_budget = 0
    assignment = {layer_idx: 0 for layer_idx in current_assignment.keys()}
    
    while used_budget < budget and moves:
        # Find move with LOWEST loss per channel (best efficiency)
        moves.sort(key=lambda x: x['loss_per_channel'])  # Ascending order
        
        best_move = None
        for move in moves:
            if used_budget + move['marginal_channels'] <= budget:
                best_move = move
                break
        
        if best_move is None:
            break  # No move fits in remaining budget
        
        # Apply the move
        layer_idx = best_move['layer_idx']
        assignment[layer_idx] = best_move['new_channels_per_head']
        used_budget += best_move['marginal_channels']
        
        # Remove this move and add next move for this layer
        moves.remove(best_move)
        
        # Add next possible move for this layer (marginal cost of going from k to k+1)
        layer_data = next(ld for ld in layer_bundles if ld['layer_idx'] == layer_idx)
        bundles = layer_data[f'{bundle_type}_bundles']
        next_k = best_move['new_channels_per_head'] + 1
        
        if next_k <= len(bundles):
            next_bundle = bundles[next_k - 1]  # k is 1-indexed, bundles are 0-indexed
            current_bundle = bundles[best_move['new_channels_per_head'] - 1]
            
            # Marginal loss = additional loss from going from k to k+1 channels per head
            marginal_loss = next_bundle['total_cost'] - current_bundle['total_cost']
            marginal_channels = next_bundle['total_channels'] - current_bundle['total_channels']
            
            moves.append({
                'layer_idx': layer_idx,
                'marginal_loss': marginal_loss,
                'marginal_channels': marginal_channels,
                'new_channels_per_head': next_k,
                'loss_per_channel': marginal_loss / marginal_channels
            })
    
    return assignment

def structured_prune_model(model, prune_amount: float, threshold_strategy: ThresholdStrategy = ThresholdStrategy.GLOBAL,
                           pruning_strategy: PruningStrategy = PruningStrategy.MULTI_HEAD_SAME_CHANNEL, 
                           importance_strategy: ImportanceStrategy = ImportanceStrategy.MAGNITUDE,
                           fisher_data_loader=None, fisher_criterion=None, fisher_config=None, 
                           order: int = 2, device='cuda', remove_empty_layers: bool = True):
    """
    Main function to perform structured pruning with configurable importance strategies.
    
    Args:
        model: The model to prune
        prune_amount: Fraction of parameters to prune (0.0 to 1.0)
        threshold_strategy: Global or local pruning strategy
        pruning_strategy: How to prune (same channel, per head, entire head)
        importance_strategy: Use magnitude or Fisher Information
        fisher_data_loader: DataLoader for Fisher computation (required if using Fisher strategy)
        fisher_criterion: Loss function for Fisher computation (required if using Fisher strategy)
        fisher_config: Fisher computation configuration
        order: Norm order for magnitude-based computation
        device: Device to run computation on
        remove_empty_layers: Whether to remove completely pruned layers
        
    Returns:
        Pruned model
    """
    
    # Get layers
    layers = get_model_layers(model)
    
    # Compute Fisher Information if needed
    fisher_info = None
    if importance_strategy == ImportanceStrategy.FISHER_INFORMATION:
        if fisher_data_loader is None and fisher_criterion is None:
            raise ValueError("fisher_data_loader and fisher_criterion are required for Fisher Information strategy")
        
        if fisher_config is None:
            fisher_config = FisherConfig()
        
        model.zero_grad()
        for p in model.parameters():
            if p.grad is not None:
                p.grad = None
        
        # Force cleanup before Fisher computation
        torch.cuda.empty_cache()
        
        print("Computing Fisher Information for importance-based pruning...")
        fisher_info = compute_fisher_information(model, fisher_data_loader, fisher_criterion, fisher_config, device)

        model.zero_grad()
        for p in model.parameters():
            p.requires_grad = False
            if p.grad is not None:
                p.grad = None
        
        # Force aggressive cleanup
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        
    model.to('cpu')
    torch.cuda.empty_cache()
    # Determine pruning plan using the specified importance strategy
    pruning_plan = determine_pruning_plan(layers, prune_amount, pruning_strategy, threshold_strategy, importance_strategy, fisher_info, order)
    
    # Apply structured pruning (reuse existing function)
    pruned_model = apply_structured_pruning(model, pruning_plan, remove_empty_layers)
    
    del model
    
    if fisher_info is not None:
        del fisher_info
    
    torch.cuda.empty_cache()
    return pruned_model


def get_model_sparsity_stats(pruned_model, original_changes: PruningChanges, detailed: bool = True):
    """
    Compute comprehensive statistics for a structurally pruned model.
    
    Args:
        pruned_model: The pruned Vision Transformer model
        original_changes: PruningChanges object containing the pruning history
        detailed: If True, returns per-layer and per-head detailed statistics
        
    Returns:
        Dictionary containing comprehensive pruning statistics
    """
    # Get current model layers
    current_layers = get_model_layers(pruned_model)
    
    # Initialize counters for original model
    original_total_params = 0
    original_total_heads = 0
    original_total_channels = 0
    original_total_layers = len(original_changes.original_config)
    
    # Initialize counters for pruned model
    pruned_total_params = 0
    pruned_total_heads = 0
    pruned_total_channels = 0
    pruned_total_layers = len(current_layers)
    
    # Calculate original model statistics
    for layer_idx, config in original_changes.original_config.items():
        
        embed_dim = config['embed_dim']
        num_heads_q = config['num_heads_q']
        num_heads_k = config['num_heads_k']
        num_heads_v = config['num_heads_v']
        num_heads_o = config['num_heads_o']
        head_dim_q = config['head_dim_q']
        head_dim_k = config['head_dim_k']
        head_dim_v = config['head_dim_v']
        head_dim_o = config['head_dim_o']
        
        # Count parameters (Q, K, V projections + output projection)
        qkv_params = embed_dim * (num_heads_q * head_dim_q + num_heads_k * head_dim_k + num_heads_v * head_dim_v)
        qkv_bias_params = num_heads_q * head_dim_q + num_heads_k * head_dim_k + num_heads_v * head_dim_v
        out_params = embed_dim * num_heads_o * head_dim_o
        
        layer_params = qkv_params + qkv_bias_params + out_params
        original_total_params += layer_params
        original_total_heads += num_heads_q  # Assuming all head counts are equal for simplicity
        original_total_channels += head_dim_q * num_heads_q  # Total channels across all heads
    
    # Calculate pruned model statistics and per-layer details
    layer_stats = []
    
    for layer_idx, sa in enumerate(current_layers):
        
        weight, _, bias, _, embed_dim, num_heads, head_dims = get_layer_weights(sa)
        
        # Get current layer configuration
        num_heads_q, num_heads_k, num_heads_v, num_heads_o = num_heads
        head_dim_q, head_dim_k, head_dim_v, head_dim_o = head_dims
        
        # Count current parameters
        qkv_params = embed_dim * (num_heads_q * head_dim_q + num_heads_k * head_dim_k + num_heads_v * head_dim_v)
        qkv_bias_params = num_heads_q * head_dim_q + num_heads_k * head_dim_k + num_heads_v * head_dim_v
        out_params = embed_dim * num_heads_o * head_dim_o
        
        layer_params = qkv_params + qkv_bias_params + out_params
        pruned_total_params += layer_params
        pruned_total_heads += num_heads_q + num_heads_k + num_heads_v + num_heads_o 
        pruned_total_channels += num_heads_q * head_dim_q + num_heads_k * head_dim_k + num_heads_v * head_dim_v + num_heads_o * head_dim_o
        
        # Find corresponding original layer configuration
        original_layer_idx = layer_idx
        # Account for removed layers by finding the actual original index
        removed_before = sum(1 for removed_idx in original_changes.removed_layers if removed_idx <= layer_idx)
        actual_original_idx = layer_idx + removed_before
        
        if actual_original_idx in original_changes.original_config:
            orig_config = original_changes.original_config[actual_original_idx]
            
            # Calculate original layer parameters
            orig_qkv_params = orig_config['embed_dim'] * (
                orig_config['num_heads_q'] * orig_config['head_dim_q'] + 
                orig_config['num_heads_k'] * orig_config['head_dim_k'] + 
                orig_config['num_heads_v'] * orig_config['head_dim_v']
            )
            orig_qkv_bias_params = (
                orig_config['num_heads_q'] * orig_config['head_dim_q'] + 
                orig_config['num_heads_k'] * orig_config['head_dim_k'] + 
                orig_config['num_heads_v'] * orig_config['head_dim_v']
            )
            orig_out_params = orig_config['embed_dim'] * orig_config['num_heads_o'] * orig_config['head_dim_o']
            orig_layer_params = orig_qkv_params + orig_qkv_bias_params + orig_out_params
            
            # Calculate reduction ratios for this layer
            param_reduction = 1.0 - (layer_params / orig_layer_params) if orig_layer_params > 0 else 0.0
            head_reduction = 1.0 - (num_heads_q / orig_config['num_heads_q']) if orig_config['num_heads_q'] > 0 else 0.0
            channel_reduction = 1.0 - (head_dim_q / orig_config['head_dim_q']) if orig_config['head_dim_q'] > 0 else 0.0
            
            layer_info = {
                'layer_idx': layer_idx,
                'original_layer_idx': actual_original_idx,
                'current_params': layer_params,
                'original_params': orig_layer_params,
                'sparsity': 1-(layer_params/orig_layer_params),
                'param_reduction': param_reduction,
                'current_heads': num_heads_q,
                'original_heads': orig_config['num_heads_q'],
                'head_reduction': head_reduction,
                'current_head_dim': head_dim_q,
                'original_head_dim': orig_config['head_dim_q'],
                'channel_reduction': channel_reduction,
                'removed_heads': len(original_changes.layer_changes.get(actual_original_idx, {}).get('removed_heads_q', set())),
                'removed_channels_qk': len(original_changes.layer_changes.get(actual_original_idx, {}).get('removed_channels_q', set())),
                'removed_channels_vo': len(original_changes.layer_changes.get(actual_original_idx, {}).get('removed_channels_v', set())),
            }
        else:
            # This shouldn't happen in normal cases, but handle it gracefully
            layer_info = {
                'layer_idx': layer_idx,
                'original_layer_idx': None,
                'current_params': layer_params,
                'original_params': None,
                'param_reduction': None,
                'current_heads': num_heads_q,
                'original_heads': None,
                'head_reduction': None,
                'current_head_dim': head_dim_q,
                'original_head_dim': None,
                'channel_reduction': None,
                'removed_heads': 0,
                'removed_channels_qk': 0,
                'removed_channels_vo': 0,
            }
        
        # Add detailed per-head analysis if requested
        if detailed:
            head_stats = []
            
            # Get weights for per-head analysis
            
            q_out_dim = num_heads_q * head_dim_q
            k_out_dim = num_heads_k * head_dim_k
            v_out_dim = num_heads_v * head_dim_v
            
            # Split weights
            q_weight = weight[:q_out_dim, :].reshape(num_heads_q, head_dim_q, -1)
            k_weight = weight[q_out_dim:q_out_dim+k_out_dim, :].reshape(num_heads_k, head_dim_k, -1)
            v_weight = weight[q_out_dim+k_out_dim:, :].reshape(num_heads_v, head_dim_v, -1)
            
            if bias is not None:
                q_bias = bias[:q_out_dim].reshape(num_heads_q, head_dim_q)
                k_bias = bias[q_out_dim:q_out_dim+k_out_dim].reshape(num_heads_k, head_dim_k)
                v_bias = bias[q_out_dim+k_out_dim:].reshape(num_heads_v, head_dim_v)
            
            for head_idx in range(num_heads_q):
                # Calculate parameters per head
                head_params = q_weight[head_idx].numel() + k_weight[head_idx].numel() + v_weight[head_idx].numel()
                if bias is not None:
                    head_params += q_bias[head_idx].numel() + k_bias[head_idx].numel() + v_bias[head_idx].numel()
                
                head_info = {
                    'head_idx': head_idx,
                    'params': head_params,
                    'channels_q': head_dim_q,
                    'channels_k': head_dim_k,
                    'channels_v': head_dim_v,
                }
                head_stats.append(head_info)
            
            layer_info['heads'] = head_stats
        
        layer_stats.append(layer_info)
    
    # Calculate overall statistics
    overall_param_reduction = 1.0 - (pruned_total_params / original_total_params) if original_total_params > 0 else 0.0
    overall_head_reduction = 1.0 - (pruned_total_heads / original_total_heads) if original_total_heads > 0 else 0.0
    overall_channel_reduction = 1.0 - (pruned_total_channels / original_total_channels) if original_total_channels > 0 else 0.0
    layer_reduction = 1.0 - (pruned_total_layers / original_total_layers) if original_total_layers > 0 else 0.0
    
    # Compile comprehensive statistics
    stats = {
        'overall': {
            'original_layers': original_total_layers,
            'pruned_layers': pruned_total_layers,
            'removed_layers': len(original_changes.removed_layers),
            'layer_reduction': layer_reduction,
            
            'original_total_params': original_total_params,
            'pruned_total_params': pruned_total_params,
            'sparsity': overall_param_reduction,
            'param_reduction': overall_param_reduction,
            'params_removed': original_total_params - pruned_total_params,
            
            'original_total_heads': original_total_heads,
            'pruned_total_heads': pruned_total_heads,
            'head_reduction': overall_head_reduction,
            'heads_removed': original_total_heads - pruned_total_heads,
            
            'original_total_channels': original_total_channels,
            'pruned_total_channels': pruned_total_channels,
            'channel_reduction': overall_channel_reduction,
            'channels_removed': original_total_channels - pruned_total_channels,
        },
        'pruning_summary': {
            'removed_layer_indices': sorted(original_changes.removed_layers),
            'modified_layers': len(original_changes.layer_changes),
            'total_layers_affected': len(original_changes.removed_layers) + len(original_changes.layer_changes),
        },
        'layers': layer_stats if detailed else None,
    }
    
    return stats