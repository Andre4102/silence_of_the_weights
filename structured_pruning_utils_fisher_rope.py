import torch
import torch.nn as nn
import torch.nn.functional as F
from enum import Enum
from collections import defaultdict
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import tf_locoformer
from custom_attention import MultiheadAttention, Attention, WhisperAttention
from utils import get_layers, get_layer_weights, get_layer_weights_fisher
from torch.nn import CrossEntropyLoss
from tqdm import tqdm
import math 
from torch.amp import autocast
import gc

NORM = False
ROPE_AWARE_PRUNING = True  # Global flag to enable/disable RoPE-aware pruning

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


# ============================================================================
# ROPE-AWARE HELPER FUNCTIONS
# ============================================================================

def detect_rope_and_get_pairs(sa, head_dim: int) -> Optional[List[Tuple[int, int]]]:
    """
    Detect if layer uses RoPE and return channel pairs.
    RoPE operates on consecutive pairs: (0,1), (2,3), (4,5), ...
    
    Args:
        sa: Attention layer
        head_dim: Head dimension to check (CHANNEL dimension, not head count)
        
    Returns:
        List of (idx1, idx2) tuples representing RoPE pairs, or None if no RoPE
    """
    if not ROPE_AWARE_PRUNING:
        return None
    
    # Check if layer has RoPE
    has_rope = ((hasattr(sa, 'rope') and sa.rope is not None) or
                (hasattr(sa, 'rotary_emb') and sa.rotary_emb is not None) or
                (hasattr(sa, 'rope_cos') and sa.rope_cos is not None)
                )
    
    if not has_rope:
        return None
    
    # RoPE uses split-half convention matching rotate_half() and LlamaRotaryEmbedding:
    #   emb = cat(freqs, freqs) so frequency i applies to channel i AND channel i+P
    #   rotate_half splits x into first half / second half and rotates together.
    # Therefore the correct pruning pair is (i, i+P), NOT (2i, 2i+1).
    P = head_dim // 2
    pairs = [(i, i + P) for i in range(P)]
    return pairs if pairs else None


def aggregate_rope_pair_scores_channels_only(scores: torch.Tensor, 
                                             rope_pairs: Optional[List[Tuple[int, int]]],
                                             is_channel_scores: bool = True) -> torch.Tensor:
    """
    Aggregate importance scores for RoPE pairs by summing them.
    This treats each pair as a single pruning unit.
    ONLY applies to channel dimension, NOT head dimension.
    
    Args:
        scores: (num_heads, head_dim) or (head_dim,) importance scores
        rope_pairs: List of RoPE pairs, or None if no RoPE
        is_channel_scores: If True, applies RoPE aggregation. If False, returns scores unchanged.
        
    Returns:
        Aggregated scores with same shape. Paired channels get the sum of their scores.
    """
    if rope_pairs is None or not is_channel_scores:
        return scores
    
    aggregated = scores.clone()
 
    # Sum scores of split-half pairs (i, i+P) and assign to both.
    # rope_pairs already encodes the correct (i, i+P) convention.
    if scores.dim() == 2:  # (num_heads, head_dim)
        for idx1, idx2 in rope_pairs:
            pair_sum = scores[:, idx1] + scores[:, idx2]
            aggregated[:, idx1] = pair_sum
            aggregated[:, idx2] = pair_sum
    else:  # (head_dim,)
        for idx1, idx2 in rope_pairs:
            pair_sum = scores[idx1] + scores[idx2]
            aggregated[idx1] = pair_sum
            aggregated[idx2] = pair_sum
 
    return aggregated


def expand_pruned_channels_to_pairs(
    prune_indices,
    rope_pairs,
    channels_to_prune=None,
) -> list[int]:
    """
    Ensure pruned channels always come in complete RoPE pairs.
    When budget is provided, we may prune slightly fewer channels than
    requested to keep pairs intact — never more.
    """
    if rope_pairs is None:
        return sorted(prune_indices if isinstance(prune_indices, list)
                      else prune_indices.tolist())

    if isinstance(prune_indices, torch.Tensor):
        prune_indices = prune_indices.tolist()

    # Build pair lookup: each channel -> its partner
    pair_map = {}
    for a, b in rope_pairs:
        pair_map[a] = b
        pair_map[b] = a

    prune_set = set(prune_indices)
    expanded  = set()

    for idx in sorted(prune_set):
        if idx in expanded:
            continue

        if idx in pair_map:
            mate = pair_map[idx]
            # Always include the full pair — never half a pair
            # If budget would be exceeded, skip this pair entirely
            if channels_to_prune is None or len(expanded) + 2 <= channels_to_prune:
                expanded.update({idx, mate})
            # else: skip both — partial pairs are never allowed
        else:
            # Unpaired channel (no RoPE constraint)
            if channels_to_prune is None or len(expanded) + 1 <= channels_to_prune:
                expanded.add(idx)

    result = sorted(expanded)

    # Sanity check — result must always be even (all pairs complete)
    assert len(result) % 2 == 0 or all(c not in pair_map for c in result), \
        f"expand_pruned_channels_to_pairs produced odd RoPE result: {len(result)}"

    return result


# ============================================================================
# PRUNING CHANGES TRACKING
# ============================================================================

class PruningChanges:
    """Track changes made during structured pruning."""
    
    def __init__(self, model=None):
        self.removed_layers = []  # List of layer indices that were completely removed
        self.layer_changes = {}   # Dict mapping layer_idx to LayerChanges
        self.original_config = {}  # Original model configuration
        self.final_config = {}     # Final model configuration after pruning
        
        if model is not None:
            layers = get_layers(model)
    
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

    del q_weight, k_weight, v_weight, o_weight
    
    if bias is not None:
        new_q_bias = q_bias[kept_heads_q][:, kept_channels_q]
        new_k_bias = k_bias[kept_heads_k][:, kept_channels_k] 
        new_v_bias = v_bias[kept_heads_v][:, kept_channels_v]

        del q_bias, k_bias, v_bias
    
    # Reshape back to linear layer format
    new_q_weight = new_q_weight.reshape(-1, new_q_weight.shape[-1]) #num_head_q * head_dim_q, embed_size
    new_k_weight = new_k_weight.reshape(-1, new_k_weight.shape[-1]) #num_head_k * head_dim_k, embed_size
    new_v_weight = new_v_weight.reshape(-1, new_v_weight.shape[-1]) #num_head_v * head_dim_v, embed_size
    new_o_weight = new_o_weight.reshape(new_o_weight.shape[0], -1) #embed_size, num_head_o * head_dim_o
    
    new_weight = torch.cat([new_q_weight, new_k_weight, new_v_weight], dim=0)

    del new_q_weight, new_k_weight, new_v_weight
    
    if bias is not None:
        new_q_bias = new_q_bias.reshape(-1)
        new_k_bias = new_k_bias.reshape(-1)
        new_v_bias = new_v_bias.reshape(-1)
        new_bias = torch.cat([new_q_bias, new_k_bias, new_v_bias], dim=0)

        del new_q_bias, new_k_bias, new_v_bias
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

    # Stack as (H, D); take rows' j-th entries and sum across heads
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
    #   - marginal "bundle" costs deltas[j] = cost to add 1 more pruned channel per head
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

    # Build a pointer for each layer's next marginal j (1..max_j)
    next_j = [1 if layer_maxj[l] > 0 else 0 for l in range(L)]

    # We'll repeatedly choose the layer with the smallest available delta for its next_j
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
    layers = get_layers(model)
    
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

        has_rope = hasattr(sa, "rope_cos") and sa.rope_cos is not None
        if has_rope:
            rope_cos_new, rope_sin_new = [], []
        
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
            
            removed_channels_qk = set()
            removed_channels_vo = set()
            
            new_head_dim_q = None
            new_head_dim_v = None

            for head_idx in all_heads_q:
                head_plan = plan.get(head_idx, {})
                
                remove_channels_qk = head_plan.get('remove_channels_qk', [])
                remove_channels_vo = head_plan.get('remove_channels_vo', [])

                # print(remove_channels_qk)
                
                #flatten in case of nested list
                if isinstance(remove_channels_qk, torch.Tensor):
                    remove_channels_qk = remove_channels_qk.tolist()
                elif isinstance(remove_channels_qk, list) and any(isinstance(x, (list, tuple)) for x in remove_channels_qk):
                    remove_channels_qk = [i for sub in remove_channels_qk for i in sub]

                if isinstance(remove_channels_vo, torch.Tensor):
                    remove_channels_vo = remove_channels_vo.tolist()
                elif isinstance(remove_channels_vo, list) and any(isinstance(x, (list, tuple)) for x in remove_channels_vo):
                    remove_channels_vo = [i for sub in remove_channels_vo for i in sub]
                
                kept_channels_q = [c for c in all_channels_q if c not in remove_channels_qk]
                kept_channels_k = [c for c in all_channels_k if c not in remove_channels_qk]
                # print(len(kept_channels_q))
                # print(kept_channels_q)
                kept_heads_q = all_heads_q
                kept_heads_k = all_heads_k
                kept_heads_v = all_heads_v
                kept_heads_o = all_heads_o
                
                if head_idx < num_heads_v:
                    kept_channels_v = [c for c in all_channels_v if c not in remove_channels_vo]
                    kept_channels_o = [c for c in all_channels_o if c not in remove_channels_vo]
                else:
                    kept_channels_v = all_channels_v
                    kept_channels_o = all_channels_o
                
                if not kept_channels_q:
                    kept_channels_q = [0]
                if not kept_channels_k:
                    kept_channels_k = [0]
                if head_idx < num_heads_v and not kept_channels_v:
                    kept_channels_v = [0]
                if head_idx < num_heads_o and not kept_channels_o:
                    kept_channels_o = [0]
                
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

                del head_q_weight, head_k_weight, head_v_weight, head_o_weight
                
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

                    del head_q_bias, head_k_bias, head_v_bias
                
                if new_head_dim_q is None:
                    new_head_dim_q = len(kept_channels_q)
                    new_head_dim_v = len(kept_channels_v) if head_idx < num_heads_v else head_dim_v
                
                removed_channels_qk.update(remove_channels_qk)
                if head_idx < num_heads_v:
                    removed_channels_vo.update(remove_channels_vo)
                
                if has_rope:
                    P_orig = sa.rope_cos.shape[2]  # original head_dim // 2
                    kept_set = set(kept_channels_q)
                    for c in kept_channels_q:
                        partner = c + P_orig if c < P_orig else c - P_orig
                        assert partner in kept_set, f"Unpaired channel at head {head_idx}: channel {c} kept but partner {partner} missing"
                    kept_pairs = sorted(c for c in kept_channels_q if c < P_orig)
                    kept_pairs_t = torch.tensor(kept_pairs, dtype=torch.long)
                    rope_cos_new.append(sa.rope_cos[head_idx, :, kept_pairs_t])
                    rope_sin_new.append(sa.rope_sin[head_idx, :, kept_pairs_t])
            
            # Reconstruct the weight matrices
            new_q_weight = torch.stack(new_q_weights, dim=0)
            new_k_weight = torch.stack(new_k_weights, dim=0)
            new_v_weight = torch.stack(new_v_weights, dim=0)
            new_o_weight = torch.stack(new_o_weights, dim=1)
            
            del new_q_weights, new_k_weights, new_v_weights, new_o_weights
            
            new_q_weight = new_q_weight.reshape(-1, embed_dim)
            new_k_weight = new_k_weight.reshape(-1, embed_dim)
            new_v_weight = new_v_weight.reshape(-1, embed_dim)
            new_o_weight = new_o_weight.reshape(embed_dim, -1)
            
            changes.add_layer_change(layer_idx, [], list(removed_channels_qk), list(removed_channels_vo))
            new_weight = torch.cat([new_q_weight, new_k_weight, new_v_weight], dim=0)
        
            if bias is not None:
                new_q_bias = torch.stack(new_q_biases, dim=0).reshape(-1)
                new_k_bias = torch.stack(new_k_biases, dim=0).reshape(-1)
                new_v_bias = torch.stack(new_v_biases, dim=0).reshape(-1)
                
                new_bias = torch.cat([new_q_bias, new_k_bias, new_v_bias], dim=0)
                del new_q_biases, new_k_biases, new_v_biases
            else:
                new_bias = None
            
            if has_rope:
                rope_cos_new = torch.stack(rope_cos_new, dim=0)
                rope_sin_new = torch.stack(rope_sin_new, dim=0)
            
        else:
            # Handle existing strategies
            kept_heads_q = [h for h in all_heads_q if h not in plan.get('remove_heads', [])]
            kept_heads_k = [h for h in all_heads_k if h not in plan.get('remove_heads', [])]
            kept_heads_v = [h for h in all_heads_v if h not in plan.get('remove_heads', [])]
            kept_heads_o = [h for h in all_heads_o if h not in plan.get('remove_heads', [])] 
            
            kept_channels_q = [c for c in all_channels_q if c not in plan.get('remove_channels_qk', [])]
            kept_channels_k = [c for c in all_channels_k if c not in plan.get('remove_channels_qk', [])]
            kept_channels_v = [c for c in all_channels_v if c not in plan.get('remove_channels_vo', [])]
            kept_channels_o = [c for c in all_channels_o if c not in plan.get('remove_channels_vo', [])]
            
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

            changes.add_layer_change(layer_idx, plan.get('remove_heads', []), 
                                   plan.get('remove_channels_qk', []), plan.get('remove_channels_vo', []))
            
            new_weight, new_bias, new_o_weight = extract_structured_weights(
                sa, kept_heads_q, kept_heads_k, kept_heads_v, kept_heads_o,
                kept_channels_q, kept_channels_k, kept_channels_v, kept_channels_o
            )
            if has_rope:
                # Convert to tensors for indexing
                kept_heads_q_tensor = torch.tensor(kept_heads_q, dtype=torch.long, device=sa.rope_cos.device)
                P_orig = sa.rope_cos.shape[2]
                kept_pairs = sorted(c for c in kept_channels_q if c < P_orig)
                kept_pairs_t = torch.tensor(kept_pairs, dtype=torch.long, device=sa.rope_cos.device)
                rope_cos_new = sa.rope_cos[kept_heads_q_tensor][:, :, kept_pairs_t]
                rope_sin_new = sa.rope_sin[kept_heads_q_tensor][:, :, kept_pairs_t]
        
        if has_rope:
            del sa.rope_cos, sa.rope_sin
            sa.register_buffer("rope_cos", rope_cos_new)
            sa.register_buffer("rope_sin", rope_sin_new)
            del rope_cos_new, rope_sin_new

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
        
        
        # Apply new weights with proper parameter deletion
        if isinstance(sa, (MultiheadAttention)):
            # Use register_parameter to properly replace parameters
            # Replace input projection with new pruned weights
            sa.in_proj_weight = nn.Linear(
                sa.embed_dim, 
                sa.q_out_dim + sa.k_out_dim + sa.v_out_dim, 
                bias=(new_bias is not None),
                device=sa.in_proj_weight.weight.device,
                dtype=sa.in_proj_weight.weight.dtype
            )
            sa.in_proj_weight.weight.data.copy_(new_weight)
            if new_bias is not None:
                sa.in_proj_weight.bias.data.copy_(new_bias)

            # Replace output projection with new pruned weights
            old_o_bias = sa.out_proj.bias.data if sa.out_proj.bias is not None else None
            sa.out_proj = nn.Linear(
                sa.v_out_dim,
                sa.embed_dim,
                bias=(old_o_bias is not None),
                device=sa.out_proj.weight.device,
                dtype=sa.out_proj.weight.dtype
            )
            sa.out_proj.weight.data.copy_(new_o_weight)
            if old_o_bias is not None:
                sa.out_proj.bias.data.copy_(old_o_bias)
                
        elif isinstance(sa, (tf_locoformer.MultiHeadSelfAttention)):
            device = next(sa.parameters()).device
            
            # Delete old modules before creating new ones
            del sa.qkv
            if hasattr(sa, 'aggregate_heads'):
                del sa.aggregate_heads
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            
            # Create new modules
            sa.qkv = nn.Linear(in_features=sa.embed_dim, 
                              out_features=sa.q_out_dim + sa.k_out_dim + sa.v_out_dim, 
                              bias=(new_bias is not None), device=device)
            sa.qkv.weight.data = new_weight.contiguous()
            if new_bias is not None:
                sa.qkv.bias.data = new_bias.contiguous()
            sa.qkv.requires_grad_(False)
            
            new_aggregate = nn.Linear(in_features=sa.v_out_dim, out_features=sa.embed_dim, bias=False, device=device)
            new_aggregate.weight.data = new_o_weight.contiguous()
            new_aggregate.requires_grad_(False)
            sa.aggregate_heads = nn.ModuleList([new_aggregate])
            
            # Handle RoPE
            if hasattr(sa, 'rope') and sa.rope is not None:
                old_freqs = sa.rope.freqs.data if isinstance(sa.rope.freqs, torch.nn.Parameter) else sa.rope.freqs
                new_dim = sa.head_dim_q
                if new_dim > 2:
                    new_rope = tf_locoformer.RotaryEmbeddingOdd(new_dim, custom_freqs=None)
                    min_dim = min(new_rope.freqs.shape[0], old_freqs.shape[0])
                    new_rope.freqs.data[:min_dim] = old_freqs[:min_dim].to(new_rope.freqs.device)
                    del sa.rope
                    sa.rope = new_rope.to(device)
                else:
                    del sa.rope
                    sa.rope = None
                    
        elif isinstance(sa, (Attention)):
            device = next(sa.parameters()).device
            old_out_bias = sa.proj.bias.data.clone() if sa.proj.bias is not None else None
            
            # Delete old modules
            del sa.qkv
            del sa.proj
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            
            # Create new modules
            sa.qkv = nn.Linear(in_features=sa.embed_dim, 
                              out_features=sa.q_out_dim + sa.k_out_dim + sa.v_out_dim, 
                              bias=(new_bias is not None), device=device)
            sa.qkv.weight.data = new_weight.contiguous()
            if new_bias is not None:
                sa.qkv.bias.data = new_bias.contiguous()
            sa.qkv.requires_grad_(False)
            
            sa.proj = nn.Linear(in_features=sa.v_out_dim, out_features=sa.embed_dim, 
                               bias=(old_out_bias is not None), device=device)
            sa.proj.weight.data = new_o_weight.contiguous()
            if old_out_bias is not None:
                sa.proj.bias.data = old_out_bias
            sa.proj.requires_grad_(False)
            
            del old_out_bias
            
        elif isinstance(sa, WhisperAttention):
            device = next(sa.parameters()).device
            
            # Extract slices
            q_w = new_weight[:sa.q_out_dim].contiguous()
            k_w = new_weight[sa.q_out_dim:sa.q_out_dim+sa.k_out_dim].contiguous()
            v_w = new_weight[sa.q_out_dim+sa.k_out_dim:].contiguous()
            
            has_bias = new_bias is not None
            if has_bias:
                q_b = new_bias[:sa.q_out_dim].contiguous()
                k_b = new_bias[sa.q_out_dim:sa.q_out_dim+sa.k_out_dim].contiguous()
                v_b = new_bias[sa.q_out_dim+sa.k_out_dim:].contiguous()
            
            o_b = sa.out_proj.bias.data.clone() if sa.out_proj.bias is not None else None
            
            # Delete old modules
            del sa.q_proj, sa.k_proj, sa.v_proj, sa.out_proj
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            
            # Create new modules
            sa.q_proj = nn.Linear(embed_dim, sa.q_out_dim, device=device, bias=has_bias)
            sa.q_proj.weight.data = q_w
            if sa.q_proj.bias is not None:
                sa.q_proj.bias.data = q_b
            
            sa.k_proj = nn.Linear(embed_dim, sa.k_out_dim, device=device, bias=False)
            sa.k_proj.weight.data = k_w
            if sa.k_proj.bias is not None:
                sa.k_proj.bias.data = k_b
            
            sa.v_proj = nn.Linear(embed_dim, sa.v_out_dim, device=device, bias=has_bias)
            sa.v_proj.weight.data = v_w
            if sa.v_proj.bias is not None:
                sa.v_proj.bias.data = v_b
            
            sa.out_proj = nn.Linear(sa.v_out_dim, embed_dim, device=device, bias=(o_b is not None))
            sa.out_proj.weight.data = new_o_weight.contiguous()
            if sa.out_proj.bias is not None:
                sa.out_proj.bias.data = o_b
            
            del q_w, k_w, v_w
            if has_bias:
                del q_b, k_b, v_b
            if o_b is not None:
                del o_b

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

        # Delete weight tensors after use
        del new_weight, new_o_weight
        if new_bias is not None:
            del new_bias
    
    # Force cleanup after all layers processed
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    gc.collect()
    
    return model

def compute_fisher_distill(
    student,
    teacher,
    processor,
    fisher_loader,
    device,
    fisher_config,
    temperature=2.0,
):
    """
    Compute Fisher importance scores for Whisper attention weights using
    the distillation KL divergence loss.

    Args:
        student       : pruned student WhisperForConditionalGeneration (fp32)
        teacher       : frozen teacher WhisperForConditionalGeneration (fp16)
        processor     : WhisperProcessor (for decoder token ids)
        fisher_loader : DataLoader with batch_size=1, shuffle=False
                        (same loader passed to prune_model for Fisher)
        device        : "cuda" or "cpu"
        fisher_config : FisherConfig(num_samples, damping, use_diagonal)
        temperature   : distillation temperature (match training value)

    Returns:
        fisher_info : defaultdict(dict) mapping
                      layer_idx -> {"qkv_weight": Tensor, "out_weight": Tensor}
                      Same format as supervised Fisher, ready for prune_model().
    """
    from utils import get_layers, get_layer_weights

    student.eval()
    teacher.eval()

    # Enable grads only on attention layer parameters
    for p in student.parameters():
        p.requires_grad = False

    layers = get_layers(student)
    for layer in layers:
        for p in layer.parameters():
            p.requires_grad = True

    fisher_sum = {}
    seen = 0

    print(f"Computing distillation Fisher Information using "
          f"{fisher_config.num_samples} samples...")

    for batch in tqdm(fisher_loader, desc="Computing Fisher (distill)",
                      total=fisher_config.num_samples):
        if seen >= fisher_config.num_samples:
            break

        # ── Move batch to device ─────────────────────────────────────────
        input_features = batch["input_features"].to(device, non_blocking=True)
        labels         = batch["labels"].to(device, non_blocking=True)
        attention_mask = batch.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(device, non_blocking=True)

        # Build decoder_input_ids (same as training loop)
        decoder_input_ids = labels.clone()
        decoder_input_ids[decoder_input_ids == -100] = \
            processor.tokenizer.pad_token_id
        decoder_input_ids = decoder_input_ids[:, :-1].contiguous()
        labels_for_loss   = labels[:, 1:].contiguous()

        student.zero_grad(set_to_none=True)

        # ── Student forward (fp32, grads enabled on attn layers) ─────────
        student_out    = student(
            input_features=input_features,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
        )
        student_logits = student_out.logits  # (1, T, V)

        # ── Teacher forward (fp16, no grad) ──────────────────────────────
        with torch.no_grad():
            teacher_out    = teacher(
                input_features=input_features.half(),
                attention_mask=attention_mask.half()
                    if attention_mask is not None else None,
                decoder_input_ids=decoder_input_ids,
            )
            teacher_logits = teacher_out.logits.float()  # (1, T, V)

        # ── KL divergence loss (distillation signal) ─────────────────────
        min_len = min(student_logits.size(1), teacher_logits.size(1))
        s_log   = F.log_softmax(student_logits[:, :min_len] / temperature, dim=-1)
        t_probs = F.softmax(teacher_logits[:, :min_len]    / temperature, dim=-1)

        # Mask padding positions
        mask = (labels_for_loss[:, :min_len] != -100).unsqueeze(-1).float()
        kl   = F.kl_div(s_log, t_probs, reduction="none")          # (1, T, V)
        loss = (temperature ** 2) * (kl * mask).sum() / mask.sum().clamp(min=1)

        loss.backward()

        # ── Accumulate squared gradients ──────────────────────────────────
        for layer_idx, layer in enumerate(layers):
            # get_layer_weights returns:
            weight, o_w = get_layer_weights_fisher(layer)
            q_w, k_w, v_w = weight
            # QKV weights — may be separate tensors or a single fused tensor
            if q_w is not None and k_w is not None and v_w is not None:
                # Separate Q, K, V projections
                grads = []
                for w in (q_w, k_w, v_w):
                    if w.grad is not None:
                        grads.append(w.grad.detach().clone() ** 2)
                if grads:
                    qkv_grad_sq = torch.cat(grads, dim=0)
                    key = (layer_idx, "qkv_weight")
                    if key not in fisher_sum:
                        fisher_sum[key] = qkv_grad_sq
                    else:
                        fisher_sum[key].add_(qkv_grad_sq)
            elif q_w is not None and q_w.grad is not None:
                # Fused QKV weight
                grad_sq = q_w.grad.detach().clone() ** 2
                key = (layer_idx, "qkv_weight")
                if key not in fisher_sum:
                    fisher_sum[key] = grad_sq
                else:
                    fisher_sum[key].add_(grad_sq)

            # Output projection
            if o_w is not None and o_w.grad is not None:
                grad_sq = o_w.grad.detach().clone() ** 2
                key = (layer_idx, "out_weight")
                if key not in fisher_sum:
                    fisher_sum[key] = grad_sq
                else:
                    fisher_sum[key].add_(grad_sq)

        seen += 1

    print(f"\nProcessed {seen} samples")
    print("Finalizing Fisher Information...")

    # ── Convert to prune_model-compatible format ──────────────────────────
    fisher_info = defaultdict(dict)
    for (layer_idx, param_name), fisher_val in fisher_sum.items():
        fisher_mean = fisher_val / max(seen, 1)
        fisher_mean.add_(fisher_config.damping)
        fisher_info[layer_idx][param_name] = fisher_mean

    fisher_sum.clear()
    del fisher_sum

    # ── Restore grad state ────────────────────────────────────────────────
    for p in student.parameters():
        p.requires_grad = False
        p.grad = None

    student.zero_grad(set_to_none=True)
    torch.cuda.empty_cache()
    gc.collect()

    student.train()
    print("Distillation Fisher Information computation completed.")

    return fisher_info

def _kl_distill_loss(student_logits, teacher_logits, temp=0.1):
    """
    KL divergence treating sharpened teacher softmax as the target distribution.
    This is a proper negative log-likelihood, making it valid for Fisher estimation.
    """
    t_dist = F.softmax(teacher_logits / temp, dim=-1).detach()
    s_log  = F.log_softmax(student_logits / temp, dim=-1)
    return F.kl_div(s_log, t_dist, reduction='batchmean')

def compute_fisher_ss_llm(
    student,
    teacher_frozen,
    fisher_loader,
    device,
    fisher_config,
):
    """
    Layer-wise Fisher Information computation for LLMs using distillation.

    For each calibration sample:
      1. Run a full forward pass through both student and teacher, collecting
         per-layer hidden state outputs (no grad, just to get targets).
      2. For each attention layer independently, recompute the student layer
         output WITH grad enabled, compute MSE against the teacher's output
         for that layer, and backward ONLY through that layer.
      3. Accumulate squared gradients as Fisher scores.

    Because each backward is local to one layer, gradients never attenuate
    across layer boundaries — layer 0 gets the same quality Fisher signal
    as layer 31.

    The student's hidden states naturally reflect its current pruned state
    (since we run a real forward pass through it), so scores reflect actual
    post-pruning importance rather than an idealized unpruned signal.

    Args:
        student:         The model being pruned (may already be partially pruned).
        teacher_frozen:  The original unpruned model, frozen, used as target.
        fisher_loader:   DataLoader yielding (input_ids, attention_mask) pairs.
        device:          Torch device.
        fisher_config:   FisherConfig(num_samples, damping).
    """
    student.eval()
    teacher_frozen.eval()

    student_layers  = get_layers(student)
    teacher_layers  = get_layers(teacher_frozen)

    # Only attention layer parameters need gradients
    for p in student.parameters():
        p.requires_grad = False
    for layer in student_layers:
        for p in layer.parameters():
            p.requires_grad = True

    fisher_sum = {}
    seen = 0

    print(f"Computing layer-wise LLM Fisher using {fisher_config.num_samples} samples...")

    for batch in tqdm(fisher_loader, desc="Fisher SS LLM", total=fisher_config.num_samples):
        if seen >= fisher_config.num_samples:
            break

        # Support dicts, BatchEncoding, (input_ids,), (input_ids, mask) tuples
        if isinstance(batch, (list, tuple)):
            input_ids      = batch[0].to(device)
            attention_mask = batch[1].to(device) if len(batch) > 1 else None
        elif hasattr(batch, 'input_ids'):
            # BatchEncoding or any object with .input_ids attribute
            input_ids      = batch.input_ids.to(device)
            attention_mask = batch.attention_mask.to(device) if hasattr(batch, 'attention_mask') else None
        elif hasattr(batch, '__getitem__'):
            # plain dict or dict-like
            input_ids      = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device) if 'attention_mask' in batch else None
        else:
            input_ids      = batch.to(device)
            attention_mask = None

        # ── Step 1: collect per-layer hidden states via forward hooks ──────
        # output_hidden_states=True may not work with the custom LlamaModel in
        # custom_attention.py, so we use hooks which are unconditional.
        student_layer_inputs  = []   # input  to each student layer (pre-hook)
        student_layer_outputs = []   # output of each student layer (post-hook)
        teacher_layer_outputs = []   # output of each teacher layer (post-hook)

        def _input_hook(store):
            def hook(module, args, kwargs):
                if len(args) > 0:
                    hidden = args[0].detach().float()
                    rest_kwargs = {k: v.detach() if isinstance(v, torch.Tensor) else v 
                                for k, v in kwargs.items()}
                else:
                    hidden = kwargs['hidden_states'].detach().float()
                    rest_kwargs = {k: v.detach() if isinstance(v, torch.Tensor) else v 
                                for k, v in kwargs.items() if k != 'hidden_states'}
                store.append((hidden, rest_kwargs))
            return hook

        def _output_hook(store):
            def hook(module, args, kwargs, out):
                if isinstance(out, tuple):
                    out = out[0]
                store.append(out)
            return hook

        s_hooks = []
        for s_layer in student_layers:
            s_hooks.append(s_layer.register_forward_pre_hook(_input_hook(student_layer_inputs), with_kwargs=True))
            s_hooks.append(s_layer.register_forward_hook(_output_hook(student_layer_outputs), with_kwargs=True))
        t_hooks = []
        for t_layer in get_layers(teacher_frozen):
            t_hooks.append(t_layer.register_forward_hook(_output_hook(teacher_layer_outputs), with_kwargs=True))

        with torch.no_grad():
            student(input_ids, attention_mask=attention_mask, use_cache=False)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
            teacher_frozen(input_ids, attention_mask=attention_mask, use_cache=False)

        for h in s_hooks + t_hooks:
            h.remove()

        # ── Step 2: per-layer backward ────────────────────────────────────
        for layer_idx, (s_layer, (s_in, s_kwargs), s_out_ref, t_out) in enumerate(
            zip(student_layers, student_layer_inputs, student_layer_outputs, teacher_layer_outputs)
        ):
            student.zero_grad(set_to_none=True)

            s_out = s_layer(s_in, **s_kwargs)
            if isinstance(s_out, tuple):
                s_out = s_out[0]

            target = t_out.float() if t_out.shape == s_out.shape else s_out_ref.float()
            loss = F.mse_loss(s_out.float(), target)
            loss.backward()

            # Accumulate squared gradients
            weight, o_weight = get_layer_weights_fisher(s_layer)

            if isinstance(weight, list):
                q, k, v = weight
                grads = []
                for w in (q, k, v):
                    grads.append(w.grad.detach().clone().to('cpu') ** 2 if w.grad is not None
                                 else torch.zeros_like(w))
                grad_sq = torch.cat(grads, dim=0)
                key = (layer_idx, 'qkv_weight')
            elif weight is not None and weight.grad is not None:
                grad_sq = weight.grad.detach().clone().to('cpu') ** 2
                key = (layer_idx, 'qkv_weight')
            else:
                grad_sq = None
                key = None

            if key is not None:
                if key not in fisher_sum:
                    fisher_sum[key] = grad_sq
                else:
                    fisher_sum[key].add_(grad_sq)

            if o_weight is not None and o_weight.grad is not None:
                o_grad_sq = o_weight.grad.detach().clone().to('cpu') ** 2
                o_key = (layer_idx, 'out_weight')
                if o_key not in fisher_sum:
                    fisher_sum[o_key] = o_grad_sq
                else:
                    fisher_sum[o_key].add_(o_grad_sq)

        seen += 1

    print(f"\nProcessed {seen} samples")
    print("Finalizing Fisher Information...")

    fisher_info = defaultdict(dict)
    for (layer_idx, param_name), fisher_val in fisher_sum.items():
        fisher_mean = fisher_val / max(seen, 1)
        fisher_mean.add_(fisher_config.damping)
        fisher_info[layer_idx][param_name] = fisher_mean

    fisher_sum.clear()
    del fisher_sum

    for p in student.parameters():
        p.requires_grad = False
        p.grad = None

    student.zero_grad(set_to_none=True)
    torch.cuda.empty_cache()
    gc.collect()

    student.train()
    print("Layer-wise LLM Fisher computation completed.")
    return fisher_info

def compute_fisher_ss(
    student,
    teacher_frozen,
    fisher_loader,
    device,
    fisher_config,
    class_loss_weight=.5,
    patch_layers=None,
):
    student.eval()

    layers = get_layers(student)

    # Enable grads only on attention layers
    for p in student.parameters():
        p.requires_grad = False
    for layer in layers:
        for p in layer.parameters():
            p.requires_grad = True

    fisher_sum = {}
    seen = 0

    # Resolve patch layers and build weights (same scheme as training)
    num_teacher_layers = teacher_frozen.config.num_hidden_layers
    if patch_layers is None:
        patch_layers = list(range(num_teacher_layers))
    n_layers = len(patch_layers)
    layer_weights = torch.linspace(0.5, 1.5, n_layers, device=device)
    layer_weights = layer_weights / layer_weights.sum()

    print(f"Computing self-supervised Fisher Information using {fisher_config.num_samples} samples...")

    for imgs, _ in tqdm(fisher_loader, desc="Computing Fisher SS", total=fisher_config.num_samples):
        if seen >= fisher_config.num_samples:
            break

        imgs = imgs.to(device)
        student.zero_grad(set_to_none=True)

        # --- Student forward (all hidden states for multi-layer patch loss) ---
        student_out = student(imgs, output_hidden_states=True)
        student_hidden = student_out.hidden_states
        student_cls = student_hidden[-1][:, 0]
        student_patch_layers = [student_hidden[i] for i in patch_layers]

        # --- Frozen teacher (fp16, no grad) ---
        with torch.no_grad():
            frozen_out = teacher_frozen(imgs.half(), output_hidden_states=True)
            frozen_hidden = frozen_out.hidden_states
            frozen_cls = frozen_hidden[-1][:, 0].float()
            frozen_patch_layers = [frozen_hidden[i].float() for i in patch_layers]

        # --- CLS loss: cosine distillation from frozen teacher ---
        cls_loss = _kl_distill_loss(student_cls, frozen_cls)

        # --- Multi-layer weighted patch loss ---
        layer_losses = torch.stack([
            _kl_distill_loss(s[:, 1:], t[:, 1:])
            for s, t in zip(student_patch_layers, frozen_patch_layers)
        ])
        patch_loss = (layer_losses * layer_weights).sum()

        loss = class_loss_weight * cls_loss + patch_loss
        loss.backward(retain_graph=False)

        # --- Accumulate squared gradients ---
        for layer_idx, layer in enumerate(layers):
            weight, o_weight = get_layer_weights_fisher(layer)

            if isinstance(weight, list):
                q, k, v = weight
                if q.grad is not None:
                    qkv_grad_sq = torch.cat([
                        q.grad.detach().clone() ** 2,
                        k.grad.detach().clone() ** 2,
                        v.grad.detach().clone() ** 2
                    ], dim=0)
                    key = (layer_idx, 'qkv_weight')
                    if key not in fisher_sum:
                        fisher_sum[key] = qkv_grad_sq
                    else:
                        fisher_sum[key].add_(qkv_grad_sq)
            elif weight is not None and weight.grad is not None:
                grad_sq = weight.grad.detach().clone() ** 2
                key = (layer_idx, 'qkv_weight')
                if key not in fisher_sum:
                    fisher_sum[key] = grad_sq
                else:
                    fisher_sum[key].add_(grad_sq)

            if o_weight is not None and o_weight.grad is not None:
                grad_sq = o_weight.grad.detach().clone() ** 2
                key = (layer_idx, 'out_weight')
                if key not in fisher_sum:
                    fisher_sum[key] = grad_sq
                else:
                    fisher_sum[key].add_(grad_sq)

        seen += 1

    print(f"\nProcessed {seen} samples")
    print("Finalizing Fisher Information...")

    fisher_info = defaultdict(dict)
    for (layer_idx, param_name), fisher_val in fisher_sum.items():
        fisher_mean = fisher_val / max(seen, 1)
        fisher_mean.add_(fisher_config.damping)
        fisher_info[layer_idx][param_name] = fisher_mean

    fisher_sum.clear()
    del fisher_sum

    for p in student.parameters():
        p.requires_grad = False
        p.grad = None

    student.zero_grad(set_to_none=True)
    torch.cuda.empty_cache()
    gc.collect()

    student.train()
    print("Self-supervised Fisher Information computation completed")

    return fisher_info


def compute_fisher_information(model, data_loader, criterion=None, config: FisherConfig = None, device='cuda'):
    """
    Compute Fisher Information incrementally - one sample at a time.
    """
    if config is None:
        config = FisherConfig()
        
    model.eval()
    
    # Initialize running statistics
    fisher_sum = {}
    sample_count = 0
    
    # Disable gradients for non-attention layers
    for p in model.parameters():
        p.requires_grad = False
    
    layers = get_layers(model)
    # print(layers)
    for layer in layers:
        for p in layer.parameters():
            p.requires_grad = True
    
    print(f"Computing Fisher Information incrementally using {config.num_samples} samples...")
    max_steps = config.num_samples // data_loader.batch_size if hasattr(data_loader, 'batch_size') else config.num_samples #assume batch size is 1 for web dataloaders

    for i, batch in enumerate(tqdm(data_loader, desc="Computing Fisher", total=max_steps)):
        if i >= max_steps:
            break
        
        # Clear gradients completely
        model.zero_grad(set_to_none=True)
        
        # with autocast('cuda', dtype=torch.bfloat16):
            # Forward pass
        if criterion is not None:
            if isinstance(batch, dict):
                inputs = batch.get("pixel_values")
                targets = batch.get("labels")
                inputs = inputs.to(device)
                targets = targets.to(device)
            else:
                inputs, targets = batch
                inputs = inputs.to(device)
                targets = targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
        else:
            batch = {
                k: v.to(device, non_blocking=True) if torch.is_tensor(v) else v
                for k, v in batch.items()
            }
            outputs = model(**batch)
            loss = outputs.loss
        
        # Backward pass - use retain_graph=False
        loss.backward(retain_graph=False)
        sample_count += 1

        # Compute squared gradients and accumulate directly
        for layer_idx, layer in enumerate(layers):
            weight, o_weight = get_layer_weights_fisher(layer)
            
            # Process QKV weights
            if isinstance(weight, list):
                q, k, v = weight
                if q.grad is not None:
                    # Detach before computing squared gradients
                    qkv_grad_sq = torch.cat([
                        q.grad.detach().clone() ** 2,
                        k.grad.detach().clone() ** 2,
                        v.grad.detach().clone() ** 2
                    ], dim=0)
                    
                    key = (layer_idx, 'qkv_weight')
                    if key not in fisher_sum:
                        fisher_sum[key] = qkv_grad_sq
                    else:
                        fisher_sum[key].add_(qkv_grad_sq)
                    
            elif weight is not None and weight.grad is not None:
                grad_sq = (weight.grad.detach().clone() ** 2)
                
                key = (layer_idx, 'qkv_weight')
                if key not in fisher_sum:
                    fisher_sum[key] = grad_sq
                else:
                    fisher_sum[key].add_(grad_sq)
            
            # Process output weights
            if o_weight is not None and o_weight.grad is not None:
                grad_sq = (o_weight.grad.detach().clone() ** 2)
                
                key = (layer_idx, 'out_weight')
                if key not in fisher_sum:
                    fisher_sum[key] = grad_sq
                else:
                    fisher_sum[key].add_(grad_sq)
                
                # del grad_sq
                # o_weight.grad = None
        
    # Convert accumulated sums to means and add damping
    print(f"\nProcessed {sample_count} samples")
    print("Finalizing Fisher Information...")

    fisher_info = defaultdict(dict)
    for (layer_idx, param_name), fisher_val in fisher_sum.items():
        fisher_mean = fisher_val.to('cpu') / sample_count
        # print(fisher_mean)
        fisher_mean.add_(config.damping)
        fisher_info[layer_idx][param_name] = fisher_mean
    
    # Clear the accumulation dict
    fisher_sum.clear()
    del fisher_sum
    
    print("Fisher Information computation completed")
    
    # Complete cleanup
    for p in model.parameters():
        p.requires_grad = False
        p.grad = None
    
    model.zero_grad(set_to_none=True)
    torch.cuda.empty_cache()
    gc.collect()
    
    return fisher_info

def compute_channel_head_importance(sa, layer_idx, importance_strategy: ImportanceStrategy = ImportanceStrategy.MAGNITUDE,
                                   fisher_info: Optional[Dict] = None, order: int = 2):
    """
    Compute channel and head importance scores using either magnitude or Fisher Information.
    MODIFIED: Handles RoPE pairs for CHANNEL importance only. HEAD importance is unaffected.
    
    Args:
        sa: Attention layer
        layer_idx: Index of the layer
        importance_strategy: Strategy for computing importance
        fisher_info: Precomputed Fisher Information (required for FISHER_INFORMATION strategy)
        order: Norm order for magnitude-based computation
        
    Returns:
        Tuple of (qk_scores, vo_scores, head_scores)
        - qk_scores: (num_heads_q, head_dim_q) importance scores (RoPE-aggregated if applicable)
        - vo_scores: (num_heads_v, head_dim_v) importance scores (RoPE-aggregated if applicable)
        - head_scores: (num_heads,) head-level importance scores (NOT RoPE-aggregated)
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
            qk_scores, vo_scores, head_scores = _gather_qk_vo_scores(
                sa, order, num_heads_q, head_dim_q, num_heads_k, head_dim_k,
                num_heads_v, head_dim_v, num_heads_o, head_dim_o,
                q_out_dim, k_out_dim, embed_dim
            )
        else:
            qk_scores, vo_scores, head_scores = _compute_fisher_importance_scores(
                sa, fisher_info[layer_idx], 
                num_heads_q, num_heads_k, num_heads_v, num_heads_o,
                head_dim_q, head_dim_k, head_dim_v, head_dim_o,
                q_out_dim, k_out_dim, v_out_dim, embed_dim
            )
    else:
        qk_scores, vo_scores, head_scores = _gather_qk_vo_scores(
            sa, order, num_heads_q, head_dim_q, num_heads_k, head_dim_k,
            num_heads_v, head_dim_v, num_heads_o, head_dim_o,
            q_out_dim, k_out_dim, embed_dim
        )
    
    # Detect RoPE pairs and aggregate scores for CHANNELS only
    rope_pairs_qk = detect_rope_and_get_pairs(sa, head_dim_q)
    # rope_pairs_vo = detect_rope_and_get_pairs(sa, head_dim_v)
    
    # Aggregate channel scores for RoPE pairs (sum the importance of paired channels)
    qk_scores = aggregate_rope_pair_scores_channels_only(qk_scores, rope_pairs_qk, is_channel_scores=True)
    # vo_scores = aggregate_rope_pair_scores_channels_only(vo_scores, rope_pairs_vo, is_channel_scores=True)
    
    # head_scores shape: (num_heads,) - DO NOT apply RoPE aggregation (heads are independent)
    # head_scores remains unchanged
    
    return qk_scores, vo_scores, head_scores

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
    Main pruning plan function - now RoPE-aware for channel pruning.
    
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

def _determine_global_pruning_plan(layers, prune_amount: float, pruning_strategy: PruningStrategy, 
                                  importance_strategy: ImportanceStrategy, fisher_info: Optional[Dict], order: int):
    """
    Global pruning plan - RoPE-aware for CHANNEL pruning, not for HEAD pruning.
    """
    plan = {}
    if pruning_strategy == PruningStrategy.MULTI_HEAD_SAME_CHANNEL:
        # CHANNEL pruning - RoPE-aware
        # Collect all channel importance scores globally
        all_qk_scores = []
        all_vo_scores = []
        qk_channel_map = []  # (layer_idx, channel_idx)
        vo_channel_map = []
        
        for layer_idx, sa in enumerate(layers):
            # Use RoPE-aware importance computation (only affects channel scores)
            qk_scores, vo_scores, _ = compute_channel_head_importance(sa, layer_idx, importance_strategy, fisher_info, order)
            
            #check if the layer implements rotary positional embedding
            has_rope = ((hasattr(sa, 'rope') and sa.rope is not None) or
                        (hasattr(sa, 'rotary_emb') and sa.rotary_emb is not None) or
                        (hasattr(sa, 'rope_cos') and sa.rope_cos is not None)
                        )
            
            # Average scores across heads (already RoPE-aggregated)
            if importance_strategy == ImportanceStrategy.FISHER_INFORMATION:
                qk_channel_scores = torch.sum(qk_scores, dim=0)  # (head_dim_q,)
                vo_channel_scores = torch.sum(vo_scores, dim=0)  # (head_dim_v,)                
            else:
                qk_channel_scores = torch.mean(qk_scores, dim=0)  # (head_dim_q,)
                vo_channel_scores = torch.mean(vo_scores, dim=0)  # (head_dim_v,)

            #if rope is implemented we need to adjust the importance of each channel 
            #to be the sum of the channels involved in each rope pairs
            if has_rope:
                for i in range(0, len(qk_channel_scores) - 1, 2):
                    pair_importance = qk_channel_scores[i] + qk_channel_scores[i + 1]
                    qk_channel_scores[i] = pair_importance
                    qk_channel_scores[i + 1] = pair_importance
                
            
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
        
        # Expand each layer's plan to include complete RoPE pairs
        for layer_idx, sa in enumerate(layers):
            _, _, _, _, _, _, head_dims = get_layer_weights(sa)
            head_dim_q, _, head_dim_v, _ = head_dims
            
            rope_pairs_qk = detect_rope_and_get_pairs(sa, head_dim_q)
            rope_pairs_vo = detect_rope_and_get_pairs(sa, head_dim_v)
            
            plan[layer_idx]['remove_channels_qk'] = expand_pruned_channels_to_pairs(
                plan[layer_idx]['remove_channels_qk'], rope_pairs_qk
            )
            plan[layer_idx]['remove_channels_vo'] = expand_pruned_channels_to_pairs(
                plan[layer_idx]['remove_channels_vo'], rope_pairs_vo
            )
    
    elif pruning_strategy == PruningStrategy.MULTI_HEAD_ENTIRE_HEAD:
        # HEAD pruning - NO RoPE handling needed (RoPE doesn't affect head dimension)
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
    """
    Local pruning plan - RoPE-aware for CHANNEL pruning, not for HEAD pruning.
    """
    plan = {}
    
    for layer_idx, sa in enumerate(layers):
        # Use RoPE-aware importance computation (only affects channel scores)
        qk_scores, vo_scores, head_scores = compute_channel_head_importance(
            sa, layer_idx, importance_strategy, fisher_info, order
        )
        
        # Get layer configuration
        _, _, _, _, _, num_heads, head_dims = get_layer_weights(sa)
        num_heads_q, _, num_heads_v, _ = num_heads
        head_dim_q, _, head_dim_v, _ = head_dims
        
        #check if the layer implements rotary positional embedding
        has_rope = ((hasattr(sa, 'rope') and sa.rope is not None) or
                        (hasattr(sa, 'rotary_emb') and sa.rotary_emb is not None) or
                        (hasattr(sa, 'rope_cos') and sa.rope_cos is not None)
                        )
        if has_rope:
            rope_pairs_qk = detect_rope_and_get_pairs(sa, head_dim_q)
        
        if pruning_strategy == PruningStrategy.MULTI_HEAD_PER_HEAD:
            plan[layer_idx] = {}
            channels_to_prune_qk = None

            for head in range(num_heads_q):
                if channels_to_prune_qk is None:
                    channels_to_prune_qk = math.ceil(head_dim_q * prune_amount)
                    if has_rope:
                        # Round DOWN to nearest even so pairs always fit within budget
                        channels_to_prune_qk = (channels_to_prune_qk // 2) * 2
                        # Ensure at least one pair survives
                        channels_to_prune_qk = min(channels_to_prune_qk, head_dim_q - 2)
                    else:
                        channels_to_prune_qk = min(channels_to_prune_qk, head_dim_q - 1)
                
                head_qk_scores = qk_scores[head].clone()
                if has_rope:
                    P_q = len(head_qk_scores) // 2
                    pair_scores = head_qk_scores[:P_q].clone() + head_qk_scores[P_q:].clone()
                    k_pairs = channels_to_prune_qk // 2
                    _, pair_idx = torch.topk(pair_scores, k_pairs, largest=False)
                    prune_indices_qk = torch.cat([pair_idx, pair_idx + P_q]).sort().values.tolist()
                else:
                    _, prune_indices_qk = torch.topk(head_qk_scores, channels_to_prune_qk, largest=False)
                    prune_indices_qk = prune_indices_qk.tolist()
                # print('final length:', len(prune_indices_qk))
                if head not in plan[layer_idx]:
                    plan[layer_idx][head] = {
                        'remove_channels_qk': prune_indices_qk,
                        'remove_channels_vo': [],
                        'remove_head': []
                    }
                else:
                    plan[layer_idx][head]['remove_channels_qk'] = prune_indices_qk

            # CHANNEL pruning per VO head
            for head in range(num_heads_v):
                channels_to_prune_vo = math.ceil(head_dim_v * prune_amount)
                channels_to_prune_vo = min(channels_to_prune_vo, head_dim_v - 1)

                head_vo_scores = vo_scores[head]
                _, prune_indices_vo = torch.topk(head_vo_scores, channels_to_prune_vo, largest=False)
                prune_indices_vo = prune_indices_vo.tolist()

                if head not in plan[layer_idx]:
                    plan[layer_idx][head] = {
                        'remove_channels_qk': [],
                        'remove_channels_vo': prune_indices_vo,
                        'remove_head': []
                    }
                else:
                    plan[layer_idx][head]['remove_channels_vo'] = prune_indices_vo
                    
        elif pruning_strategy == PruningStrategy.MULTI_HEAD_SAME_CHANNEL:
            # CHANNEL pruning across heads - RoPE-aware
            # Average scores across heads (already RoPE-aggregated)
            if importance_strategy == ImportanceStrategy.FISHER_INFORMATION:
                qk_channel_scores = torch.sum(qk_scores, dim=0)
                vo_channel_scores = torch.sum(vo_scores, dim=0)
            else:
                qk_channel_scores = torch.mean(qk_scores, dim=0)
                vo_channel_scores = torch.mean(vo_scores, dim=0)

            channels_to_prune_qk = math.ceil(head_dim_q * prune_amount)

            if has_rope:
                channels_to_prune_qk = (channels_to_prune_qk // 2) * 2
                channels_to_prune_qk = min(channels_to_prune_qk, head_dim_q - 2)
                P_q = len(head_qk_scores) // 2
                pair_scores = head_qk_scores[:P_q].clone() + head_qk_scores[P_q:].clone()
                k_pairs = channels_to_prune_qk // 2
                _, pair_idx = torch.topk(pair_scores, k_pairs, largest=False)
                prune_indices_qk = torch.cat([pair_idx, pair_idx + P_q]).sort().values.tolist()
            else:
                channels_to_prune_qk = min(channels_to_prune_qk, head_dim_q - 1)
                _, prune_indices_qk = torch.topk(head_qk_scores, channels_to_prune_qk, largest=False)
                prune_indices_qk = prune_indices_qk.tolist()
            channels_to_prune_vo = math.ceil(head_dim_v * prune_amount)
            channels_to_prune_vo = min(channels_to_prune_vo, head_dim_v - 1)
            
            _, prune_indices_vo = torch.topk(vo_channel_scores, channels_to_prune_vo, largest=False)
            
            plan[layer_idx] = {
                'remove_heads': [],
                'remove_channels_qk': prune_indices_qk,
                'remove_channels_vo': prune_indices_vo
            }
            
        elif pruning_strategy == PruningStrategy.MULTI_HEAD_ENTIRE_HEAD:
            # HEAD pruning - NO RoPE handling needed (RoPE doesn't affect head dimension)
            heads_to_prune = math.ceil(num_heads_q * prune_amount)
            heads_to_prune = min(heads_to_prune, num_heads_q - 1)
            
            _, prune_indices = torch.topk(head_scores, heads_to_prune, largest=False)
            plan[layer_idx] = {
                'remove_heads': prune_indices.tolist(),
                'remove_channels_qk': [],
                'remove_channels_vo': []
            }
    
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
        # Use RoPE-aware importance (channel scores are already aggregated)
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
        
        #check if the layer implements rotary positional embedding
        has_rope = ((hasattr(sa, 'rope') and sa.rope is not None) or
                    (hasattr(sa, 'rotary_emb') and sa.rotary_emb is not None) or
                    (hasattr(sa, 'rope_cos') and sa.rope_cos is not None)
                    )

        # QK bundles
        if has_rope:
            P_q = head_dim_q // 2
            max_qk_per_head = P_q - 1
        else:
            max_qk_per_head = head_dim_q - 1
 
        for k in range(1, max_qk_per_head + 1):
            total_cost = 0
            for head_idx in range(num_heads_q):
                head_scores = qk_scores[head_idx].clone()  # never mutate originals
 
                if has_rope:
                    # Combine half-split partners into a single pair score,
                    # then select the k cheapest pairs atomically.
                    pair_scores = head_scores[:P_q] + head_scores[P_q:]
                    k_smallest_costs, _ = torch.topk(pair_scores, k, largest=False)
                    total_cost += k_smallest_costs.sum().item()
                    channels_this_k = k * 2
                else:
                    k_smallest_costs, _ = torch.topk(head_scores, k, largest=False)
                    total_cost += k_smallest_costs.sum().item()
                    channels_this_k = k
            # print(total_cost / (channels_this_k * num_heads_q))
            qk_bundles.append({
                'channels_per_head': channels_this_k,
                'total_channels': channels_this_k * num_heads_q,
                'total_cost': total_cost,
                'cost_per_channel': total_cost / (channels_this_k * num_heads_q)
            })
        
        # VO bundles
        max_vo_per_head = head_dim_v - 1
        for k in range(1, max_vo_per_head + 1):
            total_cost = 0
            for head_idx in range(num_heads_v):
                head_scores = vo_scores[head_idx]
                k_smallest_costs, k_indices = torch.topk(head_scores, k, largest=False)
                
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
    if has_rope:
        qk_budget = (qk_budget // 2) * 2  # keep even for RoPE pairs
    vo_budget = int(total_vo_channels * prune_amount)
    # print(qk_budget)
    # print(total_qk_channels)
    # Step 3: Solve optimization problem using greedy approach
    qk_assignment = _solve_channel_assignment(layer_bundles, qk_budget, 'qk')
    vo_assignment = _solve_channel_assignment(layer_bundles, vo_budget, 'vo')
    
    # Step 4: Convert assignment to pruning plan
    plan = defaultdict(lambda: defaultdict(dict))
    
    for layer_idx, info in enumerate(layer_info):
        qk_channels_per_head = qk_assignment.get(layer_idx, 0)
        vo_channels_per_head = vo_assignment.get(layer_idx, 0)
        # print(qk_channels_per_head)
        
        num_heads_q = info['num_heads_q']
        num_heads_v = info['num_heads_v']
        qk_scores = info['qk_scores']
        vo_scores = info['vo_scores']
        
        # Get RoPE pairs
        head_dim_q = info['head_dim_q']
        head_dim_v = info['head_dim_v']
        sa = layers[layer_idx]
        
        # For each head, find the specific channels to prune
        max_heads = max(num_heads_q, num_heads_v)

        #check if the layer implements rotary positional embedding
        has_rope = ((hasattr(sa, 'rope') and sa.rope is not None) or
                    (hasattr(sa, 'rotary_emb') and sa.rotary_emb is not None) or
                    (hasattr(sa, 'rope_cos') and sa.rope_cos is not None)
                    )  
        qk_indices_per_head = []
        if has_rope:
            P_q = head_dim_q // 2
            k_pairs = qk_channels_per_head // 2  # channels_per_head is already even
            for h in range(num_heads_q):
                if k_pairs > 0:
                    scores_h = qk_scores[h].clone()
                    pair_scores = scores_h[:P_q] + scores_h[P_q:]
                    _, pair_idx = torch.topk(pair_scores, k_pairs, largest=False)
                    pidx = torch.cat([pair_idx, pair_idx + P_q]).sort().values.tolist()
                else:
                    pidx = []
                qk_indices_per_head.append(pidx)
        else:
            for h in range(num_heads_q):
                if qk_channels_per_head > 0:
                    scores_h = qk_scores[h].clone()
                    _, pidx = torch.topk(scores_h, qk_channels_per_head, largest=False)
                    pidx = pidx.tolist()
                else:
                    pidx = []
                qk_indices_per_head.append(pidx)

        uniform_qk = min((len(p) for p in qk_indices_per_head), default=0)
        if has_rope:
            uniform_qk = (uniform_qk // 2) * 2

        # Collect VO indices per head with same uniform-count enforcement
        vo_indices_per_head = []
        for h in range(num_heads_v):
            if vo_channels_per_head > 0:
                scores_h = vo_scores[h].clone()
                _, pidx = torch.topk(scores_h, vo_channels_per_head, largest=False)
                pidx = pidx.tolist()
            else:
                pidx = []
            vo_indices_per_head.append(pidx)

        uniform_vo = min((len(p) for p in vo_indices_per_head), default=0)

        for head_idx in range(max_heads):
            plan[layer_idx][head_idx]['remove_head'] = []

            if head_idx < num_heads_q:
                plan[layer_idx][head_idx]['remove_channels_qk'] = (
                    qk_indices_per_head[head_idx][:uniform_qk]
                )
            else:
                plan[layer_idx][head_idx]['remove_channels_qk'] = []

            if head_idx < num_heads_v:
                plan[layer_idx][head_idx]['remove_channels_vo'] = (
                    vo_indices_per_head[head_idx][:uniform_vo]
                )
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
            # Marginal from 0->1: previous state is (0 cost, 0 channels),
            # so marginal cost == total cost. Use same formula as subsequent moves
            # to keep greedy comparisons consistent.
            marginal_channels = first_bundle['total_channels']
            marginal_loss = first_bundle['total_cost']
            moves.append({
                'layer_idx': layer_idx,
                'marginal_loss': marginal_loss,
                'marginal_channels': marginal_channels,
                'new_channels_per_head': first_bundle['channels_per_head'],
                'bundle_index': 1,
                'loss_per_channel': marginal_loss / marginal_channels if marginal_channels > 0 else float('inf')
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
        
        layer_data = next(ld for ld in layer_bundles if ld['layer_idx'] == layer_idx)
        bundles = layer_data[f'{bundle_type}_bundles']
        next_bundle_idx = best_move['bundle_index'] + 1
        
        if next_bundle_idx <= len(bundles):
            next_bundle = bundles[next_bundle_idx - 1]
            current_bundle = bundles[best_move['bundle_index'] - 1]
            
            marginal_loss = next_bundle['total_cost'] - current_bundle['total_cost']
            marginal_channels = next_bundle['total_channels'] - current_bundle['total_channels']
            
            moves.append({
                'layer_idx': layer_idx,
                'marginal_loss': marginal_loss,
                'marginal_channels': marginal_channels,
                'new_channels_per_head': next_bundle['channels_per_head'],
                'bundle_index': next_bundle_idx,
                'loss_per_channel': marginal_loss / marginal_channels if marginal_channels > 0 else float('inf')
            })
    
    return assignment

def structured_prune_model(model, prune_amount: float, 
                            threshold_strategy: ThresholdStrategy = ThresholdStrategy.GLOBAL,
                           pruning_strategy: PruningStrategy = PruningStrategy.MULTI_HEAD_SAME_CHANNEL, 
                           importance_strategy: ImportanceStrategy = ImportanceStrategy.MAGNITUDE,
                           fisher_data_loader=None, fisher_criterion=None, fisher_config=None, 
                           order: int = 2, device='cuda', remove_empty_layers: bool = True, 
                           teacher = None, class_loss_weight=None, patch_layers=None, processor= None, layerwise=False):
    """
    Main function to perform structured pruning with configurable importance strategies.
    with rope-awareness!
    
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
    # Compute Fisher Information if needed
    fisher_info = None
    if importance_strategy == ImportanceStrategy.FISHER_INFORMATION:
        if fisher_data_loader is None:
            raise ValueError("fisher_data_loader is required for Fisher Information strategy")
        
        if fisher_config is None:
            fisher_config = FisherConfig()
        
        model.zero_grad()
        for p in model.parameters():
            if p.grad is not None:
                p.grad = None
        
        # Force cleanup before Fisher computation
        torch.cuda.empty_cache()
        
        print("Computing Fisher Information for importance-based pruning...")
        if teacher is None:
            fisher_info = compute_fisher_information(model, fisher_data_loader, fisher_criterion, fisher_config, device)
        elif class_loss_weight is not None:
            fisher_info = compute_fisher_ss(model, teacher, fisher_data_loader, device, fisher_config, class_loss_weight, patch_layers)
        elif layerwise:
            fisher_info = compute_fisher_ss_llm(model, teacher, fisher_data_loader, device, fisher_config)
        else:
            fisher_info = compute_fisher_distill(model, teacher, processor, fisher_data_loader, device, fisher_config)
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

    # Get layers
    layers = get_layers(model)
    # Determine pruning plan using the specified importance strategy
    pruning_plan = determine_pruning_plan(layers, prune_amount, pruning_strategy, threshold_strategy, importance_strategy, fisher_info, order)
    
    if fisher_info is not None:
        del fisher_info
    
    # Apply structured pruning (reuse existing function)
    pruned_model = apply_structured_pruning(model, pruning_plan, remove_empty_layers)
    
    del pruning_plan, layers, model
    
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
    current_layers = get_layers(pruned_model)
    
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