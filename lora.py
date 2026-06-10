import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Callable
from torch.utils.checkpoint import checkpoint
import math
import copy
import gc

class MaskedLoRALinear(nn.Module):
    """
    Drop-in replacement for a frozen nn.Linear that adds a low-rank
    trainable delta:  y = base(x) + scaling * (B @ A) x

    Used inside an already mask-wrapped model. The base layer is frozen;
    only A and B are trainable. Output-side mask multiplication elsewhere
    in the model handles structural pruning of channels.
    """

    def __init__(self, base: nn.Linear, rank: int = 16, alpha: float = 32.0):
        super().__init__()
        self.base = base
        for p in base.parameters():
            p.requires_grad = False

        self.rank = rank
        self.scaling = alpha / rank

        in_f, out_f = base.in_features, base.out_features
        dtype = base.weight.dtype
        device = base.weight.device

        self.A = nn.Parameter(torch.zeros(rank, in_f, dtype=dtype, device=device))
        self.B = nn.Parameter(torch.zeros(out_f, rank, dtype=dtype, device=device))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        nn.init.zeros_(self.B)

    def forward(self, x):
        # F.linear applies y = x A^T, so chaining A then B gives
        # (x @ A^T) @ B^T which equals (B @ A) x — the standard LoRA delta.
        return self.base(x) + F.linear(F.linear(x, self.A), self.B) * self.scaling


# Default targets: every Linear inside a Vicuna decoder block. Embedding /
# lm_head are deliberately excluded — they're large, not pruning-relevant,
# and rarely benefit from LoRA in pruning-recovery setups.
DEFAULT_TARGETS = (
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
)


def inject_lora_into_masked_model(
    masked_model: nn.Module,
    rank: int = 16,
    alpha: float = 32.0,
    target_names=DEFAULT_TARGETS,
) -> List[nn.Parameter]:
    """
    Replace every Linear whose attribute name is in target_names with a
    MaskedLoRALinear, in-place. Returns the list of newly-trainable
    LoRA params (A and B for each adapter).

    Call this AFTER wrap_model_with_masks and AFTER the model is on its
    final device, but BEFORE the optimizer add_param_group call.
    """
    lora_params: List[nn.Parameter] = []
    for module in masked_model.modules():
        for child_name, child in list(module.named_children()):
            if child_name in target_names and isinstance(child, nn.Linear):
                lora = MaskedLoRALinear(child, rank=rank, alpha=alpha)
                setattr(module, child_name, lora)
                lora_params.extend([lora.A, lora.B])
    return lora_params


def merge_lora_into_masked_base(masked_model: nn.Module) -> int:
    """
    Bake every MaskedLoRALinear's delta into its base Linear weight, then
    replace the adapter with the (now updated) base Linear. Returns the
    number of adapters merged.

    Run before saving the joint-trained state if you want a single
    state_dict that loads cleanly into a fresh wrap_model_with_masks
    pipeline without needing the LoRA module class at load time.
    """
    n_merged = 0
    for module in masked_model.modules():
        for child_name, child in list(module.named_children()):
            if isinstance(child, MaskedLoRALinear):
                with torch.no_grad():
                    delta = (child.B @ child.A) * child.scaling
                    child.base.weight.data.add_(
                        delta.to(child.base.weight.dtype)
                    )
                setattr(module, child_name, child.base)
                n_merged += 1
    return n_merged


def count_masked_lora_params(masked_model: nn.Module) -> int:
    """Total parameters across all LoRA adapters (for logging)."""
    n = 0
    for module in masked_model.modules():
        if isinstance(module, MaskedLoRALinear):
            n += module.A.numel() + module.B.numel()
    return n

def gradient_mask_hook(mask):
    """
    Returns a hook function that masks the gradient of a linear layer.
    `mask` should be broadcastable to the weight shape.
    """
    def hook(grad):
        return grad * mask.to(grad.device)
    return hook

class LoRALayer(nn.Module):
    """
    Standard LoRA applied to one Linear layer (W x).
    Mask is applied ONLY to B (output channels).
    """
    def __init__(self, original: nn.Linear, rank=16, alpha=16.0, use_checkpoint=False):
        super().__init__()
        self.original = original
        self.rank = rank
        self.scaling = alpha / rank
        self.use_checkpoint = use_checkpoint

        for p in original.parameters():
            p.requires_grad = False

        in_f = original.in_features
        out_f = original.out_features

        self.A = nn.Parameter(torch.zeros(rank, in_f))
        self.B = nn.Parameter(torch.zeros(out_f, rank))

        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        nn.init.zeros_(self.B)

    def set_mask(self, mask_in: Optional[torch.Tensor] = None, mask_out: Optional[torch.Tensor]= None):
        """
        mask: shape [out_features], dtype bool or {0,1}
        """
        
        if mask_in is not None:
            mask = mask_in.view(-1, 1)  # broadcast over rank
            mask = mask.to(self.B.device).float()
            # Zero masked rows immediately
            with torch.no_grad():
                self.B.mul_(mask)
            
            if hasattr(self.B, "_grad_hook_handle"):
                self.B._grad_hook_handle.remove()
            # Block gradients for masked rows
            self.B._grad_hook_handle = self.B.register_hook(gradient_mask_hook(mask))
        else:
            mask = mask_out.view(1, -1)
            
            mask = mask.to(self.A.device).float()
            # Zero masked rows immediately
            with torch.no_grad():
                self.A.mul_(mask)

            if hasattr(self.A, "_grad_hook_handle"):
                self.A._grad_hook_handle.remove()

            self.A._grad_hook_handle = self.A.register_hook(gradient_mask_hook(mask))
        

    def forward(self, x):
        def lora_forward(x):
            base = self.original(x)
            delta = F.linear(x, self.B @ self.A) * self.scaling
            return base + delta
        if self.use_checkpoint and self.training:
            return checkpoint(lora_forward, x, use_reentrant=False)
        return lora_forward(x)

class LoRAPackedProj(nn.Module):
    """
    LoRA for packed QKV projections.
    Masks are applied ONLY to B_q, B_k, B_v.
    """
    def __init__(self, original, q_dim, k_dim, v_dim, rank=16, alpha=16, use_checkpoint=False):
        super().__init__()
        self.original = original
        self.rank = rank
        self.scaling = alpha / rank
        self.use_checkpoint = use_checkpoint

        for p in original.parameters():
            p.requires_grad = False

        self.in_features = original.in_features

        self.A_q = nn.Parameter(torch.zeros(rank, self.in_features))
        self.B_q = nn.Parameter(torch.zeros(q_dim, rank))

        self.A_k = nn.Parameter(torch.zeros(rank, self.in_features))
        self.B_k = nn.Parameter(torch.zeros(k_dim, rank))

        self.A_v = nn.Parameter(torch.zeros(rank, self.in_features))
        self.B_v = nn.Parameter(torch.zeros(v_dim, rank))

        for A in (self.A_q, self.A_k, self.A_v):
            nn.init.kaiming_uniform_(A, a=math.sqrt(5))
        for B in (self.B_q, self.B_k, self.B_v):
            nn.init.zeros_(B)

    def _set_mask(self, B, mask):
        mask = mask.to(B.device).float().view(-1, 1)

        with torch.no_grad():
            B.mul_(mask)

        # Block gradients for masked rows
        if hasattr(B, "_grad_hook_handle"):
            B._grad_hook_handle.remove()
        B._grad_hook_handle = B.register_hook(gradient_mask_hook(mask))

    def set_masks(self, mask_q=None, mask_k=None, mask_v=None):
        if mask_q is not None:
            self._set_mask(self.B_q, mask_q)
        if mask_k is not None:
            self._set_mask(self.B_k, mask_k)
        if mask_v is not None:
            self._set_mask(self.B_v, mask_v)

    def forward(self, x):
        def lora_forward(x):
            base = self.original(x)
            dq = (x @ self.A_q.T) @ self.B_q.T * self.scaling
            dk = (x @ self.A_k.T) @ self.B_k.T * self.scaling
            dv = (x @ self.A_v.T) @ self.B_v.T * self.scaling
            return base + torch.cat([dq, dk, dv], dim=-1)
        if self.use_checkpoint and self.training:
            return checkpoint(lora_forward, x, use_reentrant=False)
        return lora_forward(x)



class LoRAWrapper(nn.Module):
    """
    Wraps a transformer model, replacing Q/K/V or packed QKV with LoRA layers.
    """
    PACKED_NAMES = ["in_proj_weight", "qkv", "qkv_proj", "Wqkv"]

    def __init__(self, model, rank=16, alpha=16.0, target_modules=None, use_checkpoint=False):
        super().__init__()
        self.model = model
        self.rank = rank
        self.alpha = alpha
        self.target_modules = target_modules
        self.use_checkpoint = use_checkpoint
        self._apply_lora(model)

    def _should_apply_lora(self, full_name):
        """Check if LoRA should be applied to this module based on target_modules."""
        if self.target_modules is None:
            return True
        return any(target in full_name for target in self.target_modules)

    def __getattr__(self, name):
        # Called only if attribute not found normally
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.model, name)

    def _apply_lora(self, module, prefix=""):
        for name, child in list(module.named_children()):
            full = f"{prefix}.{name}" if prefix else name

            if not self._should_apply_lora(full):
                # Still recurse to check children
                self._apply_lora(child, full)
                continue

            # Packed projection layer
            if any(k in name for k in self.PACKED_NAMES) and isinstance(child, nn.Linear):
                #pass the mask as well (if present)
                #get the dimensions from the father layer
                #if the layer is packed projection, the father always has the dimension information
                embed_dim, num_heads, head_dims = module.get_attn_shapes()

                qk_head_dim, vo_head_dim, = head_dims

                qk_dim = num_heads * qk_head_dim
                vo_dim = num_heads * vo_head_dim

                lora = LoRAPackedProj(
                    original=child,
                    q_dim=qk_dim,
                    k_dim=qk_dim,
                    v_dim=vo_dim,
                    rank=self.rank,
                    alpha=self.alpha,
                    use_checkpoint=self.use_checkpoint,
                )
                if hasattr(module, '_grad_hook_handle'):
                    lora.set_masks(module.qk_channel_mask, module.qk_channel_mask, module.v_channel_mask)
                setattr(module, name, lora)
                continue

            # Any other linear layer
            elif isinstance(child, nn.Linear):
                #pass the mask as well (if present)
                lora = LoRALayer(child, rank=self.rank, alpha=self.alpha, use_checkpoint=self.use_checkpoint)

                # Replace the module FIRST
                setattr(module, name, lora)
                if hasattr(module, '_grad_hook_handle'):
                    # Now configure the LoRA layer using buffers from the parent
                    if name in {"q_proj", "k_proj"}:
                        lora.set_mask(mask_in = module.qk_channel_mask)

                    elif name == "v_proj":
                        lora.set_mask(mask_in = module.v_channel_mask)

                    elif name == "out_proj":
                        lora.set_mask(mask_out = module.v_channel_mask)
                    continue

            # Recurse
            self._apply_lora(child, full)

    def forward(self, *a, **k):
        return self.model(*a, **k)
        
    def get_lora_parameters(self) -> List[nn.Parameter]:
        """Get only LoRA parameters for optimization."""
        lora_params = []
        for module in self.modules():
            if isinstance(module, LoRALayer):
                lora_params.extend([
                    module.A, module.B
                ])
            elif isinstance(module, LoRAPackedProj):
                lora_params.extend([
                    module.A_q, module.B_q,
                    module.A_k, module.B_k,
                    module.A_v, module.B_v,
                ])
        return lora_params
        
    def save_lora_weights(self, path: str):
        """Save only LoRA weights."""
        lora_state_dict = {}
        for name, module in self.named_modules():
            if isinstance(module, LoRALayer):
                lora_state_dict[f"{name}.A"] = module.A.data
                lora_state_dict[f"{name}.B"] = module.B.data
            elif isinstance(module, LoRAPackedProj):
                lora_state_dict[f"{name}.A_q"] = module.A_q.data
                lora_state_dict[f"{name}.B_q"] = module.B_q.data
                lora_state_dict[f"{name}.A_k"] = module.A_k.data
                lora_state_dict[f"{name}.B_k"] = module.B_k.data
                lora_state_dict[f"{name}.A_v"] = module.A_v.data
                lora_state_dict[f"{name}.B_v"] = module.B_v.data
        torch.save(lora_state_dict, path)
        
    def load_lora_weights(self, path: str):
        """Load LoRA weights."""
        lora_state_dict = torch.load(path)
        self.load_state_dict(lora_state_dict, strict=False)

def merge_lora_in_place(wrapped_parent):
    """
    Traverse both the wrapped model (with LoRA) and original model (without LoRA) in parallel.
    Replace LoRA layers in wrapped model with merged original layers.
    """
    for name in list(wrapped_parent._modules.keys()):
        wrapped_child = wrapped_parent._modules[name]
        
        #Single linear layer
        if isinstance(wrapped_child, LoRALayer):
            # Handle LoRA Linear layers
            original_layer = wrapped_child.original
            
            # Merge weights IN-PLACE on original layer
            with torch.no_grad():
                
                lora_delta = (wrapped_child.B @ wrapped_child.A) * wrapped_child.scaling
                original_layer.weight.data.add_(lora_delta)
            
            # Replace LoRA wrapper with original in the wrapped model
            setattr(wrapped_parent, name, original_layer)
        #Packed projection
        elif isinstance(wrapped_child, LoRAPackedProj):
            original_layer = wrapped_child.original
            
            with torch.no_grad():
                
                # Calculate LoRA deltas
                q_delta = (wrapped_child.B_q @ wrapped_child.A_q) * wrapped_child.scaling
        
                k_delta = (wrapped_child.B_k @ wrapped_child.A_k) * wrapped_child.scaling

                v_delta = (wrapped_child.B_v @ wrapped_child.A_v) * wrapped_child.scaling
                
                original_layer.weight.data.add_(torch.cat([q_delta, k_delta, v_delta], dim=0))
                
            # Replace LoRA wrapper with original in the wrapped model
            setattr(wrapped_parent, name, original_layer)
            
        else:
            # Recursively process children
            if hasattr(wrapped_child, '_modules'):
                merge_lora_in_place(wrapped_child)

def merge_lora_weights(wrapper):
    """
    Merge LoRA weights into the frozen base model and remove LoRA parameters.
    """
    # Handle DDP wrapper
    if isinstance(wrapper, torch.nn.DataParallel) or isinstance(wrapper, torch.nn.parallel.DistributedDataParallel):
        wrapper = wrapper.module
    
    # Check if this is actually a LoRAWrapper
    if not isinstance(wrapper, LoRAWrapper):
        # Not a LoRA model, return as-is
        return wrapper

    #merge in place to avoid leaking parameters
    merge_lora_in_place(wrapper)

    result_model = wrapper.model
    
    # Clean up wrapper references
    wrapper.model = None
    del wrapper
    
    # Force cleanup
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    gc.collect()
    
    return result_model