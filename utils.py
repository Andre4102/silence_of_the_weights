import os
import gc
import io
import time
import math
import random
import time
import datetime
import psutil
import numpy as np
import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import seaborn as sns
import json
import argparse
from fvcore.nn import FlopCountAnalysis
from fvcore.nn.jit_handles import get_shape

try:
    pass # bypassed pynvml to prevent AMD crash
except ImportError:
    pass

IS_ROCM: bool = torch.version.hip is not None
# On both CUDA and ROCm, PyTorch surfaces GPUs through the torch.cuda namespace,
# so "cuda" is the correct device/autocast string in both cases.
ACCELERATOR: str = "cuda" if torch.cuda.is_available() else "cpu"

def init_distributed():
    # ── Detect launcher ──────────────────────────────────────────────────────
    # torchrun sets RANK/LOCAL_RANK/WORLD_SIZE before spawning each process.
    # SLURM srun sets SLURM_PROCID/SLURM_LOCALID/SLURM_NTASKS instead.
    # If neither is present we are running single-process with no distributed
    # context (e.g. a quick debug run), so bail out early.

    torchrun_active = "RANK" in os.environ and "WORLD_SIZE" in os.environ
    slurm_active    = "SLURM_PROCID" in os.environ

    if not torchrun_active and not slurm_active:
        return 0, 1, 0   # single-process, no dist

    if torchrun_active:
        # torchrun already wrote RANK / LOCAL_RANK / WORLD_SIZE into the env.
        rank       = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
    else:
        # SLURM srun path: populate the env vars that dist.init_process_group
        # expects when init_method="env://".
        rank = int(os.environ.get("SLURM_PROCID", 0))
        local_rank = int(os.environ.get("SLURM_LOCALID", 0))
        world_size = int(os.environ.get("SLURM_NTASKS", 1))
        os.environ["RANK"]       = str(rank)
        os.environ["LOCAL_RANK"] = str(local_rank)
        os.environ["WORLD_SIZE"] = str(world_size)

    # ── Device assignment for this process ───────────────────────────────────
    # Each process uses local_rank as its device index (rank 0 -> cuda:0, etc).
    # Every process MUST see ALL GPUs (CUDA: device_count() == ntasks-per-node;
    # ROCm: all visible GCDs) so that local_rank correctly identifies each
    # process's device. Do NOT restrict CUDA_VISIBLE_DEVICES / ROCR_VISIBLE_DEVICES
    # and do NOT pass --gpu-bind in the SLURM script: binding renumbers each
    # task's GPU to index 0, so set_device(local_rank) for local_rank > 0 then
    # lands on an invalid device and NCCL segfaults during init_process_group.
    # Device pinning is owned entirely by torch.cuda.set_device(local_rank) below.

    # ── Validate ─────────────────────────────────────────────────────────────
    n_visible = torch.cuda.device_count()
    if n_visible == 0:
        raise RuntimeError(
            "torch.cuda.device_count()==0. "
            "Check your ROCm/CUDA installation and GPU allocation."
        )
    if local_rank >= n_visible:
        raise RuntimeError(
            f"local_rank={local_rank} but torch.cuda.device_count()={n_visible}. "
            f"Check {'ROCR_VISIBLE_DEVICES' if IS_ROCM else 'CUDA_VISIBLE_DEVICES'}="
            f"{os.environ.get('ROCR_VISIBLE_DEVICES' if IS_ROCM else 'CUDA_VISIBLE_DEVICES', 'unset')}."
        )

    # ── Initialise process group ──────────────────────────────────────────────
    # rccl is AMD's NCCL equivalent and is the correct backend for ROCm.
    # Fall back to nccl for CUDA, or if rccl is somehow not registered.
    if IS_ROCM:
        backend = "rccl" if "rccl" in torch.distributed.Backend.default_device_backend_map else "nccl"
    else:
        backend = "nccl"

    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    dist.init_process_group(
        backend=backend,
        init_method="env://",
        timeout=datetime.timedelta(hours=2),
    )
    return rank, world_size, local_rank


def is_main(rank: int) -> bool: return rank == 0
def barrier(world_size: int):
    if world_size > 1: dist.barrier()
def cleanup_distributed(world_size: int):
    if world_size > 1 and dist.is_initialized(): dist.destroy_process_group()

def log_vram(tag: str, rank: int, local_rank: int):
    """Safely logs VRAM/HBM usage across all processes without synchronization hangs."""
    if not torch.cuda.is_available(): return
    allocated = torch.cuda.memory_allocated(local_rank) / (1024 ** 3)
    reserved  = torch.cuda.memory_reserved(local_rank)  / (1024 ** 3)
    label = "HBM" if IS_ROCM else "VRAM"
    if is_main(rank):
        print(f"  [{label}] {tag:.<35} Allocated: {allocated:>5.2f} GB | Reserved: {reserved:>5.2f} GB")


# ─────────────────────────────────────────────────────────────────────────────
# Model Interface Helpers & Sparsity Tracking
# ─────────────────────────────────────────────────────────────────────────────

def count_parameters(model):
    return sum(p.numel() for p in model.parameters())

# ============================================================================
# General Sparsity Tracking Utilities
# ============================================================================

def capture_original_params(model, get_layers_fn=lambda m: m.model.layers, get_blocks_fn=lambda layer: (layer.self_attn, layer.mlp)):
    """Captures the parameter count per layer to calculate accurate sparsity later."""
    original_config = {}
    for i, layer in enumerate(get_layers_fn(model)):
        sa, mlp = get_blocks_fn(layer)
        
        attn_params = 0
        if hasattr(sa, "get_attn_weights"):
            attn_ws, o_w, _, _ = sa.get_attn_weights()
            attn_params += o_w.numel() + (sum(w.numel() for w in attn_ws) if isinstance(attn_ws, list) else attn_ws.numel())
            
        mlp_params = 0
        if hasattr(mlp, "get_mlp_weights"):
            mlp_ws, _, down_w, _ = mlp.get_mlp_weights()
            mlp_params += down_w.numel() + (sum(w.numel() for w in mlp_ws) if isinstance(mlp_ws, list) else mlp_ws.numel())
            
        original_config[i] = {
            "attn_params": attn_params,
            "mlp_params": mlp_params
        }
    return original_config

def get_model_sparsity(model, original_config=None, get_layers_fn=lambda m: m.model.layers, get_blocks_fn=lambda layer: (layer.self_attn, layer.mlp)):
    """
    Calculates true sparsity by comparing current parameter count against the captured baseline.
    Returns: A dictionary with overall sparsity stats and layer-wise sparsity stats.
    """
    layers_stats =[]
    overall_base = 0
    overall_curr = 0
    
    for i, layer in enumerate(get_layers_fn(model)):
        sa, mlp = get_blocks_fn(layer)
        
        curr_attn_params = 0
        if hasattr(sa, "get_attn_weights"):
            attn_ws, o_w, _, _ = sa.get_attn_weights()
            curr_attn_params += o_w.numel() + (sum(w.numel() for w in attn_ws) if isinstance(attn_ws, list) else attn_ws.numel())
            
        curr_mlp_params = 0
        if hasattr(mlp, "get_mlp_weights"):
            mlp_ws, _, down_w, _ = mlp.get_mlp_weights()
            curr_mlp_params += down_w.numel() + (sum(w.numel() for w in mlp_ws) if isinstance(mlp_ws, list) else mlp_ws.numel())
            
        curr_total = curr_attn_params + curr_mlp_params
        
        base_attn_params = original_config[i]["attn_params"] if original_config and i in original_config else curr_attn_params
        base_mlp_params = original_config[i]["mlp_params"] if original_config and i in original_config else curr_mlp_params
        base_total = base_attn_params + base_mlp_params
        
        sparsity = 1.0 - (curr_total / base_total) if base_total > 0 else 0.0
        attn_sparsity = 1.0 - (curr_attn_params / base_attn_params) if base_attn_params > 0 else 0.0
        mlp_sparsity = 1.0 - (curr_mlp_params / base_mlp_params) if base_mlp_params > 0 else 0.0
        
        layers_stats.append({
            "layer": i,
            "sparsity": sparsity,
            "attn_sparsity": attn_sparsity,
            "mlp_sparsity": mlp_sparsity,
            "curr_params": curr_total,
            "base_params": base_total
        })
        
        overall_base += base_total
        overall_curr += curr_total
        
    overall_sparsity = 1.0 - (overall_curr / overall_base) if overall_base > 0 else 0.0
    
    return {
        "overall": {
            "sparsity": overall_sparsity,
            "curr_params": overall_curr,
            "base_params": overall_base
        },
        "layers": layers_stats
    }


# ─────────────────────────────────────────────────────────────────────────────
# TensorBoard Logging (Heatmaps & Sparsity)
# ─────────────────────────────────────────────────────────────────────────────

def init_index_mapping(model, get_layers_fn, get_blocks_fn):
    """Creates the baseline index mapping for Iteration 0.
 
    Each layer entry holds, in ORIGINAL index space:
        heads:            list of surviving original head indices
        qk, vo:           dict[orig_h] -> list of surviving original channel indices
        mlp:              list of surviving original MLP neuron indices
        orig_num_heads, orig_hd_qk, orig_hd_vo, orig_hidden_dim: dimensions of
                          the un-pruned model (unchanged across iterations).
    """
    mapping = {}
    for i, layer in enumerate(get_layers_fn(model)):
        sa, mlp = get_blocks_fn(layer)
        entry = {}
 
        if hasattr(sa, "get_attn_shapes"):
            _, num_heads, head_dims = sa.get_attn_shapes()
            hd_qk, hd_vo = head_dims
            entry.update({
                "heads": list(range(num_heads)),
                "qk": {h: list(range(hd_qk)) for h in range(num_heads)},
                "vo": {h: list(range(hd_vo)) for h in range(num_heads)},
                "orig_num_heads": num_heads,
                "orig_hd_qk": hd_qk,
                "orig_hd_vo": hd_vo,
            })
 
        if hasattr(mlp, "get_mlp_shapes"):
            _, hidden_dim, _ = mlp.get_mlp_shapes()
            entry.update({
                "mlp": list(range(hidden_dim)),
                "orig_hidden_dim": hidden_dim,
            })
 
        mapping[i] = entry
    return mapping
 
 
def update_index_mapping(mapping, plans):
    """Updates the mapping using the structural plan from the pruner.
 
    `plans[i]['attn']` and `plans[i]['mlp']` contain indices in the CURRENT
    (already-pruned) model's coordinate system. This function remaps them
    back to the ORIGINAL coordinate system so the mapping always tracks
    which original units have survived.
    """
    for i, plan in plans.items():
 
        # --- Attention: unchanged logic ---
        if 'attn' in plan:
            kept_heads, kept_channels_qk, kept_channels_vo = plan['attn']
 
            # kept_heads: indices into the CURRENT surviving head list
            new_heads = [mapping[i]["heads"][h] for h in kept_heads]
 
            new_qk, new_vo = {}, {}
            for current_h in kept_heads:
                orig_h = mapping[i]["heads"][current_h]
                # kept_channels_*[current_h]: indices into the CURRENT surviving
                # channel list for that head
                new_qk[orig_h] = [mapping[i]["qk"][orig_h][c]
                                  for c in kept_channels_qk[current_h]]
                new_vo[orig_h] = [mapping[i]["vo"][orig_h][c]
                                  for c in kept_channels_vo[current_h]]
 
            mapping[i]["heads"] = new_heads
            mapping[i]["qk"] = new_qk
            mapping[i]["vo"] = new_vo
 
        # --- MLP: new tracking ---
        if 'mlp' in plan and "mlp" in mapping[i]:
            kept_neurons = plan['mlp']  # indices into CURRENT surviving neuron list
            mapping[i]["mlp"] = [mapping[i]["mlp"][n] for n in kept_neurons]
 
    return mapping

def save_mapping(mapping, path):
    with open(path, 'w') as f: json.dump(mapping, f)

def load_mapping(path):
    with open(path, 'r') as f: raw = json.load(f)
    mapping = {}
    for layer_k, layer_v in raw.items():
        l_idx = int(layer_k)
        mapping[l_idx] = {
            "heads": layer_v["heads"],
            "qk": {int(k): v for k, v in layer_v["qk"].items()},
            "vo": {int(k): v for k, v in layer_v["vo"].items()},
            "orig_num_heads": layer_v["orig_num_heads"],
            "orig_hd_qk": layer_v["orig_hd_qk"],
            "orig_hd_vo": layer_v["orig_hd_vo"]
        }
    return mapping

def log_structural_shapes(writer: SummaryWriter, model, iteration: int, get_layers_fn, get_blocks_fn):
    """Logs the exact number of heads, channels, and neurons per layer to TensorBoard."""
    
    if writer is None:
        return

    for i, layer in enumerate(get_layers_fn(model)):
        sa, mlp = get_blocks_fn(layer)
        
        # 1. Log Attention Shapes
        if hasattr(sa, "get_attn_shapes"):
            embed_dim, num_heads, head_dims = sa.get_attn_shapes()
            hd_qk, hd_vo = head_dims
            
            # Grouped under "Structure_Attn" in TensorBoard
            writer.add_scalar(f"Structure_Attn/Layer_{i}_NumHeads", num_heads, iteration)
            writer.add_scalar(f"Structure_Attn/Layer_{i}_HeadDim_QK", hd_qk, iteration)
            writer.add_scalar(f"Structure_Attn/Layer_{i}_HeadDim_VO", hd_vo, iteration)
            
        # 2. Log MLP Shapes
        if hasattr(mlp, "get_mlp_shapes"):
            in_features, hidden_dim, out_features = mlp.get_mlp_shapes()
            
            # Grouped under "Structure_MLP" in TensorBoard
            writer.add_scalar(f"Structure_MLP/Layer_{i}_HiddenDim", hidden_dim, iteration)

def pad_along_dim1(tensors):
    """Pad tensors along dimension 1 (columns/channels)"""
    max_len = max(t.size(1) for t in tensors)
    padded =[]
    for t in tensors:
        pad_len = max_len - t.size(1)
        padded.append(F.pad(t, (0, pad_len)))  
    return padded

def pad_along_dim0(tensors):
    """Pad tensors along dimension 0 (rows/heads)"""
    max_len = max(t.size(0) for t in tensors)
    padded =[]
    for t in tensors:
        pad_len = max_len - t.size(0)
        padded.append(F.pad(t, (0, 0, 0, pad_len)))  
    return padded

def log_attention_heatmaps(writer: SummaryWriter, model, iteration: int, mapping: dict, get_layers_fn, get_blocks_fn, order=2):
    """Logs heatmaps maintaining original size, plotting pruned connections as Black."""
    if writer is None: return
    
    for i, layer in enumerate(get_layers_fn(model)):
        sa, _ = get_blocks_fn(layer)
        if not hasattr(sa, "get_attn_shapes"): continue
        
        embed_dim, curr_num_heads, head_dims = sa.get_attn_shapes()
        curr_hd_qk, curr_hd_vo = head_dims
        attn_ws, _, _, _ = sa.get_attn_weights()
        
        if isinstance(attn_ws, list):
            q_w = attn_ws[0].view(curr_num_heads, curr_hd_qk, embed_dim).detach().cpu()
            k_w = attn_ws[1].view(curr_num_heads, curr_hd_qk, embed_dim).detach().cpu()
            v_w = attn_ws[2].view(curr_num_heads, curr_hd_vo, embed_dim).detach().cpu()
        else:
            qk_out = curr_num_heads * curr_hd_qk
            q_w = attn_ws[:qk_out].view(curr_num_heads, curr_hd_qk, embed_dim).detach().cpu()
            k_w = attn_ws[qk_out:2*qk_out].view(curr_num_heads, curr_hd_qk, embed_dim).detach().cpu()
            v_w = attn_ws[2*qk_out:].view(curr_num_heads, curr_hd_vo, embed_dim).detach().cpu()
            
        q_norm = torch.norm(q_w, p=order, dim=-1)
        k_norm = torch.norm(k_w, p=order, dim=-1)
        v_norm = torch.norm(v_w, p=order, dim=-1)
        
        # 1. Setup the Dense Canvas using Original Dimensions
        orig_heads = mapping[i]["orig_num_heads"]
        orig_qk = mapping[i]["orig_hd_qk"]
        orig_vo = mapping[i]["orig_hd_vo"]
        max_orig_channels = max(orig_qk, orig_vo)
        
        # Fill with NaN (Seaborn makes NaN transparent)
        canvas = np.full((3 * orig_heads, max_orig_channels), np.nan)
        
        # 2. Map surviving weights back to their exact original coordinates
        for curr_h_idx in range(curr_num_heads):
            orig_h_idx = mapping[i]["heads"][curr_h_idx]
            
            # Place Q and K channels
            for curr_c_idx in range(curr_hd_qk):
                orig_c_idx = mapping[i]["qk"][orig_h_idx][curr_c_idx]
                canvas[orig_h_idx, orig_c_idx] = q_norm[curr_h_idx, curr_c_idx]
                canvas[orig_heads + orig_h_idx, orig_c_idx] = k_norm[curr_h_idx, curr_c_idx]
                
            # Place V channels
            for curr_c_idx in range(curr_hd_vo):
                orig_c_idx = mapping[i]["vo"][orig_h_idx][curr_c_idx]
                canvas[2 * orig_heads + orig_h_idx, orig_c_idx] = v_norm[curr_h_idx, curr_c_idx]
        
        # 3. Plotting
        labels = ([f"Q{h}" for h in range(orig_heads)] +[f"K{h}" for h in range(orig_heads)] +[f"V{h}" for h in range(orig_heads)])
        
        fig, ax = plt.subplots(figsize=(max(8, max_orig_channels / 8), max(6, (3 * orig_heads) / 4)))
        
        # Set background to black (shows through the NaN transparent cells)
        ax.set_facecolor('black')
        
        sns.heatmap(canvas, cmap="viridis", ax=ax, cbar=True, yticklabels=labels)
        ax.hlines([orig_heads, 2 * orig_heads], *ax.get_xlim(), colors="red", linestyles="dashed", linewidth=1.5)
        
        ax.set_xlabel("Original Channels")
        ax.set_ylabel("Original Heads (Q, K, V)")
        ax.set_title(f"Layer {i} Attention - Black = Pruned")
        
        fig.tight_layout()
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        image = plt.imread(buf)
        writer.add_image(f"Attention_Heatmaps/Layer_{i}", image, global_step=iteration, dataformats='HWC')
        plt.close(fig)

def log_sparsity(writer: SummaryWriter, attention_layers_stats, iteration: int):
    for i, stat in enumerate(attention_layers_stats):
        sparsity = stat['sparsity']
        writer.add_scalar(f"Sparsity/Layer_{i}_overall", sparsity, iteration)


# ──────────────────────────────────────────────────────────────────────────
# System Monitoring & Miscellaneous
# ──────────────────────────────────────────────────────────────────────────
def log_system_metrics_to_tensorboard(writer, metrics, step, prefix=""):
    for key, value in metrics.items():
        if isinstance(value, (int, float)):
            writer.add_scalar(f'{prefix}{key}', value, step)

def get_current_memory_usage():
    memory = psutil.virtual_memory()
    cpu_memory_gb = memory.used / (1024**3)
    gpu_memory_gb = 0
    if torch.cuda.is_available():
        gpu_memory_gb = torch.cuda.memory_allocated() / (1024**3)
    return {'cpu_memory_gb': cpu_memory_gb, 'gpu_memory_gb': gpu_memory_gb}
    
def set_seed(seed: int):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.mha.set_fastpath_enabled(False)

def print_gpu_memory(tag=""):
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        print(f"[{tag}] GPU Memory: {allocated:.3f}GB allocated, {reserved:.3f}GB reserved")
        return allocated, reserved
    return 0, 0

def aggressive_cleanup():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    gc.collect() 

def _make_handlers():
    """Custom fvcore FLOP handlers for ops it doesn't natively support."""
    def elementwise(inputs, outputs):
        out_shape = get_shape(outputs[0])
        return int(np.prod(out_shape)) if out_shape else 0
    def softmax(inputs, outputs):
        out_shape = get_shape(outputs[0])
        return int(np.prod(out_shape)) * 2 if out_shape else 0
    def gelu(inputs, outputs):
        in_shape = get_shape(inputs[0])
        return int(np.prod(in_shape)) * 8 if in_shape else 0
    def silu(inputs, outputs):
        in_shape = get_shape(inputs[0])
        return int(np.prod(in_shape)) * 4 if in_shape else 0
    def power(inputs, outputs):
        out_shape = get_shape(outputs[0])
        return int(np.prod(out_shape)) if out_shape else 0
    def sdpa(inputs, outputs):
        q_shape = get_shape(inputs[0])
        k_shape = get_shape(inputs[1])
        if not q_shape or not k_shape:
            return 0
        batch, heads, seq_len, head_dim = q_shape
        kv_seq_len = k_shape[2]
        return (
            batch * heads * seq_len * kv_seq_len * head_dim * 2  # QK^T
            + batch * heads * seq_len * kv_seq_len * 2           # softmax
            + batch * heads * seq_len * head_dim * kv_seq_len * 2  # AV
        )

    def matmul(inputs, outputs):
        shape_a = get_shape(inputs[0])
        out_shape = get_shape(outputs[0])
        if not shape_a or not out_shape: return 0
        k = shape_a[-1] if len(shape_a) > 0 else 1
        out_volume = int(np.prod(out_shape)) if len(out_shape) > 0 else 1
        return 2 * k * out_volume

    # We force math-SDP in compute_flops, which decomposes attention into
    # bmm+softmax+bmm at the trace level. The matmul/softmax handlers below
    # then count those decomposed ops. Registering an SDPA handler in
    # parallel double-counts attention FLOPs (~10% inflation on ViTs), so
    # we deliberately leave SDPA out.
    return {
        "aten::add":   elementwise,
        "aten::mul":   elementwise,
        "aten::div":   elementwise,
        "aten::sub":   elementwise,
        "aten::pow":   power,
        "aten::softmax":  softmax,
        "aten::gelu":  gelu,
        "aten::silu":  silu,
        "aten::matmul": matmul,
        "aten::mm":     matmul,
        "aten::bmm":    matmul,
    }


def compute_flops(model, inputs):
    model.eval()

    prev_use_cache = None
    if hasattr(model, "config") and hasattr(model.config, "use_cache"):
        prev_use_cache = model.config.use_cache
        model.config.use_cache = False

    # transformers >= 4.45 builds the causal mask via torch.vmap inside
    # `sdpa_mask_recent_torch`. fvcore's JIT tracer can't go through vmap
    # (raises "RuntimeError: unordered_map::at"), so force the eager mask
    # interface for the trace.
    prev_attn_impl = None
    if hasattr(model, "config"):
        prev_attn_impl = getattr(model.config, "_attn_implementation", None)
        try:
            model.config._attn_implementation = "eager"
        except Exception:
            prev_attn_impl = None

    import torch.backends.cuda
    prev_flash = torch.backends.cuda.flash_sdp_enabled()
    prev_mem = torch.backends.cuda.mem_efficient_sdp_enabled()
    prev_math = torch.backends.cuda.math_sdp_enabled()

    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)

    try:
        analyzer = FlopCountAnalysis(model, inputs)

        handlers = _make_handlers()
        analyzer.set_op_handle(**handlers)

        return int(analyzer.total())

    finally:
        if prev_use_cache is not None:
            model.config.use_cache = prev_use_cache

        if prev_attn_impl is not None:
            try:
                model.config._attn_implementation = prev_attn_impl
            except Exception:
                pass

        torch.backends.cuda.enable_flash_sdp(prev_flash)
        torch.backends.cuda.enable_mem_efficient_sdp(prev_mem)
        torch.backends.cuda.enable_math_sdp(prev_math)


def compute_theoretical_flops(model, dense_flops):
    """
    Theoretical FLOPs assuming a sparse kernel that skips zeros. Returns
    None when there is no zero pattern to exploit — the dense reference,
    or any structurally-pruned model whose linears have been resized
    rather than zeroed. In those cases the effective FLOPs measurement
    is already the truthful number, and a "theoretical" version would
    just collapse onto the dense constant.
    """
    total_params = 0
    nonzero_params = 0
    for module in model.modules():
        if isinstance(module, torch.nn.Linear):
            w = module.weight.data
            total_params += w.numel()
            nonzero_params += (w != 0).sum().item()

    if total_params == 0 or nonzero_params == total_params:
        return None
    sparsity = 1 - nonzero_params / total_params
    return int(dense_flops * (1 - sparsity))


@torch.no_grad()
def benchmark_latency_llm(model, seq_len=2048, batch_size=1,
                          warmup=10, iterations=100, disable_flash=False):
    """Wall-clock per-forward latency in milliseconds.

    disable_flash: if True, force SDPA onto the math kernel and skip the
    in-attention Q/K/V flash-padding. Useful as a control to measure
    "vanilla without flash" — i.e., the kernel path the irregularly-
    pruned baselines are stuck on.
    """
    import torch.backends.cuda as bc
    device = next(model.parameters()).device
    ids = torch.randint(0, model.config.vocab_size,
                        (batch_size, seq_len), device=device)

    model.eval()

    prev_use_cache = None
    if hasattr(model, "config") and hasattr(model.config, "use_cache"):
        prev_use_cache = model.config.use_cache
        model.config.use_cache = False

    # Push our custom attention onto SDPA's flash/efficient path. The
    # default forward keeps HF's 4-D causal mask, which routes SDPA to
    # the math kernel and dominates per-layer cost (especially under
    # irregular per-layer head_dim). For un-padded benchmark inputs the
    # is_causal=True dispatch is correct. When disable_flash=True we
    # leave the mask in place and force the math kernel.
    prev_force_flash = []
    if not disable_flash:
        for m in model.modules():
            if hasattr(m, "force_flash_attn"):
                prev_force_flash.append((m, m.force_flash_attn))
                m.force_flash_attn = True

    prev_flash = bc.flash_sdp_enabled()
    prev_mem   = bc.mem_efficient_sdp_enabled()
    prev_math  = bc.math_sdp_enabled()
    if disable_flash:
        bc.enable_flash_sdp(False)
        bc.enable_mem_efficient_sdp(False)
        bc.enable_math_sdp(True)

    try:
        times = []
        for _ in range(iterations + warmup):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            _ = model(ids)
            torch.cuda.synchronize()
            times.append(time.perf_counter() - t0)
    finally:
        if prev_use_cache is not None:
            model.config.use_cache = prev_use_cache
        for m, prev in prev_force_flash:
            m.force_flash_attn = prev
        bc.enable_flash_sdp(prev_flash)
        bc.enable_mem_efficient_sdp(prev_mem)
        bc.enable_math_sdp(prev_math)

    return (sum(times[warmup:]) / iterations) * 1000


@torch.no_grad()
def benchmark_latency_vit(model, image_size=224, batch_size=1,
                           warmup=10, iterations=100):
    """Wall-clock per-forward latency for ViTs (image input)."""
    device = next(model.parameters()).device
    inputs = torch.randn(batch_size, 3, image_size, image_size, device=device)
    
    model.eval()
    times = []
    for _ in range(iterations + warmup):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        _ = model(inputs)
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
    
    return (sum(times[warmup:]) / iterations) * 1000


def compute_flops_llm_analytical(model, seq_len):
    """
    Analytical forward FLOPs for a Llama-style LLM at batch_size=1.

    Avoids fvcore's JIT tracer, which fails on transformers >= 4.45 with
    "RuntimeError: unordered_map::at" because the causal-mask path goes
    through torch.vmap. We instead walk the actual nn.Linear modules and
    add an SDPA term per attention layer, so the count automatically
    reflects structural pruning (resized linears, fewer heads).
    """
    flops = 0

    # Every transformer Linear runs once per token: 2 * seq * in * out FMAs.
    # This covers QKV, O, MLP gate/up/down, and the LM head.
    for m in model.modules():
        if isinstance(m, torch.nn.Linear):
            flops += 2 * seq_len * m.in_features * m.out_features

    # SDPA per attention layer. Prefer per-module num_heads/head_dim so
    # structurally-pruned models with heterogeneous layers are counted
    # correctly; fall back to a uniform config-based estimate if no
    # attention module exposes those attributes.
    sdpa_layers_seen = 0
    for m in model.modules():
        cls_name = m.__class__.__name__
        if not cls_name.endswith("Attention"):
            continue
        num_heads = getattr(m, "num_heads", None)
        head_dim  = getattr(m, "head_dim",  None)
        if num_heads is None or head_dim is None:
            continue
        flops += 4 * num_heads * seq_len * seq_len * head_dim   # QK^T + AV
        flops += 3 * num_heads * seq_len * seq_len              # softmax
        sdpa_layers_seen += 1

    if sdpa_layers_seen == 0:
        cfg = getattr(model, "config", None)
        if cfg is not None:
            n_layers = getattr(cfg, "num_hidden_layers", 0)
            n_heads  = getattr(cfg, "num_attention_heads", 0)
            h_size   = getattr(cfg, "hidden_size", 0)
            head_dim = getattr(cfg, "head_dim", None)
            if head_dim is None and n_heads:
                head_dim = h_size // n_heads
            if n_layers and n_heads and head_dim:
                flops += n_layers * (
                    4 * n_heads * seq_len * seq_len * head_dim
                    + 3 * n_heads * seq_len * seq_len
                )

    return flops


def benchmark_full(model, model_type="llm", seq_len=2048, image_size=224,
                   dense_flops=None, disable_flash=False):
    """
    Run all benchmarks. Returns a dict of measurements.

    model_type: "llm" or "vit"
    dense_flops: if provided, also reports theoretical sparse FLOPs.
                 Get this by running compute_flops on the unpruned model once.
    """
    device = next(model.parameters()).device

    results = {
        "model_type": model_type,
    }

    if model_type == "llm":
        results["effective_flops_g"] = compute_flops_llm_analytical(
            model, seq_len) / 1e9
        results["latency_bs1_ms"] = benchmark_latency_llm(
            model, seq_len=seq_len, batch_size=1, disable_flash=disable_flash)
        results["latency_bs16_ms"] = benchmark_latency_llm(
            model, seq_len=seq_len, batch_size=16, disable_flash=disable_flash)
    else:
        flops_input = torch.randn(1, 3, image_size, image_size, device=device)
        results["effective_flops_g"] = compute_flops(model, flops_input) / 1e9
        results["latency_bs1_ms"] = benchmark_latency_vit(
            model, image_size=image_size, batch_size=1)
        results["latency_bs64_ms"] = benchmark_latency_vit(
            model, image_size=image_size, batch_size=64)

    if dense_flops is not None:
        theoretical = compute_theoretical_flops(model, dense_flops)
        if theoretical is not None:
            results["theoretical_flops_g"] = theoretical / 1e9
            results["sparsity_realizable"] = 1 - theoretical / dense_flops

    return results

def print_benchmark_report(bench):
    for k, v in bench.items():
        print(f"{k}: {v}")

def build_warmup_scheduler(optimizer, num_epochs: int, dataloader_len: int, grad_accumulation_steps: int = 1, warmup_ratio: float = 0.05, schedule_type: str = "cosine"):
    total_steps = (num_epochs * dataloader_len) // grad_accumulation_steps
    warmup_steps = int(warmup_ratio * total_steps)

    def lr_lambda(step: int):
        if warmup_steps > 0 and step < warmup_steps:
            return step / max(1, warmup_steps)
        if schedule_type == "constant":
            return 1.0
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return LambdaLR(optimizer, lr_lambda)

def str2bool(v):
    if isinstance(v, bool): return v
    if v.lower() in ("yes", "true", "t", "1"): return True
    elif v.lower() in ("no", "false", "f", "0"): return False
    else: raise argparse.ArgumentTypeError("Boolean value expected.")
