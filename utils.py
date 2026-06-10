import os
import io
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import psutil
import pynvml
import threading
import time
import os
import gc
from fvcore.nn import FlopCountAnalysis
from fvcore.nn.jit_handles import elementwise_flop_counter
import math
from torch.optim.lr_scheduler import LambdaLR
from fvcore.nn.jit_handles import elementwise_flop_counter, get_shape
import numpy as np
import argparse

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from custom_attention import (MultiheadAttention, Attention, WhisperAttention, LayerWiseWhisperConfig)
import tf_locoformer

ATTENTION_CLASSES = (
    MultiheadAttention,
    WhisperAttention,
    Attention,
    tf_locoformer.MultiHeadSelfAttention,
)

def update_config_from_model(model, config):
    """
    Update LayerwiseBertConfig based on the actual pruned model structure.
    This assumes the model has encoder.layer[i].attention.self.{query,key,value}
    and intermediate/dense layers like Hugging Face BertModel.
    """

    if isinstance(config, LayerWiseWhisperConfig):
        num_layers = len(config.encoder_self_qkv_config)+len(config.decoder_self_qkv_config)

    # Hidden size = output dim of embeddings / encoder outputs
    config.hidden_size = model.config.hidden_size
    
    for i in range(num_layers):
        if isinstance(config, LayerWiseWhisperConfig):
            if i < len(config.encoder_self_qkv_config):
                self_attn = model.model.encoder.layers[i].self_attn
                hidden_size = self_attn.embed_dim
                # query
                q_heads = self_attn.num_heads_q
                q_dim = self_attn.head_dim_q

                # value
                v_heads = self_attn.num_heads_v
                v_dim = self_attn.head_dim_v
                config.encoder_self_qkv_config[i] = {
                    "hidden_size": hidden_size,
                    "num_attention_heads": q_heads,
                    "num_attention_heads_v": v_heads,
                    "head_dim": q_dim,
                    "head_dim_v": v_dim,
                    "attention_dropout": self_attn.dropout,
                }
            else:
                idx = i - len(config.encoder_self_qkv_config)
                self_attn = model.model.decoder.layers[idx].self_attn
                hidden_size = self_attn.embed_dim
                # query
                q_heads = self_attn.num_heads_q
                q_dim = self_attn.head_dim_q

                # value
                v_heads = self_attn.num_heads_v
                v_dim = self_attn.head_dim_v

                config.decoder_self_qkv_config[idx] = {
                    "hidden_size": hidden_size,
                    "num_attention_heads": q_heads,
                    "num_attention_heads_v": v_heads,
                    "head_dim": q_dim,
                    "head_dim_v": v_dim,
                    "attention_dropout": self_attn.dropout,
                }

                cross_attn = model.model.decoder.layers[idx].encoder_attn
                hidden_size = cross_attn.embed_dim
                # query
                q_heads = cross_attn.num_heads_q
                q_dim = cross_attn.head_dim_q

                # value
                v_heads = cross_attn.num_heads_v
                v_dim = cross_attn.head_dim_v

                config.decoder_cross_qkv_config[idx] = {
                    "hidden_size": hidden_size,
                    "num_attention_heads": q_heads,
                    "num_attention_heads_v": v_heads,
                    "head_dim": q_dim,
                    "head_dim_v": v_dim,
                    "attention_dropout": self_attn.dropout,
                }


    model.config = config
    return config

def count_parameters(model):
    """Count the number of trainable parameters in the model."""
    return sum(p.numel() for p in model.parameters())

def build_layer_index(model):
    index = []
    for name, module in model.named_modules():
        if is_encoder_layer(module):
            index.append(name)
    return index

def get_layer_by_name(model, name):
    cur = model
    for attr in name.split("."):
        cur = getattr(cur, attr)
    return cur

def get_model_layers(model):
    """Extract encoder layers from model."""
    layers = []
    if isinstance(model, torch.nn.DataParallel):
        base_model = model.module
    else:
        base_model = model
        
    for i, name in enumerate(base_model._encoder_layer_names):
        layer = get_layer_by_name(model, name)
        layers.append(layer)
    
    return layers

def get_layers(model):
    attn_layers = []

    for name in model._encoder_layer_names:
        layer = get_layer_by_name(model, name)
        matches = []

        for _, module in layer.named_modules():
            if isinstance(module, ATTENTION_CLASSES):
                matches.append(module)

        if len(matches) == 0:
            raise RuntimeError(
                f"No attention module found in encoder layer {layer.__class__.__name__}"
            )

        if len(matches) > 2:
            raise RuntimeError(
                f"Too many attention modules found in encoder layer {layer.__class__.__name__}"
            )

        attn_layers.append(matches[0])  

    return attn_layers


def capture_encoder_layer_names(model):
    if isinstance(model, torch.nn.DataParallel):
        base_model = model.module
    else:
        base_model = model

    names = []
    for name, module in base_model.named_modules():
        # print(name)
        if is_encoder_layer(module):
            names.append(name)

    # preserve semantic order
    names.sort(key=lambda x: (x.count("."), x))
    return names

def _contains_linear(module):
    if isinstance(module, nn.Linear):
        return True
    return any(_contains_linear(child) for child in module.children())

def _is_norm_layer(module):
    NORM_TYPES = (
        nn.LayerNorm, nn.BatchNorm1d, nn.BatchNorm2d,
        nn.BatchNorm3d, nn.GroupNorm, nn.InstanceNorm1d,
        nn.InstanceNorm2d, nn.InstanceNorm3d,
    )
    if isinstance(module, NORM_TYPES):
        return True
    # Catch RMSNorm, LlamaRMSNorm, T5LayerNorm, etc. by name
    return 'norm' in type(module).__name__.lower()

def is_encoder_layer(module):
    # print(module)
    # Don't match attention modules themselves
    if isinstance(module, ATTENTION_CLASSES):
        # print('attention class, not encoder layer')
        return False

    # Count attention modules at shallow depth (direct children or one level down)
    # to avoid counting all attentions in the entire stack
    attention_count = 0
    for child in module.children():
        if isinstance(child, ATTENTION_CLASSES):
            attention_count += 1
        # else:
        #     # one level deeper to handle wrappers like BertAttention -> BertSelfAttention
        #     for grandchild in child.children():
        #         if isinstance(grandchild, ATTENTION_CLASSES):
        #             attention_count += 1

    # Encoder layer: exactly 1 self-attention
    # Decoder layer: exactly 2 (self-attention + cross-attention)
    if attention_count not in (1, 2):
        # print('too many or too few attention classes', attention_count)
        return False

    # Must have a direct MLP/FFN child
    has_ffn = any(
        not isinstance(child, ATTENTION_CLASSES)
        and not _is_norm_layer(child)
        and not isinstance(child, nn.Dropout)
        and _contains_linear(child)
        for child in module.children()
    )
    # print('FFN', has_ffn)
    return has_ffn


def get_layer_weights_fisher(sa):
    #EXTRACT MODULE WEIGHTS for fisher information
    
    if isinstance(sa, (MultiheadAttention)):
        in_weight = sa.in_proj_weight.weight
        out_weight = sa.out_proj.weight
    elif isinstance(sa, (tf_locoformer.MultiHeadSelfAttention)):
        in_weight = sa.qkv.weight
        out_weight = sa.aggregate_heads[0].weight
    elif isinstance(sa, (Attention)):
        in_weight = sa.qkv.weight
        out_weight = sa.proj.weight
    elif isinstance(sa, WhisperAttention):
        q_weight = sa.q_proj.weight
        k_weight = sa.k_proj.weight
        v_weight = sa.v_proj.weight
        in_weight = [q_weight, k_weight, v_weight]
        out_weight = sa.out_proj.weight
    else:
        print("Error, wrong type of attention module!")
        exit(-1)
    
    return in_weight, out_weight 

def get_layer_weights(sa, need_grad=False):
    #EXTRACT MODULE WEIGHTS
    in_bias = None
    out_bias = None
    
    if isinstance(sa, (MultiheadAttention)):
        in_weight = sa.in_proj_weight.weight
        out_weight = sa.out_proj.weight
        if sa.in_proj_weight.bias is not None:
            in_bias = sa.in_proj_weight.bias
        if sa.out_proj.bias is not None:
            out_bias = sa.out_proj.bias
        embed_dim = sa.embed_dim
    elif isinstance(sa, (tf_locoformer.MultiHeadSelfAttention)):
        in_weight = sa.qkv.weight
        out_weight = sa.aggregate_heads[0].weight
        if sa.qkv.bias is not None:
            in_bias = sa.qkv.bias 
        if sa.aggregate_heads[0].bias is not None:
            out_bias = sa.aggregate_heads[0].bias
        embed_dim = sa.embed_dim
    elif isinstance(sa, (Attention)):
        in_weight = sa.qkv.weight
        out_weight = sa.proj.weight
        if sa.qkv.bias is not None:
            in_bias = sa.qkv.bias 
        if sa.proj.bias is not None:
            out_bias = sa.proj.bias
        embed_dim = sa.embed_dim
    elif isinstance(sa, WhisperAttention):
        q_weight = sa.q_proj.weight
        q_bias = sa.q_proj.bias
        k_weight = sa.k_proj.weight
        k_bias = sa.k_proj.bias
        v_weight = sa.v_proj.weight
        v_bias = sa.v_proj.bias
        
        embed_dim = sa.embed_dim
        
        in_weight = torch.cat((q_weight, k_weight, v_weight), dim = 0)
        if q_bias is not None:
            in_bias = torch.cat((q_bias, q_bias, v_bias), dim = 0) #k_bias is false for whisper, so we just copy q_bias twice, but we do not copy it to key.bias later on
        out_weight = sa.out_proj.weight
        if sa.out_proj.bias is not None: 
            out_bias = sa.out_proj.bias
    else:
        print("Error, wrong type of attention module!")
        exit(-1)
        
    if hasattr(sa, 'num_heads_q'):
        num_heads_q = sa.num_heads_q
        num_heads_k = sa.num_heads_k 
        num_heads_v = sa.num_heads_v
        num_heads_o = sa.num_heads_out 
        
        head_dim_q = sa.head_dim_q
        head_dim_k = sa.head_dim_k
        head_dim_v = sa.head_dim_v
        head_dim_o = sa.head_dim_out
    
    num_heads = [num_heads_q, num_heads_k, num_heads_v, num_heads_o]
    head_dims = [head_dim_q, head_dim_k, head_dim_v, head_dim_o]
    
    return in_weight, out_weight, in_bias, out_bias, embed_dim, num_heads, head_dims 

def pad_along_dim1(tensors):
    """Pad tensors along dimension 1 (columns/channels)"""
    # Find maximum size along dim=1
    max_len = max(t.size(1) for t in tensors)
   
    padded = []
    for t in tensors:
        pad_len = max_len - t.size(1)
        # Pad only along dim=1 (columns), (left, right)
        padded.append(F.pad(t, (0, pad_len)))  
    return padded

def pad_along_dim0(tensors):
    """Pad tensors along dimension 0 (rows/heads)"""
    # Find maximum size along dim=0
    max_len = max(t.size(0) for t in tensors)
   
    padded = []
    for t in tensors:
        pad_len = max_len - t.size(0)
        # Pad only along dim=0 (rows), (top, bottom)
        # F.pad format: (left, right, top, bottom, front, back, ...)
        padded.append(F.pad(t, (0, 0, 0, pad_len)))  
    return padded

# Logging attention weight heatmaps to TensorBoard
def log_heatmap(writer: SummaryWriter,
                sa: torch.nn.Module,
                tag: str,
                iteration: int,
                order: int,
                strategy_name):
    
    weight, _, _, _, embed_dim, num_heads, head_dims= get_layer_weights(sa)
    
    num_heads_q, num_heads_k, num_heads_v, _ = num_heads
    head_dim_q, head_dim_k, head_dim_v, _ = head_dims
    
    q_out_dim = num_heads_q * head_dim_q
    k_out_dim = num_heads_k * head_dim_k
    
    q_weight = weight[:q_out_dim, :].view(num_heads_q, head_dim_q, embed_dim).detach().cpu()
    k_weight = weight[q_out_dim:q_out_dim+k_out_dim, :].view(num_heads_k, head_dim_k, embed_dim).detach().cpu()
    v_weight = weight[q_out_dim+k_out_dim:, :].view(num_heads_v, head_dim_v, embed_dim).detach().cpu()
    
    q_norm = torch.norm(q_weight, p = order, dim = -1)
    k_norm = torch.norm(k_weight, p = order, dim = -1)
    v_norm = torch.norm(v_weight, p = order, dim = -1)
    
    if strategy_name in ['MULTI_HEAD_SAME_CHANNEL', 'MULTI_HEAD_PER_HEAD']:
        # Pad along dimension 1 (channels)
        if not (head_dim_q == head_dim_k == head_dim_v):
            q_norm, k_norm, v_norm = pad_along_dim1([q_norm, k_norm, v_norm])
    elif strategy_name == 'MULTI_HEAD_ENTIRE_HEAD':
        # Pad along dimension 0 (heads)
        if not (num_heads_q == num_heads_k == num_heads_v):
            q_norm, k_norm, v_norm = pad_along_dim0([q_norm, k_norm, v_norm])
    else:
        # Default behavior - pad along dim 1 if head dimensions differ
        if not (head_dim_q == head_dim_k == head_dim_v):
            q_norm, k_norm, v_norm = pad_along_dim1([q_norm, k_norm, v_norm])
    
   
    matrix = torch.cat([q_norm, k_norm, v_norm], dim=0)
    matrix = matrix.numpy() > 0
    
    labels = (
        [f"Q{h}" for h in range(num_heads_q)] +
        [f"K{h}" for h in range(num_heads_k)] +
        [f"V{h}" for h in range(num_heads_v)]
    )
    
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(matrix, cmap="viridis", ax=ax, cbar=True, yticklabels=labels, xticklabels=[f"C{c}" for c in range(matrix.shape[1])])
    
    q_end = num_heads_q
    k_end = q_end + num_heads_k
    ax.hlines([q_end, k_end], *ax.get_xlim(), colors="red", linestyles="dashed", linewidth=1.5)

    ax.set_xlabel("Channels")
    ax.set_ylabel("Heads")
    ax.set_title("Q / K / V Head x Channel")
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    image = plt.imread(buf)
    writer.add_image(tag, image, global_step=iteration, dataformats='HWC')
    plt.close(fig)

# Logging sparsity metrics per attention layer
def log_sparsity(writer: SummaryWriter,
                 attention_layers_stats,
                 iteration: int):
    
    for i, stat in enumerate(attention_layers_stats):
        sparsity = stat['sparsity']
        writer.add_scalar(f"Sparsity/Layer_{i}_in_proj_weight", sparsity, iteration)

class SystemMonitor:
    def __init__(self, interval=0.5):
        self.interval = interval
        self.monitoring = False
        self.monitor_thread = None
        self.metrics_data = []
        self.start_time = None
        self.peak_cpu_memory = 0
        self.peak_gpu_memory = 0
        self.total_energy = 0
        self.power_readings = []
        
        # Initialize NVIDIA ML
        try:
            pynvml.nvmlInit()
            self.gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            self.gpu_available = True
        except:
            self.gpu_available = False
            print("Warning: GPU monitoring not available")
    
    def _get_current_metrics(self):
        """Get current system metrics"""
        # CPU memory
        memory = psutil.virtual_memory()
        cpu_memory_gb = memory.used / (1024**3)
        cpu_percent = psutil.cpu_percent(interval=None)  # Non-blocking
        
        # GPU metrics
        gpu_memory_gb = 0
        gpu_power_w = 0
        gpu_util = 0
        
        if self.gpu_available:
            try:
                # GPU memory
                if torch.cuda.is_available():
                    gpu_memory_gb = torch.cuda.memory_allocated() / (1024**3)
                
                # GPU power
                gpu_power_w = pynvml.nvmlDeviceGetPowerUsage(self.gpu_handle) / 1000.0
                
                # GPU utilization
                utilization = pynvml.nvmlDeviceGetUtilizationRates(self.gpu_handle)
                gpu_util = utilization.gpu
                
            except Exception as e:
                print(f"GPU metrics error: {e}")
        
        return {
            'timestamp': time.time(),
            'cpu_memory_gb': cpu_memory_gb,
            'gpu_memory_gb': gpu_memory_gb,
            'cpu_percent': cpu_percent,
            'gpu_power_w': gpu_power_w,
            'gpu_util': gpu_util
        }
    
    def _monitor_loop(self):
        """Background monitoring loop"""
        while self.monitoring:
            try:
                metrics = self._get_current_metrics()
                self.metrics_data.append(metrics)
                
                # Update peaks
                self.peak_cpu_memory = max(self.peak_cpu_memory, metrics['cpu_memory_gb'])
                self.peak_gpu_memory = max(self.peak_gpu_memory, metrics['gpu_memory_gb'])
                
                # Store power for energy calculation
                if metrics['gpu_power_w'] > 0:
                    self.power_readings.append(metrics['gpu_power_w'])
                
            except Exception as e:
                print(f"Monitoring error: {e}")
            
            time.sleep(self.interval)
    
    def start_monitoring(self):
        """Start background monitoring"""
        if self.monitoring:
            return
        
        self.monitoring = True
        self.start_time = time.time()
        self.metrics_data = []
        self.peak_cpu_memory = 0
        self.peak_gpu_memory = 0
        self.power_readings = []
        
        self.monitor_thread = threading.Thread(target=self._monitor_loop)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
        print("Started system monitoring")
    
    def stop_monitoring(self):
        """Stop monitoring and calculate final metrics"""
        if not self.monitoring:
            return {}
        
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=2)
        
        # Calculate total energy (approximation)
        if self.power_readings:
            avg_power = sum(self.power_readings) / len(self.power_readings)
            duration_hours = (time.time() - self.start_time) / 3600
            self.total_energy = avg_power * duration_hours  # Wh
        
        final_metrics = {
            'duration_seconds': time.time() - self.start_time,
            'peak_cpu_memory_gb': self.peak_cpu_memory,
            'peak_gpu_memory_gb': self.peak_gpu_memory,
            'total_energy_wh': self.total_energy,
            'avg_gpu_power_w': sum(self.power_readings) / len(self.power_readings) if self.power_readings else 0,
            'num_samples': len(self.metrics_data)
        }
        
        print("Stopped system monitoring")
        return final_metrics

# Integration functions for your main code
def log_system_metrics_to_tensorboard(writer, metrics, step, prefix=""):
    """Log system metrics to TensorBoard"""
    for key, value in metrics.items():
        if isinstance(value, (int, float)):
            writer.add_scalar(f'{prefix}{key}', value, step)

def get_current_memory_usage():
    """Get current memory usage (for immediate logging)"""
    memory = psutil.virtual_memory()
    cpu_memory_gb = memory.used / (1024**3)
    
    gpu_memory_gb = 0
    if torch.cuda.is_available():
        gpu_memory_gb = torch.cuda.memory_allocated() / (1024**3)
    
    return {
        'cpu_memory_gb': cpu_memory_gb,
        'gpu_memory_gb': gpu_memory_gb
    }
    
# Set reproducibility
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

# Compute evaluation metrics
def metrics(preds: List[int], labels: List[int]) -> Tuple[float, float, float, float]:
    all_preds = np.array(preds)
    all_labels = np.array(labels)

    acc = accuracy_score(all_labels, all_preds)
    prec = precision_score(all_labels, all_preds, average='macro', zero_division=0.0)
    rec = recall_score(all_labels, all_preds, average='macro', zero_division=0.0)
    f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0.0)

    # print(f"Accuracy: {acc:.4f}")
    # print(f"Precision: {prec:.4f}")
    # print(f"Recall: {rec:.4f}")
    # print(f"F1 Score: {f1:.4f}")
    
    return acc, prec, rec, f1

# Training loop with validation
def training(model: torch.nn.Module,
             train_loader: DataLoader,
             epoch: int,
             epochs: int,
             device: torch.device,
             optimizer: torch.optim.Optimizer,
             criterion,
             warmup_scheduler = None) -> float:

    total_training_loss = 0.0

    model.train()
    for batch in tqdm(train_loader, desc=f"Epoch {epoch+1} Training"):

        optimizer.zero_grad()

        batch = {
            k: v.to(device, non_blocking=True) if torch.is_tensor(v) else v
            for k, v in batch.items()
        }
        outputs = model(batch['pixel_values'])
        loss = criterion(outputs, batch['labels'])
        
        loss.backward()

        optimizer.step()
        if warmup_scheduler is not None:
            warmup_scheduler.step()

        total_training_loss += loss.item()

    avg_loss = total_training_loss / len(train_loader)
    print(f"Epoch {epoch + 1}/{epochs}, Training Loss: {avg_loss:.4f}")

def print_gpu_memory(tag=""):
    """Print detailed GPU memory usage"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        print(f"[{tag}] GPU Memory: {allocated:.3f}GB allocated, {reserved:.3f}GB reserved")
        return allocated, reserved
    return 0, 0

def aggressive_cleanup():
    """Most aggressive cleanup possible"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        torch.cuda.ipc_collect()
    gc.collect() 

def compute_flops_with_handlers(model, flop_inputs):
    """Compute FLOPs with custom handlers for unsupported ops."""
    
    def softmax_flop_jit(inputs, outputs):
        output_shape = get_shape(outputs[0])
        flops = int(np.prod(output_shape)) * 2
        return flops

    def gelu_flop_jit(inputs, outputs):
        input_shape = get_shape(inputs[0])
        flops = int(np.prod(input_shape)) * 8
        return flops

    def sub_flop_jit(inputs, outputs):
        # Element-wise subtraction: 1 flop per element
        output_shape = get_shape(outputs[0])
        return int(np.prod(output_shape))

    def pow_flop_jit(inputs, outputs):
        # Element-wise power: 1 flop per element
        output_shape = get_shape(outputs[0])
        return int(np.prod(output_shape))

    def upsample_bicubic2d_flop_jit(inputs, outputs):
        # Bicubic interpolation: ~32 flops per output element
        # (4x4 neighborhood * 2 ops per point for interpolation)
        output_shape = get_shape(outputs[0])
        return int(np.prod(output_shape)) * 32

    def scaled_dot_product_attention_flop_jit(inputs, outputs):
        # Q, K, V are inputs[0], inputs[1], inputs[2]
        # FLOPs = 2 * seq_len^2 * head_dim (QK^T) + 2 * seq_len^2 * head_dim (AV)
        q_shape = get_shape(inputs[0])  # [B, H, S, D]
        k_shape = get_shape(inputs[1])  # [B, H, S, D]
        
        # q_shape: (batch, heads, seq_len, head_dim)
        batch, heads, seq_len, head_dim = q_shape
        kv_seq_len = k_shape[2]
        
        # QK^T matmul: [B, H, S, D] x [B, H, D, S] -> [B, H, S, S]
        qk_flops = batch * heads * seq_len * kv_seq_len * head_dim * 2
        # Softmax over attention weights
        softmax_flops = batch * heads * seq_len * kv_seq_len * 2
        # AV matmul: [B, H, S, S] x [B, H, S, D] -> [B, H, S, D]
        av_flops = batch * heads * seq_len * head_dim * kv_seq_len * 2
        
        return qk_flops + softmax_flops + av_flops

    model.eval()
    with torch.no_grad():
        flop_counter = FlopCountAnalysis(model, flop_inputs)
        
        flop_counter.set_op_handle("aten::add", elementwise_flop_counter(1, 0))
        flop_counter.set_op_handle("aten::mul", elementwise_flop_counter(1, 0))
        flop_counter.set_op_handle("aten::div", elementwise_flop_counter(1, 0))
        flop_counter.set_op_handle("aten::softmax", softmax_flop_jit)
        flop_counter.set_op_handle("aten::gelu", gelu_flop_jit)
        
        # New handlers for unsupported ops
        flop_counter.set_op_handle("aten::sub", sub_flop_jit)
        flop_counter.set_op_handle("aten::pow", pow_flop_jit)
        flop_counter.set_op_handle("aten::upsample_bicubic2d", upsample_bicubic2d_flop_jit)
        flop_counter.set_op_handle("aten::scaled_dot_product_attention", scaled_dot_product_attention_flop_jit)

    return flop_counter.total()
    
def build_warmup_scheduler(
    optimizer,
    num_epochs: int,
    dataloader_len: int,
    grad_accumulation_steps: int = 1,
    warmup_ratio: float = 0.05,
    schedule_type: str = "cosine",
):
    """
    Builds a LambdaLR scheduler with linear warmup and optional cosine decay.

    Args:
        optimizer: torch optimizer
        num_epochs: number of training epochs
        dataloader_len: len(train_dataloader)
        grad_accumulation_steps: gradient accumulation steps
        warmup_ratio: fraction of total steps for warmup (e.g., 0.05)
        schedule_type: "cosine" or "constant"

    Returns:
        scheduler (LambdaLR)
    """

    total_steps = (num_epochs * dataloader_len) // grad_accumulation_steps
    warmup_steps = int(warmup_ratio * total_steps)

    def lr_lambda(step: int):
        if warmup_steps > 0 and step < warmup_steps:
            return step / max(1, warmup_steps)

        if schedule_type == "constant":
            return 1.0

        # cosine decay
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return LambdaLR(optimizer, lr_lambda)

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")