import torch
import math
import torch.nn as nn
import torch.nn.functional as F  # You were using F.dropout
from torch.nn.functional import (
    # _in_projection,
    # _mha_shape_check,
    dropout,
    linear
)
# from typing import List, Union
from torch import Tensor
from torch.nn import Module
from typing import Optional, Tuple
from torch.nn.init import constant_, xavier_normal_, xavier_uniform_
from timm.models.layers import DropPath
from timm.models.vision_transformer import Mlp
# from torch.nn.modules.linear import NonDynamicallyQuantizableLinear
from transformers.cache_utils import Cache, EncoderDecoderCache
from ragged_attn.ragged_multihead_attention import ragged_attention_kernel
 #, LlamaRotaryEmbedding, rotate_half, apply_rotary_pos_emb
from transformers.models.whisper.modeling_whisper import WhisperEncoder, WhisperDecoder, WhisperEncoderLayer, WhisperDecoderLayer
from transformers.models.vit.modeling_vit import ViTLayer, ViTEncoder, ViTAttention, ViTSelfOutput, ViTConfig
from transformers.models.whisper.configuration_whisper import WhisperConfig
from transformers.models.dinov2.modeling_dinov2 import Dinov2Attention, Dinov2Layer, Dinov2Encoder
from transformers import Dinov2Config
# from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS, dynamic_rope_update
# from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from transformers.activations import ACT2FN
from transformers.models.clip.modeling_clip import CLIPEncoderLayer, CLIPMLP#, eager_attention_forward
from transformers.models.clip.configuration_clip import CLIPConfig
from transformers import BertConfig
# from ragged_attn import ragged_multihead_attention

class AttrDict(dict):
    """Dictionary that allows attribute access."""

    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError:
            raise AttributeError(key)

    def __setattr__(self, key, value):
        self[key] = value


# @torch._dynamo.disable
def memory_efficient_attention(
    q, k, v,
    *,
    valid_mask=None,
    attention_mask=None,
    dropout_p=0.0,
    training=False,
):
    B, H, Tq, Dq = q.shape
    _, _, Tk, _ = k.shape

    # Raw attention scores
    attn = torch.matmul(q, k.transpose(-2, -1))  # [B, H, Tq, Tk]

    # Apply optional attention mask
    if attention_mask is not None:
        attn = attn + attention_mask

    # Mask invalid heads/channels
    if valid_mask is not None:
        mask = valid_mask.any(dim=-1)  # [H]
        h_eff = mask.sum(dtype=torch.float32).clamp_(1)
        d_eff = valid_mask.sum(dtype=torch.float32) / h_eff
        d_eff = d_eff.clamp_(1)
        mask = mask.view(1, H, 1, 1)
        attn = attn.masked_fill(~mask, float('-inf'))
    else:
        d_eff = torch.tensor(Dq, device=q.device, dtype=q.dtype)

    # Scale
    scale = torch.rsqrt(d_eff)
    attn = attn * scale

    # Softmax along last dimension
    attn_final = F.softmax(attn, dim=-1)

    # Optional dropout
    # if training and dropout_p > 0.0:
    #     attn_final = F.dropout(attn_final, p=dropout_p)

    # Weighted sum
    out = torch.matmul(attn_final, v)
    return out


@staticmethod
def chunked_attention(query, key, value, attention_mask=None, chunk_size=1024, dropout_p=0.0):
    """
    Compute attention in chunks to reduce memory usage.
    Works with different head dimensions for Q, K, V.
    """
    batch_size, num_heads_q, seq_len_q, head_dim_q = query.shape
    _, num_heads_k, seq_len_k, head_dim_k = key.shape
    _, num_heads_v, _, head_dim_v = value.shape

    int(batch_size)
    int(num_heads_q)
    int(seq_len_q)
    int(head_dim_q)
    int(num_heads_k)
    int(seq_len_k)
    int(head_dim_k)
    int(num_heads_v)
    int(head_dim_v)
    
    # For small sequences, use standard attention
    if seq_len_q * seq_len_k <= chunk_size * chunk_size:
        # Standard scaled dot-product attention
        scale = 1.0 / math.sqrt(head_dim_q)
        attn_weights = torch.matmul(query, key.transpose(-2, -1)) * scale
        
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask
        
        attn_weights = F.softmax(attn_weights, dim=-1)
        
        if dropout_p > 0.0:
            attn_weights = F.dropout(attn_weights, p=dropout_p)
        
        output = torch.matmul(attn_weights, value)
        return output
    
    # Chunked computation for large sequences
    output = torch.zeros(
        batch_size, num_heads_q, seq_len_q, head_dim_v,
        dtype=query.dtype, device=query.device
    )

    scale = 1.0 / math.sqrt(head_dim_q)
    
    # Process query in chunks
    for q_start in range(0, seq_len_q, chunk_size):
        q_end = min(q_start + chunk_size, seq_len_q)
        query_chunk = query[:, :, q_start:q_end, :]
        
        # Compute attention scores for this query chunk
        attn_weights_chunk = torch.matmul(query_chunk, key.transpose(-2, -1)) * scale
        
        if attention_mask is not None:
            mask_chunk = attention_mask[:, :, q_start:q_end, :]
            attn_weights_chunk = attn_weights_chunk + mask_chunk
        
        attn_weights_chunk = F.softmax(attn_weights_chunk, dim=-1)
        
        if dropout_p > 0.0:
            attn_weights_chunk = F.dropout(attn_weights_chunk, p=dropout_p)
        
        # Compute output for this chunk
        output[:, :, q_start:q_end, :] = torch.matmul(attn_weights_chunk, value)
        
        # Clear intermediate tensors
        del attn_weights_chunk
    
    return output


class RaggedFlashAttentionFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q_list, K_list, V_list, is_causal=False):
        """
        Q_list, K_list, V_list: lists of tensors, each tensor is (B, S, head_dim)
        """
        head_outputs = []
        for q, k, v in zip(Q_list, K_list, V_list):
            # Use fused scaled_dot_product_attention per head
            out = F.scaled_dot_product_attention(q, k, v, is_causal=is_causal)
            head_outputs.append(out)
        # Concatenate along channel dim
        return torch.cat(head_outputs, dim=-1)

    @staticmethod
    def backward(ctx, grad_output):
        # Could implement efficient backward later
        raise NotImplementedError

ragged_flash_attention = RaggedFlashAttentionFunction.apply

class PythonFlashAttention(torch.nn.Module):
    def __init__(self, embed_dim, num_heads, qk_head_dims=None, v_head_dims=None, is_causal=False, original_attn=None):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.is_causal = is_causal
        self.SEQ_BLOCK = 64
       
        # Default: equal split for QK heads
        if qk_head_dims is None:
            d_h = embed_dim // num_heads
            qk_head_dims = [d_h] * num_heads
            remainder = embed_dim - d_h * num_heads
            for i in range(remainder):
                qk_head_dims[i] += 1
               
        # Default: same as QK heads for V heads
        if v_head_dims is None:
            v_head_dims = qk_head_dims.copy()
           
        # Validate dimensions
        assert len(qk_head_dims) == num_heads
        assert len(v_head_dims) == num_heads
       
        self.qk_head_dims = torch.tensor(qk_head_dims, dtype=torch.long)
        self.v_head_dims = torch.tensor(v_head_dims, dtype=torch.long)
        self.total_qk_dims = sum(qk_head_dims)
        self.total_v_dim = sum(v_head_dims)
       
        # Compute offsets for indexing
        self.qk_head_offsets = torch.cat([torch.tensor([0]), self.qk_head_dims.cumsum(0)])
        self.v_head_offsets = torch.cat([torch.tensor([0]), self.v_head_dims.cumsum(0)])
       
        # Projection layers - CORRECTED
        self.q_proj = torch.nn.Linear(embed_dim, self.total_qk_dims)  # embed_dim -> total_qk_dims
        self.k_proj = torch.nn.Linear(embed_dim, self.total_qk_dims)  # embed_dim -> total_qk_dims
        self.v_proj = torch.nn.Linear(embed_dim, self.total_v_dim)    # embed_dim -> total_v_dim
        self.out_proj = torch.nn.Linear(self.total_v_dim, embed_dim)  # total_v_dim -> embed_dim
        
        # Calculate proper BLOCK_SIZE
        max_head_dim = max(max(self.qk_head_dims).item(), max(self.v_head_dims).item())
        self.block_size = max(64, 2**((max_head_dim-1).bit_length()))  # Next power of 2
        
        if original_attn is not None:
            self._copy_attributes(original_attn)
            self._copy_weights(original_attn)
    
    def _copy_attributes(self, original_attention):
        """Copy all relevant attributes from the original attention module"""
        
        # Standard ViT attention attributes
        if hasattr(original_attention, 'num_heads'):
            self.num_heads = original_attention.num_heads
            
        if hasattr(original_attention, 'head_dim'):
            self.head_dim = original_attention.head_dim
        else:
            self.head_dim = self.embed_dim // self.num_heads
            
        if hasattr(original_attention, 'scale'):
            self.scale = original_attention.scale
        else:
            self.scale = self.head_dim ** -0.5
        
        # Copy dropout layers - these are crucial!
        if hasattr(original_attention, 'attn_drop'):
            self.attn_drop = original_attention.attn_drop
        else:
            self.attn_drop = torch.nn.Identity()
            
        if hasattr(original_attention, 'proj_drop'):
            self.proj_drop = original_attention.proj_drop  
        else:
            self.proj_drop = torch.nn.Identity()
            
        # Copy any other attributes that might be important
        for attr_name in ['qk_scale', 'attn_head_dim', 'fused_attn']:
            if hasattr(original_attention, attr_name):
                setattr(self, attr_name, getattr(original_attention, attr_name))
    
    def _copy_weights(self, original_attention):
        """Copy the trained weights from original attention to our projections"""
        with torch.no_grad():
            # Get the combined QKV weights and biases
            qkv_weight = original_attention.qkv.weight.data  # [3*embed_dim, embed_dim]
            qkv_bias = original_attention.qkv.bias.data      # [3*embed_dim]
            
            embed_dim = self.embed_dim
            
            # Split and copy Q, K, V weights
            self.q_proj.weight.copy_(qkv_weight[:embed_dim, :])
            self.k_proj.weight.copy_(qkv_weight[embed_dim:2*embed_dim, :])
            self.v_proj.weight.copy_(qkv_weight[2*embed_dim:, :])
            
            # Split and copy Q, K, V biases
            self.q_proj.bias.copy_(qkv_bias[:embed_dim])
            self.k_proj.bias.copy_(qkv_bias[embed_dim:2*embed_dim])
            self.v_proj.bias.copy_(qkv_bias[2*embed_dim:])
            
            # Copy output projection weights
            self.out_proj.weight.copy_(original_attention.proj.weight.data)
            self.out_proj.bias.copy_(original_attention.proj.bias.data)
    
    def forward(self, x):
        B, S, _ = x.shape  # (batch, sequence, embed_dim)

        # Project inputs
        Q = self.q_proj(x)  # (B, S, total_qk_dims)
        K = self.k_proj(x)  # (B, S, total_qk_dims)
        V = self.v_proj(x)  # (B, S, total_v_dim)

        outputs = []  # collect each head's output

        for h in range(self.num_heads):
            # --- Slice per head ---
            q_start, q_end = self.qk_head_offsets[h].item(), self.qk_head_offsets[h + 1].item()
            v_start, v_end = self.v_head_offsets[h].item(), self.v_head_offsets[h + 1].item()

            q = Q[:, :, q_start:q_end]  # (B, S, d_qk)
            k = K[:, :, q_start:q_end]  # (B, S, d_qk)
            v = V[:, :, v_start:v_end]  # (B, S, d_v)

            # --- Add head dimension ---
            q = q.unsqueeze(1)  # (B, 1, S, d_qk)
            k = k.unsqueeze(1)  # (B, 1, S, d_qk)
            v = v.unsqueeze(1)  # (B, 1, S, d_v)

            # --- Per-head attention ---
            out = torch.nn.functional.scaled_dot_product_attention(
                q, k, v, is_causal=self.is_causal
            )  # (B, 1, S, d_v)

            outputs.append(out.squeeze(1))  # (B, S, d_v)

        # --- Concatenate heads ---
        output = torch.cat(outputs, dim=-1)  # (B, S, total_v_dim)

        # --- Final projection ---
        out = self.out_proj(output)         # (B, S, embed_dim)
        return self.proj_drop(out)


class FlexibleFlashAttention(torch.nn.Module):
    def __init__(self, embed_dim, num_heads, qk_head_dims=None, v_head_dims=None, is_causal=False, original_attn=None):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.is_causal = is_causal
        self.SEQ_BLOCK = 64
       
        # Default: equal split for QK heads
        if qk_head_dims is None:
            d_h = embed_dim // num_heads
            qk_head_dims = [d_h] * num_heads
            remainder = embed_dim - d_h * num_heads
            for i in range(remainder):
                qk_head_dims[i] += 1
               
        # Default: same as QK heads for V heads
        if v_head_dims is None:
            v_head_dims = qk_head_dims.copy()
           
        # Validate dimensions
        assert len(qk_head_dims) == num_heads
        assert len(v_head_dims) == num_heads
       
        self.qk_head_dims = torch.tensor(qk_head_dims, dtype=torch.long)
        self.v_head_dims = torch.tensor(v_head_dims, dtype=torch.long)
        self.total_qk_dims = sum(qk_head_dims)
        self.total_v_dim = sum(v_head_dims)
       
        # Compute offsets for indexing
        self.qk_head_offsets = torch.cat([torch.tensor([0]), self.qk_head_dims.cumsum(0)])
        self.v_head_offsets = torch.cat([torch.tensor([0]), self.v_head_dims.cumsum(0)])
       
        # Projection layers - CORRECTED
        self.q_proj = torch.nn.Linear(embed_dim, self.total_qk_dims)  # embed_dim -> total_qk_dims
        self.k_proj = torch.nn.Linear(embed_dim, self.total_qk_dims)  # embed_dim -> total_qk_dims
        self.v_proj = torch.nn.Linear(embed_dim, self.total_v_dim)    # embed_dim -> total_v_dim
        self.out_proj = torch.nn.Linear(self.total_v_dim, embed_dim)  # total_v_dim -> embed_dim
        
        # Calculate proper BLOCK_SIZE
        max_head_dim = max(max(self.qk_head_dims).item(), max(self.v_head_dims).item())
        self.block_size = max(64, 2**((max_head_dim-1).bit_length()))  # Next power of 2
        
        if original_attn is not None:
            self._copy_attributes(original_attn)
            self._copy_weights(original_attn)
    
    def _copy_attributes(self, original_attention):
        """Copy all relevant attributes from the original attention module"""
        
        # Standard ViT attention attributes
        if hasattr(original_attention, 'num_heads'):
            self.num_heads = original_attention.num_heads
            
        if hasattr(original_attention, 'head_dim'):
            self.head_dim = original_attention.head_dim
        else:
            self.head_dim = self.embed_dim // self.num_heads
            
        if hasattr(original_attention, 'scale'):
            self.scale = original_attention.scale
        else:
            self.scale = self.head_dim ** -0.5
        
        # Copy dropout layers - these are crucial!
        if hasattr(original_attention, 'attn_drop'):
            self.attn_drop = original_attention.attn_drop
        else:
            self.attn_drop = torch.nn.Identity()
            
        if hasattr(original_attention, 'proj_drop'):
            self.proj_drop = original_attention.proj_drop  
        else:
            self.proj_drop = torch.nn.Identity()
            
        # Copy any other attributes that might be important
        for attr_name in ['qk_scale', 'attn_head_dim', 'fused_attn']:
            if hasattr(original_attention, attr_name):
                setattr(self, attr_name, getattr(original_attention, attr_name))
    
    def _copy_weights(self, original_attention):
        """Copy the trained weights from original attention to our projections"""
        with torch.no_grad():
            # Get the combined QKV weights and biases
            qkv_weight = original_attention.qkv.weight.data  # [3*embed_dim, embed_dim]
            qkv_bias = original_attention.qkv.bias.data      # [3*embed_dim]
            
            embed_dim = self.embed_dim
            
            # Split and copy Q, K, V weights
            self.q_proj.weight.copy_(qkv_weight[:embed_dim, :])
            self.k_proj.weight.copy_(qkv_weight[embed_dim:2*embed_dim, :])
            self.v_proj.weight.copy_(qkv_weight[2*embed_dim:, :])
            
            # Split and copy Q, K, V biases
            self.q_proj.bias.copy_(qkv_bias[:embed_dim])
            self.k_proj.bias.copy_(qkv_bias[embed_dim:2*embed_dim])
            self.v_proj.bias.copy_(qkv_bias[2*embed_dim:])
            
            # Copy output projection weights
            self.out_proj.weight.copy_(original_attention.proj.weight.data)
            self.out_proj.bias.copy_(original_attention.proj.bias.data)
    
    def forward(self, x):
        B, S, _ = x.shape
        
        # Project inputs
        Q = self.q_proj(x)  # (B, S, total_qk_dims)
        K = self.k_proj(x)  # (B, S, total_qk_dims)
        V = self.v_proj(x)  # (B, S, total_v_dim)
        
        # Flatten for kernel
        Q_flat = Q.reshape(B * S * self.total_qk_dims)
        K_flat = K.reshape(B * S * self.total_qk_dims)
        V_flat = V.reshape(B * S * self.total_v_dim)
        
        # Output tensor
        output = torch.zeros((B * S * self.total_v_dim), device=x.device, dtype=x.dtype)
        
        # FIXED: Correct grid - one thread per (batch, sequence_position, head)
        grid = (B, S, self.num_heads)
        
        ragged_attention_kernel[grid](
            Q_flat, K_flat, V_flat, output,
            B, S,
            self.qk_head_offsets.to(x.device),
            self.v_head_offsets.to(x.device),
            self.total_qk_dims,
            self.total_v_dim,
            BLOCK_SIZE=self.block_size,
            BLOCK_SIZE_K=self.SEQ_BLOCK,
            CAUSAL=self.is_causal
        )
        
        output = output.reshape(B, S, self.total_v_dim)
        
        # Final projection
        out = self.out_proj(output)
        return self.proj_drop(out)

# class RaggedAttention(torch.nn.Module):
#     def __init__(self, embed_dim, num_heads, qk_head_dims=None, v_head_dims=None):
#         super().__init__()
#         self.embed_dim = embed_dim
#         self.num_heads = num_heads
        
#         # Default: equal split for QK heads
#         if qk_head_dims is None:
#             d_h = embed_dim // num_heads
#             qk_head_dims = [d_h] * num_heads
#             print(d_h)
#             remainder = embed_dim - d_h * num_heads
#             for i in range(remainder):
#                 qk_head_dims[i] += 1
                
#         # Default: same as QK heads for V heads
#         if v_head_dims is None:
#             v_head_dims = qk_head_dims.copy()
            
#         # Validate dimensions
#         assert len(qk_head_dims) == num_heads, f"qk_head_dims length {len(qk_head_dims)} != num_heads {num_heads}"
#         assert len(v_head_dims) == num_heads, f"v_head_dims length {len(v_head_dims)} != num_heads {num_heads}"
#         assert sum(qk_head_dims) == embed_dim, f"qk_head_dims sum {sum(qk_head_dims)} != embed_dim {embed_dim}"
        
#         self.qk_head_dims = torch.tensor(qk_head_dims, dtype=torch.long)
#         self.v_head_dims = torch.tensor(v_head_dims, dtype=torch.long)
#         self.total_v_dim = sum(v_head_dims)
        
#         # Compute offsets for indexing
#         self.qk_head_offsets = torch.cat([torch.tensor([0]), self.qk_head_dims.cumsum(0)])
#         self.v_head_offsets = torch.cat([torch.tensor([0]), self.v_head_dims.cumsum(0)])
        
#         # Projection layers
#         self.q_proj = torch.nn.Linear(embed_dim, embed_dim)  # Q: embed_dim -> embed_dim (QK dims)
#         self.k_proj = torch.nn.Linear(embed_dim, embed_dim)  # K: embed_dim -> embed_dim (QK dims)
#         self.v_proj = torch.nn.Linear(embed_dim, self.total_v_dim)  # V: embed_dim -> total_v_dim (V dims)
#         self.out_proj = torch.nn.Linear(self.total_v_dim, embed_dim)  # Output: total_v_dim -> embed_dim
        
#         # Precompute scale factor
#         self.scale = 1.0 / (sum(qk_head_dims) / num_heads) ** 0.5  # Average head dimension for scaling
        
#     def forward(self, x, mask: Optional[torch.Tensor] = None):
#         B, T, C = x.shape
#         device = x.device
        
#         # Project inputs
#         Q = self.q_proj(x).contiguous()
#         K = self.k_proj(x).contiguous() 
#         V = self.v_proj(x).contiguous()
        
#         # Move head dimensions to device
#         qk_dims = self.qk_head_dims.to(device)
#         v_dims = self.v_head_dims.to(device)
        
#         # Call the CUDA kernel with separate QK and V head dimensions
#         # Note: The mask parameter isn't used in the current CUDA implementation
#         # which uses causal masking. You may want to extend the CUDA code to support custom masks.
#         out = ragged_multihead_attention(
#             Q, K, V, 
#             qk_dims.tolist(),  # Convert to list for the C++ interface
#             v_dims.tolist(),   # Convert to list for the C++ interface
#             self.scale
#         )
        
#         # Final output projection
#         out = self.out_proj(out)  # [B, T, embed_dim]
#         return out

def softmax(weights, scores, dim):
    
    num = weights*torch.exp(scores)
    den = torch.sum(weights*torch.exp(scores), dim = dim)
    
    return num/(den.unsqueeze(-1) + 1e-8)

# def _in_projection_packed(
#     q: Tensor,
#     k: Tensor,
#     v: Tensor,
#     w: Tensor,
#     num_heads: List[int],  # [num_heads_q, num_heads_k, num_heads_v]
#     head_dims: List[int],  # [head_dim_q, head_dim_k, head_dim_v]
#     b: Optional[Tensor] = None,
# ) -> List[Tensor]:
#     r"""Perform the in-projection step of the attention operation, using packed weights.

#     Output is a triple containing projection tensors for query, key and value.

#     Args:
#         q, k, v: query, key and value tensors to be projected. For self-attention,
#             these are typically the same tensor; for encoder-decoder attention,
#             k and v are typically the same tensor. (We take advantage of these
#             identities for performance if they are present.) Regardless, q, k and v
#             must share a common embedding dimension; otherwise their shapes may vary.
#         w: projection weights for q, k and v, packed into a single tensor. Weights
#             are packed along dimension 0, in q, k, v order.
#         num_heads: list of number of heads for [q, k, v]
#         head_dims: list of head dimensions for [q, k, v]
#         b: optional projection biases for q, k and v, packed into a single tensor
#             in q, k, v order.

#     Shape:
#         Inputs:
#         - q: :math:`(..., E)` where E is the embedding dimension
#         - k: :math:`(..., E)` where E is the embedding dimension
#         - v: :math:`(..., E)` where E is the embedding dimension
#         - w: :math:`(num_heads[0] * head_dims[0] + num_heads[1] * head_dims[1] + num_heads[2] * head_dims[2], E)`
#         - b: :math:`num_heads[0] * head_dims[0] + num_heads[1] * head_dims[1] + num_heads[2] * head_dims[2]`

#         Output:
#         - in output list :math:`[q', k', v']`, each output tensor will have the
#             shape corresponding to its specific num_heads * head_dim configuration.
#     """
#     E = q.size(-1)
    
#     # Calculate output dimensions for each projection
#     q_out_dim = num_heads[0] * head_dims[0]
#     k_out_dim = num_heads[1] * head_dims[1]
#     v_out_dim = num_heads[2] * head_dims[2]
    
#     if k is v:
#         if q is k:
#             # self-attention - but now with potentially different output dimensions
#             proj = linear(q, w, b)
#             # Split based on actual output dimensions
#             q_proj = proj[..., :q_out_dim]
#             k_proj = proj[..., q_out_dim:q_out_dim + k_out_dim]
#             v_proj = proj[..., q_out_dim + k_out_dim:q_out_dim + k_out_dim + v_out_dim]
#             return q_proj, k_proj, v_proj
#         else:
#             # encoder-decoder attention
#             w_q, w_kv = w.split([q_out_dim, k_out_dim + v_out_dim])
#             if b is None:
#                 b_q = b_kv = None
#             else:
#                 b_q, b_kv = b.split([q_out_dim, k_out_dim + v_out_dim])
#             q_proj = linear(q, w_q, b_q)
#             kv_proj = linear(k, w_kv, b_kv)
#             k_proj = kv_proj[..., :k_out_dim]
#             v_proj = kv_proj[..., k_out_dim:]
#             return q_proj, k_proj, v_proj
#     else:
#         # Different tensors for q, k, v
#         w_q, w_k, w_v = w.split([q_out_dim, k_out_dim, v_out_dim])
#         if b is None:
#             b_q = b_k = b_v = None
#         else:
#             b_q, b_k, b_v = b.split([q_out_dim, k_out_dim, v_out_dim])
#         return linear(q, w_q, b_q), linear(k, w_k, b_k), linear(v, w_v, b_v)
    

# def scaled_dot_product_attention(q, k, v,
#                                  attn_mask=None,
#                                  dropout_p=0.0,
#                                  is_causal=False,
#                                  scale=None):
#     """
#     Pure-PyTorch scaled dot-product attention.

#     Args:
#       q, k, v: tensors with shape (..., seq_len, dim)
#       attn_mask: optional mask (..., L, S), bool or float
#       dropout_p: dropout probability
#       is_causal: if True, use lower-triangular causal mask
#       scale: optional scale, default = 1/sqrt(dim)
#     Returns:
#       output tensor (..., L, dim)
#     """
#     dim = q.size(-1)
#     scale_factor = scale if scale is not None else 1.0 / math.sqrt(dim)
    
#     scores = torch.matmul(q, k.transpose(-2, -1)) * scale_factor
    
#     if attn_mask is not None:
#         if attn_mask.dtype == torch.bool:
#             scores = scores.masked_fill(~attn_mask, float('-inf'))
#         else:
#             scores = scores + attn_mask
    
#     if is_causal:
#         L, S = scores.shape[-2], scores.shape[-1]
#         causal_mask = torch.triu(torch.ones(L, S, dtype=torch.bool, device=scores.device), diagonal=1)
#         scores = scores.masked_fill(causal_mask, float('-inf'))
    
#     # if WEIGHTED_SOFTMAX and not STRUCTURAL_PRUNING:
#     #     weights = (scores != 0).to(torch.float)
#     #     attn = softmax(weights, scores, dim=-1)
#     #     del weights  # Free weights tensor
#     # else:
#     attn = torch.softmax(scores, dim=-1)
    
#     #attn = softmax(weights, scores, dim=-1)
#     if dropout_p > 0:
#         attn = F.dropout(attn, p=dropout_p, training=True)
    
#     return torch.matmul(attn, v)

# def multi_head_attention_forward(
#     query: Tensor,
#     key: Tensor,
#     value: Tensor,
#     embed_dim_to_check: int,
#     num_heads: List[int],  # [num_heads_q, num_heads_k, num_heads_v] 
#     head_dim: List[int],  # [num_heads_q, num_heads_k, num_heads_v] 
#     in_proj_weight: Optional[Tensor],
#     in_proj_bias: Optional[Tensor],
#     bias_k: Optional[Tensor],
#     bias_v: Optional[Tensor],
#     add_zero_attn: bool,
#     dropout_p: float,
#     out_proj_weight: Tensor,
#     out_proj_bias: Optional[Tensor],
#     training: bool = True,
#     key_padding_mask: Optional[Tensor] = None,
#     need_weights: bool = True,
#     attn_mask: Optional[Tensor] = None,
#     use_separate_proj_weight: bool = False,
#     q_proj_weight: Optional[Tensor] = None,
#     k_proj_weight: Optional[Tensor] = None,
#     v_proj_weight: Optional[Tensor] = None,
#     static_k: Optional[Tensor] = None,
#     static_v: Optional[Tensor] = None,
#     average_attn_weights: bool = True,
#     is_causal: bool = False,
# ) -> Tuple[Tensor, Optional[Tensor]]:
#     tens_ops = (
#         query,
#         key,
#         value,
#         in_proj_weight,
#         in_proj_bias,
#         bias_k,
#         bias_v,
#         out_proj_weight,
#         out_proj_bias,
#     )
    
#     num_heads_q, num_heads_k, num_heads_v = num_heads
#     head_dim_q, head_dim_k, head_dim_v = head_dim

#     is_batched = _mha_shape_check(
#         query, key, value, key_padding_mask, attn_mask, num_heads_q
#     )

#     # For unbatched input, we unsqueeze at the expected batch-dim to pretend that the input
#     # is batched, run the computation and before returning squeeze the
#     # batch dimension so that the output doesn't carry this temporary batch dimension.
#     if not is_batched:
#         # unsqueeze if the input is unbatched
#         query = query.unsqueeze(1)
#         key = key.unsqueeze(1)
#         value = value.unsqueeze(1)
#         if key_padding_mask is not None:
#             key_padding_mask = key_padding_mask.unsqueeze(0)

#     # set up shape vars
#     tgt_len, bsz, embed_dim = query.shape
#     src_len, _, _ = key.shape

#     #
#     # compute in-projection
#     #
#     if not use_separate_proj_weight:
#         assert (
#             in_proj_weight is not None
#         ), "use_separate_proj_weight is False but in_proj_weight is None"
#         q, k, v = _in_projection_packed(query, key, value, in_proj_weight, num_heads, head_dim, in_proj_bias)
#     else:
#         assert (
#             q_proj_weight is not None
#         ), "use_separate_proj_weight is True but q_proj_weight is None"
#         assert (
#             k_proj_weight is not None
#         ), "use_separate_proj_weight is True but k_proj_weight is None"
#         assert (
#             v_proj_weight is not None
#         ), "use_separate_proj_weight is True but v_proj_weight is None"
#         if in_proj_bias is None:
#             b_q = b_k = b_v = None
#         else:
#             b_q, b_k, b_v = in_proj_bias.split([num_heads_q * head_dim_q, num_heads_k * head_dim_k, num_heads_v * head_dim_v])
#         q, k, v = _in_projection(
#             query,
#             key,
#             value,
#             q_proj_weight,
#             k_proj_weight,
#             v_proj_weight,
#             b_q,
#             b_k,
#             b_v,
#         )

#     # prep attention mask

#     if attn_mask is not None:
#         # ensure attn_mask's dim is 3
#         if attn_mask.dim() == 2:
#             correct_2d_size = (tgt_len, src_len)
#             if attn_mask.shape != correct_2d_size:
#                 raise RuntimeError(
#                     f"The shape of the 2D attn_mask is {attn_mask.shape}, but should be {correct_2d_size}."
#                 )
#             attn_mask = attn_mask.unsqueeze(0)
#         elif attn_mask.dim() == 3:
#             correct_3d_size = (bsz * num_heads, tgt_len, src_len)
#             if attn_mask.shape != correct_3d_size:
#                 raise RuntimeError(
#                     f"The shape of the 3D attn_mask is {attn_mask.shape}, but should be {correct_3d_size}."
#                 )
#         else:
#             raise RuntimeError(
#                 f"attn_mask's dimension {attn_mask.dim()} is not supported"
#             )

#     # add bias along batch dimension (currently second)
#     if bias_k is not None and bias_v is not None:
#         assert static_k is None, "bias cannot be added to static key."
#         assert static_v is None, "bias cannot be added to static value."
#         k = torch.cat([k, bias_k.repeat(1, bsz, 1)])
#         v = torch.cat([v, bias_v.repeat(1, bsz, 1)])
#     else:
#         assert bias_k is None
#         assert bias_v is None

#     #
#     # reshape q, k, v for multihead attention and make them batch first
#     #
#     q = q.contiguous().view(tgt_len, bsz * num_heads_q, head_dim_q).transpose(0, 1)
#     if static_k is None:
#         k = k.contiguous().view(k.shape[0], bsz * num_heads_k, head_dim_k).transpose(0, 1)
#     else:
#         # TODO finish disentangling control flow so we don't do in-projections when statics are passed
#         assert (
#             static_k.size(0) == bsz * num_heads
#         ), f"expecting static_k.size(0) of {bsz * num_heads}, but got {static_k.size(0)}"
#         assert (
#             static_k.size(2) == head_dim
#         ), f"expecting static_k.size(2) of {head_dim}, but got {static_k.size(2)}"
#         k = static_k
#     if static_v is None:
#         v = v.contiguous().view(v.shape[0], bsz * num_heads_v, head_dim_v).transpose(0, 1)
#     else:
#         # TODO finish disentangling control flow so we don't do in-projections when statics are passed
#         assert (
#             static_v.size(0) == bsz * num_heads
#         ), f"expecting static_v.size(0) of {bsz * num_heads}, but got {static_v.size(0)}"
#         assert (
#             static_v.size(2) == head_dim
#         ), f"expecting static_v.size(2) of {head_dim}, but got {static_v.size(2)}"
#         v = static_v

#     # add zero attention along batch dimension (now first)
#     if add_zero_attn:
#         zero_attn_shape = (bsz * num_heads, 1, head_dim)
#         k = torch.cat(
#             [k, torch.zeros(zero_attn_shape, dtype=k.dtype, device=k.device)], dim=1
#         )
#         v = torch.cat(
#             [v, torch.zeros(zero_attn_shape, dtype=v.dtype, device=v.device)], dim=1
#         )

#     # update source sequence length after adjustments
#     src_len = k.size(1)

#     # merge key padding and attention masks
#     if key_padding_mask is not None:
#         assert key_padding_mask.shape == (
#             bsz,
#             src_len,
#         ), f"expecting key_padding_mask shape of {(bsz, src_len)}, but got {key_padding_mask.shape}"
#         key_padding_mask = (
#             key_padding_mask.view(bsz, 1, 1, src_len)
#             .expand(-1, num_heads, -1, -1)
#             .reshape(bsz * num_heads, 1, src_len)
#         )
#         if attn_mask is None:
#             attn_mask = key_padding_mask
#         else:
#             attn_mask = attn_mask + key_padding_mask

#     # adjust dropout probability
#     if not training:
#         dropout_p = 0.0

#     #
#     # (deep breath) calculate attention and out projection
#     #

#     if need_weights:
#         B, Nt, E = q.shape
#         q_scaled = q * math.sqrt(1.0 / float(E))

#         assert not (
#             is_causal and attn_mask is None
#         ), "FIXME: is_causal not implemented for need_weights"

#         if attn_mask is not None:
#             attn_output_weights = torch.baddbmm(
#                 attn_mask, q_scaled, k.transpose(-2, -1)
#             )
#         else:
#             attn_output_weights = torch.bmm(q_scaled, k.transpose(-2, -1))
#         attn_output_weights = softmax(attn_output_weights, dim=-1)
#         if dropout_p > 0.0:
#             attn_output_weights = dropout(attn_output_weights, p=dropout_p)

#         attn_output = torch.bmm(attn_output_weights, v)

#         attn_output = (
#             attn_output.transpose(0, 1).contiguous().view(tgt_len * bsz, num_heads_v * head_dim_v)
#         )
#         attn_output = linear(attn_output, out_proj_weight, out_proj_bias)
#         attn_output = attn_output.view(tgt_len, bsz, attn_output.size(1))

#         # optionally average attention weights over heads
#         attn_output_weights = attn_output_weights.view(bsz, num_heads_v, tgt_len, src_len)
#         if average_attn_weights:
#             attn_output_weights = attn_output_weights.mean(dim=1)

#         if not is_batched:
#             # squeeze the output if input was unbatched
#             attn_output = attn_output.squeeze(1)
#             attn_output_weights = attn_output_weights.squeeze(0)
#         return attn_output, attn_output_weights
#     else:
#         # attn_mask can be either (L,S) or (N*num_heads, L, S)
#         # if attn_mask's shape is (1, L, S) we need to unsqueeze to (1, 1, L, S)
#         # in order to match the input for SDPA of (N, num_heads, L, S)
#         if attn_mask is not None:
#             if attn_mask.size(0) == 1 and attn_mask.dim() == 3:
#                 attn_mask = attn_mask.unsqueeze(0)
#             else:
#                 attn_mask = attn_mask.view(bsz, num_heads, -1, src_len)

#         q = q.view(bsz, num_heads_q, tgt_len, head_dim_q)
#         k = k.view(bsz, num_heads_k, src_len, head_dim_k)
#         v = v.view(bsz, num_heads_v, src_len, head_dim_v)

#         attn_output = scaled_dot_product_attention(
#             q, k, v, attn_mask, dropout_p, is_causal
#         )
#         # attn_output shape: (bsz, num_heads, tgt_len, head_dim_v)
        
#         attn_output = (
#             attn_output.permute(2, 0, 1, 3).contiguous().view(bsz * tgt_len, num_heads_v * head_dim_v)
#         )

#         attn_output = linear(attn_output, out_proj_weight, out_proj_bias)
#         attn_output = attn_output.view(tgt_len, bsz, attn_output.size(1))
#         if not is_batched:
#             # squeeze the output if input was unbatched
#             attn_output = attn_output.squeeze(1)
#         return attn_output, None

class MultiheadAttention(Module):
    __constants__ = ["batch_first"]
    bias_k: Optional[torch.Tensor]
    bias_v: Optional[torch.Tensor]

    def __init__(
        self,
        embed_dim,
        num_heads,
        num_heads_k=None,
        num_heads_v=None,
        num_heads_out=None,
        head_dim=None,
        head_dim_k=None, 
        head_dim_v=None,
        head_dim_out=None,
        dropout=0.0,
        bias=True,
        add_bias_kv=False,
        add_zero_attn=False,
        kdim=None,
        vdim=None,
        batch_first=False,
        device=None,
        dtype=None,
    ) -> None:
        if embed_dim <= 0 or num_heads <= 0:
            raise ValueError(
                f"embed_dim and num_heads must be greater than 0,"
                f" got embed_dim={embed_dim} and num_heads={num_heads} instead"
            )
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.embed_dim = embed_dim
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim
        self._qkv_same_embed_dim = self.kdim == embed_dim and self.vdim == embed_dim

        self.num_heads_q = num_heads
        self.num_heads_k = num_heads_k if num_heads_k is not None else num_heads
        self.num_heads_v = num_heads_v if num_heads_v is not None else num_heads
        self.num_heads_out = num_heads_out if num_heads_out is not None else num_heads
        
        self.dropout = dropout
        self.batch_first = batch_first
        self.add_zero_attn = add_zero_attn
        self.head_dim_q = head_dim if head_dim is not None else embed_dim // self.num_heads_q
        self.head_dim_k = head_dim_k if head_dim_k is not None else embed_dim // self.num_heads_k  
        self.head_dim_v = head_dim_v if head_dim_v is not None else embed_dim // self.num_heads_v
        self.head_dim_out = head_dim_out if head_dim_out is not None else embed_dim // self.num_heads_out
        
        self.q_out_dim = self.num_heads_q * self.head_dim_q
        self.k_out_dim = self.num_heads_k * self.head_dim_k
        self.v_out_dim = self.num_heads_v * self.head_dim_v
        
        self.in_proj_weight = torch.nn.Linear(self.embed_dim, self.q_out_dim+self.k_out_dim+self.v_out_dim, bias=bias, **factory_kwargs)

        self.out_proj = torch.nn.Linear(self.v_out_dim, embed_dim, bias=bias, **factory_kwargs)

        self.add_zero_attn = add_zero_attn

        # self._reset_parameters()

    def _reset_parameters(self):
        if self._qkv_same_embed_dim:
            xavier_uniform_(self.in_proj_weight)
        else:
            xavier_uniform_(self.q_proj_weight)
            xavier_uniform_(self.k_proj_weight)
            xavier_uniform_(self.v_proj_weight)

        if self.in_proj_bias is not None:
            constant_(self.in_proj_bias, 0.0)
            constant_(self.out_proj.bias, 0.0)
        if self.bias_k is not None:
            xavier_normal_(self.bias_k)
        if self.bias_v is not None:
            xavier_normal_(self.bias_v)

    def __setstate__(self, state):
        # Support loading old MultiheadAttention checkpoints generated by v1.1.0
        if "_qkv_same_embed_dim" not in state:
            state["_qkv_same_embed_dim"] = True

        super().__setstate__(state)

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        key_padding_mask: Optional[Tensor] = None,
        need_weights: bool = True,
        attn_mask: Optional[Tensor] = None,
        average_attn_weights: bool = True,
        is_causal: bool = False,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        is_batched = query.dim() == 3

        key_padding_mask = F._canonical_mask(
            mask=key_padding_mask,
            mask_name="key_padding_mask",
            other_type=F._none_or_dtype(attn_mask),
            other_name="attn_mask",
            target_type=query.dtype,
        )

        attn_mask = F._canonical_mask(
            mask=attn_mask,
            mask_name="attn_mask",
            other_type=None,
            other_name="",
            target_type=query.dtype,
            check_other=False,
        )

        if not is_batched:
            # unsqueeze if the input is unbatched
            query = query.unsqueeze(1)
            key = key.unsqueeze(1)
            value = value.unsqueeze(1)
            if key_padding_mask is not None:
                key_padding_mask = key_padding_mask.unsqueeze(0)

        if self.batch_first and is_batched:
            # make sure that the transpose op does not affect the "is" property
            if key is value:
                if query is key:
                    query = key = value = query.transpose(1, 0)
                else:
                    query, key = (x.transpose(1, 0) for x in (query, key))
                    value = key
            else:
                query, key, value = (x.transpose(1, 0) for x in (query, key, value))
        
        # set up shape vars
        tgt_len, bsz, embed_dim = query.shape
        src_len, _, _ = key.shape
        
        if key is query and value is query:
            # Self-attention
            proj = self.in_proj_weight(query)
            q = proj[..., :self.q_out_dim]
            k = proj[..., self.q_out_dim:self.q_out_dim + self.k_out_dim]
            v = proj[..., self.q_out_dim + self.k_out_dim:]
        else:
            # Cross-attention - use F.linear with sliced weights
            q = F.linear(
                query,
                self.in_proj_weight.weight[:self.q_out_dim],
                self.in_proj_weight.bias[:self.q_out_dim] if self.in_proj_weight.bias is not None else None
            )
            k = F.linear(
                key,
                self.in_proj_weight.weight[self.q_out_dim:self.q_out_dim + self.k_out_dim],
                self.in_proj_weight.bias[self.q_out_dim:self.q_out_dim + self.k_out_dim] if self.in_proj_weight.bias is not None else None
            )
            v = F.linear(
                value,
                self.in_proj_weight.weight[self.q_out_dim + self.k_out_dim:],
                self.in_proj_weight.bias[self.q_out_dim + self.k_out_dim:] if self.in_proj_weight.bias is not None else None
            )

        # reshape q, k, v for multihead attention and make them batch first
        q = q.contiguous().view(tgt_len, bsz * self.num_heads_q, self.head_dim_q).transpose(0, 1)
        k = k.contiguous().view(src_len, bsz * self.num_heads_k, self.head_dim_k).transpose(0, 1)
        v = v.contiguous().view(src_len, bsz * self.num_heads_v, self.head_dim_v).transpose(0, 1)

        # add zero attention along batch dimension (now first)
        if self.add_zero_attn:
            zero_attn_shape = (bsz * self.num_heads_q, 1, self.head_dim_q)
            k = torch.cat([k, torch.zeros(zero_attn_shape, dtype=k.dtype, device=k.device)], dim=1)
            v = torch.cat([v, torch.zeros(zero_attn_shape, dtype=v.dtype, device=v.device)], dim=1)

        # update source sequence length after adjustments
        src_len = k.size(1)

        # merge key padding and attention masks
        if key_padding_mask is not None:
            assert key_padding_mask.shape == (
                bsz,
                src_len,
            ), f"expecting key_padding_mask shape of {(bsz, src_len)}, but got {key_padding_mask.shape}"
            key_padding_mask = (
                key_padding_mask.view(bsz, 1, 1, src_len)
                .expand(-1, self.num_heads_k, -1, -1)
                .reshape(bsz * self.num_heads_k, 1, src_len)
            )
            if attn_mask is None:
                attn_mask = key_padding_mask
            else:
                attn_mask = attn_mask + key_padding_mask

        # adjust dropout probability
        dropout_p = self.dropout if self.training else 0.0
        
        if need_weights:
            B, Nt, E = q.shape
            q_scaled = q * math.sqrt(1.0 / float(E))

            assert not (
                is_causal and attn_mask is None
            ), "FIXME: is_causal not implemented for need_weights"

            if attn_mask is not None:
                attn_output_weights = torch.baddbmm(
                    attn_mask, q_scaled, k.transpose(-2, -1)
                )
            else:
                attn_output_weights = torch.bmm(q_scaled, k.transpose(-2, -1))
            attn_output_weights = softmax(attn_output_weights, dim=-1)
            if dropout_p > 0.0:
                attn_output_weights = dropout(attn_output_weights, p=dropout_p)

            attn_output = torch.bmm(attn_output_weights, v)

            attn_output = (
                attn_output.transpose(0, 1).contiguous().view(tgt_len * bsz, self.num_heads_v * self.head_dim_v)
            )
            attn_output = self.out_proj(attn_output)
            attn_output = attn_output.view(tgt_len, bsz, attn_output.size(1))

            # optionally average attention weights over heads
            attn_output_weights = attn_output_weights.view(bsz, self.num_heads_v, tgt_len, src_len)
            if average_attn_weights:
                attn_output_weights = attn_output_weights.mean(dim=1)

            if not is_batched:
                # squeeze the output if input was unbatched
                attn_output = attn_output.squeeze(1)
                attn_output_weights = attn_output_weights.squeeze(0)
            
            if self.batch_first and is_batched:
                return attn_output.transpose(1, 0), attn_output_weights
            else:
                return attn_output, attn_output_weights
        else:
            # attn_mask can be either (L,S) or (N*num_heads, L, S)
            # if attn_mask's shape is (1, L, S) we need to unsqueeze to (1, 1, L, S)
            # in order to match the input for SDPA of (N, num_heads, L, S)
            if attn_mask is not None:
                if attn_mask.size(0) == 1 and attn_mask.dim() == 3:
                    attn_mask = attn_mask.unsqueeze(0)
                else:
                    attn_mask = attn_mask.view(bsz, self.num_heads_k, -1, src_len)

            q = q.view(bsz, self.num_heads_q, tgt_len, self.head_dim_q)
            k = k.view(bsz, self.num_heads_k, src_len, self.head_dim_k)
            v = v.view(bsz, self.num_heads_v, src_len, self.head_dim_v)
            # with torch.nn.attention.sdpa_kernel(backends=[torch.nn.attention.SDPBackend.MATH]):
            attn_output = torch.nn.functional.scaled_dot_product_attention(
                q, k, v, attn_mask, dropout_p, is_causal, 
                scale=1.0 / math.sqrt(self.head_dim_q)
            )
            # attn_output shape: (bsz, num_heads_v, tgt_len, head_dim_v)
            
            attn_output = (
                attn_output.permute(2, 0, 1, 3)
                .contiguous()
                .view(bsz * tgt_len, self.num_heads_v * self.head_dim_v)
            )

            attn_output = self.out_proj(attn_output)
            attn_output = attn_output.view(tgt_len, bsz, attn_output.size(1))
            
            if not is_batched:
                # squeeze the output if input was unbatched
                attn_output = attn_output.squeeze(1)

            if self.batch_first and is_batched:
                return attn_output.transpose(1, 0), None
            else:
                return attn_output, None
        
def _check_arg_device(x: Optional[torch.Tensor]) -> bool:
    if x is not None:
        return x.device.type in [
            "cpu",
            "cuda",
            torch.utils.backend_registration._privateuse1_backend_name,
        ]
    return True


def _arg_requires_grad(x: Optional[torch.Tensor]) -> bool:
    if x is not None:
        return x.requires_grad
    return False


def _is_make_fx_tracing():
    if not torch.jit.is_scripting():
        torch_dispatch_mode_stack = (
            torch.utils._python_dispatch._get_current_dispatch_mode_stack()
        )
        return any(
            type(x) == torch.fx.experimental.proxy_tensor.ProxyTorchDispatchMode
            for x in torch_dispatch_mode_stack
        )
    else:
        return False
    
    
class Attention(torch.nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., 
                 num_heads_k=None, num_heads_v=None, num_heads_o=None, 
                 head_dim_q=None, head_dim_k=None, head_dim_v=None, head_dim_o=None):
        super().__init__()
        self.num_heads_q = num_heads
        self.num_heads_k = num_heads_k if num_heads_k is not None else num_heads
        self.num_heads_v = num_heads_v if num_heads_v is not None else num_heads
        self.num_heads_out = num_heads_o if num_heads_o is not None else num_heads
        
        self.head_dim_q = head_dim_q if head_dim_q is not None else dim // num_heads
        self.head_dim_k = head_dim_k if head_dim_k is not None else dim // num_heads
        self.head_dim_v = head_dim_v if head_dim_v is not None else dim // num_heads
        self.head_dim_out = head_dim_o if head_dim_o is not None else dim // num_heads
        
        self.q_out_dim = self.head_dim_q * self.num_heads_q
        self.k_out_dim = self.head_dim_k * self.num_heads_k
        self.v_out_dim = self.head_dim_v * self.num_heads_v
        
        self.embed_dim = dim
        
        self.scale = qk_scale or self.head_dim_q ** -0.5

        self.qkv = torch.nn.Linear(self.embed_dim, (self.q_out_dim+self.k_out_dim+self.v_out_dim), bias=qkv_bias)
        self.attn_drop = torch.nn.Dropout(attn_drop)
        self.proj = torch.nn.Linear(self.v_out_dim, self.embed_dim)
        self.proj_drop = torch.nn.Dropout(proj_drop)

    def forward(self, x, key=None, value=None):
        B, N, C = x.shape #batch, num_token, embed_dim
        qkv = self.qkv(x).reshape(B, N, (self.q_out_dim+self.k_out_dim+self.v_out_dim))
        q, k, v = qkv[..., :self.q_out_dim], qkv[..., self.q_out_dim:self.q_out_dim+self.k_out_dim], qkv[..., self.q_out_dim+self.k_out_dim:]   # make torchscript happy (cannot use tensor as tuple)

        q = q.contiguous().view(B, N, self.num_heads_q, self.head_dim_q).transpose(1, 2)
        k = k.contiguous().view(B, N, self.num_heads_k, self.head_dim_k).transpose(1, 2)
        v = v.contiguous().view(B, N, self.num_heads_v, self.head_dim_v).transpose(1, 2)
        
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, self.v_out_dim)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x, None
    
class Block(Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=torch.nn.GELU, norm_layer=torch.nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else torch.nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        attn_out, _ = self.attn(self.norm1(x))
        x = x + self.drop_path(attn_out)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

class LayerwiseBertConfig(BertConfig):
    """
    Extends BertConfig to store per-layer attention parameters.
    Each entry in qkv_config corresponds to one layer:
      {"q_heads": int, "k_heads": int, "v_heads": int, "q_dim": int, "k_dim": int, "v_dim": int}
    """
    model_type = "custom-bert"

    def __init__(self, qkv_config=None, **kwargs):
        super().__init__(**kwargs)
        if qkv_config is not None:
            self.qkv_config = qkv_config
        elif hasattr(self, "qkv_config") and self.qkv_config is not None:
            pass
        else:
            # fallback: same shape for all layers
            self.qkv_config = [{
                'hidden_size': self.hidden_size,
                "q_heads": self.num_attention_heads,
                "k_heads": self.num_attention_heads,
                "v_heads": self.num_attention_heads,
                "q_dim": self.hidden_size // self.num_attention_heads,
                "k_dim": self.hidden_size // self.num_attention_heads,
                "v_dim": self.hidden_size // self.num_attention_heads,
            } for _ in range(self.num_hidden_layers)]

class BertSelfAttention(torch.nn.Module):
    def __init__(self, config, layer_idx=None):
        super().__init__()
        layer_config = config.qkv_config[layer_idx]
        self.hidden_size = config.hidden_size
        self.num_attention_heads_q = layer_config['q_heads']
        self.num_attention_heads_k = layer_config['k_heads']
        self.num_attention_heads_v = layer_config['v_heads']
        
        self.head_dim_q = layer_config['q_dim']
        self.head_dim_k = layer_config['k_dim']
        self.head_dim_v = layer_config['v_dim']
        self.all_head_size_q = self.num_attention_heads_q * self.head_dim_q
        self.all_head_size_k = self.num_attention_heads_k * self.head_dim_k
        self.all_head_size_v = self.num_attention_heads_v * self.head_dim_v

        #should create single layer?
        self.query = torch.nn.Linear(self.hidden_size, self.all_head_size_q)
        self.key = torch.nn.Linear(self.hidden_size, self.all_head_size_k)
        self.value = torch.nn.Linear(self.hidden_size, self.all_head_size_v)

        self.dropout = torch.nn.Dropout(config.attention_probs_dropout_prob)
        self.position_embedding_type = getattr(config, "position_embedding_type", "absolute")
        if self.position_embedding_type == "relative_key" or self.position_embedding_type == "relative_key_query":
            self.max_position_embeddings = config.max_position_embeddings
            self.distance_embedding = torch.nn.Embedding(2 * config.max_position_embeddings - 1, self.head_dim_q)

        self.is_decoder = config.is_decoder
        self.layer_idx = layer_idx

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        past_key_values = None,
        output_attentions: Optional[bool] = False,
        cache_position: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor]:
        batch_size, seq_length, _ = hidden_states.shape
        query_layer = self.query(hidden_states)
        query_layer = query_layer.view(batch_size, -1, self.num_attention_heads_q, self.head_dim_q).transpose(1, 2)

        is_cross_attention = encoder_hidden_states is not None
        if past_key_values is not None:
            if isinstance(past_key_values, EncoderDecoderCache):
                is_updated = past_key_values.is_updated.get(self.layer_idx)
                if is_cross_attention:
                    # after the first generated id, we can subsequently re-use all key/value_layer from cache
                    curr_past_key_value = past_key_values.cross_attention_cache
                else:
                    curr_past_key_value = past_key_values.self_attention_cache
            else:
                curr_past_key_value = past_key_values

        current_states = encoder_hidden_states if is_cross_attention else hidden_states
        if is_cross_attention and past_key_values is not None and is_updated:
            # reuse k,v, cross_attentions
            key_layer = curr_past_key_value.layers[self.layer_idx].keys
            value_layer = curr_past_key_value.layers[self.layer_idx].values
        else:
            key_layer = self.key(current_states)
            key_layer = key_layer.view(batch_size, -1, self.num_attention_heads_k, self.head_dim_k).transpose(1, 2)
            value_layer = self.value(current_states)
            value_layer = value_layer.view(batch_size, -1, self.num_attention_heads_v, self.head_dim_v).transpose(1, 2)

            if past_key_values is not None:
                # save all key/value_layer to cache to be re-used for fast auto-regressive generation
                cache_position = cache_position if not is_cross_attention else None
                key_layer, value_layer = curr_past_key_value.update(
                    key_layer, value_layer, self.layer_idx, {"cache_position": cache_position}
                )
                # set flag that curr layer for cross-attn is already updated so we can re-use in subsequent calls
                if is_cross_attention:
                    past_key_values.is_updated[self.layer_idx] = True

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

        if self.position_embedding_type == "relative_key" or self.position_embedding_type == "relative_key_query":
            query_length, key_length = query_layer.shape[2], key_layer.shape[2]
            if past_key_values is not None:
                position_ids_l = torch.tensor(key_length - 1, dtype=torch.long, device=hidden_states.device).view(
                    -1, 1
                )
            else:
                position_ids_l = torch.arange(query_length, dtype=torch.long, device=hidden_states.device).view(-1, 1)
            position_ids_r = torch.arange(key_length, dtype=torch.long, device=hidden_states.device).view(1, -1)
            distance = position_ids_l - position_ids_r

            positional_embedding = self.distance_embedding(distance + self.max_position_embeddings - 1)
            positional_embedding = positional_embedding.to(dtype=query_layer.dtype)  # fp16 compatibility

            if self.position_embedding_type == "relative_key":
                relative_position_scores = torch.einsum("bhld,lrd->bhlr", query_layer, positional_embedding)
                attention_scores = attention_scores + relative_position_scores
            elif self.position_embedding_type == "relative_key_query":
                relative_position_scores_query = torch.einsum("bhld,lrd->bhlr", query_layer, positional_embedding)
                relative_position_scores_key = torch.einsum("bhrd,lrd->bhlr", key_layer, positional_embedding)
                attention_scores = attention_scores + relative_position_scores_query + relative_position_scores_key

        attention_scores = attention_scores / math.sqrt(self.all_head_size_q)
        if attention_mask is not None:
            # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
            attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = torch.nn.functional.softmax(attention_scores, dim=-1)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = torch.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size_v,)
        context_layer = context_layer.view(new_context_layer_shape)

        return context_layer, attention_probs
     
class BertSdpaSelfAttention(BertSelfAttention):
    def __init__(self, config, layer_idx):
        super().__init__(config, layer_idx=layer_idx)
        self.dropout_prob = config.attention_probs_dropout_prob

    # Adapted from BertSelfAttention
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        past_key_values: Optional[Cache] = None,
        output_attentions: Optional[bool] = False,
        cache_position: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor]:
        if self.position_embedding_type != "absolute" or output_attentions or head_mask is not None:
            return super().forward(
                hidden_states,
                attention_mask,
                head_mask,
                encoder_hidden_states,
                past_key_values,
                output_attentions,
                cache_position,
            )

        bsz, tgt_len, _ = hidden_states.size()

        query_layer = (
            self.query(hidden_states).view(bsz, -1, self.num_attention_heads_q, self.head_dim_q).transpose(1, 2)
        )

        is_cross_attention = encoder_hidden_states is not None
        current_states = encoder_hidden_states if is_cross_attention else hidden_states
        if past_key_values is not None:
            if isinstance(past_key_values, EncoderDecoderCache):
                is_updated = past_key_values.is_updated.get(self.layer_idx)
                if is_cross_attention:
                    # after the first generated id, we can subsequently re-use all key/value_states from cache
                    curr_past_key_value = past_key_values.cross_attention_cache
                else:
                    curr_past_key_value = past_key_values.self_attention_cache
            else:
                curr_past_key_value = past_key_values

        current_states = encoder_hidden_states if is_cross_attention else hidden_states
        if is_cross_attention and past_key_values is not None and is_updated:
            # reuse k,v, cross_attentions
            key_layer = curr_past_key_value.layers[self.layer_idx].keys
            value_layer = curr_past_key_value.layers[self.layer_idx].values
        else:
            key_layer = (self.key(current_states).view(bsz, -1, self.num_attention_heads_k, self.head_dim_k).transpose(1, 2))
            value_layer = (self.value(current_states).view(bsz, -1, self.num_attention_heads_v, self.head_dim_v).transpose(1, 2))

            if past_key_values is not None:
                # save all key/value_layer to cache to be re-used for fast auto-regressive generation
                cache_position = cache_position if not is_cross_attention else None
                key_layer, value_layer = curr_past_key_value.update(
                    key_layer, value_layer, self.layer_idx, {"cache_position": cache_position}
                )
                # set flag that curr layer for cross-attn is already updated so we can re-use in subsequent calls
                if is_cross_attention:
                    past_key_values.is_updated[self.layer_idx] = True

        # We dispatch to SDPA's Flash Attention or Efficient kernels via this `is_causal` if statement instead of an inline conditional assignment
        # in SDPA to support both torch.compile's dynamic shapes and full graph options. An inline conditional prevents dynamic shapes from compiling.
        # The tgt_len > 1 is necessary to match with AttentionMaskConverter.to_causal_4d that does not create
        # a causal mask in case tgt_len == 1.
        is_causal = self.is_decoder and not is_cross_attention and attention_mask is None and tgt_len > 1
        # with torch.nn.attention.sdpa_kernel(backends=[torch.nn.attention.SDPBackend.MATH]):
        attn_output = torch.nn.functional.scaled_dot_product_attention(
            query_layer,
            key_layer,
            value_layer,
            attn_mask=attention_mask,
            dropout_p=self.dropout_prob if self.training else 0.0,
            is_causal=is_causal,
        )

        # attn_output = memory_efficient_attention(query_layer, key_layer, value_layer, attention_mask=attention_mask, dropout_p=self.dropout_prob if self.training else 0.0, training=self.training)

        attn_output = attn_output.transpose(1, 2).reshape(bsz, tgt_len, self.all_head_size_v).contiguous()

        return attn_output, None
    
class BertSelfOutput(torch.nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        layer_config = config.qkv_config[layer_idx]
        self.dense = torch.nn.Linear(layer_config['v_heads']*layer_config['v_dim'], config.hidden_size)
        self.LayerNorm = torch.nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = torch.nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states

class BertAttention(torch.nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.self = BertSdpaSelfAttention(config, layer_idx)
        self.output = BertSelfOutput(config, layer_idx)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        past_key_values: Optional[Cache] = None,
        output_attentions: Optional[bool] = False,
        cache_position: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor]:
        self_outputs = self.self(
            hidden_states,
            attention_mask=attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            past_key_values=past_key_values,
            output_attentions=output_attentions,
            cache_position=cache_position,
        )
        attention_output = self.output(self_outputs[0], hidden_states)
        outputs = (attention_output,) + self_outputs[1:]  # add attentions if we output them
        return outputs


class ASTSelfAttention(torch.nn.Module):
    def __init__(self, config, layer_config):
        super().__init__()

        self.config = config
        self.num_attention_heads_q = layer_config['q_heads']
        self.num_attention_heads_k = layer_config['k_heads']
        self.num_attention_heads_v = layer_config['v_heads']
        
        self.head_dim_q = layer_config['q_dim']
        self.head_dim_k = layer_config['k_dim']
        self.head_dim_v = layer_config['v_dim']
        self.all_head_size_q = self.num_attention_heads_q * self.head_dim_q
        self.all_head_size_k = self.num_attention_heads_k * self.head_dim_k
        self.all_head_size_v = self.num_attention_heads_v * self.head_dim_v
        
        self.dropout_prob = config.attention_probs_dropout_prob
        self.scaling = self.all_head_size_q**-0.5
        self.is_causal = False

        self.query = torch.nn.Linear(config.hidden_size, self.all_head_size_q, bias=config.qkv_bias)
        self.key = torch.nn.Linear(config.hidden_size, self.all_head_size_k, bias=config.qkv_bias)
        self.value = torch.nn.Linear(config.hidden_size, self.all_head_size_v, bias=config.qkv_bias)
        
        
    def forward(
        self, hidden_states, head_mask = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size = hidden_states.shape[0]

        key_layer = self.key(hidden_states).view(batch_size, -1, self.num_attention_heads_q, self.head_dim_q).transpose(1, 2)
        value_layer = self.value(hidden_states).view(batch_size, -1, self.num_attention_heads_k, self.head_dim_k).transpose(1, 2)
        query_layer = self.query(hidden_states).view(batch_size, -1, self.num_attention_heads_v, self.head_dim_v).transpose(1, 2)
        # with torch.nn.attention.sdpa_kernel(backends=[torch.nn.attention.SDPBackend.MATH]):
        context_layer = torch.nn.functional.scaled_dot_product_attention(
            query_layer,
            key_layer,
            value_layer,
            head_mask,
            is_causal=self.is_causal,
            scale=self.scaling,
            dropout=0.0 if not self.training else self.dropout_prob,
        )

        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size_v,)
        context_layer = context_layer.reshape(new_context_layer_shape)

        return context_layer
    
class ASTSelfOutput(torch.nn.Module):
    """
    The residual connection is defined in ASTLayer instead of here (as is the case with other models), due to the
    layernorm applied before each block.
    """

    def __init__(self, config, layer_config):
        super().__init__()
        self.dense = torch.nn.Linear(layer_config['v_heads']*layer_config['v_dim'], config.hidden_size)
        self.dropout = torch.nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        return self.dropout(hidden_states)
    

class CLIPAttention(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        if isinstance(config, dict):
            self.embed_dim = config['hidden_size']
            self.num_heads_q = config['num_attention_heads']
            self.num_heads_k = self.num_heads_q
            self.num_heads_v = config['num_attention_heads_v']
            self.num_heads_out = self.num_heads_v
            self.head_dim_q = config.get('head_dim', self.embed_dim // self.num_heads_q)
            self.head_dim_k = config.get('head_dim', self.embed_dim // self.num_heads_q)
            self.head_dim_v = config.get('head_dim_v', self.embed_dim // self.num_heads_v)
            self.head_dim_out = self.head_dim_v
            self.dropout = config['attention_dropout']
            self._attn_implementation = config.get('_attn_implementation', None)
        else:
            self.embed_dim = config.hidden_size
            self.num_heads_q = config.num_attention_heads
            self.num_heads_k = self.num_heads_q
            self.num_heads_v = config.num_attention_heads_v
            self.num_heads_out =self.num_heads_v
            self.head_dim_q = getattr(config, 'head_dim', self.embed_dim // self.num_heads_q)
            self.head_dim_k = getattr(config, 'head_dim', self.embed_dim // self.num_heads_q)
            self.head_dim_v = getattr(config, 'head_dim_v', self.embed_dim // self.num_heads_v)
            self.head_dim_out = self.head_dim_v
            self.dropout = config.attention_dropout
            self._attn_implementation = getattr(config, '_attn_implementation', None)
        
        self.scale = self.head_dim_q**-0.5
        self.is_causal = False

        self.q_out_dim = self.num_heads_q * self.head_dim_q
        self.k_out_dim = self.num_heads_k * self.head_dim_k
        self.v_out_dim = self.num_heads_v * self.head_dim_v

        self.q_proj = nn.Linear(self.embed_dim, self.q_out_dim)
        self.k_proj = nn.Linear(self.embed_dim, self.k_out_dim)
        self.v_proj = nn.Linear(self.embed_dim, self.v_out_dim)
        self.out_proj = nn.Linear(self.v_out_dim, self.embed_dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        causal_attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = False,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Input shape: Batch x Token number x Channel"""

        batch_size, seq_length, embed_dim = hidden_states.shape

        queries = self.q_proj(hidden_states)
        keys = self.k_proj(hidden_states)
        values = self.v_proj(hidden_states)

        queries = queries.view(batch_size, seq_length, self.num_heads_q, self.head_dim_q).transpose(1, 2)
        keys = keys.view(batch_size, seq_length, self.num_heads_k, self.head_dim_k).transpose(1, 2)
        values = values.view(batch_size, seq_length, self.num_heads_v, self.head_dim_v).transpose(1, 2)
        # CLIP text model uses both `causal_attention_mask` and `attention_mask`
        # in case FA2 kernel is called, `is_causal` should be inferred from `causal_attention_mask`
        if self._attn_implementation == "flash_attention_2":
            self.is_causal = causal_attention_mask is not None
        else:
            if attention_mask is not None and causal_attention_mask is not None:
                attention_mask = attention_mask + causal_attention_mask
            elif causal_attention_mask is not None:
                attention_mask = causal_attention_mask
        
        # scale = 1 / torch.sqrt(self.head_dim_q)
        #debug this
        head_valid_mask = self.qk_channel_mask if hasattr(self, "qk_channel_mask") else None
        # if hasattr(self, "qk_channel_mask") and hasattr(self, "v_channel_mask"):
        #     qk_mask = self.qk_channel_mask.view(self.num_heads_q, self.head_dim_q)  # (num_heads_q, head_dim_q)
        #     v_mask = self.v_channel_mask.view(self.num_heads_v, self.head_dim_v)    # (num_heads_v, head_dim_v)

        #     queries = queries * qk_mask.unsqueeze(0).unsqueeze(2)  # (1, num_heads_q, 1, head_dim_q)
        #     keys = keys * qk_mask.unsqueeze(0).unsqueeze(2)
        #     values = values * v_mask.unsqueeze(0).unsqueeze(2)
        #     head_valid_mask = self.qk_channel_mask

        # attn_output = memory_efficient_attention(
        #     queries, keys, values,
        #     valid_mask=head_valid_mask,
        #     attention_mask=attention_mask,
        #     dropout_p=self.dropout,
        #     training=self.training,
        # )
        # with torch.nn.attention.sdpa_kernel(backends=[torch.nn.attention.SDPBackend.MATH]):
        attn_output = torch.nn.functional.scaled_dot_product_attention(
            queries,
            keys,
            values,
            attn_mask=attention_mask,
            is_causal=self.is_causal,
            scale=self.scale,
            dropout_p=0.0 if not self.training else self.dropout
        )

        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_length, self.v_out_dim)
        attn_output = self.out_proj(attn_output)

        return attn_output, None

class CustomCLIPEncoderLayer(CLIPEncoderLayer):
    def __init__(self, config, layer_idx):
        super().__init__(config)
        self.embed_dim = config.hidden_size
        attn_cfg = config.qkv_config[layer_idx]
        self.self_attn = CLIPAttention(attn_cfg)
        self.layer_norm1 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)
        self.mlp = CLIPMLP(config)
        self.layer_norm2 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)

class LayerwiseCLIPConfig(CLIPConfig):
    #we only care for the vision encoder config for now
    model_type = "custom-clip"

    def __init__(self, qkv_config=None, **kwargs):
        super().__init__(**kwargs)

        if hasattr(self.vision_config, "qkv_config") and self.vision_config.qkv_config is not None:
            pass
        else:
            # fallback: same shape for all layers
            self.vision_config.qkv_config = [{
                'hidden_size': self.vision_config.hidden_size,
                "num_attention_heads": self.vision_config.num_attention_heads,
                "num_attention_heads_v": self.vision_config.num_attention_heads,
                "head_dim": self.vision_config.hidden_size // self.vision_config.num_attention_heads,
                "head_dim_v": self.vision_config.hidden_size // self.vision_config.num_attention_heads,
                "attention_dropout": self.vision_config.attention_dropout,
                '_attn_implementation': self.vision_config._attn_implementation
            } for _ in range(self.vision_config.num_hidden_layers)]

#This can be copied for any encoder-decoder architecture
class LayerWiseWhisperConfig(WhisperConfig):
    #we only care for the vision encoder config for now
    model_type = "custom-whisper"

    def __init__(self, qkv_config=None, **kwargs):
        super().__init__(**kwargs)

        if hasattr(self, "encoder_self_qkv_config"):
            pass
        else:
            # fallback: same shape for all layers
            self.encoder_self_qkv_config = [{
                'hidden_size': self.d_model,
                "num_attention_heads": self.encoder_attention_heads,
                "num_attention_heads_v": self.encoder_attention_heads,
                "head_dim": self.d_model // self.encoder_attention_heads,
                "head_dim_v": self.d_model // self.encoder_attention_heads,
                "attention_dropout": self.attention_dropout,
                # '_attn_implementation': self._attn_implementation
            } for _ in range(self.encoder_layers)]
        
        if hasattr(self, "decoder_self_qkv_config"):
            pass
        else:
            # fallback: same shape for all layers
            self.decoder_self_qkv_config = [{
                'hidden_size': self.d_model,
                "num_attention_heads": self.decoder_attention_heads,
                "num_attention_heads_v": self.decoder_attention_heads,
                "head_dim": self.d_model // self.decoder_attention_heads,
                "head_dim_v": self.d_model // self.decoder_attention_heads,
                "attention_dropout": self.attention_dropout,
                # '_attn_implementation': self._attn_implementation
            } for _ in range(self.decoder_layers)]
        
        if hasattr(self, "decoder_cross_qkv_config"):
            pass
        else:
            # fallback: same shape for all layers
            self.decoder_cross_qkv_config = [{
                'hidden_size': self.d_model,
                "num_attention_heads": self.decoder_attention_heads,
                "num_attention_heads_v": self.decoder_attention_heads,
                "head_dim": self.d_model // self.decoder_attention_heads,
                "head_dim_v": self.d_model // self.decoder_attention_heads,
                "attention_dropout": self.attention_dropout,
                # '_attn_implementation': self._attn_implementation
            } for _ in range(self.decoder_layers)]

class WhisperAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        is_decoder: bool = False,
        bias: bool = True,
        is_causal: bool = False,
        layer_idx: Optional[int] = None,
        config = None,
    ):
        super().__init__()
        cfg = config
        if isinstance(cfg, dict):
            self.embed_dim = cfg['hidden_size']
            self.num_heads_q = cfg['num_attention_heads']
            self.num_heads_k = self.num_heads_q
            self.num_heads_v = cfg['num_attention_heads_v']
            self.num_heads_out = self.num_heads_v
            self.head_dim_q = cfg.get('head_dim', self.embed_dim // self.num_heads_q)
            self.head_dim_k = cfg.get('head_dim', self.embed_dim // self.num_heads_q)
            self.head_dim_v = cfg.get('head_dim_v', self.embed_dim // self.num_heads_v)
            self.head_dim_out = self.head_dim_v
            self.dropout = cfg['attention_dropout']
        else:
            self.embed_dim = cfg.hidden_size
            self.num_heads_q = cfg.num_attention_heads
            self.num_heads_k = self.num_heads_q
            self.num_heads_v = cfg.num_attention_heads_v
            self.num_heads_out =self.num_heads_v
            self.head_dim_q = getattr(cfg, 'head_dim', self.embed_dim // self.num_heads_q)
            self.head_dim_k = getattr(cfg, 'head_dim', self.embed_dim // self.num_heads_q)
            self.head_dim_v = getattr(cfg, 'head_dim_v', self.embed_dim // self.num_heads_v)
            self.head_dim_out = self.head_dim_v
            self.dropout = cfg.attention_dropout
       
        self.config = config

        self.scaling = self.head_dim_q**-0.5
        self.is_decoder = is_decoder
        self.is_causal = is_causal
        self.layer_idx = layer_idx
        self.q_out_dim = self.head_dim_q*self.num_heads_q
        self.k_out_dim = self.head_dim_k*self.num_heads_k
        self.v_out_dim = self.head_dim_v*self.num_heads_v
        self.q_proj = nn.Linear(embed_dim, self.head_dim_q*self.num_heads_q, bias=bias)
        self.k_proj = nn.Linear(embed_dim, self.head_dim_k*self.num_heads_k, bias=False)
        self.v_proj = nn.Linear(embed_dim, self.head_dim_v*self.num_heads_v, bias=bias)
        self.out_proj = nn.Linear(self.head_dim_out*self.num_heads_out, embed_dim, bias=bias)

    def forward(self, hidden_states, key_value_states=None, past_key_values=None,
            attention_mask=None, layer_head_mask=None, output_attentions=False,
            cache_position=None, **kwargs):
        try:
            is_cross_attention = key_value_states is not None
            bsz, tgt_len = hidden_states.shape[:-1]

            # ← only change: use head_dim_q and num_heads_q
            query_states = self.q_proj(hidden_states)
            query_states = query_states.view(bsz, tgt_len, self.num_heads_q, self.head_dim_q)
            query_states = query_states.transpose(1, 2).contiguous()

            # ← exact copy from original
            if past_key_values is not None and isinstance(past_key_values, EncoderDecoderCache):
                is_updated = past_key_values.is_updated.get(self.layer_idx)
                if is_cross_attention:
                    past_key_values.is_updated[self.layer_idx] = True
                    past_key_values = past_key_values.cross_attention_cache
                else:
                    past_key_values = past_key_values.self_attention_cache

            current_states = key_value_states if key_value_states is not None else hidden_states

            if is_cross_attention and past_key_values and is_updated:
                print(f"[CACHE HIT] layer={self.layer_idx} "
                    f"keys shape={past_key_values.layers[self.layer_idx].keys.shape} "
                    f"values shape={past_key_values.layers[self.layer_idx].values.shape}")
                # ← exact copy from original — .layers[].keys is correct for this transformers version
                key_states   = past_key_values.layers[self.layer_idx].keys
                value_states = past_key_values.layers[self.layer_idx].values
            else:
                # ← only change: use per-role num_heads and head_dim
                key_states   = self.k_proj(current_states).view(bsz, -1, self.num_heads_k, self.head_dim_k).transpose(1, 2).contiguous()
                value_states = self.v_proj(current_states).view(bsz, -1, self.num_heads_v, self.head_dim_v).transpose(1, 2).contiguous()
                if past_key_values is not None:
                    cache_position = cache_position if not is_cross_attention else None
                    key_states, value_states = past_key_values.update(
                        key_states, value_states, self.layer_idx, {"cache_position": cache_position}
                    )
            # with torch.nn.attention.sdpa_kernel(backends=[torch.nn.attention.SDPBackend.MATH]):
            attn_output = torch.nn.functional.scaled_dot_product_attention(
                query_states, key_states, value_states,
                attn_mask=attention_mask if not self.is_causal else None,
                dropout_p=0.0 if not self.training else self.dropout,
                is_causal=self.is_causal,
                scale=self.scaling,
            )

            attn_output = attn_output.transpose(1, 2).reshape(bsz, tgt_len, -1).contiguous()
            attn_output = self.out_proj(attn_output)
        except Exception as e:
            import traceback
            print(f"[FORWARD ERROR] layer={self.layer_idx} cross={is_cross_attention}: {e}")
            traceback.print_exc()
            raise

        return attn_output, None

class CustomWhisperEncoderLayer(WhisperEncoderLayer):
    def __init__(self, config: LayerWiseWhisperConfig, layer_idx):
        super().__init__(config)
        #overwrite the attention layer
        self.self_attn = WhisperAttention(
            embed_dim=self.embed_dim,
            num_heads=config.encoder_attention_heads,
            dropout=config.attention_dropout,
            layer_idx=layer_idx,
            config=config.encoder_self_qkv_config[layer_idx],
        )

class CustomWhisperEncoder(WhisperEncoder):
    def __init__(self, config: WhisperConfig):
        super().__init__(config)

        self.layers = nn.ModuleList([CustomWhisperEncoderLayer(config, layer_idx) for layer_idx in range(config.encoder_layers)])
        # Initialize weights and apply final processing
        self.post_init()

class CustomWhisperDecoderLayer(WhisperDecoderLayer):
    def __init__(self, config: LayerWiseWhisperConfig, layer_idx: Optional[int] = None):
        super().__init__(config, layer_idx)

        self.self_attn = WhisperAttention(
            embed_dim=self.embed_dim,
            num_heads=config.decoder_attention_heads,
            dropout=config.attention_dropout,
            is_decoder=True,
            is_causal=True,
            layer_idx=layer_idx,
            config=config.decoder_self_qkv_config[layer_idx],
        )

        self.encoder_attn = WhisperAttention(
            self.embed_dim,
            config.decoder_attention_heads,
            dropout=config.attention_dropout,
            is_decoder=True,
            layer_idx=layer_idx,
            config=config.decoder_cross_qkv_config[layer_idx],
        )

class CustomWhisperDecoder(WhisperDecoder):

    main_input_name = "input_ids"

    def __init__(self, config: WhisperConfig):
        super().__init__(config)

        self.layers = nn.ModuleList([CustomWhisperDecoderLayer(config, layer_idx) for layer_idx in range(config.decoder_layers)])

        self.post_init()

class LayerWiseVitConfig(ViTConfig):
    model_type = "custom-vit"

    def __init__(self, qkv_config=None, **kwargs):
        super().__init__(**kwargs)
        if qkv_config is not None:
            self.qkv_config = qkv_config
            print("✅ Loaded per-layer QKV config from checkpoint.")
        else:
            # fallback: same shape for all layers
            self.qkv_config = [
                AttrDict({
                    'hidden_size': self.hidden_size,
                    'q_heads': self.num_attention_heads,
                    'k_heads': self.num_attention_heads,
                    'v_heads': self.num_attention_heads,
                    'q_dim': self.hidden_size // self.num_attention_heads,
                    'k_dim': self.hidden_size // self.num_attention_heads,
                    'v_dim': self.hidden_size // self.num_attention_heads,
                    'attention_dropout': self.attention_probs_dropout_prob,
                    'qkv_bias': self.qkv_bias,
                    'hidden_dropout_prob': self.hidden_dropout_prob,
                })
                 for _ in range(self.num_hidden_layers)]
            
class ViTSelfAttention(nn.Module):
    def __init__(self, config: LayerWiseVitConfig):
        super().__init__()
        if isinstance(config, dict):
            self.embed_dim = config['hidden_size']
            self.num_heads_q = config['q_heads']
            self.num_heads_k = self.num_heads_q
            self.num_heads_v = config['v_heads']
            self.num_heads_out = self.num_heads_v
            self.head_dim_q = config.get('q_dim', self.embed_dim // self.num_heads_q)
            self.head_dim_k = config.get('k_dim', self.embed_dim // self.num_heads_k)
            self.head_dim_v = config.get('v_dim', self.embed_dim // self.num_heads_v)
            self.head_dim_out = self.head_dim_v
            self.dropout = config['attention_dropout']
            self.qkv_bias = config['qkv_bias']
            # self._attn_implementation = cfg.get('_attn_implementation', None)
        else:
            self.embed_dim = config.hidden_size
            self.num_heads_q = config.q_heads
            self.num_heads_k = self.num_heads_q
            self.num_heads_v = config.v_heads
            self.num_heads_out =self.num_heads_v
            self.head_dim_q = getattr(config, 'q_dim', self.embed_dim // self.num_heads_q)
            self.head_dim_k = getattr(config, 'k_dim', self.embed_dim // self.num_heads_k)
            self.head_dim_v = getattr(config, 'v_dim', self.embed_dim // self.num_heads_v)
            self.head_dim_out = self.head_dim_v
            self.dropout = config.attention_dropout
            self.qkv_bias = config.qkv_bias
            # self._attn_implementation = getattr(cfg, '_attn_implementation', None)

        self.scaling = self.head_dim_q**-0.5
        self.q_out_dim = self.head_dim_q*self.num_heads_q
        self.k_out_dim = self.head_dim_k*self.num_heads_k
        self.v_out_dim = self.head_dim_v*self.num_heads_v
        self.query = nn.Linear(self.embed_dim, self.head_dim_q*self.num_heads_q, bias=self.qkv_bias)
        self.key = nn.Linear(self.embed_dim, self.head_dim_k*self.num_heads_k, bias=self.qkv_bias)
        self.value = nn.Linear(self.embed_dim, self.head_dim_v*self.num_heads_v, bias=self.qkv_bias)

    def forward(self, hidden_states: torch.Tensor, head_mask: Optional[torch.Tensor] = None) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size = hidden_states.shape[0]

        key_layer = self.key(hidden_states).view(batch_size, -1, self.num_heads_k, self.head_dim_k).transpose(1, 2)
        value_layer = self.value(hidden_states).view(batch_size, -1, self.num_heads_v, self.head_dim_v).transpose(1, 2)
        query_layer = self.query(hidden_states).view(batch_size, -1, self.num_heads_q, self.head_dim_q).transpose(1, 2)

        # scaling = self.head_dim_q**-0.5
        # with torch.nn.attention.sdpa_kernel(backends=[torch.nn.attention.SDPBackend.MATH]):
        context_layer = torch.nn.functional.scaled_dot_product_attention(
            query_layer,
            key_layer,
            value_layer,
            head_mask,
            is_causal=False,
            scale=self.scaling,
            dropout_p=0.0 if not self.training else self.dropout,
        )
        context_layer = context_layer.transpose(1, 2).reshape(batch_size, -1, self.num_heads_v * self.head_dim_v)

        return context_layer, None

class CustomViTSelfOutput(ViTSelfOutput):
    def __init__(self, config: LayerWiseVitConfig):
        super().__init__(config)
        if isinstance(config, dict):
            self.embed_dim = config['hidden_size']
            self.num_heads_v = config['v_heads']
            self.head_dim_v = config.get('v_dim', self.embed_dim // self.num_heads_v)
            self.dropout_prob = config['hidden_dropout_prob']
        else:
            self.embed_dim = config.hidden_size
            self.num_heads_v = config.v_heads
            self.head_dim_v = getattr(config, 'v_dim', self.embed_dim // self.num_heads_v)
            self.dropout_prob = config.hidden_dropout_prob
            self.qkv_bias = config.qkv_bias

        self.dense = nn.Linear(self.num_heads_v*self.head_dim_v, self.embed_dim)
        self.dropout = nn.Dropout(self.dropout_prob)

class CustomViTAttention(ViTAttention):
    def __init__(self, config: LayerWiseVitConfig, layer_idx):
        super().__init__(config)
        cfg = config.qkv_config[layer_idx]
        self.attention = ViTSelfAttention(cfg)
        self.output = CustomViTSelfOutput(cfg)

class CustomViTLayer(ViTLayer):
    def __init__(self, config: LayerWiseVitConfig, layer_idx):
        super().__init__(config)

        self.attention = CustomViTAttention(config, layer_idx)

class CustomViTEncoder(ViTEncoder):
    def __init__(self, config: LayerWiseVitConfig):
        super().__init__(config)
        self.config = config
        self.layer = nn.ModuleList([CustomViTLayer(config, idx) for idx in range(config.num_hidden_layers)])
        self.gradient_checkpointing = False

class LayerWiseDinov2Config(Dinov2Config):
    def __init__(self, qkv_config=None, **kwargs):
        super().__init__(**kwargs)
        if qkv_config is not None:
            self.qkv_config = qkv_config
            print("✅ Loaded per-layer QKV config from checkpoint.")
        else:
            # fallback: same shape for all layers
            self.qkv_config = [
                AttrDict({
                    'hidden_size': self.hidden_size,
                    'q_heads': self.num_attention_heads,
                    'k_heads': self.num_attention_heads,
                    'v_heads': self.num_attention_heads,
                    'q_dim': self.hidden_size // self.num_attention_heads,
                    'k_dim': self.hidden_size // self.num_attention_heads,
                    'v_dim': self.hidden_size // self.num_attention_heads,
                    'attention_dropout': self.attention_probs_dropout_prob,
                    'qkv_bias': self.qkv_bias,
                    'hidden_dropout_prob': self.hidden_dropout_prob,
                })
                 for _ in range(self.num_hidden_layers)]


class Dinov2SelfAttention(nn.Module):
    def __init__(self, config: LayerWiseDinov2Config):
        super().__init__()
        if isinstance(config, dict):
            self.embed_dim = config['hidden_size']
            self.num_heads_q = config['q_heads']
            self.num_heads_k = self.num_heads_q
            self.num_heads_v = config['v_heads']
            self.num_heads_out = self.num_heads_v
            self.head_dim_q = config.get('q_dim', self.embed_dim // self.num_heads_q)
            self.head_dim_k = config.get('k_dim', self.embed_dim // self.num_heads_k)
            self.head_dim_v = config.get('v_dim', self.embed_dim // self.num_heads_v)
            self.head_dim_out = self.head_dim_v
            self.dropout = config['attention_dropout']
            self.qkv_bias = config['qkv_bias']
            # self._attn_implementation = cfg.get('_attn_implementation', None)
        else:
            self.embed_dim = config.hidden_size
            self.num_heads_q = config.q_heads
            self.num_heads_k = self.num_heads_q
            self.num_heads_v = config.v_heads
            self.num_heads_out =self.num_heads_v
            self.head_dim_q = getattr(config, 'q_dim', self.embed_dim // self.num_heads_q)
            self.head_dim_k = getattr(config, 'k_dim', self.embed_dim // self.num_heads_k)
            self.head_dim_v = getattr(config, 'v_dim', self.embed_dim // self.num_heads_v)
            self.head_dim_out = self.head_dim_v
            self.dropout = config.attention_dropout
            self.qkv_bias = config.qkv_bias
        self.is_causal =False
        self.scaling = self.head_dim_q**-0.5

        self.all_head_size_q = self.head_dim_q*self.num_heads_q
        self.all_head_size_k = self.head_dim_k*self.num_heads_k
        self.all_head_size_v = self.head_dim_v*self.num_heads_v
        self.all_head_size_out = self.head_dim_out*self.num_heads_out
        
        self.query = nn.Linear(self.embed_dim, self.all_head_size_q, bias=self.qkv_bias)
        self.key = nn.Linear(self.embed_dim, self.all_head_size_k, bias=self.qkv_bias)
        self.value = nn.Linear(self.embed_dim, self.all_head_size_v, bias=self.qkv_bias)

    def forward(
        self, hidden_states: torch.Tensor, head_mask: Optional[torch.Tensor] = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size = hidden_states.shape[0]
        tgt_len = hidden_states.shape[1]

        key_layer = self.key(hidden_states).view(batch_size, tgt_len, self.num_heads_k, self.head_dim_k).transpose(1, 2)
        value_layer = self.value(hidden_states).view(batch_size, tgt_len, self.num_heads_v, self.head_dim_v).transpose(1, 2)
        query_layer = self.query(hidden_states).view(batch_size, tgt_len, self.num_heads_q, self.head_dim_q).transpose(1, 2)
        
        # with torch.nn.attention.sdpa_kernel(backends=[torch.nn.attention.SDPBackend.MATH]):
        context_layer = torch.nn.functional.scaled_dot_product_attention(
            query_layer,
            key_layer,
            value_layer,
            head_mask,
            is_causal=self.is_causal,
            scale=self.scaling,
            dropout_p=0.0 if not self.training else self.dropout,
        )

        context_layer = context_layer.transpose(1, 2).reshape((batch_size, tgt_len, self.all_head_size_out))

        return context_layer, None


# Copied from transformers.models.vit.modeling_vit.ViTSelfOutput with ViT->Dinov2
class Dinov2SelfOutput(nn.Module):

    def __init__(self, config: LayerWiseDinov2Config):
        super().__init__()
        if isinstance(config, dict):
            self.embed_dim = config['hidden_size']
            self.num_heads_out = config['v_heads']
            self.head_dim_out = config.get('v_dim', self.embed_dim // self.num_heads_out)
            self.dropout_prob = config['hidden_dropout_prob']
            self.qkv_bias = config['qkv_bias']
            # self._attn_implementation = cfg.get('_attn_implementation', None)
        else:
            self.embed_dim = config.hidden_size
            self.num_heads_out = config.v_heads
            self.head_dim_out = getattr(config, 'v_dim', self.embed_dim // self.num_heads_out)
            self.dropout_prob = config.hidden_dropout_prob
            self.qkv_bias = config.qkv_bias

        self.all_head_size_out = self.head_dim_out*self.num_heads_out
        self.dense = nn.Linear(self.all_head_size_out, self.embed_dim)
        self.dropout = nn.Dropout(self.dropout_prob)

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        return hidden_states

class CustomDinov2Attention(Dinov2Attention):
    def __init__(self, config: LayerWiseDinov2Config, idx):
        super().__init__(config)
        cfg = config.qkv_config[idx]
        self.attention = Dinov2SelfAttention(cfg)
        self.output = Dinov2SelfOutput(cfg)
        self.pruned_heads = set()

    def forward(self, hidden_states: torch.Tensor, head_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        self_attn_output, _ = self.attention(hidden_states, head_mask)
        output = self.output(self_attn_output, hidden_states)
        return output


class CustomDinov2Layer(Dinov2Layer):
    """This corresponds to the Block class in the original implementation."""

    def __init__(self, config: LayerWiseDinov2Config, idx) -> None:
        super().__init__(config)

        self.attention = CustomDinov2Attention(config, idx)
        

class CustomDinov2Encoder(Dinov2Encoder):
    def __init__(self, config: LayerWiseDinov2Config):
        super().__init__(config)
        self.config = config
        self.layer = nn.ModuleList([CustomDinov2Layer(config, idx) for idx in range(config.num_hidden_layers)])
        self.gradient_checkpointing = False