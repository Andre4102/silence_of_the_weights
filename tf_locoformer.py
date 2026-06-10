import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Conv1d, ConvTranspose1d
import math
from rotary_embedding_torch import RotaryEmbedding
import torchaudio
from einops import rearrange, repeat
from torch import einsum, broadcast_tensors, is_tensor, tensor, Tensor
from typing import Literal

def rotate_half(x):
    x1, x2 = x[..., :x.shape[-1] // 2], x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)

def exists(val):
        return val is not None
    
def default(val, d):
    return val if exists(val) else d

class RotaryEmbeddingOdd(RotaryEmbedding):
    def __init__(self, dim, *args, **kwargs):
        super().__init__(dim, *args, **kwargs)
        self.dim = dim
        self.rot_dim = dim // 2 * 2   # largest even number ≤ dim

    @staticmethod
    def rotate_half(x):
        # only apply rotation to even part
        even_dim = (x.shape[-1] // 2) * 2
        x_rot = x[..., :even_dim]
        x_rem = x[..., even_dim:]   # remainder (if odd)

        x_rot = rearrange(x_rot, '... (d r) -> ... d r', r=2)
        x1, x2 = x_rot.unbind(dim=-1)
        x_rot = torch.stack((-x2, x1), dim=-1)
        x_rot = rearrange(x_rot, '... d r -> ... (d r)')

        if x_rem.shape[-1] > 0:
            return torch.cat([x_rot, x_rem], dim=-1)
        return x_rot

    def rotate_queries_or_keys(self, t, seq_dim=None, offset=0, scale=None):
        """
        Apply rotary embeddings to queries or keys.
        Supports both even and odd head_dim.
        """
        seq_dim = default(seq_dim, self.default_seq_dim)

        assert not self.use_xpos or exists(scale), \
            'Use `.rotate_queries_and_keys` for xpos with both queries and keys'

        device, dtype, seq_len = t.device, t.dtype, t.shape[seq_dim]

        # sequence positions
        seq = self.get_seq_pos(seq_len, device=device, dtype=dtype, offset=offset)

        # build frequencies
        freqs = self.forward(seq, seq_len=seq_len, offset=offset)

        if seq_dim == -3:
            freqs = rearrange(freqs, 'n d -> n 1 d')

        # split into even (rotated) and remainder (untouched)
        rot_dim = self.rot_dim
        t_rot, t_rem = t[..., :rot_dim], t[..., rot_dim:]

        freqs_rot = freqs[..., :rot_dim]

        # apply rotary on even part only
        t_rotated = (
            t_rot * freqs_rot.cos() * default(scale, 1.)
            + self.rotate_half(t_rot) * freqs_rot.sin() * default(scale, 1.)
        )

        if t_rem.shape[-1] > 0:
            out = torch.cat([t_rotated, t_rem], dim=-1)
        else:
            out = t_rotated

        return out.type(dtype)
    
    def forward(self, t: Tensor, seq_len: int | None = None, offset=0):
        should_cache = (
            self.cache_if_possible and
            not self.learned_freq and
            exists(seq_len) and
            self.freqs_for != 'pixel' and
            (offset + seq_len) <= self.cache_max_seq_len
        )

        if (
            should_cache and
            exists(self.cached_freqs) and
            (offset + seq_len) <= self.cached_freqs_seq_len
        ):
            return self.cached_freqs[offset:(offset + seq_len)].detach()

        freqs = self.freqs

        # Only build frequencies up to the even rot_dim
        freqs_even = freqs[: self.rot_dim // 2]
        freqs = einsum('..., f -> ... f', t.type(freqs_even.dtype), freqs_even)
        freqs = repeat(freqs, '... n -> ... (n r)', r=2)

        # If odd dim, pad with zeros (so last feature passes unchanged)
        if self.rot_dim < self.dim:
            pad = torch.zeros(*freqs.shape[:-1], 1, device=freqs.device, dtype=freqs.dtype)
            freqs = torch.cat([freqs, pad], dim=-1)

        if should_cache and offset == 0:
            self.cached_freqs[:seq_len] = freqs.detach()
            self.cached_freqs_seq_len = seq_len

        return freqs


class GradientMonitor:
    def __init__(self, model, log_file="gradient_log.txt"):
        self.model = model
        self.log_file = log_file
        self.step_count = 0
        
    def log_stats(self, name, tensor, tensor_type="activation"):
        if tensor is None:
            return
            
        stats = {
            'mean': tensor.mean().item() if tensor.numel() > 0 else 0,
            'std': tensor.std().item() if tensor.numel() > 0 else 0,
            'min': tensor.min().item() if tensor.numel() > 0 else 0,
            'max': tensor.max().item() if tensor.numel() > 0 else 0,
            'nan_count': torch.isnan(tensor).sum().item(),
            'inf_count': torch.isinf(tensor).sum().item(),
        }
        
        # Check for problematic values
        if stats['nan_count'] > 0 or stats['inf_count'] > 0:
            print(f"⚠️  WARNING: {name} has {stats['nan_count']} NaN and {stats['inf_count']} Inf values")
        
        if tensor_type == "gradient" and (abs(stats['mean']) > 1e3 or stats['std'] > 1e3):
            print(f"⚠️  GRADIENT EXPLOSION in {name}: mean={stats['mean']:.2e}, std={stats['std']:.2e}")
            
        if tensor_type == "activation" and (abs(stats['mean']) > 1e6 or stats['std'] > 1e6):
            print(f"⚠️  ACTIVATION EXPLOSION in {name}: mean={stats['mean']:.2e}, std={stats['std']:.2e}")
            
        return stats
    
    def check_gradients(self):
        
        gradient_problems = []
        
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                stats = self.log_stats(name, param.grad, "gradient")
                
                # Flag problematic gradients
                if stats['nan_count'] > 0 or stats['inf_count'] > 0:
                    gradient_problems.append(f"{name}: NaN={stats['nan_count']}, Inf={stats['inf_count']}")
                elif abs(stats['mean']) > 1e2 or stats['std'] > 1e2:
                    gradient_problems.append(f"{name}: mean={stats['mean']:.2e}, std={stats['std']:.2e}")
        
        if gradient_problems:
            print(f"\n=== Gradient Check Step {self.step_count} ===")
            print("🚨 GRADIENT PROBLEMS DETECTED:")
            for problem in gradient_problems[:10]:  # Show first 10
                print(f"  - {problem}")
        
        self.step_count += 1
        return len(gradient_problems) == 0

def check_nan_inf(tensor, name, raise_error=False):
    """Enhanced NaN/Inf checker with more info"""
    if tensor is None:
        return
        
    nan_count = torch.isnan(tensor).sum().item()
    inf_count = torch.isinf(tensor).sum().item()
    
    if nan_count > 0 or inf_count > 0:
        print(f"❌ {name}: NaN={nan_count}, Inf={inf_count}, shape={tensor.shape}")
        print(f"   Stats: mean={tensor.mean().item():.2e}, std={tensor.std().item():.2e}")
        if raise_error:
            raise ValueError(f"NaN/Inf detected in {name}")


class MultiHeadSelfAttention(nn.Module):
    def __init__(
        self,
        emb_dim,
        attention_dim,
        n_heads=8,
        dropout=0.0,
        rope=None,
        flash_attention=False,
        kdim = None,
        vdim = None,
        num_heads_k = None,
        num_heads_v = None,
        num_heads_out = None,
        head_dim = None,
        head_dim_k = None,
        head_dim_v = None,
        head_dim_out = None,
        
    ):
        super().__init__()

        self.embed_dim = emb_dim
        self.kdim = kdim if kdim is not None else emb_dim
        self.vdim = vdim if vdim is not None else emb_dim
        self._qkv_same_embed_dim = self.kdim == emb_dim and self.vdim == emb_dim

        self.num_heads_q = n_heads
        self.num_heads_k = num_heads_k if num_heads_k is not None else n_heads
        self.num_heads_v = num_heads_v if num_heads_v is not None else n_heads
        self.num_heads_out = num_heads_out if num_heads_out is not None else n_heads
        
        self.dropout = dropout
        self.head_dim_q = head_dim if head_dim is not None else attention_dim // self.num_heads_q
        self.head_dim_k = head_dim_k if head_dim_k is not None else attention_dim // self.num_heads_k  
        self.head_dim_v = head_dim_v if head_dim_v is not None else attention_dim // self.num_heads_v
        self.head_dim_out = head_dim_out if head_dim_out is not None else attention_dim // self.num_heads_out
        
        self.q_out_dim = self.num_heads_q * self.head_dim_q
        self.k_out_dim = self.num_heads_k * self.head_dim_k
        self.v_out_dim = self.num_heads_v * self.head_dim_v
        self.n_heads = n_heads
        self.dropout = dropout
        self.head_dim = attention_dim // n_heads
        self.scale = self.head_dim ** -0.5  # Add explicit scaling
        
        #standard rotary embedding
        #self.rope = RotaryEmbedding(self.head_dim_q)
        #rotary embedding to handle odd head size
        if rope is not None:
            self.rope = RotaryEmbeddingOdd(self.head_dim_q)
        else:
            self.rope = rope

        
        self.qkv = nn.Linear(in_features = self.embed_dim, out_features=self.q_out_dim + self.k_out_dim + self.v_out_dim, bias=False)
        self.aggregate_heads = nn.Sequential(
            nn.Linear(in_features = self.v_out_dim, out_features = self.embed_dim, bias=False), 
            nn.Dropout(dropout)
        )

        if flash_attention:
            self.flash_attention_config = dict(enable_flash=True, enable_math=False, enable_mem_efficient=False)
        else:
            self.flash_attention_config = dict(enable_flash=False, enable_math=True, enable_mem_efficient=True)

    def forward(self, input, k, v):
        # get query, key, and value
        query, key, value = self.get_qkv(input)

        # rotary positional encoding
        if self.rope is not None:
            query, key = self.apply_rope(query, key)
        # pytorch 2.0 flash attention: q, k, v, mask, dropout, softmax_scale
        output = F.scaled_dot_product_attention(
            query=query,
            key=key,
            value=value,
            attn_mask=None,
            dropout_p=self.dropout if self.training else 0.0,
            scale=self.scale  # Use proper scaling
        )

        output = output.transpose(1, 2)  # (batch, seq_len, head, -1)
        output = output.reshape(output.shape[:2] + (-1,))
        return self.aggregate_heads(output), None

    def get_qkv(self, input):
        n_batch, seq_len = input.shape[:2]
        qkv_output = self.qkv(input)
        
        q_flat = qkv_output[:, :, :self.q_out_dim]  # (batch, seq_len, q_out_dim)
        k_flat = qkv_output[:, :, self.q_out_dim:self.q_out_dim + self.k_out_dim]  # (batch, seq_len, k_out_dim)
        v_flat = qkv_output[:, :, self.q_out_dim + self.k_out_dim:]  # (batch, seq_len, v_out_dim)
        
        # Reshape each component according to its specific head configuration
        query = q_flat.reshape(n_batch, seq_len, self.num_heads_q, self.head_dim_q)
        key = k_flat.reshape(n_batch, seq_len, self.num_heads_k, self.head_dim_k)
        value = v_flat.reshape(n_batch, seq_len, self.num_heads_v, self.head_dim_v)
        
        # Move head dimension to position 1: (batch, head, seq_len, head_dim)
        query = query.transpose(1, 2)  # (batch, num_heads_q, seq_len, head_dim_q)
        key = key.transpose(1, 2)      # (batch, num_heads_k, seq_len, head_dim_k)
        value = value.transpose(1, 2)  # (batch, num_heads_v, seq_len, head_dim_v)

        return query, key, value

    def apply_rope(self, query, key):
        query = self.rope.rotate_queries_or_keys(query)
        key = self.rope.rotate_queries_or_keys(key)
        return query, key

# TODO: FINISH THIS 
    # def reconstruct_weights():
    #     del sa.qkv
    #     if hasattr(sa, 'aggregate_heads'):
    #         del sa.aggregate_heads
    #     torch.cuda.empty_cache() if torch.cuda.is_available() else None
    #     sa.qkv = nn.Linear(sa.embed_dim, sa.q_out_dim + sa.k_out_dim + sa.v_out_dim,
    #                        bias=(new_bias is not None), device=device)
    #     sa.qkv.weight.data = new_weight.contiguous()
    #     if new_bias is not None:
    #         sa.qkv.bias.data = new_bias.contiguous()
    #     sa.qkv.requires_grad_(False)
    #     agg = nn.Linear(sa.v_out_dim, sa.embed_dim, bias=False, device=device)
    #     agg.weight.data = new_o_weight.contiguous()
    #     agg.requires_grad_(False)
    #     sa.aggregate_heads = nn.ModuleList([agg])
    #     if hasattr(sa, 'rope') and sa.rope is not None:
    #         old_freqs = sa.rope.freqs.data \
    #             if isinstance(sa.rope.freqs, nn.Parameter) else sa.rope.freqs
    #         if sa.head_dim_q > 2:
    #             new_rope = tf_locoformer.RotaryEmbeddingOdd(sa.head_dim_q)
    #             m = min(new_rope.freqs.shape[0], old_freqs.shape[0])
    #             new_rope.freqs.data[:m] = old_freqs[:m].to(new_rope.freqs.device)
    #             del sa.rope
    #             sa.rope = new_rope.to(device)
    #         else:
    #             del sa.rope
    #             sa.rope = None

class RMSGroupNorm(nn.Module):
    def __init__(self, num_groups, dim, eps=1e-8, bias=False):
        """
        Root Mean Square Group Normalization (RMSGroupNorm).
        Unlike Group Normalization in vision, RMSGroupNorm
        is applied to each TF bin.

        Args:
            num_groups: int
                Number of groups
            dim: int
                Number of dimensions
            eps: float
                Small constant to avoid division by zero.
            bias: bool
                Whether to add a bias term. RMSNorm does not use bias.
        """
        super().__init__()

        assert dim % num_groups == 0, (dim, num_groups)
        self.num_groups = num_groups
        self.dim_per_group = dim // self.num_groups

        self.gamma = nn.Parameter(torch.ones(dim))
        
        self.bias = bias
        if self.bias:
            self.beta = nn.Parameter(torch.zeros(dim))
        
        self.eps = eps
        self.num_groups = num_groups

    def forward(self, input):
        others = input.shape[:-1]
        input = input.reshape(others + (self.num_groups, self.dim_per_group))

        # normalization with improved numerical stability
        norm_ = input.norm(2, dim=-1, keepdim=True)
        rms = norm_ * (self.dim_per_group ** (-0.5))
        output = input / (rms + self.eps)

        # reshape and affine transformation
        output = output.reshape(others + (-1,))
        output = output * self.gamma
        if self.bias:
            output = output + self.beta

        return output

class SwigluConvBlock(nn.Module):
    def __init__(self, num_group, emb_dim, eps, hidden_dim, kernel_size, stride, dropout):
        super().__init__()
        self.norm = RMSGroupNorm(num_group, emb_dim, eps)
        
        # Use smaller initialization scale for conv layers
        self.conv1 = Conv1d(emb_dim, hidden_dim, kernel_size, padding=kernel_size // 2)
        # Initialize with smaller scale
        nn.init.xavier_uniform_(self.conv1.weight, gain=0.1)
        if self.conv1.bias is not None:
            nn.init.zeros_(self.conv1.bias)
            
        self.swish = nn.SiLU()
        self.dropout = nn.Dropout(dropout)
        
        self.deconv = ConvTranspose1d(hidden_dim // 2, emb_dim, kernel_size, padding=kernel_size // 2)
        # Initialize with smaller scale
        nn.init.xavier_uniform_(self.deconv.weight, gain=0.1)
        if self.deconv.bias is not None:
            nn.init.zeros_(self.deconv.bias)
            
        self.hidden_dim = hidden_dim // 2
        self.diff_ks = kernel_size - stride
        self.conv1d_kernel = kernel_size
        self.conv1d_shift = stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x  # Store residual
        x = self.norm(x)
        b, s1, s2, h = x.shape
        x = x.contiguous().view(b * s1, s2, h)
        x = x.transpose(-1, -2)

        # padding
        seq_len = (
            math.ceil((s2 + 2 * self.diff_ks - self.conv1d_kernel) / self.conv1d_shift) * self.conv1d_shift
            + self.conv1d_kernel
        )
        x = F.pad(x, (self.diff_ks, seq_len - s2 - self.diff_ks))
        
        x = self.conv1(x)
        x = F.silu(x)  # Apply SiLU activation
        
        # Split for gating
        gate = self.swish(x[..., self.hidden_dim:, :])
        x = x[..., :self.hidden_dim, :] * gate
        
        x = self.dropout(x)
        
        x = self.deconv(x).transpose(-1, -2)
        
        x = x[..., self.diff_ks : self.diff_ks + s2, :]
        
        x = self.dropout(x).reshape(b, s1, s2, h)
        
        # Add residual connection with proper scaling
        return x + residual  # Scale down the output


class LocoFormerBlock(nn.Module):
    def __init__(self, config, mode, block):
        super().__init__()
        self.mode = mode
        
        #FF1
        self.ffn1 = SwigluConvBlock(
            config["num_groups"], 
            config["emb_dim"], 
            float(config.get("eps", 1e-8)), 
            config["ffn_hidden_dim"][0], 
            config["conv1d_kernel"], 
            config["stride"], 
            config["dropout"]
        )
        
        #self attention
        
        self.attn_norm = RMSGroupNorm(config["num_groups"], config["emb_dim"], float(config.get("eps", 1e-8)))
        self.rope = RotaryEmbeddingOdd(config["emb_dim"] // config["n_heads"])
        if 'qkv_config' in config and config['qkv_config'] is not None:
            
            qkv_config = config['qkv_config'][block]
            if qkv_config["q_dim"] > 2:
                rope = RotaryEmbeddingOdd(qkv_config["q_dim"])
            else:
                rope = None
            self.attn = MultiHeadSelfAttention(
                qkv_config['hidden_size'],
                qkv_config["q_heads"]*qkv_config["q_dim"], 
                qkv_config["q_heads"], 
                dropout=config["dropout"],
                rope=rope,
                flash_attention=False,
                kdim=qkv_config["k_heads"]*qkv_config["k_dim"],
                vdim=qkv_config["v_heads"]*qkv_config["v_dim"],
                num_heads_k = qkv_config["k_heads"],
                num_heads_v = qkv_config["v_heads"],
                num_heads_out = qkv_config["v_heads"],
                head_dim = qkv_config["q_dim"],
                head_dim_k = qkv_config["k_dim"],
                head_dim_v = qkv_config["v_dim"],
                head_dim_out = qkv_config["v_dim"],
            )
        else:
            rope = RotaryEmbeddingOdd(config["emb_dim"] // config["n_heads"])
            self.attn = MultiHeadSelfAttention(
                config["emb_dim"], 
                config["emb_dim"], 
                config["n_heads"], 
                dropout=config["dropout"],
                rope=rope
            )
        
        #FF2
        self.ffn2 = SwigluConvBlock(
            config["num_groups"], 
            config["emb_dim"], 
            float(config.get("eps", 1e-8)), 
            config["ffn_hidden_dim"][1], 
            config["conv1d_kernel"], 
            config["stride"], 
            config["dropout"]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.mode == "freq":
            B, T, F, C = x.shape
        elif self.mode == "time":
            B, F, T, C = x.shape
        
        # FFN1 with residual
        ffn1_out = self.ffn1(x)
        att_input = ffn1_out 
        
        # Self-attention with proper normalization
        normed_input = self.attn_norm(att_input)
        
        if self.mode == "freq":
            x_reshaped = normed_input.reshape(B * T, F, C)
            attn_out, _ = self.attn(x_reshaped, x_reshaped, x_reshaped)
            att_out_reshaped = attn_out.reshape(B, T, F, C) + att_input

        elif self.mode == "time":
            x_reshaped = normed_input.reshape(B * F, T, C)
            attn_out, _ = self.attn(x_reshaped, x_reshaped, x_reshaped)
            att_out_reshaped = attn_out.reshape(B, F, T, C) + att_input

        # FFN2 with residual
        ffn2_out = self.ffn2(att_out_reshaped)
        return ffn2_out


class TFLocoFormerBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        t_ksize = 3
        ks, padding = (t_ksize, 3), (t_ksize // 2, 1)
        
        in_channels = config.get("in_channels", 2)
        out_channels = 2 * config.get("num_spk", 2)

        # Encoder with better initialization
        self.encoder = nn.Conv2d(in_channels, config["emb_dim"], kernel_size=ks, padding=padding)
        nn.init.xavier_uniform_(self.encoder.weight, gain=0.1)  # Smaller initialization
        if self.encoder.bias is not None:
            nn.init.zeros_(self.encoder.bias)
        
        self.enc_norm = nn.GroupNorm(1, config["emb_dim"], eps=float(config.get("eps", 1e-8)))

        # Decoder with better initialization
        self.decoder = nn.ConvTranspose2d(config["emb_dim"], out_channels, kernel_size=ks, padding=padding)
        nn.init.xavier_uniform_(self.decoder.weight, gain=0.1)  # Smaller initialization
        if self.decoder.bias is not None:
            nn.init.zeros_(self.decoder.bias)

        # Build blocks
        self.blocks = nn.ModuleList()
        for i in range(config["n_layers"]):
            self.blocks.append(LocoFormerBlock(config, mode="freq" if config["tf_order"] == "ft" else "time", block=i*2))
            self.blocks.append(LocoFormerBlock(config, mode="time" if config["tf_order"] == "ft" else "freq", block=i*2+1))

    def forward(self, x):  # x: (B, 2, T, F)
        x = self.encoder(x)  # (B, D, T, F)
        # check_nan_inf(x, "encoder output")
        
        x = self.enc_norm(x)
        # check_nan_inf(x, "norm output")

        if self.config["tf_order"] == "ft":
            x = x.permute(0, 2, 3, 1)  # (B, T, F, D)
        else:
            x = x.permute(0, 3, 2, 1)  # (B, F, T, D)

        for i, (block1, block2) in enumerate(zip(self.blocks[::2], self.blocks[1::2])):
            x = block1(x).transpose(1, 2)
            # check_nan_inf(x, f"block {i*2} output")
            
            x = block2(x).transpose(1, 2)
            # check_nan_inf(x, f"block {i*2+1} output")

        if self.config["tf_order"] == "ft":
            x = x.permute(0, 3, 1, 2)  # (B, D, T, F)
        else:
            x = x.permute(0, 3, 2, 1)  # (B, D, T, F)

        x = self.decoder(x)  # (B, 2*num_spk, T, F)
        # check_nan_inf(x, "decoder output")
        
        B, C, T, F = x.shape
        assert C == 2*self.config["num_spk"]
        x = x.contiguous().view(B, 2, self.config["num_spk"], T, F)
        
        # Apply tanh to bound the output
        # x = torch.tanh(x)  # This helps prevent extreme mask values
        
        return [torch.complex(x[:, 0, i], x[:, 1, i]) for i in range(self.config["num_spk"])]


class TFLocoformerWithSTFT(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        self.base_model = TFLocoFormerBlock(config)
        self.n_fft = config['n_fft']
        self.hop_length = config['hop_length']
        
        self.register_buffer('window', torch.hann_window(self.n_fft), persistent=False)

    def forward(self, x):  # x: waveform (B, T)
        T_orig = x.shape[-1]

        stft = torch.stft(
            x,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            window=self.window.to(x.device),
            return_complex=True
        )  # (B, F, T)
        
    
        real = stft.real
        imag = stft.imag
        tf_input = torch.stack([real, imag], dim=1)  # (B, 2, F, T)
        
        tf_input = tf_input.permute(0, 1, 3, 2)  # (B, 2, T, F)
        
        
        reconstructed_spectrograms  = self.base_model(tf_input) # (B, 2, num_spk, T, F)
        
        outputs = []
        for recon_spec in reconstructed_spectrograms:
            # Apply mask to original STFT
            
            wav = torch.istft(
                recon_spec.transpose(-2, -1),
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                window=self.window.to(x.device),
                length=T_orig
            )
            outputs.append(wav)

        return torch.stack(outputs, dim=1)  # (B, num_spk, T)

def create_locoformer(config):
    return TFLocoformerWithSTFT(config)