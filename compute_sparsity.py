"""
Computes attention parameter sparsity across pruning iterations and logs to TensorBoard.

Sparsity definition is consistent with get_model_sparsity_stats():
    sparsity = 1 - (pruned_params / original_params)

Parameter count per attention block layer:
    qkv_params      = embed_dim * (nq*dq + nk*dk + nv*dv)
    qkv_bias_params =              nq*dq + nk*dk + nv*dv
    out_params      = embed_dim *  no*do

In the config, Q and K share num_attention_heads / head_dim,
V uses num_attention_heads_v / head_dim_v,
and the output projection mirrors V (no=nv, do=dv).

Usage:
    python compute_sparsity.py \
        --experiment_dir /path/to/global_per_head_fisher_information \
        --base_model_dir /path/to/basemodel
"""

import re
import json
import argparse
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter
from transformers import WhisperConfig


# ── Config class ─────────────────────────────────────────────────────────────

class LayerWiseWhisperConfig(WhisperConfig):
    model_type = "custom-whisper"

    def __init__(self, qkv_config=None, **kwargs):
        super().__init__(**kwargs)

        if not hasattr(self, "encoder_self_qkv_config"):
            self.encoder_self_qkv_config = [{
                'hidden_size': self.d_model,
                "num_attention_heads": self.encoder_attention_heads,
                "num_attention_heads_v": self.encoder_attention_heads,
                "head_dim": self.d_model // self.encoder_attention_heads,
                "head_dim_v": self.d_model // self.encoder_attention_heads,
                "attention_dropout": self.attention_dropout,
            } for _ in range(self.encoder_layers)]

        if not hasattr(self, "decoder_self_qkv_config"):
            self.decoder_self_qkv_config = [{
                'hidden_size': self.d_model,
                "num_attention_heads": self.decoder_attention_heads,
                "num_attention_heads_v": self.decoder_attention_heads,
                "head_dim": self.d_model // self.decoder_attention_heads,
                "head_dim_v": self.d_model // self.decoder_attention_heads,
                "attention_dropout": self.attention_dropout,
            } for _ in range(self.decoder_layers)]

        if not hasattr(self, "decoder_cross_qkv_config"):
            self.decoder_cross_qkv_config = [{
                'hidden_size': self.d_model,
                "num_attention_heads": self.decoder_attention_heads,
                "num_attention_heads_v": self.decoder_attention_heads,
                "head_dim": self.d_model // self.decoder_attention_heads,
                "head_dim_v": self.d_model // self.decoder_attention_heads,
                "attention_dropout": self.attention_dropout,
            } for _ in range(self.decoder_layers)]


# ── Parameter counting (mirrors get_model_sparsity_stats) ────────────────────

def layer_attention_params(layer: dict, embed_dim: int) -> int:
    """
    Count attention parameters for a single layer config dict.

    Matches the formula in get_model_sparsity_stats():
        qkv_params      = embed_dim * (nq*dq + nk*dk + nv*dv)
        qkv_bias_params =              nq*dq + nk*dk + nv*dv
        out_params      = embed_dim *  no*do

    Q and K share num_attention_heads / head_dim.
    V uses num_attention_heads_v / head_dim_v.
    Output projection is assumed to mirror V (no=nv, do=dv),
    which is standard for Whisper-style cross/self attention.
    """
    nq = layer["num_attention_heads"]
    dq = layer["head_dim"]
    nk = nq          # K always matches Q in Whisper configs
    dk = dq
    nv = layer["num_attention_heads_v"]
    dv = layer["head_dim_v"]
    no = nv           # output projection mirrors V
    do = dv

    qkv_params      = embed_dim * (nq * dq + nk * dk + nv * dv)
    qkv_bias_params =              nq * dq + nk * dk + nv * dv
    out_params      = embed_dim *  no * do

    return qkv_params + qkv_bias_params + out_params


def block_params(qkv_config: list, embed_dim: int) -> int:
    """Total attention parameters across all layers of one block type."""
    return sum(layer_attention_params(layer, embed_dim) for layer in qkv_config)


def get_attention_params(cfg: LayerWiseWhisperConfig) -> dict:
    embed_dim = cfg.d_model
    return {
        "encoder_self":  block_params(cfg.encoder_self_qkv_config,  embed_dim),
        "decoder_self":  block_params(cfg.decoder_self_qkv_config,  embed_dim),
        "decoder_cross": block_params(cfg.decoder_cross_qkv_config, embed_dim),
    }


# ── I/O helpers ───────────────────────────────────────────────────────────────

def load_config(folder: Path) -> LayerWiseWhisperConfig:
    with open(folder / "config.json") as f:
        cfg_dict = json.load(f)
    for key in ("model_type", "architectures", "transformers_version", "dtype"):
        cfg_dict.pop(key, None)
    return LayerWiseWhisperConfig(**cfg_dict)


def extract_param_count(folder_name: str) -> float:
    m = re.search(r"model_(\d+(?:\.\d+)?)M_params", folder_name)
    if m:
        return float(m.group(1))
    raise ValueError(f"Cannot parse param count from folder name: {folder_name!r}")


def find_model_folders(experiment_dir: Path) -> list[Path]:
    """Return model folders sorted by param count descending (biggest = iteration 1)."""
    folders = [
        p for p in experiment_dir.iterdir()
        if p.is_dir() and re.match(r"model_\d+(?:\.\d+)?M_params", p.name)
    ]
    folders.sort(key=lambda p: extract_param_count(p.name), reverse=True)
    return folders


# ── Main ──────────────────────────────────────────────────────────────────────

def main(args):
    exp_path  = Path(args.experiment_dir)
    base_path = Path(args.base_model_dir)
    assert exp_path.is_dir(),  f"Not a directory: {exp_path}"
    assert base_path.is_dir(), f"Not a directory: {base_path}"

    folders = find_model_folders(exp_path)
    assert len(folders) >= 1, "No model_XXXM_params folders found."

    print(f"Found {len(folders)} model folders:")
    for i, f in enumerate(folders, 1):
        print(f"  it {i}: {f.name}")

    # Reference = fixed unpruned base model
    ref_cfg    = load_config(base_path)
    ref_params = get_attention_params(ref_cfg)
    ref_total  = sum(ref_params.values())
    print(f"\nReference attention params: {ref_params}  ->  total={ref_total:,}")

    writer = SummaryWriter(log_dir=str(exp_path / "tensorboard_logs"))

    for iteration, folder in enumerate(folders, start=1):
        cfg    = load_config(folder)
        params = get_attention_params(cfg)
        total  = sum(params.values())

        global_sparsity    = 1.0 - total                    / ref_total
        enc_self_sparsity  = 1.0 - params["encoder_self"]  / ref_params["encoder_self"]
        dec_self_sparsity  = 1.0 - params["decoder_self"]  / ref_params["decoder_self"]
        dec_cross_sparsity = 1.0 - params["decoder_cross"] / ref_params["decoder_cross"]

        print(
            f"  it {iteration:>3d} | {folder.name:<32s} | "
            f"attn_params={total:>10,} | sparsity={global_sparsity:.4f}"
        )

        writer.add_scalar("sparsity/global",        global_sparsity,    iteration)
        writer.add_scalar("sparsity/encoder_self",  enc_self_sparsity,  iteration)
        writer.add_scalar("sparsity/decoder_self",  dec_self_sparsity,  iteration)
        writer.add_scalar("sparsity/decoder_cross", dec_cross_sparsity, iteration)

        # Raw param counts for reference
        writer.add_scalar("attn_params/total",         total,                   iteration)
        writer.add_scalar("attn_params/encoder_self",  params["encoder_self"],  iteration)
        writer.add_scalar("attn_params/decoder_self",  params["decoder_self"],  iteration)
        writer.add_scalar("attn_params/decoder_cross", params["decoder_cross"], iteration)

        writer.add_scalar("model/params_M", extract_param_count(folder.name), iteration)

    writer.close()
    print(f"\nTensorBoard logs written to: {exp_path / 'tensorboard_logs'}")
    print(f"Run:  tensorboard --logdir '{exp_path / 'tensorboard_logs'}'")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Log attention sparsity to TensorBoard.")
    parser.add_argument("--experiment_dir", required=True,
                        help="Directory containing model_XXXM_params folders.")
    parser.add_argument("--base_model_dir", required=True,
                        help="Unpruned base model directory (contains config.json).")
    args = parser.parse_args()
    main(args)