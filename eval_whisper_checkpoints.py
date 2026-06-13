#!/usr/bin/env python3
"""
STAGE 2 of the Whisper pruning pipeline: evaluate already-saved checkpoints.

`pruning_whisper.py` (stage 1) prunes + distill-finetunes and writes, per
iteration, a checkpoint dir `model_<N>M_params/` containing the HF model + a
`sparsity.json` (overall + per-layer attention sparsity). This script loads each
such checkpoint, runs the full multilingual eval suite (WER + CoVoST BLEU), and
optionally benchmarks FLOPs + forward latency, then writes one aggregated JSON
keyed by attention sparsity so results map straight onto the sparsity axis.

Reconstruction is automatic: `save_checkpoint` already called
`update_config_from_model`, so the saved `LayerWiseWhisperConfig` encodes the
pruned per-layer head dims and `load_checkpoint` rebuilds the right shapes.

Usage:
  # one strategy dir (contains model_*M_params/ subdirs):
  python eval_whisper_checkpoints.py --strategy_dir RESULTS/global_per_head_fisher_information

  # whole sweep (all strategy dirs under a root) + dense baseline:
  python eval_whisper_checkpoints.py --sweep_root RESULTS --include_dense

  # quick: only the seen-language ASR sets, no benchmarking:
  python eval_whisper_checkpoints.py --strategy_dir DIR --asr_preset seen --no-bench
"""
import argparse
import glob
import json
import os
import time

import torch

import config_whisper as config
from pruning_whisper import load_checkpoint
from whisper_utils import load_model
from whisper_eval import (
    evaluate_all_languages,
    PRESET_SEEN, PRESET_UNSEEN, PRESET_ALL_ASR, PRESET_ALL_BLEU,
)

ASR_PRESETS = {"seen": PRESET_SEEN, "unseen": PRESET_UNSEEN,
               "all": PRESET_ALL_ASR, "none": []}


def _ckpt_dirs(strategy_dir):
    """Checkpoint subdirs sorted by param count DESC (sparsity ascending)."""
    dirs = []
    for d in glob.glob(os.path.join(strategy_dir, "model_*M_params")):
        sp = os.path.join(d, "sparsity.json")
        if os.path.isdir(d) and os.path.exists(sp):
            try:
                pc = int(os.path.basename(d).split("_")[1].rstrip("M"))
            except Exception:
                pc = 0
            dirs.append((pc, d))
    return [d for _, d in sorted(dirs, key=lambda x: -x[0])]


@torch.no_grad()
def benchmark(model, device, seeds=5, reps=200, warmup=50, decoder_len=16):
    """Mean +/- std forward latency (ms, batch=1) over `seeds` random inputs,
    plus best-effort FLOPs. Run AFTER evaluate_all_languages, so the model is
    already on `device` in its eval dtype (bf16). Whisper encoder self-attn over
    the full 1500-frame sequence is exactly where entire-head pruning pays off.
    """
    mel = getattr(model.config, "num_mel_bins", 80)
    frames = 2 * getattr(model.config, "max_source_positions", 1500)  # 3000 = 30s
    start_id = (getattr(model.config, "decoder_start_token_id", None)
                or getattr(model.config, "bos_token_id", 50257))
    dtype = next(model.parameters()).dtype

    dec = torch.full((1, decoder_len), start_id, dtype=torch.long, device=device)

    def one_forward(x):
        model(input_features=x, decoder_input_ids=dec)

    # FLOPs (best effort; architecture-determined, dtype-agnostic). fvcore needs
    # an nn.Module and Whisper wants decoder_input_ids by keyword, so wrap it.
    flops_g = None
    try:
        from utils import compute_flops_with_handlers

        class _FwdWrap(torch.nn.Module):
            def __init__(self, m, dec):
                super().__init__(); self.m = m; self.dec = dec
            def forward(self, x):
                return self.m(input_features=x, decoder_input_ids=self.dec)

        x = torch.randn(1, mel, frames, device=device, dtype=dtype)
        flops_g = compute_flops_with_handlers(_FwdWrap(model, dec), x) / 1e9
    except Exception as e:
        print(f"[bench] FLOPs skipped: {type(e).__name__}: {e}")

    cuda = device.startswith("cuda")
    all_means = []
    for s in range(seeds):
        torch.manual_seed(1000 + s)
        x = torch.randn(1, mel, frames, device=device, dtype=dtype)
        for _ in range(warmup):
            one_forward(x)
        if cuda:
            torch.cuda.synchronize()
        t = []
        for _ in range(reps):
            if cuda:
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            one_forward(x)
            if cuda:
                torch.cuda.synchronize()
            t.append(time.perf_counter() - t0)
        all_means.append(sum(t) / len(t) * 1000.0)

    mean = sum(all_means) / len(all_means)
    var = sum((m - mean) ** 2 for m in all_means) / max(len(all_means) - 1, 1)
    return {"latency_ms_mean": mean, "latency_ms_std": var ** 0.5,
            "flops_g": flops_g, "bench_seeds": seeds, "bench_reps": reps}


def eval_one(ckpt_dir, device, asr_tags, trans_tags, batch_size, do_bench,
             seeds, reps):
    with open(os.path.join(ckpt_dir, "sparsity.json")) as f:
        sp = json.load(f)

    model, processor, *_ = load_checkpoint(ckpt_dir, device=device, load_lora=False)

    results = evaluate_all_languages(
        model=model, processor=processor,
        transcription_tags=asr_tags, translation_tags=trans_tags,
        batch_size=batch_size, save_results=False,
    )

    rec = {
        "checkpoint": ckpt_dir,
        "iteration": sp.get("iteration"),
        "param_count": sp.get("param_count"),
        "attention_sparsity": sp.get("attention_sparsity"),
        "eval": results,
    }
    # convenience aggregates
    wers = [v["wer"] for v in results.values() if v.get("wer") is not None and v.get("seen")]
    if wers:
        rec["seen_avg_wer"] = sum(wers) / len(wers)

    if do_bench:
        try:
            rec.update(benchmark(model, device, seeds=seeds, reps=reps))
        except Exception as e:
            print(f"[bench] skipped for {ckpt_dir}: {type(e).__name__}: {e}")

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return rec


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--strategy_dir", help="one dir with model_*M_params/ subdirs")
    g.add_argument("--sweep_root", help="root holding many strategy dirs")
    ap.add_argument("--include_dense", action="store_true",
                    help="also eval config.MODEL_NAME as the 0%% baseline")
    ap.add_argument("--asr_preset", choices=list(ASR_PRESETS), default="all")
    ap.add_argument("--no-trans", dest="trans", action="store_false",
                    help="skip CoVoST translation/BLEU sets")
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--bench", dest="bench", action="store_true", default=True)
    ap.add_argument("--no-bench", dest="bench", action="store_false")
    ap.add_argument("--bench_seeds", type=int, default=5)
    ap.add_argument("--bench_reps", type=int, default=200)
    ap.add_argument("--out", default=None, help="output json (default per dir)")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    asr_tags = ASR_PRESETS[args.asr_preset] or None
    trans_tags = None if args.trans else []

    if args.strategy_dir:
        strategy_dirs = [args.strategy_dir]
    else:
        strategy_dirs = sorted(
            d for d in glob.glob(os.path.join(args.sweep_root, "*"))
            if os.path.isdir(d) and _ckpt_dirs(d)
        )

    out_path = args.out or os.path.join(
        args.strategy_dir or args.sweep_root, "eval_results.json")
    records = []

    def flush():
        with open(out_path, "w") as f:
            json.dump(records, f, indent=2)

    if args.include_dense:
        print(f"\n##### DENSE baseline: {config.MODEL_NAME}")
        try:
            model, processor = load_model(config.MODEL_NAME)
            model = model.to(device)
            res = evaluate_all_languages(model=model, processor=processor,
                                         transcription_tags=asr_tags,
                                         translation_tags=trans_tags,
                                         batch_size=args.batch_size,
                                         save_results=False)
            rec = {"checkpoint": "DENSE", "iteration": 0,
                   "attention_sparsity": 0.0, "eval": res}
            if args.bench:
                rec.update(benchmark(model, device, args.bench_seeds, args.bench_reps))
            records.append(rec); flush()
            del model
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
        except Exception as e:
            print(f"[dense] failed: {type(e).__name__}: {e}")

    for sdir in strategy_dirs:
        for ckpt in _ckpt_dirs(sdir):
            print(f"\n##### {os.path.basename(sdir)} :: {os.path.basename(ckpt)}")
            try:
                rec = eval_one(ckpt, device, asr_tags, trans_tags,
                               args.batch_size, args.bench,
                               args.bench_seeds, args.bench_reps)
                rec["strategy"] = os.path.basename(sdir)
                records.append(rec); flush()
            except Exception as e:
                print(f"[eval] failed for {ckpt}: {type(e).__name__}: {e}")
                records.append({"checkpoint": ckpt, "strategy": os.path.basename(sdir),
                                "error": f"{type(e).__name__}: {e}"})
                flush()

    print(f"\nWrote {len(records)} records -> {out_path}")


if __name__ == "__main__":
    main()
