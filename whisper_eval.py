import os
import torch
import sacrebleu
from tqdm import tqdm
from jiwer import wer
from torch.utils.data import DataLoader
import json
from datetime import datetime
import threading
import time
import subprocess
import sys

import config_whisper as config

from whisper_utils import (
    LibriSpeechDataset,
    FleursDataset,
    CoVoSTDataset,
    CommonVoiceDataset,
    MLSDataset,
    MeanwhileDataset,
    whisper_collate_fn,
    normalize_whisper,
    normalize_librispeech,
    to_whisper_lang,
    load_model,
)

# -----------------------------------------------------------------------
# EVALUATION SETS
#
# Edit these lists to control what gets evaluated.
# Set to None to skip that entire category.
#
# Transcription sets — each entry:
#   (tag, dataset_type, root_config_attr, dataset_kwargs, lang_code, seen_in_training)
#
# Translation sets — each entry:
#   (tag, covost_pair, source_lang_code, seen_in_training)
# -----------------------------------------------------------------------

TRANSCRIPTION_SETS = [
    # ── Seen languages ──────────────────────────────────────────────────
    ("librispeech_en",  "librispeech",  "LIBRISPEECH_ROOT",  {"split": "test-clean"},              "en",  True),
    ("commonvoice_fr",  "commonvoice",  "COMMON_VOICE_ROOT", {"lang": "fr", "split": "test"},       "fr",  True),
    ("commonvoice_de",  "commonvoice",  "COMMON_VOICE_ROOT", {"lang": "de", "split": "test"},       "de",  True),
    ("commonvoice_es",  "commonvoice",  "COMMON_VOICE_ROOT", {"lang": "es", "split": "test"},       "es",  True),
    ("commonvoice_it",  "commonvoice",  "COMMON_VOICE_ROOT", {"lang": "it", "split": "test"},       "it",  True),
    ("commonvoice_zh",  "commonvoice",  "COMMON_VOICE_ROOT", {"lang": "zh", "split": "test"},       "zh",  True),
    ("mls_pl",          "mls",          "MLS_ROOT",          {"lang": "polish", "split": "test"},   "pl",  True),
    # ── Unseen languages ─────────────────────────────────────────────────
    ("commonvoice_ru",  "commonvoice",  "COMMON_VOICE_ROOT", {"lang": "ru", "split": "test"},       "ru",  False),
    ("commonvoice_ar",  "commonvoice",  "COMMON_VOICE_ROOT", {"lang": "ar", "split": "test"},       "ar",  False),
    ("fleurs_hi",       "fleurs",       "FLEURS_ROOT",       {"lang": "hi_in", "split": "test",
                                                              "task": "transcribe"},                 "hi",  False),
]

TRANSLATION_SETS = [
    # ── Seen pairs ────────────────────────────────────────────────────────
    ("covost_de_en",    "de_en",    "de",   True),
    ("covost_zh-CN_en", "zh-CN_en", "zh",   True),
    # ── Unseen pairs ──────────────────────────────────────────────────────
    ("covost_ar_en",    "ar_en",    "ar",   False),
]

# -----------------------------------------------------------------------
# To restrict evaluation to a subset, pass tag names to evaluate_all():
#
#   evaluate_all(model, processor,
#       transcription_tags=["librispeech_en", "commonvoice_de"],
#       translation_tags=["covost_de_en"])
#
# Or use the convenience presets below.
# -----------------------------------------------------------------------

PRESET_SEEN      = [t[0] for t in TRANSCRIPTION_SETS if t[5]]
PRESET_UNSEEN    = [t[0] for t in TRANSCRIPTION_SETS if not t[5]]
PRESET_ALL_ASR   = [t[0] for t in TRANSCRIPTION_SETS]
PRESET_ALL_BLEU  = [t[0] for t in TRANSLATION_SETS]


# -----------------------------------------------------------------------
# GPU monitoring
# -----------------------------------------------------------------------
def print_gpu_stats(label=""):
    try:
        out = subprocess.check_output(
            ["nvidia-smi",
             "--query-gpu=index,name,utilization.gpu,memory.used,memory.total",
             "--format=csv,noheader,nounits"],
            encoding="utf-8"
        )
        prefix = f"[GPU @ {label}] " if label else "[GPU] "
        for line in out.strip().splitlines():
            idx, name, util, mem_used, mem_total = [x.strip() for x in line.split(",")]
            print(f"{prefix}GPU {idx} ({name}) | Util: {util}% | VRAM: {mem_used}/{mem_total} MB")
    except Exception as e:
        print(f"[GPU stats unavailable: {e}]")


class GPUMonitor:
    def __init__(self, interval_seconds=30, label=""):
        self.interval = interval_seconds
        self.label = label
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        elapsed = 0
        while not self._stop_event.is_set():
            time.sleep(1)
            elapsed += 1
            if elapsed % self.interval == 0:
                print_gpu_stats(f"{self.label} [{elapsed}s]")
                sys.stdout.flush()

    def start(self):
        print_gpu_stats(f"{self.label} [start]")
        sys.stdout.flush()
        self._thread.start()
        return self

    def stop(self):
        self._stop_event.set()
        self._thread.join()
        print_gpu_stats(f"{self.label} [end]")
        sys.stdout.flush()

    def __enter__(self):  return self.start()
    def __exit__(self, *_): self.stop()


# -----------------------------------------------------------------------
# Dataset builder
# -----------------------------------------------------------------------
def _build_dataset(ds_type, root, kwargs):
    if ds_type == "librispeech":
        return LibriSpeechDataset(root=root, **kwargs)
    elif ds_type == "commonvoice":
        return CommonVoiceDataset(root=root, **kwargs)
    elif ds_type == "mls":
        return MLSDataset(root=root, **kwargs)
    elif ds_type == "fleurs":
        return FleursDataset(root=root, **kwargs)
    else:
        raise ValueError(f"Unknown dataset type: {ds_type}")


def _make_loader(ds, processor, batch_size=48, num_workers=4):
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=lambda b: whisper_collate_fn(b, processor),
    )


# -----------------------------------------------------------------------
# Inference loops
# -----------------------------------------------------------------------
def _run_asr(model, processor, device, loader, lang_code):
    whisper_lang = to_whisper_lang(lang_code)
    preds, refs = [], []
    for batch in tqdm(loader):
        input_features = batch["input_features"].to(device, dtype=torch.bfloat16)
        attention_mask = batch["attention_mask"].to(device, dtype=torch.bfloat16)
        with torch.no_grad():
            predicted_ids = model.generate(
                input_features,
                attention_mask=attention_mask,
                task="transcribe",
                language=whisper_lang,
                max_new_tokens=444,
            )
        batch_preds = processor.batch_decode(predicted_ids, skip_special_tokens=True)
        preds.extend([normalize_librispeech(p) for p in batch_preds])
        refs.extend([normalize_librispeech(r) for r in batch["texts"]])
    return preds, refs


def _run_translate(model, processor, device, loader, source_lang_code):
    whisper_lang = to_whisper_lang(source_lang_code)
    preds, refs = [], []
    for batch in tqdm(loader):
        input_features = batch["input_features"].to(device, dtype=torch.bfloat16)
        attention_mask = batch["attention_mask"].to(device, dtype=torch.bfloat16)
        with torch.no_grad():
            predicted_ids = model.generate(
                input_features,
                attention_mask=attention_mask,
                task="translate",
                language=whisper_lang,
                max_new_tokens=444,
            )
        batch_preds = processor.batch_decode(predicted_ids, skip_special_tokens=True)
        preds.extend([normalize_librispeech(p) for p in batch_preds])
        refs.extend([normalize_librispeech(r) for r in batch["texts"]])
    return preds, refs


# -----------------------------------------------------------------------
# Meanwhile (long-form, separate loop)
# -----------------------------------------------------------------------
def evaluate_meanwhile(model, processor, device):
    from whisper_utils import transcribe_long_audio
    results = {}
    tag = "meanwhile_test"
    print(f"\n{'='*60}\nEvaluating {tag}\n{'='*60}")
    try:
        ds = MeanwhileDataset(root=config.MEANWHILE_ROOT, split="test")
        preds, refs = [], []
        for i in tqdm(range(len(ds))):
            sample = ds[i]
            hyp = transcribe_long_audio(sample["audio"], processor, model, device)
            preds.append(normalize_whisper(hyp))
            refs.append(normalize_whisper(sample["text"]))
        score = wer(refs, preds)
        print(f"WER: {score:.4f}")
        results[tag] = {"wer": score, "seen": True}
    except Exception as e:
        print(f"Error: {e}")
        results[tag] = {"wer": None, "seen": True}
    return results


# -----------------------------------------------------------------------
# Main evaluation entry point
# -----------------------------------------------------------------------
def evaluate_all_languages(
    model=None,
    processor=None,
    transcription_tags=None,
    translation_tags=None,
    include_meanwhile=False,
    batch_size=64,
    num_workers=2,
    save_results=True,
):
    """
    Run evaluation on the configured sets.

    Args:
        model              : WhisperForConditionalGeneration (loaded if None)
        processor          : WhisperProcessor (loaded if None)
        transcription_tags : list of tag strings from TRANSCRIPTION_SETS to run,
                             or None to run all. Use PRESET_* constants for convenience.
                             Examples:
                               transcription_tags=PRESET_SEEN
                               transcription_tags=["librispeech_en", "commonvoice_de"]
        translation_tags   : list of tag strings from TRANSLATION_SETS to run,
                             or None to run all.
                             Examples:
                               translation_tags=["covost_de_en"]
                               translation_tags=PRESET_ALL_BLEU
        include_meanwhile  : whether to run Meanwhile long-form eval (slow)
        batch_size         : inference batch size
        num_workers        : DataLoader workers
        save_results       : whether to write JSON results file

    Returns:
        dict mapping tag -> {"wer"|"bleu": float|None, "seen": bool}
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if model is None or processor is None:
        model, processor = load_model(config.MODEL_NAME)
    model = model.to(torch.bfloat16)
    model.to(device)
    model.eval()
    print(f"Model ready on {device}")

    # Resolve which sets to run
    asr_to_run = (
        [s for s in TRANSCRIPTION_SETS if s[0] in transcription_tags]
        if transcription_tags is not None
        else TRANSCRIPTION_SETS
    )
    trans_to_run = (
        [s for s in TRANSLATION_SETS if s[0] in translation_tags]
        if translation_tags is not None
        else TRANSLATION_SETS
    )

    unknown_asr = (
        set(transcription_tags or []) - {s[0] for s in TRANSCRIPTION_SETS}
    )
    unknown_trans = (
        set(translation_tags or []) - {s[0] for s in TRANSLATION_SETS}
    )
    if unknown_asr:
        print(f"WARNING: unknown transcription tags: {unknown_asr}")
    if unknown_trans:
        print(f"WARNING: unknown translation tags: {unknown_trans}")

    results = {}

    # ── Transcription ──────────────────────────────────────────────────
    if asr_to_run:
        seen_wers, unseen_wers = [], []
        print(f"\n{'='*60}\nTRANSCRIPTION  ({len(asr_to_run)} sets)\n{'='*60}")

        for tag, ds_type, root_cfg, kwargs, lang_code, seen in asr_to_run:
            label = f"{'[SEEN]  ' if seen else '[UNSEEN]'} {tag}"
            print(f"\n--- {label} ---")
            with GPUMonitor(interval_seconds=60, label=tag):
                try:
                    root   = getattr(config, root_cfg)
                    ds     = _build_dataset(ds_type, root, kwargs)
                    loader = _make_loader(ds, processor, batch_size, num_workers)
                    preds, refs = _run_asr(model, processor, device, loader, lang_code)
                    score = wer(refs, preds)
                    print(f"WER: {score:.4f}")
                    results[tag] = {"wer": score, "seen": seen}
                    (seen_wers if seen else unseen_wers).append(score)
                except Exception as e:
                    print(f"Error: {e}")
                    results[tag] = {"wer": None, "seen": seen}

        if seen_wers:
            print(f"\nSeen    avg WER : {sum(seen_wers)  / len(seen_wers):.4f}  "
                  f"({len(seen_wers)} sets)")
        if unseen_wers:
            print(f"Unseen  avg WER : {sum(unseen_wers) / len(unseen_wers):.4f}  "
                  f"({len(unseen_wers)} sets)")

    # ── Translation ────────────────────────────────────────────────────
    if trans_to_run:
        seen_bleus, unseen_bleus = [], []
        print(f"\n{'='*60}\nTRANSLATION  ({len(trans_to_run)} pairs)\n{'='*60}")

        for tag, lang_pair, source_lang, seen in trans_to_run:
            label = f"{'[SEEN]  ' if seen else '[UNSEEN]'} {tag}"
            print(f"\n--- {label} ---")
            with GPUMonitor(interval_seconds=60, label=tag):
                try:
                    ds = CoVoSTDataset(
                        root=config.COVOST_ROOT,
                        lang_pair=lang_pair,
                        split="test",
                        task="translate",
                    )
                    loader = _make_loader(ds, processor, batch_size, num_workers)
                    preds, refs = _run_translate(
                        model, processor, device, loader, source_lang
                    )
                    bleu = sacrebleu.corpus_bleu(preds, [refs])
                    print(f"BLEU: {bleu.score:.2f}")
                    results[tag] = {"bleu": bleu.score, "seen": seen}
                    (seen_bleus if seen else unseen_bleus).append(bleu.score)
                except Exception as e:
                    print(f"Error: {e}")
                    results[tag] = {"bleu": None, "seen": seen}

        if seen_bleus:
            print(f"\nSeen    avg BLEU: {sum(seen_bleus)  / len(seen_bleus):.2f}  "
                  f"({len(seen_bleus)} pairs)")
        if unseen_bleus:
            print(f"Unseen  avg BLEU: {sum(unseen_bleus) / len(unseen_bleus):.2f}  "
                  f"({len(unseen_bleus)} pairs)")

    # ── Meanwhile (optional) ───────────────────────────────────────────
    if include_meanwhile:
        results.update(evaluate_meanwhile(model, processor, device))

    # ── Summary ────────────────────────────────────────────────────────
    print(f"\n{'='*80}\nEVALUATION SUMMARY\n{'='*80}")
    for tag, entry in sorted(results.items()):
        seen_label = "[SEEN]  " if entry.get("seen") else "[UNSEEN]"
        if "wer" in entry:
            val = f"WER  {entry['wer']:.4f}" if entry["wer"] is not None else "WER  FAILED"
        else:
            val = f"BLEU {entry['bleu']:.2f}" if entry["bleu"] is not None else "BLEU FAILED"
        print(f"  {seen_label} {tag:40s}  {val}")

    if save_results:
        timestamp   = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = os.path.join(
            getattr(config, "results_root", "."),
            f"eval_{timestamp}.json"
        )
        with open(output_file, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {output_file}")

    return results


# -----------------------------------------------------------------------
# CLI entry point
# -----------------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Evaluate Whisper model on a configurable set of languages."
    )
    parser.add_argument(
        "--asr", nargs="*", default=None,
        help=(
            "Transcription tags to evaluate. "
            "Omit to run all. Use 'seen' or 'unseen' as shortcuts. "
            f"Available: {[s[0] for s in TRANSCRIPTION_SETS]}"
        )
    )
    parser.add_argument(
        "--bleu", nargs="*", default=None,
        help=(
            "Translation tags to evaluate. "
            "Omit to run all. "
            f"Available: {[s[0] for s in TRANSLATION_SETS]}"
        )
    )
    parser.add_argument(
        "--meanwhile", action="store_true",
        help="Include Meanwhile long-form evaluation"
    )
    parser.add_argument("--batch_size",  type=int, default=96)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--no_save",     action="store_true")
    args = parser.parse_args()

    # Resolve shortcut keywords
    asr_tags = args.asr
    if asr_tags == ["seen"]:
        asr_tags = PRESET_SEEN
    elif asr_tags == ["unseen"]:
        asr_tags = PRESET_UNSEEN

    evaluate_all_languages(
        transcription_tags=asr_tags,
        translation_tags=args.bleu,
        include_meanwhile=args.meanwhile,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        save_results=not args.no_save,
    )