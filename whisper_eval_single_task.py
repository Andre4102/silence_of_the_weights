"""
eval_task.py  —  evaluate ONE task on ONE model and log to TensorBoard.

This script is meant to be launched in parallel, one process per task per model.
The launcher (eval_all_models.py) submits one job per combination.

Usage:
    python eval_task.py \\
        --model_path /path/to/model_1501M_params \\
        --task librispeech_en \\
        --tb_logdir /path/to/tensorboard_logs \\
        [--batch_size 48] [--num_workers 4]

TensorBoard layout:
    Each task_name is a separate run so you get one curve per language.
    X-axis = pruning iteration (1 = largest model, N = most pruned).
    Y-axis = WER or BLEU depending on task.

    Example tag: librispeech_en/WER
                 covost_de_en/BLEU
"""

import os
import sys
import re
import argparse
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from jiwer import wer, cer
import sacrebleu
from torch.utils.tensorboard import SummaryWriter
import numpy as np

import config_whisper as config
from whisper_utils import (
    LibriSpeechDataset,
    FleursDataset,
    CoVoSTDataset,
    CommonVoiceDataset,
    MLSDataset,
    whisper_collate_fn,
    normalize_librispeech,
    to_whisper_lang,
    load_model,
    load_original_model
)
# import inspect
# import custom_attention
# print(inspect.getsource(custom_attention.WhisperAttention.forward))
# print(custom_attention.__file__)

# -----------------------------------------------------------------------
# Full evaluation set — same definition as whisper_eval.py
# -----------------------------------------------------------------------
TRANSCRIPTION_SETS = {
    "librispeech_en":  ("librispeech",  "LIBRISPEECH_ROOT",  {"split": "test-clean"},               "en",  True),
    "commonvoice_fr":  ("commonvoice",  "COMMON_VOICE_ROOT", {"lang": "fr", "split": "test"},        "fr",  True),
    "commonvoice_de":  ("commonvoice",  "COMMON_VOICE_ROOT", {"lang": "de", "split": "test"},        "de",  True),
    "commonvoice_es":  ("commonvoice",  "COMMON_VOICE_ROOT", {"lang": "es", "split": "test"},        "es",  True),
    "commonvoice_it":  ("commonvoice",  "COMMON_VOICE_ROOT", {"lang": "it", "split": "test"},        "it",  True),
    "commonvoice_zh":  ("commonvoice",  "COMMON_VOICE_ROOT", {"lang": "zh", "split": "test"},        "zh",  True),
    "mls_pl":          ("mls",          "MLS_ROOT",          {"lang": "polish", "split": "test"},    "pl",  True),
    "commonvoice_ru":  ("commonvoice",  "COMMON_VOICE_ROOT", {"lang": "ru", "split": "test"},        "ru",  False),
    "commonvoice_ar":  ("commonvoice",  "COMMON_VOICE_ROOT", {"lang": "ar", "split": "test"},        "ar",  False),
    "fleurs_hi":       ("fleurs",       "FLEURS_ROOT",       {"lang": "hi_in", "split": "test",
                                                              "task": "transcribe"},                  "hi",  False),
}

TRANSLATION_SETS = {
    "covost_de_en":    ("de_en",    "de",  True),
    "covost_zh-CN_en": ("zh-CN_en", "zh",  True),
    "covost_ar_en":    ("ar_en",    "ar",  False),
}

ALL_TASKS = list(TRANSCRIPTION_SETS.keys()) + list(TRANSLATION_SETS.keys())


# -----------------------------------------------------------------------
# Helpers
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


def _make_loader(ds, processor, batch_size, num_workers):
    def eval_collate(batch):
        audios = []
        texts  = []
        for item in batch:
            raw = item["audio"]
            if not isinstance(raw, np.ndarray):
                raw = np.array(raw, dtype=np.float32)
            # Pad/truncate to 30s
            if len(raw) < 480_000:
                raw = np.pad(raw, (0, 480_000 - len(raw)))
            else:
                raw = raw[:480_000]
            audios.append(raw)
            texts.append(item.get("text", ""))
        # print("[DEBUG] audio[0] mean:", audios[0].mean())
        # print("[DEBUG] audio[0] std:", audios[0].std())
        # print("[DEBUG] audio[0] max:", np.abs(audios[0]).max())

        inputs = processor.feature_extractor(
            audios,
            sampling_rate=16000,
            return_tensors="pt",
            return_attention_mask=True,
        )
        return {
            "input_features": inputs.input_features,
            "attention_mask":  inputs.attention_mask,
            "texts":           texts,
        }

    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=eval_collate,
    )


def _infer(model, processor, device, loader, lang_code, task):
    whisper_lang = to_whisper_lang(lang_code)
    preds, refs = [], []
    for batch in tqdm(loader, desc=f"{task}"):
        feats = batch["input_features"].to(device, dtype=torch.bfloat16)
        mask  = batch["attention_mask"].to(device)
        with torch.no_grad():
            predicted_ids = model.generate(
                feats, attention_mask=mask,
                task="transcribe" if task in TRANSCRIPTION_SETS else "translate",
                language=whisper_lang,
                max_new_tokens=444,
                use_cache=False
            )
        decoded = processor.batch_decode(predicted_ids, skip_special_tokens=True)

        # if not preds:
        #     feats_check = batch["input_features"]
        #     print("[DEBUG] input_features shape:", feats_check.shape)
        #     print("[DEBUG] input_features mean:", feats_check.mean().item())
        #     print("[DEBUG] input_features std:", feats_check.std().item())
        #     print("[DEBUG] input_features min:", feats_check.min().item())
        #     print("[DEBUG] input_features max:", feats_check.max().item())
        #     print("[DEBUG] generated ids:", predicted_ids[0].tolist())
        #     print("[DEBUG] generated ids length:", len(predicted_ids[0]))
        #     print("[DEBUG] RAW pred[0]:", decoded[0])
        #     print("[DEBUG] RAW ref[0] :", batch["texts"][0])
        #     forced = model.config.forced_decoder_ids
        #     print("forced_decoder_ids:", forced)
        #     ids = processor.get_decoder_prompt_ids(language="english", task="transcribe")
        #     print("decoder prompt ids:", ids)
        #     print("prompt length:", len(ids))
        #     print("[DEBUG] norm pred  :", normalize_librispeech(decoded[0]))
        #     print("[DEBUG] norm ref   :", normalize_librispeech(batch["texts"][0]))
        preds.extend([normalize_librispeech(p) for p in decoded])
        refs.extend([normalize_librispeech(r) for r in batch["texts"]])
    return preds, refs


def check():
    import torch
    import numpy as np
    import soundfile as sf
    from transformers import WhisperForConditionalGeneration, WhisperConfig, WhisperProcessor
    from custom_attention import LayerWiseWhisperConfig
    from whisper_utils import CustomWhisperForConditionalGeneration

    path = "/leonardo_scratch/large/userexternal/adiecidu/pruning/results/whisper/basemodel"
    processor = WhisperProcessor.from_pretrained(path)

    orig   = WhisperForConditionalGeneration.from_pretrained(path).eval()
    custom = CustomWhisperForConditionalGeneration.from_pretrained(
        path, config=LayerWiseWhisperConfig.from_pretrained(path)
    ).eval()

    # Same input for both
    audio, sr = sf.read("/leonardo_scratch/large/userexternal/adiecidu/pruning/data/audio/LibriSpeech/test-clean/1089/134691/1089-134691-0000.flac")
    audio = audio.astype(np.float32)
    inputs = processor(audio, sampling_rate=16000, return_tensors="pt")
    feats  = inputs.input_features  # fp32, no casting yet

    # with torch.no_grad():
    #     enc_orig   = orig.model.encoder(feats).last_hidden_state
    #     enc_custom = custom.model.encoder(feats).last_hidden_state

    #     dec_input = torch.tensor([[50258, 50259, 50359, 50363]])  # SOT en transcribe notimestamps

    #     dec_orig   = orig.model.decoder(
    #         input_ids=dec_input,
    #         encoder_hidden_states=enc_orig
    #     ).last_hidden_state

    #     dec_custom = custom.model.decoder(
    #         input_ids=dec_input,
    #         encoder_hidden_states=enc_custom
    #     ).last_hidden_state

    #     print("Decoder match:", torch.allclose(dec_orig, dec_custom, atol=1e-4))
    #     print("Decoder max diff:", (dec_orig - dec_custom).abs().max().item())

    #     logits_orig   = orig.proj_out(dec_orig)
    #     logits_custom = custom.proj_out(dec_custom)
    #     print("Logits match:", torch.allclose(logits_orig, logits_custom, atol=1e-4))
    #     print("Logits max diff:", (logits_orig - logits_custom).abs().max().item())
    #     print("Orig   top5:", logits_orig[0, -1].topk(5).indices.tolist())
    #     print("Custom top5:", logits_custom[0, -1].topk(5).indices.tolist())

    #     # Also check layer by layer in decoder
    #     for i in range(orig.config.decoder_layers):
    #         layer_orig   = orig.model.decoder.layers[i]
    #         layer_custom = custom.model.decoder.layers[i]
    #         # Compare self_attn weights
    #         sa_match = torch.allclose(
    #             layer_orig.self_attn.q_proj.weight,
    #             layer_custom.self_attn.q_proj.weight, atol=1e-6
    #         )
    #         ca_match = torch.allclose(
    #             layer_orig.encoder_attn.q_proj.weight,
    #             layer_custom.encoder_attn.q_proj.weight, atol=1e-6
    #         )
    #         print(f"  Layer {i:2d}  self_attn match: {sa_match}  cross_attn match: {ca_match}")

    with torch.no_grad():
        # Test generate with cache disabled
        ids_no_cache = orig.generate(
            feats, language="english", task="transcribe",
            use_cache=False, max_new_tokens=50
        )
        ids_custom_no_cache = custom.generate(
            feats, language="english", task="transcribe",
            use_cache=False, max_new_tokens=50
        )
        print("Orig (no cache):  ", processor.batch_decode(ids_no_cache, skip_special_tokens=True))
        print("Custom (no cache):", processor.batch_decode(ids_custom_no_cache, skip_special_tokens=True))

# -----------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------
def run(model_path, task, tb_logdir, iteration, n_params, batch_size, num_workers):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype  = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32

    print(f"\n{'='*60}")
    print(f"  Task  : {task}")
    print(f"  Model : {model_path}")
    print(f"  Device: {device}  dtype={dtype}")
    print(f"{'='*60}\n")

    print(to_whisper_lang('en'))

    # ── Pruning iteration and param count (passed explicitly by launcher) ──
    print(f"  Pruning iteration {iteration}  ({n_params}M params)")

    # check()

    # ── Load model ─────────────────────────────────────────────────────
    model, processor = load_model(model_path)
    model = model.to(dtype).to(device).eval()

    # ── Run evaluation ─────────────────────────────────────────────────
    score = None

    if task in TRANSCRIPTION_SETS:
        ds_type, root_cfg, kwargs, lang_code, seen = TRANSCRIPTION_SETS[task]
        root   = getattr(config, root_cfg)
        ds     = _build_dataset(ds_type, root, kwargs)
        loader = _make_loader(ds, processor, batch_size, num_workers)
        preds, refs = _infer(model, processor, device, loader, lang_code, task)
        if lang_code in ("zh", "ar", "hi"):
            score = cer(refs, preds)
            metric = "CER"
        else:
            score = wer(refs, preds)
            metric = "WER"
        print(f"{metric}: {score:.4f}")

    elif task in TRANSLATION_SETS:
        lang_pair, source_lang, seen = TRANSLATION_SETS[task]
        ds = CoVoSTDataset(
            root=config.COVOST_ROOT, lang_pair=lang_pair,
            split="test", task="translate",
        )
        loader = _make_loader(ds, processor, batch_size, num_workers)
        preds, refs = _infer(model, processor, device, loader, source_lang, task)
        bleu  = sacrebleu.corpus_bleu(preds, [refs])
        score = bleu.score
        lang_code = lang_pair
        metric = "BLEU"
        print(f"BLEU: {score:.2f}")

    else:
        raise ValueError(f"Unknown task: {task}. Available: {ALL_TASKS}")

    # ── Log to TensorBoard ─────────────────────────────────────────────
    # Each task gets its own run directory so they appear as separate
    # curves in TensorBoard. X-axis = pruning iteration.
    # task_logdir = os.path.join(tb_logdir, task)
    writer = SummaryWriter(log_dir=tb_logdir)
    writer.add_scalar(f"{metric}_{lang_code}", score, global_step=iteration)
    # Also log param count on a shared axis for reference
    writer.add_scalar("n_params_M", n_params, global_step=iteration)
    writer.close()

    print(f"\nLogged {metric}={score:.4f} at step={iteration} "
          f"to TensorBoard run '{task}'")

    return score, metric


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate one task on one model and log to TensorBoard."
    )
    parser.add_argument("--model_path",  default='/leonardo_scratch/large/userexternal/adiecidu/pruning/results/whisper/basemodel', help="Path to model folder")
    parser.add_argument("--task", default="librispeech_en", choices=ALL_TASKS, help="Which evaluation task to run")
    parser.add_argument("--tb_logdir",   required=True, help="TensorBoard log directory (shared across all tasks/models)")
    parser.add_argument("--batch_size",  type=int, default=48)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--iteration", type=int, required=True,
                        help="Pruning iteration index (0 unpruned, 1 = least pruned, N = most pruned)")
    parser.add_argument("--n_params", type=int, required=True,
                        help="Number of parameters in millions (from folder name)")
    args = parser.parse_args()

    run(
        model_path = args.model_path,
        task       = args.task,
        tb_logdir  = args.tb_logdir,
        iteration  = args.iteration,
        n_params   = args.n_params,
        batch_size = args.batch_size,
        num_workers= args.num_workers,
    )