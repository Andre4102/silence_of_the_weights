import os
import soundfile as sf
import torch
from torch.utils.data import Dataset, ConcatDataset, DataLoader
import tarfile
import numpy as np
import librosa
import glob
import pandas as pd
import re
import io
import unicodedata
import random

# -----------------------------------------------------------------------
# Lazy parquet row reader  (used by VoxPopuli and Meanwhile)
# -----------------------------------------------------------------------
def _read_parquet_row(fpath: str, local_row: int, columns: list) -> dict:
    """
    Read a single row from a parquet file using pyarrow, correctly handling
    nested struct columns (e.g. HuggingFace audio dicts).

    Scans row-groups sequentially until the target row is found, so only
    one row-group is decompressed per call. Row-group size is typically
    1 000-5 000 rows, so this is efficient enough for DataLoader workers.
    """
    import pyarrow.parquet as pq
    pf = pq.ParquetFile(fpath)
    offset = 0
    for rg_idx in range(pf.num_row_groups):
        rg_size = pf.metadata.row_group(rg_idx).num_rows
        if local_row < offset + rg_size:
            local_offset = local_row - offset
            table = pf.read_row_group(rg_idx, columns=columns)
            row = {col: table.column(col)[local_offset].as_py() for col in columns}
            return row
        offset += rg_size
    raise IndexError(f"Row {local_row} out of range for {fpath}")

from transformers.models.whisper import WhisperModel, WhisperProcessor, WhisperForConditionalGeneration, WhisperConfig
from custom_attention import LayerWiseWhisperConfig, CustomWhisperEncoder, CustomWhisperDecoder


# -----------------------------------------------------------------------
# Language code mapping: dataset-specific codes -> Whisper language names
# -----------------------------------------------------------------------
LANG_CODE_TO_WHISPER = {
    # ISO 639-1 standard codes
    "ar": "arabic",
    "ca": "catalan",
    "cs": "czech",
    "cy": "welsh",
    "de": "german",
    "en": "english",
    "es": "spanish",
    "et": "estonian",
    "fa": "persian",
    "fi": "finnish",
    "fr": "french",
    "hr": "croatian",
    "hu": "hungarian",
    "id": "indonesian",
    "it": "italian",
    "ja": "japanese",
    "lt": "lithuanian",
    "lv": "latvian",
    "mn": "mongolian",
    "nl": "dutch",
    "pl": "polish",
    "pt": "portuguese",
    "ro": "romanian",
    "ru": "russian",
    "sk": "slovak",
    "sl": "slovenian",
    "sv": "swedish",
    "ta": "tamil",
    "th": "thai",
    "tr": "turkish",
    "uk": "ukrainian",
    "zh": "chinese",
    # Non-standard / compound codes used in datasets
    "sv-SE":       "swedish",
    "sv-se":       "swedish",
    "zh-CN":       "chinese",
    "zh-cn":       "chinese",
    "zh-TW":       "chinese",
    "zh-tw":       "chinese",
    "en_accented": "english",
    # MLS full language names
    "dutch":       "dutch",
    "french":      "french",
    "german":      "german",
    "italian":     "italian",
    "polish":      "polish",
    "portuguese":  "portuguese",
    "spanish":     "spanish",
}



def to_whisper_lang(lang: str) -> str:
    """Convert any dataset language code to a Whisper language name."""
    if lang in LANG_CODE_TO_WHISPER:
        return LANG_CODE_TO_WHISPER[lang]
    if lang.lower() in LANG_CODE_TO_WHISPER:
        return LANG_CODE_TO_WHISPER[lang.lower()]
    # Assume it's already a full name (e.g. "english")
    return lang.lower()


# -----------------------------------------------------------------------
# Unified audio loader  (handles mp3 / ogg / opus via librosa)
# -----------------------------------------------------------------------
def load_audio(path: str, target_sr: int = 16000) -> np.ndarray:
    """
    Load any audio file as float32 mono at target_sr.
    soundfile cannot decode MP3/OGG/OPUS, so those are routed through librosa.
    """
    ext = os.path.splitext(path)[1].lower()
    if ext in (".mp3", ".ogg", ".opus"):
        audio, sr = librosa.load(path, sr=None, mono=True)
    else:
        audio, sr = sf.read(path)
        if audio.ndim == 2:
            audio = audio.mean(axis=1)

    audio = audio.astype(np.float32)
    if sr != target_sr:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
    return audio


class CustomWhisper(WhisperModel):
    def __init__(self, config: LayerWiseWhisperConfig):
        super().__init__(config)
        self.encoder = CustomWhisperEncoder(config)
        self.decoder = CustomWhisperDecoder(config)

class CustomWhisperForConditionalGeneration(WhisperForConditionalGeneration):
    def __init__(self, config):
        super().__init__(config)
        self.model = CustomWhisper(config)


def load_model(path):
    cfg = LayerWiseWhisperConfig.from_pretrained(path)
    processor = WhisperProcessor.from_pretrained(path)
    model = CustomWhisperForConditionalGeneration.from_pretrained(path, config=cfg)
    return model, processor

def load_original_model(path):
    cfg_orig   = WhisperConfig.from_pretrained(path)
    cfg_custom = LayerWiseWhisperConfig.from_pretrained(path)

    print("d_model match:", cfg_orig.d_model == cfg_custom.d_model)
    print("encoder_layers match:", cfg_orig.encoder_layers == cfg_custom.encoder_layers)
    print("encoder_attention_heads match:", cfg_orig.encoder_attention_heads == cfg_custom.encoder_attention_heads)
    print("decoder_layers match:", cfg_orig.decoder_layers == cfg_custom.decoder_layers)

    # Check what the per-layer config looks like
    print("\nencoder_self_qkv_config[0]:", cfg_custom.encoder_self_qkv_config[0])
    print("Expected num_heads:", cfg_orig.encoder_attention_heads)
    print("Expected head_dim:", cfg_orig.d_model // cfg_orig.encoder_attention_heads)
    processor = WhisperProcessor.from_pretrained(path)
    model = WhisperForConditionalGeneration.from_pretrained(path, config=cfg_orig)
    return model, processor
# -----------------------------------------------------------------------
# LibriSpeech
# -----------------------------------------------------------------------
class LibriSpeechDataset(Dataset):
    def __init__(self, root, split="test-clean", max_samples=None):
        """
        root:  path to LibriSpeech/
        split: test-clean | test-other | dev-clean | dev-other
        """
        self.samples = []
        split_dir = os.path.join(root, split)
        assert os.path.isdir(split_dir), f"Missing split dir: {split_dir}"
        self._load_metadata(split_dir)
        if max_samples is not None:
            self.samples = self.samples[:max_samples]
        print(f"Loaded LibriSpeech {split}: {len(self.samples)} samples")

    def _load_metadata(self, split_dir):
        for speaker in sorted(os.listdir(split_dir)):
            speaker_dir = os.path.join(split_dir, speaker)
            if not os.path.isdir(speaker_dir):
                continue
            for chapter in sorted(os.listdir(speaker_dir)):
                chapter_dir = os.path.join(speaker_dir, chapter)
                if not os.path.isdir(chapter_dir):
                    continue
                trans_path = os.path.join(chapter_dir, f"{speaker}-{chapter}.trans.txt")
                if not os.path.exists(trans_path):
                    continue
                with open(trans_path, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        parts = line.split(" ", 1)
                        if len(parts) < 2:
                            continue
                        utt_id, text = parts
                        audio_path = os.path.join(chapter_dir, f"{utt_id}.flac")
                        self.samples.append((audio_path, text))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        audio_path, text = self.samples[idx]
        return {"audio": load_audio(audio_path), "text": text}


# -----------------------------------------------------------------------
# FLEURS
# -----------------------------------------------------------------------
class FleursDataset(Dataset):
    def __init__(self, root, lang="fr_fr", split="test", task="transcribe"):
        """
        root:  path to FLEURS/
        lang:  e.g. fr_fr, de_de, es_es
        split: train | validation | test
        """
        assert task in ["transcribe", "translate"]
        self.task = task

        lang_dir  = os.path.join(root, lang)
        assert os.path.isdir(lang_dir), f"Missing language dir: {lang_dir}"

        tsv_path  = os.path.join(lang_dir, f"{split}.tsv")
        audio_dir = os.path.join(lang_dir, "audio", split)

        assert os.path.exists(tsv_path),   f"Missing {tsv_path}"
        assert os.path.isdir(audio_dir),   f"Missing audio dir"
        self._ensure_fleurs_audio_extracted(audio_dir)
        self._load_metadata(tsv_path, audio_dir)

    def _ensure_fleurs_audio_extracted(self, audio_dir: str):
        lock_file = os.path.join(audio_dir, ".extracted")
        if os.path.exists(lock_file):
            return
        tar_files = [f for f in os.listdir(audio_dir) if f.endswith(".tar.gz")]
        if not tar_files:
            open(lock_file, "w").close()
            return
        for tar_name in tar_files:
            with tarfile.open(os.path.join(audio_dir, tar_name), "r:gz") as tar:
                tar.extractall(path=audio_dir)
        open(lock_file, "w").close()

    def _load_metadata(self, tsv_path, audio_dir):
        self.metadata = []
        skipped = 0
        with open(tsv_path, "r", encoding="utf-8") as f:
            for line in f:
                cols = line.strip().split("\t")
                if len(cols) < 4:
                    continue
                audio_path = os.path.join(audio_dir, cols[1])
                if not os.path.exists(audio_path):
                    skipped += 1
                    continue
                self.metadata.append({
                    "audio_path": audio_path,
                    "text":       cols[2],
                    "norm_text":  cols[3],
                })
        if skipped:
            print(f"  FLEURS: skipped {skipped} missing audio files")

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        row = self.metadata[idx]
        return {
            "audio":     load_audio(row["audio_path"]),
            "text":      row["text"],
            "norm_text": row["norm_text"],
        }


# -----------------------------------------------------------------------
# Meanwhile
# -----------------------------------------------------------------------
class MeanwhileDataset(Dataset):
    """
    Memory-safe Meanwhile loader.

    Only the 'text' column is read at init; audio bytes are loaded lazily
    per sample by re-reading just the required row from the parquet shard.
    """
    def __init__(self, root, split="test", sampling_rate=16000, max_samples=None):
        parquet_dir   = os.path.join(root, "data")
        parquet_files = sorted(glob.glob(os.path.join(parquet_dir, f"{split}-*.parquet")))
        if not parquet_files:
            raise FileNotFoundError(f"No parquet files found for split={split}")

        self.sampling_rate = sampling_rate
        self._index = []   # (parquet_path, local_row_index)
        self._texts = []

        for fpath in parquet_files:
            df = pd.read_parquet(fpath, columns=["text"])
            if "text" not in df.columns:
                raise ValueError(f"Missing 'text' column in {fpath}")
            for local_row, text in enumerate(df["text"]):
                self._index.append((fpath, local_row))
                self._texts.append(text)
                if max_samples is not None and len(self._texts) >= max_samples:
                    break
            if max_samples is not None and len(self._texts) >= max_samples:
                break

        print(f"Loaded MEANWHILE {split}: {len(self._texts)} samples (audio loaded lazily)")

    def __len__(self):
        return len(self._texts)

    def _decode_audio(self, audio):
        if isinstance(audio, dict):
            if "array" in audio and "sampling_rate" in audio:
                wav = np.array(audio["array"], dtype=np.float32)
                sr  = audio["sampling_rate"]
            elif "bytes" in audio:
                wav, sr = sf.read(io.BytesIO(audio["bytes"]))
                wav = wav.astype(np.float32)
            else:
                raise KeyError(f"Unexpected audio dict keys: {list(audio.keys())}")
        elif isinstance(audio, str):
            wav, sr = sf.read(audio)
            wav = wav.astype(np.float32)
        else:
            raise TypeError(f"Unsupported audio type: {type(audio)}")
        if wav.ndim > 1:
            wav = wav.mean(axis=1)
        if sr != self.sampling_rate:
            wav = librosa.resample(wav, orig_sr=sr, target_sr=self.sampling_rate)
        return wav.astype(np.float32)

    def __getitem__(self, idx):
        fpath, local_row = self._index[idx]
        row = _read_parquet_row(fpath, local_row, columns=["audio"])
        return {"audio": self._decode_audio(row["audio"]), "text": self._texts[idx]}


# -----------------------------------------------------------------------
# CoVOST
# -----------------------------------------------------------------------
class CoVoSTDataset(Dataset):
    """
    CoVOST speech translation dataset.

    TSV columns: audio  src_text  tgt_text
    The audio column holds full absolute paths to Common Voice MP3 files.
    """
    def __init__(self, root, lang_pair="ar_en", split="test",
                 task="translate", max_samples=None):
        assert task in ["translate", "transcribe"]
        self.task        = task
        self.source_lang = lang_pair.split("_")[0]

        tsv_path = os.path.join(root, lang_pair, f"{split}.tsv")
        assert os.path.exists(tsv_path), f"Missing CoVOST file: {tsv_path}"
        self.metadata = self._load_metadata(tsv_path, max_samples)
        print(f"Loaded CoVOST {lang_pair} {split}: {len(self.metadata)} samples")

    def _load_metadata(self, tsv_path, max_samples):
        metadata = []
        with open(tsv_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
        header    = lines[0].strip().split("\t")
        audio_idx = header.index("audio")
        src_idx   = header.index("src_text") if "src_text" in header else None
        tgt_idx   = header.index("tgt_text") if "tgt_text" in header else None

        skipped = 0
        for line in lines[1:]:
            if max_samples is not None and len(metadata) >= max_samples:
                break
            cols = line.strip().split("\t")
            if len(cols) <= audio_idx:
                continue
            audio_path = cols[audio_idx]
            src_text   = cols[src_idx] if src_idx is not None and len(cols) > src_idx else ""
            tgt_text   = cols[tgt_idx] if tgt_idx is not None and len(cols) > tgt_idx else ""
            if not audio_path or not os.path.exists(audio_path):
                skipped += 1
                continue
            metadata.append({"audio_path": audio_path, "src_text": src_text, "tgt_text": tgt_text})
        if skipped:
            print(f"  CoVoST: skipped {skipped} missing audio files")
        return metadata

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        row = self.metadata[idx]
        text = row["tgt_text"] if self.task == "translate" else row["src_text"]
        return {
            "audio":    load_audio(row["audio_path"]),
            "text":     text,
            "src_text": row["src_text"],
            "tgt_text": row["tgt_text"],
        }


# -----------------------------------------------------------------------
# Common Voice
# -----------------------------------------------------------------------
class CommonVoiceDataset(Dataset):
    """
    Mozilla Common Voice.

    Structure:
        root/<lang>/clips/*.mp3
        root/<lang>/test.tsv   (columns: client_id, path, sentence, ...)
    """
    def __init__(self, root, lang="en", split="test", max_samples=None):
        lang_dir       = os.path.join(root, lang)
        self.clips_dir = os.path.join(lang_dir, "clips")
        assert os.path.isdir(lang_dir),       f"Missing language dir: {lang_dir}"
        assert os.path.isdir(self.clips_dir), f"Missing clips dir: {self.clips_dir}"

        tsv_path = os.path.join(lang_dir, f"{split}.tsv")
        assert os.path.exists(tsv_path), f"Missing TSV: {tsv_path}"
        self.metadata = self._load_metadata(tsv_path, max_samples)
        print(f"Loaded Common Voice {lang} {split}: {len(self.metadata)} samples")

    def _load_metadata(self, tsv_path, max_samples):
        metadata = []
        skipped = 0
        with open(tsv_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
        header       = lines[0].strip().split("\t")
        path_idx     = header.index("path")
        sentence_idx = header.index("sentence")
        for line in lines[1:]:
            if max_samples is not None and len(metadata) >= max_samples:
                break
            cols = line.strip().split("\t")
            if len(cols) <= max(path_idx, sentence_idx):
                continue
            audio_path = os.path.join(self.clips_dir, cols[path_idx])
            if not os.path.exists(audio_path):
                skipped += 1
                continue
            metadata.append({"audio_path": audio_path, "text": cols[sentence_idx]})
        if skipped:
            print(f"  CommonVoice: skipped {skipped} missing clips")
        return metadata

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        row = self.metadata[idx]
        return {"audio": load_audio(row["audio_path"]), "text": row["text"]}


# -----------------------------------------------------------------------
# VoxPopuli
# -----------------------------------------------------------------------
class VoxPopuliDataset(Dataset):
    """
    VoxPopuli: HuggingFace parquet format.

    Memory-safe: only text columns are read at init. Audio is loaded lazily
    per sample by seeking into the parquet shard at __getitem__ time, so
    raw audio bytes never accumulate in RAM across the dataset.

    Structure:
        root/<lang>/train-*.parquet
        root/<lang>/test-*.parquet
        root/<lang>/validation-*.parquet
    """
    def __init__(self, root, lang="en", split="test", max_samples=None):
        lang_dir = os.path.join(root, lang)
        assert os.path.isdir(lang_dir), f"Missing language dir: {lang_dir}"

        parquet_files = sorted(glob.glob(os.path.join(lang_dir, f"{split}-*.parquet")))
        if not parquet_files:
            raise FileNotFoundError(
                f"No parquet files found: {lang_dir}/{split}-*.parquet"
            )

        print(f"Loading VoxPopuli {lang} {split} metadata from {len(parquet_files)} file(s)...")

        # Discover text column by peeking at column names only (no data read)
        import pyarrow.parquet as pq
        schema_cols = pq.read_schema(parquet_files[0]).names
        self.text_col = next(
            (c for c in ("normalized_text", "raw_text", "text") if c in schema_cols),
            None
        )
        if self.text_col is None:
            raise ValueError(f"No text column found. Available: {schema_cols}")

        self._index = []   # (parquet_path, local_row_index)
        self._texts = []

        for fpath in parquet_files:
            df = pd.read_parquet(fpath, columns=[self.text_col])
            for local_row, text in enumerate(df[self.text_col]):
                self._index.append((fpath, local_row))
                self._texts.append(text)
                if max_samples is not None and len(self._texts) >= max_samples:
                    break
            if max_samples is not None and len(self._texts) >= max_samples:
                break

        print(f"Loaded VoxPopuli {lang} {split}: {len(self._texts)} samples "
              f"(text col: {self.text_col}, audio loaded lazily)")

    def __len__(self):
        return len(self._texts)

    def _decode_audio(self, audio, target_sr=16000):
        if isinstance(audio, dict):
            if "array" in audio and "sampling_rate" in audio:
                wav = np.array(audio["array"], dtype=np.float32)
                sr  = audio["sampling_rate"]
            elif "bytes" in audio:
                wav, sr = sf.read(io.BytesIO(audio["bytes"]))
                wav = wav.astype(np.float32)
            else:
                raise KeyError(f"Unexpected audio dict keys: {list(audio.keys())}")
        else:
            raise TypeError(f"Unsupported audio type: {type(audio)}")
        if wav.ndim > 1:
            wav = wav.mean(axis=1)
        if sr != target_sr:
            wav = librosa.resample(wav, orig_sr=sr, target_sr=target_sr)
        return wav.astype(np.float32)

    def __getitem__(self, idx):
        fpath, local_row = self._index[idx]
        row = _read_parquet_row(fpath, local_row, columns=["audio"])
        return {"audio": self._decode_audio(row["audio"]), "text": self._texts[idx]}


# -----------------------------------------------------------------------
# MLS (Multilingual LibriSpeech)
# -----------------------------------------------------------------------
class MLSDataset(Dataset):
    """
    Multilingual LibriSpeech.

    Structure:
        root/mls_<lang>/<split>/
            audio/<utt_id>.flac   (flat — e.g. 1406_1028_000000.flac)
            transcripts.txt       (<utt_id>\t<text>)
            segments.txt
    """
    def __init__(self, root, lang="dutch", split="test", max_samples=None):
        lang_dir = os.path.join(root, f"mls_{lang}", split)
        assert os.path.isdir(lang_dir), f"Missing split dir: {lang_dir}"

        self.audio_dir   = os.path.join(lang_dir, "audio")
        trans_path       = os.path.join(lang_dir, "transcripts.txt")
        segments_path    = os.path.join(lang_dir, "segments.txt")

        assert os.path.isdir(self.audio_dir),  f"Missing audio dir: {self.audio_dir}"
        assert os.path.exists(trans_path),     f"Missing transcripts: {trans_path}"
        assert os.path.exists(segments_path),  f"Missing segments: {segments_path}"

        self.metadata = self._load_metadata(trans_path, max_samples)
        print(f"Loaded MLS {lang} {split}: {len(self.metadata)} samples")

    def _load_metadata(self, trans_path, max_samples):
        metadata = []
        skipped = 0
        with open(trans_path, "r", encoding="utf-8") as f:
            for line in f:
                if max_samples is not None and len(metadata) >= max_samples:
                    break
                parts = line.strip().split("\t")
                if len(parts) < 2:
                    continue
                utt_id, text = parts[0], parts[1]
                audio_path = os.path.join(self.audio_dir, f"{utt_id}.flac")
                if not os.path.exists(audio_path):
                    audio_path = os.path.join(self.audio_dir, f"{utt_id}.opus")
                if not os.path.exists(audio_path):
                    skipped += 1
                    continue
                metadata.append({"audio_path": audio_path, "text": text, "utt_id": utt_id})
        if skipped:
            print(f"  MLS: skipped {skipped} missing audio files")
        return metadata

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        row = self.metadata[idx]
        return {"audio": load_audio(row["audio_path"]), "text": row["text"]}


# -----------------------------------------------------------------------
# Collate function
# -----------------------------------------------------------------------
def whisper_collate_fn(batch, processor):
    """
    Collate a batch of samples into model-ready tensors.

    Each sample dict must contain:
        audio  : np.ndarray  (float32, 16 kHz mono)
        text   : str         (target transcript or translation)
        task   : str         "transcribe" | "translate"  (default: "transcribe")
        lang   : str         Whisper language name, e.g. "english" (default: "english")
    """
    from collections import defaultdict

    texts      = [item["text"]                        for item in batch]
    tasks      = [item.get("task", "transcribe")      for item in batch]
    langs      = [item.get("lang", "english")         for item in batch]
    norm_texts = [item.get("norm_text", item["text"]) for item in batch]

    # Validate audio arrays before passing to feature extractor.
    # Samples that are too short (corrupt parquet reads) are replaced
    # with silence so a single bad sample never crashes the whole run.
    MIN_SAMPLES = 400  # 0.025 s at 16 kHz
    audios = []
    for i, item in enumerate(batch):
        raw = item["audio"]

        # Handle case where __getitem__ returned a raw HuggingFace audio dict
        # instead of a decoded numpy array (e.g. from parquet struct columns)
        if isinstance(raw, dict):
            if "array" in raw and "sampling_rate" in raw:
                wav = np.array(raw["array"], dtype=np.float32)
                sr  = raw["sampling_rate"]
                if wav.ndim > 1:
                    wav = wav.mean(axis=1)
                if sr != 16000:
                    import librosa
                    wav = librosa.resample(wav, orig_sr=sr, target_sr=16000)
                raw = wav.astype(np.float32)
            elif "bytes" in raw:
                import soundfile as sf, io
                wav, sr = sf.read(io.BytesIO(raw["bytes"]))
                wav = wav.astype(np.float32)
                if wav.ndim > 1:
                    wav = wav.mean(axis=1)
                if sr != 16000:
                    import librosa
                    wav = librosa.resample(wav, orig_sr=sr, target_sr=16000)
                raw = wav.astype(np.float32)
            else:
                print(f"[collate] WARNING: sample {i} has unknown audio dict keys: "
                      f"{list(raw.keys())} — replacing with silence")
                raw = np.zeros(16000, dtype=np.float32)

        if not isinstance(raw, np.ndarray):
            print(f"[collate] WARNING: sample {i} audio is type {type(raw)} — replacing with silence")
            raw = np.zeros(16000, dtype=np.float32)
        if raw.ndim > 1:
            raw = raw.mean(axis=1)
        raw = raw.astype(np.float32)
        if len(raw) < MIN_SAMPLES:
            print(f"[collate] WARNING: sample {i} audio has only {len(raw)} samples "
                  f"(text='{texts[i][:60]}') — replacing with silence")
            raw = np.zeros(16000, dtype=np.float32)

        # Pad / truncate to exactly 30 s (480 000 samples) in the time domain.
        # This makes the feature extractor's own padding logic irrelevant and
        # guarantees exactly 3000 mel frames regardless of transformers version.
        TARGET_SAMPLES = 480_000  # 30 s × 16 000 Hz
        if len(raw) < TARGET_SAMPLES:
            raw = np.pad(raw, (0, TARGET_SAMPLES - len(raw)), mode="constant")
        else:
            raw = raw[:TARGET_SAMPLES]
        audios.append(raw)

    audio_inputs = processor.feature_extractor(
        audios,
        sampling_rate=16000,
        return_tensors="pt",
        return_attention_mask=True,
    )
    assert audio_inputs.input_features.shape[-1] == 3000, \
        f"Feature extractor produced {audio_inputs.input_features.shape[-1]} frames, expected 3000"

    all_labels = [None] * len(batch)
    groups = defaultdict(list)
    for i, (task, lang) in enumerate(zip(tasks, langs)):
        groups[(task, lang)].append(i)

    for (task, lang), indices in groups.items():
        processor.tokenizer.set_prefix_tokens(language=lang, task=task)
        group_texts = [texts[i] for i in indices]
        encoded = processor.tokenizer(
            group_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=448,
        )
        label_ids = encoded.input_ids
        label_ids[label_ids == processor.tokenizer.pad_token_id] = -100
        for j, idx in enumerate(indices):
            all_labels[idx] = label_ids[j]

    max_label_len = max(t.size(0) for t in all_labels)
    padded_labels = torch.full(
        (len(batch), max_label_len), fill_value=-100, dtype=torch.long
    )
    for i, label in enumerate(all_labels):
        padded_labels[i, :label.size(0)] = label

    return {
        "input_features": audio_inputs.input_features,
        "attention_mask": audio_inputs.attention_mask,
        "labels":         padded_labels,
        "texts":          texts,
        "norm_texts":     norm_texts,
        "tasks":          tasks,
        "langs":          langs,
    }


def merge_datasets(paths, processor, batch_size=16, shuffle=True, num_workers=4):
    """
    Build a DataLoader from the curated distillation dataset mix.

    Target mix (~70% transcription / ~30% translation):
      LibriSpeech  EN               1 x 1500  =  1 500
      MLS          EN/ES/FR/DE/IT/PT 6 x 1500  =  9 000
      CommonVoice  EN/ES/FR/DE      4 x 1000  =  4 000
      VoxPopuli    EN/DE/FR/ES/IT   5 x 1000  =  5 000
      FLEURS       EN/ES/FR/DE/IT/ZH 6 x  500  =  3 000
      CoVoST EN->X (translate)      5 x 1500  =  7 500
      CoVoST X->EN (translate)      3 x 1000  =  3 000
                                            Total ~ 33 000

    max_samples is passed directly into each Dataset constructor so files
    are never fully read into RAM — only the required rows are loaded.
    """
    datasets = []

    class TaggedDataset(Dataset):
        def __init__(self, ds, task, lang):
            self.ds   = ds
            self.task = task
            self.lang = lang
        def __len__(self):
            return len(self.ds)
        def __getitem__(self, idx):
            item = self.ds[idx]
            item["task"] = self.task
            item["lang"] = self.lang
            return item

    def add(ds, task, lang):
        datasets.append(TaggedDataset(ds, task, lang))

    # LibriSpeech
    if "librispeech" in paths and os.path.exists(paths["librispeech"]):
        ds = LibriSpeechDataset(paths["librispeech"], split="train-clean-100", max_samples=1500)
        add(ds, "transcribe", "english")
    else:
        print("Skipping LibriSpeech: path not provided or does not exist")

    # MLS
    if "mls" in paths and os.path.exists(paths["mls"]):
        mls_langs = {
            "dutch":      "dutch",
            "french":     "french",
            "german":     "german",
            "italian":    "italian",
            "spanish":    "spanish",
            "polish":     "polish",
            "portuguese": "portuguese",
        }
        for mls_name, whisper_name in mls_langs.items():
            lang_path = os.path.join(paths["mls"], f"mls_{mls_name}")
            if os.path.exists(lang_path):
                ds = MLSDataset(paths["mls"], lang=mls_name, split="train", max_samples=1500)
                add(ds, "transcribe", whisper_name)
            else:
                print(f"Skipping MLS {mls_name}: directory does not exist")
    else:
        print("Skipping MLS: path not provided or does not exist")

    # Common Voice
    if "commonvoice" in paths and os.path.exists(paths["commonvoice"]):
        cv_langs = {
            "en": "english",
            "fr": "french",
            "de": "german",
            "es": "spanish",
            "zh": "chinese",
            "it": "italian",
        }
        for code, whisper_name in cv_langs.items():
            lang_path = os.path.join(paths["commonvoice"], code)
            if os.path.exists(lang_path):
                ds = CommonVoiceDataset(paths["commonvoice"], lang=code, split="train", max_samples=1000)
                add(ds, "transcribe", whisper_name)
            else:
                print(f"Skipping Common Voice {code}: directory does not exist")
    else:
        print("Skipping Common Voice: path not provided or does not exist")

    # VoxPopuli
    if "voxpopuli" in paths and os.path.exists(paths["voxpopuli"]):
        vox_langs = {
            "en": "english",
            "de": "german",
            "fr": "french",
            "es": "spanish",
            "it": "italian",
        }
        for code, whisper_name in vox_langs.items():
            lang_path = os.path.join(paths["voxpopuli"], code)
            if os.path.exists(lang_path):
                ds = VoxPopuliDataset(paths["voxpopuli"], lang=code, split="train", max_samples=1000)
                add(ds, "transcribe", whisper_name)
            else:
                print(f"Skipping VoxPopuli {code}: directory does not exist")
    else:
        print("Skipping VoxPopuli: path not provided or does not exist")

    # FLEURS
    if "fleurs" in paths and os.path.exists(paths["fleurs"]):
        fleurs_langs = {
            "fr_fr":       "french",
            "de_de":       "german",
            "es_419":      "spanish",
            "cmn_hans_cn": "chinese",
            "it_it":       "italian",
        }
        for fleurs_code, whisper_name in fleurs_langs.items():
            lang_path = os.path.join(paths["fleurs"], fleurs_code)
            if os.path.exists(lang_path):
                ds = FleursDataset(paths["fleurs"], lang=fleurs_code, split="train", task="transcribe")
                add(subsample_dataset(ds, 500), "transcribe", whisper_name)
            else:
                print(f"Skipping FLEURS {fleurs_code}: directory does not exist")
    else:
        print("Skipping FLEURS: path not provided or does not exist")

    # CoVoST-2 translation
    if "covost" in paths and os.path.exists(paths["covost"]):
        covost_pairs = {
            "de_en":     ("german",   1500),
            "fr_en":     ("french",   1500),
            "es_en":     ("spanish",  1500),
            "en_de":     ("english",  1500),
            "it_en":     ("italian",  1000),
            "zh-CN_en":  ("chinese",  1000),
            "en_zh-CN":  ("english",  1000),
        }
        for pair, (src_whisper, n) in covost_pairs.items():
            pair_path = os.path.join(paths["covost"], pair)
            if os.path.exists(pair_path):
                ds = CoVoSTDataset(paths["covost"], lang_pair=pair, split="train", task="translate", max_samples=n)
                add(ds, "translate", src_whisper)
            else:
                print(f"Skipping CoVoST {pair}: directory does not exist")
    else:
        print("Skipping CoVoST: path not provided or does not exist")

    if not datasets:
        raise ValueError("No datasets found! Check paths dictionary and filesystem.")

    merged_dataset = ConcatDataset(datasets)

    n_transcribe = sum(len(d) for d in datasets if d.task == "transcribe")
    n_translate  = sum(len(d) for d in datasets if d.task == "translate")
    total = len(merged_dataset)
    print(
        f"Total samples: {total}  "
        f"(transcribe: {n_transcribe} = {100*n_transcribe/total:.0f}%,  "
        f"translate: {n_translate} = {100*n_translate/total:.0f}%)"
    )

    dataloader = DataLoader(
        merged_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=lambda batch: whisper_collate_fn(batch, processor),
        pin_memory=True,
    )

    print("DataLoader ready for training")
    return dataloader

# -----------------------------------------------------------------------
# Long-audio helpers
# -----------------------------------------------------------------------
def chunk_audio(audio, sr=16000, chunk_length_s=30, overlap_s=4):
    chunk_len = int(chunk_length_s * sr)
    step      = chunk_len - int(overlap_s * sr)
    chunks, start = [], 0
    while start < len(audio):
        chunk = audio[start:start + chunk_len]
        if len(chunk) < chunk_len:
            chunk = np.pad(chunk, (0, chunk_len - len(chunk)), mode="constant")
        chunks.append(chunk)
        start += step
    return chunks


def stitch_with_overlap(prev, curr, max_overlap_words=20):
    pw, cw = prev.split(), curr.split()
    for k in range(max_overlap_words, 0, -1):
        if pw[-k:] == cw[:k]:
            return prev + " " + " ".join(cw[k:])
    return prev + " " + curr


def transcribe_chunk(chunk, processor, model, device):
    inputs = processor(
        [chunk], sampling_rate=16000, return_tensors="pt",
        padding=True, return_attention_mask=True,
    )
    with torch.no_grad():
        predicted_ids = model.generate(
            inputs.input_features.to(device),
            attention_mask=inputs.attention_mask.to(device),
            max_new_tokens=444,
            temperature=0.0,
            no_speech_threshold=0.0,
            logprob_threshold=-float("inf"),
            return_timestamps=True,
        )
    return processor.batch_decode(predicted_ids, decode_with_timestamps=True)[0]


TIMESTAMP_RE = re.compile(r"<\|\d+\.\d+\|>")
TS_RE        = re.compile(r"<\|(\d+\.\d+)\|>")


def remove_timestamps(text: str) -> str:
    return re.sub(r"\s+", " ", TIMESTAMP_RE.sub("", text)).strip()


def parse_timestamped_text(text):
    parts, segments, current_time = TS_RE.split(text), [], None
    for part in parts:
        if re.fullmatch(r"\d+\.\d+", part):
            current_time = float(part)
        elif current_time is not None and part.strip():
            segments.append((current_time, part.strip()))
    return segments


def merge_chunk_segments(all_segs, chunk_segs, chunk_start_s, last_t, time_eps=0.05):
    for t_local, txt in chunk_segs:
        t_abs = chunk_start_s + t_local
        if t_abs > last_t + time_eps:
            all_segs.append((t_abs, txt))
            last_t = t_abs
    return all_segs, last_t


def transcribe_long_audio(audio, processor, model, device,
                          sr=16000, chunk_length_s=30, overlap_s=4):
    chunks   = chunk_audio(audio, sr=sr, chunk_length_s=chunk_length_s, overlap_s=overlap_s)
    step_s   = chunk_length_s - overlap_s
    all_segs = []
    last_t   = -1.0
    for i, chunk in enumerate(chunks):
        segs     = parse_timestamped_text(transcribe_chunk(chunk, processor, model, device))
        all_segs, last_t = merge_chunk_segments(all_segs, segs, i * step_s, last_t)
    return " ".join(txt for _, txt in all_segs)


# -----------------------------------------------------------------------
# Normalisation helpers
# -----------------------------------------------------------------------
def _normalize(text):
    text = text.lower()
    text = unicodedata.normalize("NFKD", text)
    text = text.encode("ascii", "ignore").decode("ascii")
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

normalize_whisper     = _normalize
normalize_librispeech = _normalize

def subsample_dataset(ds, max_samples):
    if len(ds) <= max_samples:
        return ds
    return torch.utils.data.Subset(ds, range(max_samples))