#!/usr/bin/env python3
"""
Prepare AST datasets into the JSON-manifest + label-CSV format expected by
`ast_utils.AudiosetDataset` and `config_ast.py`.

Targets (paths taken from config_ast.py dataset_config[...]):
  speechcommands  data_root/
      datafiles/speechcommand_train_data.json
      datafiles/speechcommand_valid_data.json
      datafiles/speechcommand_eval_data.json
      speechcommands_class_labels_indices.csv
  audioset        data_root/
      balanced_train_data.json
      eval_data.json
      class_labels_indices.csv

Manifest format:  {"data": [{"wav": "/abs/file.wav", "labels": "mid1,mid2"}, ...]}
Label CSV format: columns index,mid,display_name  (mid must match manifest labels)

Run on the LOGIN node (needs internet). Sources:
  speechcommands -> Google canonical tarball speech_commands_v0.02.tar.gz
                    (uses official validation_list.txt / testing_list.txt splits)
  audioset       -> HF dataset agkphysics/AudioSet, config "balanced"
                    (labels are AudioSet mids; index order from the official
                     class_labels_indices.csv so it matches the pretrained head)

Usage:
  python prepare_ast_data.py --dataset speechcommands
  python prepare_ast_data.py --dataset audioset
  python prepare_ast_data.py --dataset both
"""
import argparse
import csv
import json
import os
import tarfile
import urllib.request
from pathlib import Path

import config_ast as config

SC_TAR_URL = "http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz"
# Class ordering the base checkpoint was trained with (AST repo recipe)
SC_LABELS_URL = (
    "https://raw.githubusercontent.com/YuanGongND/ast/master/"
    "egs/speechcommands/data/speechcommands_class_labels_indices.csv"
)
# AudioSet 527-class index file (AST repo copy; matches mit/ast-finetuned-audioset)
AUDIOSET_LABELS_URL = (
    "https://raw.githubusercontent.com/YuanGongND/ast/master/"
    "egs/audioset/data/class_labels_indices.csv"
)


# --------------------------------------------------------------------------- #
# Speech Commands v0.02 (canonical tarball)
# --------------------------------------------------------------------------- #
def prepare_speechcommands():
    """Mirror YuanGongND/ast egs/speechcommands/prep_sc.py exactly, so the
    class ordering and label ids match the base checkpoint."""
    dc = config.dataset_config["speechcommands"]
    root = Path(dc["data_root"])
    sc_dir = root / "speech_commands_v0.02"
    datafiles_dir = root / "datafiles"
    sc_dir.mkdir(parents=True, exist_ok=True)
    datafiles_dir.mkdir(parents=True, exist_ok=True)

    # download + extract the canonical 35-class v0.02 tarball
    if not (sc_dir / "yes").is_dir():
        tar_path = root / "speech_commands_v0.02.tar.gz"
        if not tar_path.exists():
            print(f"[sc] downloading {SC_TAR_URL}")
            urllib.request.urlretrieve(SC_TAR_URL, tar_path)
        print(f"[sc] extracting -> {sc_dir}")
        with tarfile.open(tar_path) as t:
            t.extractall(sc_dir)

    # AST-recipe label CSV (index,mid,display_name); display_name -> index
    label_csv = root / dc["label_csv"]
    if not label_csv.exists():
        print(f"[sc] fetching AST label CSV -> {label_csv}")
        urllib.request.urlretrieve(SC_LABELS_URL, label_csv)
    label_map = {}
    with open(label_csv) as f:
        for row in csv.DictReader(f):
            label_map[row["display_name"]] = row["index"]
    print(f"[sc] {len(label_map)} classes (expected {dc['n_class']})")

    # official splits; train = all - validation - testing
    def load_list(name):
        p = sc_dir / name
        return set(p.read_text().splitlines()) if p.exists() else set()
    val_list = load_list("validation_list.txt")
    test_list = load_list("testing_list.txt")

    cmds = [d.name for d in sc_dir.iterdir()
            if d.is_dir() and d.name != "_background_noise_"]
    splits = {"train": [], "valid": [], "eval": []}
    for cmd in cmds:
        if cmd not in label_map:
            print(f"[sc] WARNING: command '{cmd}' not in label CSV, skipping")
            continue
        mid = "/m/spcmd" + str(label_map[cmd]).zfill(2)
        for wav in (sc_dir / cmd).glob("*.wav"):
            rel = f"{cmd}/{wav.name}"
            entry = {"wav": str(wav.resolve()), "labels": mid}
            if rel in test_list:
                splits["eval"].append(entry)
            elif rel in val_list:
                splits["valid"].append(entry)
            else:
                splits["train"].append(entry)

    for split, name in (("train", dc["tr_data"]),
                        ("valid", dc["val_data"]),
                        ("eval", dc["eval_data"])):
        out = root / name
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w") as f:
            json.dump({"data": splits[split]}, f)
        print(f"[sc] {split}: {len(splits[split])} samples -> {out}")


# --------------------------------------------------------------------------- #
# AudioSet (balanced) via HF agkphysics/AudioSet
# --------------------------------------------------------------------------- #
def prepare_audioset():
    import soundfile as sf
    from datasets import load_dataset

    dc = config.dataset_config["audioset"]
    root = Path(dc["data_root"])
    audio_dir = root / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)

    # Official 527-class index file (canonical ordering for the pretrained head)
    label_csv = root / dc["label_csv"]
    if not label_csv.exists():
        print(f"[as] downloading official labels -> {label_csv}")
        urllib.request.urlretrieve(AUDIOSET_LABELS_URL, label_csv)
    valid_mids = set()
    with open(label_csv) as f:
        for row in csv.DictReader(f):
            valid_mids.add(row["mid"])
    print(f"[as] {len(valid_mids)} classes in {label_csv}")

    ds = load_dataset("agkphysics/AudioSet", "balanced")
    split_out = {"train": dc["tr_data"], "test": dc["eval_data"]}
    for split, out_name in split_out.items():
        entries = []
        for ex in ds[split]:
            mids = [m for m in ex["labels"] if m in valid_mids]
            if not mids:
                continue
            arr = ex["audio"]["array"]
            sr = ex["audio"]["sampling_rate"]
            wav_path = audio_dir / f"{ex['video_id']}.wav"
            if not wav_path.exists():
                sf.write(str(wav_path), arr, sr)
            entries.append({"wav": str(wav_path.resolve()), "labels": ",".join(mids)})
        out = root / out_name
        with open(out, "w") as f:
            json.dump({"data": entries}, f)
        print(f"[as] {split}: {len(entries)} samples -> {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["speechcommands", "audioset", "both"],
                    required=True)
    args = ap.parse_args()
    if args.dataset in ("speechcommands", "both"):
        prepare_speechcommands()
    if args.dataset in ("audioset", "both"):
        prepare_audioset()


if __name__ == "__main__":
    main()
