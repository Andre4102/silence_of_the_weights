# The Silence of the Weights

Structural pruning of attention-based audio architectures with second-order
(Fisher information) metrics.

> A. Diecidue, C. A. Barbano, P. Fraternali, M. Fontaine, E. Tartaglione,
> *"The silence of the weights: a structural pruning strategy for
> attention-based audio signal architectures with second order metrics."*

We propose a channel-wise structural pruning strategy targeting the attention
block. Each head and each of the four projection matrices (query, key, value,
output) is pruned independently, and parameters are scored with **Fisher
information** rather than raw magnitude. Pruning 50% of the attention-block
parameters preserves performance on **AST** (audio classification) and
**Whisper** (transcription / translation).

## Method axes

The experiments sweep three independent axes (see the paper, Sec. 2):

- **Pruning scheme** — `EH` entire-head vs `PH` per-head channel pruning.
- **Scoring metric** — `MAG` magnitude vs `FI` Fisher information.
- **Thresholding** — `G` global (cross-layer budget) vs `L` local (per-layer).

## Repository layout

### AST (audio classification)
- `pruning_ast.py` — iterative attention-block pruning loop for AST.
- `ast_utils.py` — AST model + dataset (Audioset, SpeechCommands) helpers.
- `config_ast.py` — AST run configuration.

### Whisper (transcription / translation)
- `pruning_whisper.py` — iterative pruning loop for Whisper.
- `whisper_eval.py`, `whisper_eval_single_task.py` — WER / BLEU evaluation
  across LibriSpeech, CommonVoice, FLEURS, CoVoST, MLS.
- `whisper_utils.py` — dataset loading and Whisper helpers.
- `config_whisper.py` — Whisper run configuration.

### Pruning core
- `structured_pruning_utils_fisher_rope.py` — Fisher-scored channel/head
  budget allocation and masking (used by both AST and Whisper pruning).
- `pruning_utils2.py` — magnitude-based pruning helpers.
- `custom_attention.py` — layer-wise prunable attention modules (AST + Whisper).
- `tf_locoformer.py` — module imported by the pruning utils (dependency).
- `lora.py` — LoRA fine-tuning wrapper used between pruning iterations.
- `utils.py` — common helpers (seeding, logging, ...).

### Analysis / figures
- `compute_sparsity.py` — per-layer attention sparsity (Fig. 3).
- `plot_fisher_importance.py` — Fisher-importance plots.

### Launchers (SLURM skeletons)
Generic SLURM templates; every input is exposed as an environment variable
with a default (see the header of each file).
- `pruning_ast.sh` — AST pruning run.
- `pruning_whisper.sh` — Whisper pruning run.
- `whisper_eval.sh` — single-task Whisper evaluation.

Example:
```bash
PRUNING_STRATEGY=entire_head THRESHOLD_STRATEGY=local \
  IMPORTANCE_STRATEGY=fisher_information DATASET=speechcommands \
  sbatch pruning_ast.sh
```
