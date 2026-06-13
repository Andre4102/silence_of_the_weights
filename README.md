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

## Results (demo)

Tables reproduced with this code. Sparsity is the **fraction of attention-block
parameters removed**. Cells are interpolated between the 10 pruning iterations;
`—` means that sparsity was not reached. Full per-language / per-iteration
tables: [`RESULTS_speechcommands.md`](RESULTS_speechcommands.md) and
[`RESULTS_whisper.md`](RESULTS_whisper.md).

**Takeaways:** Fisher beats magnitude at every comparable point; entire-head is
the most robust scheme and the only one that also reduces latency (it removes
whole attention maps rather than thinning each head).

### AST — SpeechCommands, accuracy (%) vs attention sparsity (dense 98.15%)

| Scheme | Threshold | Importance | 10% | 20% | 30% | 40% | 50% | 60% | 70% | 80% | 90% |
|---|---|---|---|---|---|---|---|---|---|---|---|
| per-head | global | Fisher | 98.15 | 97.88 | 97.76 | 97.96 | 97.92 | 97.71 | — | — | — |
| per-head | global | magnitude | 97.86 | 97.87 | 97.75 | 97.44 | 97.07 | 96.54 | — | — | — |
| per-head | local | Fisher | 97.90 | 98.05 | 97.92 | 97.58 | 97.59 | 97.55 | 97.24 | — | — |
| per-head | local | magnitude | 97.65 | 97.33 | 97.91 | 97.77 | 97.66 | 97.49 | 97.01 | — | — |
| entire-head | global | Fisher | 98.02 | 97.92 | 98.12 | 98.12 | 97.78 | 97.58 | — | — | — |
| entire-head | global | magnitude | 97.86 | 97.74 | 97.91 | 97.74 | 97.63 | — | — | — | — |
| entire-head | local | Fisher | 98.14 | 98.12 | 97.69 | 97.77 | 97.86 | 97.52 | 96.95 | 95.82 | 90.31 |
| entire-head | local | magnitude | 98.13 | 98.12 | 97.91 | 97.87 | 97.53 | 97.12 | 96.46 | 94.25 | 83.80 |

### AST — SpeechCommands, latency (ms/forward, batch=1, dense 8.99 ms)

FLOPs are scheme-invariant at equal sparsity, but latency is not: entire-head
drops below dense while per-head can sit above it (it keeps all 12 attention
maps and softmaxes).

| Scheme | Threshold | Importance | 10% | 30% | 50% | 70% | 90% |
|---|---|---|---|---|---|---|---|
| per-head | global | Fisher | 9.45 | 9.32 | 9.28 | — | — |
| per-head | global | magnitude | 9.94 | 9.24 | 9.27 | — | — |
| entire-head | global | Fisher | 9.17 | 9.23 | 9.19 | — | — |
| entire-head | local | Fisher | 8.97 | 9.02 | 9.03 | 9.02 | 8.82 |
| entire-head | local | magnitude | 9.01 | 9.08 | 9.08 | 9.07 | 8.94 |

### AST — AudioSet, mAP vs attention sparsity (dense 0.3226)

Multi-label 527-class; mAP higher is better. Full FLOPs/latency/trajectories in
[`RESULTS_audioset.md`](RESULTS_audioset.md).

| Scheme | Threshold | Importance | 10% | 20% | 30% | 40% | 50% | 60% | 70% | 80% | 90% |
|---|---|---|---|---|---|---|---|---|---|---|---|
| per-head | global | Fisher | 0.3245 | 0.3235 | 0.3206 | 0.3172 | 0.3123 | 0.3086 | — | — | — |
| per-head | global | magnitude | 0.3224 | 0.3197 | 0.3148 | 0.3047 | 0.2820 | 0.2590 | — | — | — |
| per-head | local | Fisher | 0.3219 | 0.3206 | 0.3179 | 0.3156 | 0.3115 | 0.3060 | 0.2956 | — | — |
| per-head | local | magnitude | 0.3216 | 0.3192 | 0.3159 | 0.3117 | 0.3074 | 0.2985 | 0.2824 | — | — |
| entire-head | global | Fisher | 0.3242 | 0.3240 | 0.3219 | 0.3201 | 0.3163 | 0.3110 | — | — | — |
| entire-head | global | magnitude | 0.3244 | 0.3221 | 0.3151 | 0.3085 | 0.2990 | — | — | — | — |
| entire-head | local | Fisher | 0.3223 | 0.3213 | 0.3199 | 0.3166 | 0.3107 | 0.3019 | 0.2880 | 0.2543 | 0.1809 |
| entire-head | local | magnitude | 0.3212 | 0.3198 | 0.3178 | 0.3110 | 0.3049 | 0.2906 | 0.2774 | 0.2475 | 0.2057 |

### Whisper — avg WER (%) over seen languages vs global attention sparsity

Average over en/de/fr/es/it/pl (dense WER_en 3.88%); per-language WER/CER and
CoVoST BLEU are in [`RESULTS_whisper.md`](RESULTS_whisper.md).

| Scheme | Threshold | Importance | 10% | 20% | 30% | 40% | 50% |
|---|---|---|---|---|---|---|---|
| per-head | global | Fisher | 7.56 | 7.83 | 8.88 | 12.71 | — |
| per-head | global | magnitude | 24.40 | 111.29 | 189.83 | 178.16 | — |
| entire-head | global | Fisher | 7.38 | 7.38 | 7.71 | 8.56 | — |
| entire-head | global | magnitude | 7.82 | 9.62 | 13.15 | 26.83 | — |
| per-head | local | Fisher | 8.07 | 9.94 | 15.65 | 36.88 | — |
| per-head | local | magnitude | 10.93 | 14.09 | 25.18 | 60.46 | — |
| entire-head | local | Fisher | 7.57 | 7.96 | 8.29 | 9.56 | 20.15 |
| entire-head | local | magnitude | 8.03 | 9.10 | 12.73 | 15.41 | 31.42 |

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
