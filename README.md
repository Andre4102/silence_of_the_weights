# The Silence of the Weights

Structural pruning of attention-based audio architectures with second-order
(Fisher information) metrics.

> A. Diecidue, C. A. Barbano, P. Fraternali, M. Fontaine, E. Tartaglione,
> *"The silence of the weights: a structural pruning strategy for
> attention-based audio signal architectures with second order metrics,"*
> Interspeech 2026.

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
`—` means that sparsity was not reached. Tables are shown over the paper's
sparsity range (≤ ~60%). Full per-language / per-iteration tables:
[`RESULTS_speechcommands.md`](RESULTS_speechcommands.md),
[`RESULTS_audioset.md`](RESULTS_audioset.md), and
[`RESULTS_whisper.md`](RESULTS_whisper.md).

**Takeaways:** Fisher beats magnitude at every comparable point; entire-head is
the most robust scheme and the only one that also reduces latency (it removes
whole attention maps rather than thinning each head).

### AST — SpeechCommands, accuracy (%) vs attention sparsity (dense 98.15%)

| Scheme | Threshold | Importance | 10% | 20% | 30% | 40% | 50% | 60% |
|---|---|---|---|---|---|---|---|---|
| per-head | global | Fisher | 98.15 | 97.88 | 97.76 | 97.96 | 97.92 | 97.71 |
| per-head | global | magnitude | 97.86 | 97.87 | 97.75 | 97.44 | 97.07 | 96.54 |
| per-head | local | Fisher | 97.90 | 98.05 | 97.92 | 97.58 | 97.59 | 97.55 |
| per-head | local | magnitude | 97.65 | 97.33 | 97.91 | 97.77 | 97.66 | 97.49 |
| entire-head | global | Fisher | 98.02 | 97.92 | 98.12 | 98.12 | 97.78 | 97.58 |
| entire-head | global | magnitude | 97.86 | 97.74 | 97.91 | 97.74 | 97.63 | 96.52 |
| entire-head | local | Fisher | 98.14 | 98.12 | 97.69 | 97.77 | 97.86 | 97.52 |
| entire-head | local | magnitude | 98.13 | 98.12 | 97.91 | 97.87 | 97.53 | 97.12 |

### AST — AudioSet, mAP vs attention sparsity (dense 0.3226)

Multi-label 527-class; mAP higher is better. Full FLOPs/latency/trajectories in
[`RESULTS_audioset.md`](RESULTS_audioset.md).

| Scheme | Threshold | Importance | 10% | 20% | 30% | 40% | 50% | 60% |
|---|---|---|---|---|---|---|---|---|
| per-head | global | Fisher | 0.3245 | 0.3235 | 0.3206 | 0.3172 | 0.3123 | 0.3086 |
| per-head | global | magnitude | 0.3224 | 0.3197 | 0.3148 | 0.3047 | 0.2820 | 0.2590 |
| per-head | local | Fisher | 0.3219 | 0.3206 | 0.3179 | 0.3156 | 0.3115 | 0.3060 |
| per-head | local | magnitude | 0.3216 | 0.3192 | 0.3159 | 0.3117 | 0.3074 | 0.2985 |
| entire-head | global | Fisher | 0.3242 | 0.3240 | 0.3219 | 0.3201 | 0.3163 | 0.3110 |
| entire-head | global | magnitude | 0.3244 | 0.3221 | 0.3151 | 0.3085 | 0.2990 | 0.2918 |
| entire-head | local | Fisher | 0.3223 | 0.3213 | 0.3199 | 0.3166 | 0.3107 | 0.3019 |
| entire-head | local | magnitude | 0.3212 | 0.3198 | 0.3178 | 0.3110 | 0.3049 | 0.2906 |

### AST — AudioSet, latency (ms/forward, batch=1, dense 20.81 ms)

FLOPs are scheme-invariant at equal sparsity, but latency is not: on AudioSet's
long input sequence, entire-head pruning removes whole attention maps/softmaxes
and is consistently faster than per-head (which keeps all heads and only thins
each one).

| Scheme | Threshold | Importance | 10% | 20% | 30% | 40% | 50% | 60% |
|---|---|---|---|---|---|---|---|---|
| per-head | global | Fisher | 20.51 | 19.93 | 19.47 | 19.00 | 18.50 | 17.95 |
| per-head | global | magnitude | 20.48 | 19.86 | 19.23 | 18.74 | 18.16 | 17.20 |
| per-head | local | Fisher | 20.72 | 20.05 | 19.23 | 18.95 | 18.20 | 18.01 |
| per-head | local | magnitude | 20.62 | 19.93 | 19.21 | 18.88 | 18.16 | 17.96 |
| entire-head | global | Fisher | 20.09 | 19.33 | 18.55 | 17.63 | 16.78 | 15.78 |
| entire-head | global | magnitude | 20.17 | 19.16 | 18.38 | 17.62 | 16.57 | 16.03 |
| entire-head | local | Fisher | 20.08 | 19.37 | 18.58 | 17.82 | 16.87 | 16.29 |
| entire-head | local | magnitude | 20.03 | 19.32 | 18.54 | 17.80 | 16.81 | 16.29 |

### Whisper — avg WER (%) over en/fr/it/pl vs global attention sparsity

Average over en/fr/it/pl (dense WER_en 3.88%); per-language WER and CoVoST BLEU
are in [`RESULTS_whisper.md`](RESULTS_whisper.md).

| Scheme | Threshold | Importance | 10% | 20% | 30% | 40% |
|---|---|---|---|---|---|---|
| per-head | global | Fisher | 8.04 | 8.21 | 9.29 | 13.27 |
| per-head | global | magnitude | 24.49 | 99.05 | 191.77 | 212.96 |
| entire-head | global | Fisher | 7.86 | 7.83 | 8.02 | 8.85 |
| entire-head | global | magnitude | 8.28 | 9.79 | 12.93 | 27.76 |
| per-head | local | Fisher | 8.54 | 10.52 | 16.01 | 37.37 |
| per-head | local | magnitude | 11.64 | 14.59 | 25.42 | 60.32 |
| entire-head | local | Fisher | 7.99 | 8.23 | 8.65 | 9.76 |
| entire-head | local | magnitude | 8.65 | 10.06 | 14.28 | 16.30 |

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

## Cite us

If you use this code or build on our results, please cite:

```bibtex
@inproceedings{diecidue2026silence,
  author    = {Diecidue, Andrea and Barbano, Carlo Alberto and Fraternali, Piero and Fontaine, Mathieu and Tartaglione, Enzo},
  title     = {The Silence of the Weights: A Structural Pruning Strategy for Attention-Based Audio Signal Architectures with Second Order Metrics},
  booktitle = {Proc. Interspeech 2026},
  year      = {2026},
}
```
