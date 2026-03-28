# autoresearch — Protein Inverse Folding

This is an experiment to have the LLM do its own protein design research.

## Setup

1. Agree on a **run tag** with the human (e.g. `mar28`). Verify that `autoresearch/<tag>` branch does not already exist.
2. Create the branch: `git checkout -b autoresearch/<tag>`
3. Read the in-scope files: `prepare.py` and `train.py`. Understand the data pipeline, model architecture, and evaluation.
4. Verify data exists: check that `~/.cache/auto-bio/invfold/processed/` contains `.pt` files. If missing, tell the human to run `uv run prepare.py` and wait.
5. Initialize `results.tsv` with the header row (see Logging Results below).
6. Run the baseline as-is — this is experiment #1. Record it as `keep baseline`.
7. Confirm with the human, then begin the experiment loop.

## Experimentation

Each experiment runs on MPS/CPU with a fixed **5-minute time budget**.

Launch command: `uv run train.py`

### What you CAN do

Modify `train.py` — architecture, optimizer, hyperparameters, training loop, batch size, model size. Everything is fair game.

### What you CANNOT do

- Modify `prepare.py`.
- Install new packages.
- Change evaluation logic.

### Model contract

The model MUST implement `predict_logits(batch) -> (total_residues, 20)`. This is called by `evaluate_recovery()` in `prepare.py`.

### Goal

Maximize `val_metric` — sequence recovery (fraction of correctly predicted amino acids, 0.0 to 1.0). Higher is better.

### VRAM

VRAM is a soft constraint. Some increase is acceptable for meaningful gains.

## Simplicity criterion

All else being equal, simpler is better. A small improvement that adds ugly complexity is not worth it. Conversely, removing something and getting equal or better results is a great outcome.

When evaluating whether to keep a change, weigh the complexity cost against the improvement magnitude:

- A +0.5% improvement that adds 30 lines of hacky code? Probably not worth it.
- A +0.5% improvement from deleting code? Definitely keep.

## Architecture search space

### Tier 1 — try first

1. **Backbone noise level:** 0.00, 0.01, 0.02, 0.05, 0.10, 0.15. Also try a noise schedule (high to low over training).
2. **Hidden dim:** 128, 192, 256. More dims = more expressive but fewer epochs in the time budget.
3. **Number of encoder layers:** 3, 4, 5, 6.
4. **Label smoothing:** 0.05-0.1.
5. **Autoregressive decoder:** Decode one residue at a time in random order (ProteinMPNN-style). Potentially more accurate but much slower training.

### Tier 2 — explore next

6. Attention in encoder (graph transformer style within k-NN neighborhood).
7. Iterative refinement: predict all, then refine for 2-3 rounds using predictions as context.
8. Multi-noise training (random noise level per batch from {0, 0.02, 0.1, 0.2}).
9. Cosine LR schedule.
10. Skip connections across encoder layers (DenseNet-style).

### Tier 3 — radical

11. GVP layers (SE(3)-equivariant, scalar + vector features).
12. Surface-derived features from geometry (contact number, solvent accessibility).
13. Confidence-aware decoding order.
14. Ensemble decoding (multiple random orders, consensus).

## Architecture DO NOTs

- Do NOT use pretrained language models (ESM2, etc.) — too large for 5-minute training.
- Do NOT change the k=48 graph — fixed in `prepare.py`.
- Do NOT change edge featurization — 15x16 RBF features are fixed.

## Known results (for context)

| Method | Recovery |
|---|---|
| Rosetta | 32.9% |
| StructGNN | 36.4% |
| GVP | 39.2% |
| ProteinMPNN baseline | 41.2% |
| ProteinMPNN + all improvements | 52.4% |
| PiFold (one-shot) | 51.7% |
| SurfFold | 64.2% |

## Output format

The training script prints metrics in this format:

```
---
val_metric:       0.424000
val_perplexity:   12.3400
training_seconds: 300.1
total_seconds:    325.9
peak_vram_mb:     4200.3
num_params:       578836
```

## Metric extraction

```bash
grep "^val_metric:\|^peak_vram_mb:" run.log
```

If empty, the run crashed. Run `tail -n 50 run.log` to read the stack trace.

## Logging results

`results.tsv` is tab-separated (NOT commas). Header and 5 columns:

```
commit	val_metric	memory_gb	status	description
```

- **commit:** git short hash (7 chars)
- **val_metric:** recovery achieved (0.000000 for crashes)
- **memory_gb:** peak VRAM in GB, rounded to .1f (0.0 for crashes)
- **status:** `keep`, `discard`, or `crash`
- **description:** short text of what was tried

Do NOT commit `results.tsv` — leave it untracked.

## The experiment loop

LOOP FOREVER:

1. Look at git state.
2. Modify `train.py` with an idea.
3. `git commit`
4. Run: `uv run train.py > run.log 2>&1`
5. Read results: `grep "^val_metric:\|^peak_vram_mb:" run.log`
6. If grep empty, the run crashed. `tail -n 50 run.log` for traceback.
7. Log to `results.tsv`.
8. If `val_metric` improved (HIGHER) — **keep** (advance branch).
9. If `val_metric` equal or worse — `git reset --hard HEAD~1` (discard).

**Timeout:** If a run exceeds 10 minutes, kill it and treat as failure.

**Crashes:** If it is a typo or easy fix, fix and re-run. If fundamentally broken, log crash and move on.

## Small dataset note

The dataset has ~18K training proteins and ~1K test proteins. On this dataset, only improvements >1% recovery should be treated as clearly meaningful signal. Smaller deltas may be noise from training randomness.

## NEVER STOP

Once the experiment loop begins, do NOT pause to ask the human if you should continue. Do NOT ask "should I keep going?" or "is this a good stopping point?". The human might be asleep or away and expects you to continue working indefinitely until manually stopped. If you run out of ideas, think harder — re-read the in-scope files, try combining previous near-misses, try radical changes. The loop runs until the human interrupts you.
