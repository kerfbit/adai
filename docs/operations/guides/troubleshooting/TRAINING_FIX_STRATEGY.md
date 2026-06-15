# Training Fix Strategy

Practical guide for diagnosing and correcting training problems on the current platform. All configuration is via `config.conf`; there are no per-run CLI flags for hyperparameters.

---

## Table of Contents

1. [Platform Overview](#platform-overview)
2. [Reading the Training Log](#reading-the-training-log)
3. [Understanding the Metrics](#understanding-the-metrics)
4. [Diagnosing Common Problems](#diagnosing-common-problems)
5. [Configuration Levers](#configuration-levers)
6. [Gradient Clipping Reference](#gradient-clipping-reference)
7. [Data Quality](#data-quality)
8. [Historical Fixes (For Reference)](#historical-fixes-for-reference)

---

## Platform Overview

Training runs through two tools:

| Tool | Role |
| --- | --- |
| `dataset_manager` | Manages the data queue — add local files, download from Gutenberg or HuggingFace |
| `incremental_trainer` | Consumes the queue and trains the model; forks to the background automatically |

**All hyperparameters live in `config.conf`.** Edit the file; the next training run picks up changes. No rebuild is required for parameter changes.

**All training runs fork into the background.** Output goes to the log file, not the terminal.

```bash
# Check what is running / what has run
./build/bin/incremental_trainer status
./build/bin/incremental_trainer history

# Follow the active log
tail -f chatbot_server.log    # or whatever LOG_FILE_PATH is set to
```

---

## Reading the Training Log

Each epoch line shows the key indicators:

```text
[info] Epoch 5/25 | loss: 2.341 | val_loss: 2.489 | perplexity: 10.4 | lr: 0.000271 | grad_norm: 3.12 | updates: 412
```

| Field | What it tells you |
| --- | --- |
| `loss` | Training loss this epoch (lower is better) |
| `val_loss` | Validation loss — generalization health; should track training loss |
| `perplexity` | `exp(val_loss)` — intuitive quality measure |
| `lr` | Current learning rate after the schedule step |
| `grad_norm` | Average gradient L2 norm for the epoch |
| `updates` | Number of weight-update steps (should be consistent across epochs) |

A NaN/Inf gradient warning looks like:

```text
[warn] ⚠️  WARNING: NaN or Inf gradient detected at sample 142! Skipping update.
```

The trainer skips bad updates automatically; occasional warnings are not fatal. Persistent warnings signal a configuration problem.

---

## Understanding the Metrics

### Loss

Cross-entropy loss averaged over all predictions.

| Range | Interpretation |
| --- | --- |
| > 6 | Model is barely above random guessing |
| 3 – 6 | Early learning; normal for epochs 1–3 on small data |
| 1 – 3 | Good convergence |
| < 1 | Excellent; risk of overfitting on tiny datasets |

Random-guess baseline: `ln(vocab_size)` — approximately 8.5 for a 5,000-token vocab, 9.2 for 10,000.

### Perplexity

`exp(loss)` — average branching factor of predictions.

| Range | Interpretation |
| --- | --- |
| > 200 | Underfitting; model not learning |
| 50 – 200 | Weak; needs more training or better data |
| 10 – 50 | Acceptable for small datasets |
| 3 – 10 | Good |
| < 3 | Excellent; watch for overfitting |

### Gradient Norm

L2 norm of all gradients before clipping.

| Range | Interpretation |
| --- | --- |
| 1 – 20 | Healthy |
| 20 – 50 | Elevated; monitor |
| > 50 | Warning; clipping is doing heavy work |
| NaN / Inf | Numerical failure; update skipped automatically |

### Validation Loss Gap

`val_loss - loss` = generalization gap.

- Gap ≈ 0 — healthy
- Gap growing each epoch — overfitting; reduce epochs, increase `WEIGHT_DECAY`, or add data
- `val_loss` lower than `loss` — data split may be too small; not a real problem

### Expected Training Dynamics

| Phase | Typical Loss | Typical Perplexity |
| --- | --- | --- |
| Epochs 1–3 | 7–8 → 4–5 | 1000+ → 50–150 |
| Epochs 4–10 | 4–5 → 2–3 | 50–150 → 10–20 |
| Epochs 11–25 | 2–3 → 1–2 | 10–20 → 3–7 |

---

## Diagnosing Common Problems

### Loss rises or oscillates after epoch 2

**Most likely cause:** Learning rate too high.

The WARMUP_COSINE schedule ramps LR up over the first ~10% of steps, then decays it. If the peak LR is too high, the model diverges after warmup ends.

**Fix:**

```ini
# config.conf
LEARNING_RATE=0.0003     # safe starting point for most datasets
GRADIENT_CLIP=1.0
```

Then retrain from scratch:

```bash
./build/bin/incremental_trainer retrain 25
```

If 0.0003 still diverges, drop to 0.0001.

---

### Persistent NaN/Inf gradient warnings

Occasional NaN/Inf warnings are harmless — the trainer skips those updates. If they appear every few steps throughout training:

1. Halve the learning rate:

   ```ini
   LEARNING_RATE=0.0001
   GRADIENT_CLIP=0.5
   ```

2. Check training data for extremely long sequences or binary/non-UTF-8 content.
3. Consider enabling adaptive clipping (see [Gradient Clipping Reference](#gradient-clipping-reference)).

---

### Perplexity stuck above 100 after 10+ epochs

**Possible causes and fixes — check in order:**

1. **Insufficient epochs.** Set `NUM_EPOCHS=40` or `50` and run `incremental_trainer train 40`.

2. **Learning rate too low.** After warmup the LR may be decaying too fast for the dataset. Try `LEARNING_RATE=0.0005` for larger datasets (> 5,000 pairs).

3. **Data format problems.** Every pair must have exact `INPUT:` / `RESPONSE:` prefixes with no extra whitespace. Verify:

   ```bash
   head -20 your_training_data.txt
   grep -c "^INPUT:" your_training_data.txt
   grep -c "^RESPONSE:" your_training_data.txt
   # Counts should be equal
   ```

4. **Vocabulary mismatch.** If vocab was rebuilt after the model was initialized, rebuild the model:

   ```bash
   ./build/bin/incremental_trainer reset --yes
   ./build/bin/incremental_trainer train 25
   ```

5. **Dataset too small.** Fewer than 500 pairs is very difficult to train on. Add more data:

   ```bash
   ./build/bin/dataset_manager gutenberg 1342 500
   ./build/bin/incremental_trainer train 25
   ```

6. **Model too large for the data.** Consider reducing architecture in `config.conf`:

   ```ini
   D_MODEL=128
   D_FF=512
   NUM_ENCODER_LAYERS=2
   NUM_DECODER_LAYERS=2
   ```

   Then reset and retrain: `./build/bin/incremental_trainer reset --yes && ./build/bin/incremental_trainer train 30`

---

### Loss decreases but chatbot responses are incoherent

Low training loss does not guarantee quality responses. Common causes:

- **Vocabulary too small** — words are over-tokenized into subwords. Rebuild with `--vocab-size 8000` or higher.
- **Overfitting** — loss is very low but validation loss is higher and growing. The model has memorized training pairs. Add more diverse data or reduce epochs.
- **Generation strategy mismatch** — try `STRATEGY=nucleus` with `TEMPERATURE=0.8` and `TOP_P=0.9`.
- **Context too short** — check `MAX_SEQ_LENGTH` covers your typical prompt+response length.

---

### Validation loss climbs while training loss falls (overfitting)

The model is memorizing rather than generalizing.

**Fixes (try in order):**

1. Add more diverse training data.
2. Increase regularization:

   ```ini
   WEIGHT_DECAY=0.05    # default is 0.01; raise it
   ```

3. Reduce epochs — stop when validation loss is at its minimum, not when training loss is lowest. The trainer automatically saves the best-validation-loss checkpoint as the active model.

4. Enable early stopping:

   ```ini
   # Not yet exposed as a config key — currently set in code defaults
   # patience = 5 (stops if val_loss doesn't improve for 5 epochs)
   ```

---

### Training is very slow

- Confirm OpenMP is active (parallel attention): `lscpu | grep CPU`. If the build has OpenMP, training uses all cores automatically.
- Reduce `BATCH_SIZE` if RAM is the bottleneck (counter-intuitively, smaller batches may train faster on memory-bound systems).
- Set `GPU_STRATEGY=full` on a dedicated training machine (default `background` yields to other GPU work).
- Reduce model size for prototyping; scale up once dynamics are validated.

---

### "No pending data. Use DatasetManager to queue training data."

The `train` command found an empty pending queue.

```bash
./build/bin/dataset_manager add your_data.txt
# or
./build/bin/dataset_manager gutenberg 1342 500
./build/bin/incremental_trainer train 25
```

---

### Training completed but model files are missing

The trainer saves epoch checkpoints in `SESSION_DIR` and promotes the best validation-loss epoch to the active model symlinks. If symlinks are missing after an interrupted run:

```bash
# Find available checkpoints
ls training_sessions/*.bin | sort

# Manually link the best epoch (e.g. session_3_checkpoint.bin)
touch chatbot_model.bin
for ext in config encoder decoder lm_head vocab; do
    ln -sf training_sessions/session_3_checkpoint.bin.${ext} chatbot_model.bin.${ext}
done
```

Or resume to let the trainer re-promote:

```bash
./build/bin/incremental_trainer resume
```

---

## Configuration Levers

All of these go in `config.conf`. Restart training after any change.

### Hyperparameters

| Key | Default | When to change |
| --- | --- | --- |
| `LEARNING_RATE` | `0.0001` | Raise to `0.0003–0.0005` for faster convergence; lower to `0.00005` if diverging |
| `NUM_EPOCHS` | `10` | `25–30` for small datasets; `15–20` for large ones |
| `BATCH_SIZE` | `1` | Raise to `4–32` for speed; lower if out of memory |
| `WEIGHT_DECAY` | `0.01` | Raise to `0.05–0.1` to combat overfitting |
| `GRADIENT_CLIP` | `1.0` | Lower to `0.5` if seeing persistent NaN/Inf warnings |

### Architecture (requires reset + retrain to take effect)

| Key | Default | Small dataset | Large dataset |
| --- | --- | --- | --- |
| `D_MODEL` | `512` | `128` | `512` |
| `D_FF` | `2048` | `512` | `2048` |
| `NUM_ENCODER_LAYERS` | `6` | `2` | `6` |
| `NUM_DECODER_LAYERS` | `6` | `2` | `6` |

After changing architecture keys:

```bash
./build/bin/incremental_trainer reset --yes
./build/bin/incremental_trainer train 25
```

### Generation Quality Scoring

Enable BLEU/ROUGE scoring during validation to measure output quality directly:

```ini
ENABLE_GENERATION_QUALITY_METRICS=true
GENERATION_QUALITY_SAMPLE_SIZE=20
GENERATION_QUALITY_MAX_TOKENS=50
```

Scores appear in the log alongside loss. Adds significant time per epoch; keep `SAMPLE_SIZE` ≤ 20 for routine training.

---

## Gradient Clipping Reference

### Fixed clipping (default)

The gradient L2 norm is capped at `GRADIENT_CLIP` (default `1.0`). Any update whose norm exceeds the threshold is rescaled to exactly the threshold.

```ini
GRADIENT_CLIP=1.0     # standard
GRADIENT_CLIP=0.5     # more aggressive; use when NaN/Inf warnings persist
```

### Adaptive clipping

Dynamically adjusts the clip threshold based on a running EMA of recent gradient norms, so it tightens during spikes and relaxes during stable phases. Activate with:

```ini
GRADIENT_CLIP_ADAPTIVE=true
GRADIENT_CLIP_MIN=0.1          # threshold never goes below this
GRADIENT_CLIP_MAX=5.0          # threshold never goes above this
GRADIENT_CLIP_EMA_DECAY=0.05   # smoothing factor (smaller = slower adaptation)
GRADIENT_CLIP_HEADROOM=2.0     # threshold = ema_norm × headroom
GRADIENT_CLIP_WARMUP_STEPS=100 # steps before adaptive logic activates
GRADIENT_CLIP_SPIKE_K=5.0      # norms > k×ema are excluded from EMA update
```

**When to use adaptive clipping:**

- Training on mixed-quality datasets where gradient magnitude varies widely
- Long training runs where optimal clip threshold changes over time
- After seeing many NaN/Inf warnings with fixed clipping

**When to stick with fixed clipping:**

- Short prototyping runs (adaptive needs warmup steps to calibrate)
- When you already know the right threshold from a previous run

---

## Data Quality

Training quality is bounded by data quality. Common issues and checks:

### Format verification

```bash
# Pair counts must match
grep -c "^INPUT:" training_data.txt
grep -c "^RESPONSE:" training_data.txt

# Spot-check the first few pairs
head -20 training_data.txt

# Check for very long lines that may exceed MAX_SEQ_LENGTH
awk 'length > 400' training_data.txt | wc -l
```

### Dataset size guidelines

| Pairs | Expected outcome |
| --- | --- |
| < 200 | Very likely to memorize and overfit; for testing only |
| 200 – 1,000 | Workable for a narrow-domain bot; use small architecture |
| 1,000 – 10,000 | Good general training range |
| > 10,000 | Can support full default architecture; increase `BATCH_SIZE` |

### Improving a low-quality dataset

1. **Add diversity** — mix genres, topics, and sentence lengths. A Gutenberg batch adds literary range quickly:

   ```bash
   ./build/bin/dataset_manager gutenberg-batch 1342,11,1661,84 400
   ```

2. **Add conversational data** — HuggingFace `daily_dialog` gives natural multi-turn pairs:

   ```bash
   ./build/bin/dataset_manager huggingface daily_dialog 500
   ```

3. **Check topic balance** — one over-represented topic causes the model to veer toward it regardless of input.

4. **Remove truncated responses** — partial sentences at end-of-file cause the model to generate incomplete replies.

---

## Historical Fixes (For Reference)

These bugs were found and fixed in earlier development. The fixes are permanently in the codebase; this section exists to explain *why* certain behaviors exist.

### Loss averaging bug (fixed)

**Symptom:** Reported epoch loss appeared artificially low and inconsistent; perplexity stuck above 1000 even after many epochs.

**Root cause:** `total_loss` was divided by `num_samples` but was only accumulated on update steps (not on every sample), producing incorrect averages when `batch_size > 1` or gradient accumulation was active.

**Fix:** Division is now by `num_updates` (the count of actual weight-update steps), giving a correct per-update average.

**Current behavior:** Loss reported in the log is accurate and comparable across epochs and runs.

---

### Missing NaN/Inf gradient safety check (fixed)

**Symptom:** A single corrupt batch could propagate NaN weights through the entire network, causing all subsequent loss values to be NaN and making the model unrecoverable without restarting.

**Root cause:** No check on `grad_norm` before calling the optimizer step.

**Fix:** After computing `grad_norm`, the trainer checks `std::isnan(grad_norm) || std::isinf(grad_norm)`. If true, it logs a warning, zeroes gradients, and continues to the next batch without updating weights.

**Current behavior:** Bad updates are skipped automatically. Occasional warnings in the log are expected and safe; persistent warnings indicate a configuration issue (see [Persistent NaN/Inf gradient warnings](#persistent-naninf-gradient-warnings) above).
