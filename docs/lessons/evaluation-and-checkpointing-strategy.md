# Evaluation and Checkpointing Strategy

*ADAI Training — Single-Point Lesson*

---

## The Core Idea

A model that trains without a coherent evaluation and checkpointing plan is running blind. Evaluation tells you whether the model is actually improving on what matters. Checkpointing ensures that the best state is recoverable when training eventually diverges, hardware fails, or a later epoch degrades quality.

These two concerns are inseparable: evaluation determines *which* checkpoint is best, and checkpointing determines whether you can *retrieve* it.

---

## The Validation Set

The validation set is the primary tool for measuring generalization during training. It must satisfy three properties:

**1. Representative** — it samples the same distribution as the target deployment use case. A validation set drawn from the training distribution measures memorization, not generalization.

**2. Held-out** — no example in the validation set has been seen during training, directly or through deduplication proximity. Contamination produces optimistically biased metrics.

**3. Fixed** — the validation set does not change between runs. Changing it invalidates comparisons across experiments.

A validation set of **1 000 – 10 000 examples** is sufficient to produce stable loss estimates for most language model training runs. Larger is better for low-frequency tasks where a small sample has high variance.

---

## What to Evaluate

### Validation Loss

Perplexity or cross-entropy loss on the validation set is the primary training signal. It is fast to compute, directly comparable across runs, and smooth enough to track convergence.

$$\mathcal{L}_{val} = -\frac{1}{|D_{val}|} \sum_{(x,y) \in D_{val}} \log P_\theta(y \mid x)$$

Loss alone is not sufficient for final model selection — a model with lower validation loss does not always perform better on downstream tasks — but it is the right signal to monitor *during* training.

### Downstream Task Metrics

Evaluate on task-specific benchmarks periodically (e.g., every 10–20% of training). These are slower to compute but reflect real capability. Examples:

- Exact match or F1 for question answering
- BLEU / ROUGE for generation tasks
- Accuracy on classification benchmarks
- Pass@k for code generation

Do not use downstream metrics for step-by-step early stopping — they are too noisy and too slow. Use them to confirm that validation loss improvements correspond to real capability gains, and to catch cases where they diverge.

### Training Loss

Always log training loss alongside validation loss. A steady decrease in training loss with a plateau or increase in validation loss is the earliest reliable signal of overfitting.

---

## Evaluation Frequency

| Phase | Recommended evaluation frequency |
| --- | --- |
| First 5% of training | Every 1–2% of total steps — instability is most likely here |
| Mid-training (5–90%) | Every 5–10% of total steps |
| Final 10% of training | Every 2–5% of total steps — loss often moves fast near the end |
| Short fine-tuning runs | Every epoch, or every 100–500 steps |

Evaluating too frequently wastes compute on validation inference. Evaluating too infrequently misses the window to identify and react to divergence before it compounds.

---

## Checkpointing Strategy

A checkpoint is a snapshot of the full training state: model weights, optimizer state, learning rate scheduler state, RNG state, and the step number.

### What Must Be Saved

| Component | Why |
| --- | --- |
| Model weights | The primary artifact |
| Optimizer state (Adam moments) | Required to resume training correctly |
| LR scheduler state | Required to resume at the correct point on the schedule |
| RNG state (CPU + GPU) | Required for exact reproducibility of data ordering |
| Step number and epoch | Required to resume the schedule and data pipeline |
| Validation loss at this step | Required to identify the best checkpoint later |

Saving only the model weights produces a checkpoint that cannot resume training — it can only be used for inference.

### Checkpoint Frequency

Save a checkpoint at every evaluation point. Storage is cheap relative to the cost of rerunning training.

Additionally, save a **rolling checkpoint** that overwrites the previous one every N steps (e.g., every 500 steps). This provides a recent recovery point in case of hardware failure between evaluation checkpoints.

### Best Checkpoint Tracking

At the end of training, the final checkpoint is not always the best one. Validation loss often reaches its minimum before the end of the run, then climbs slightly as the model begins to overfit or as the learning rate bottoms out.

Track the checkpoint with the lowest validation loss throughout training and preserve it separately:

``` python
if val_loss < best_val_loss:
    best_val_loss = val_loss
    save_checkpoint("best_model.bin")
```

For fine-tuning runs, the best checkpoint is almost always not the final one.

---

## Early Stopping

Early stopping halts training when validation loss has not improved for a fixed number of evaluation intervals (the **patience**):

``` python
patience = 5   # evaluation intervals without improvement
no_improve_count = 0

if val_loss < best_val_loss - min_delta:
    best_val_loss = val_loss
    no_improve_count = 0
else:
    no_improve_count += 1
    if no_improve_count >= patience:
        stop training
```

**When to use early stopping**: fine-tuning runs where overfitting is the primary risk and compute is limited.

**When not to use early stopping**: large-scale pretraining, where validation loss rarely increases and early stopping would terminate a run that would have continued improving with more data or a decaying learning rate.

---

## Test Set Discipline

The test set is used **once**, at the end of the project, to report final numbers. It is not a validation set.

Using the test set iteratively — selecting among models based on test performance, then reporting that performance — is data leakage. It produces numbers that are optimistic by a margin proportional to how many decisions were made using the test set.

Rules:

- Never use test set results to make training decisions.
- Never tune hyperparameters to improve test performance.
- Run the test set evaluation once, after all model selection is complete.

---

## Multi-Metric Model Selection

Validation loss alone can be misleading for final model selection. When multiple checkpoints have similar validation loss, use downstream task performance to break ties.

A practical approach:

1. Filter to checkpoints within $\delta = 0.01$ nats of the best validation loss.
2. Among those, select the checkpoint with the best downstream task score.
3. If downstream tasks conflict (model A is better at task X, model B at task Y), define a weighted composite metric before evaluating — do not select after seeing the results.

---

## Logging and Experiment Tracking

Each training run should log, at minimum:

- Step number and wall-clock time
- Training loss (smoothed and raw)
- Validation loss at each evaluation point
- Learning rate at each step
- Gradient norm at each step
- Loss scale value (for fp16 runs)
- Hardware utilization (MFU or GPU utilization %)

Store hyperparameters and git commit hash alongside each run. Without a record of what was run and under what conditions, results from different runs cannot be compared reliably.

---

## Common Mistakes

- **Saving only the model weights.** This prevents resuming training and forces a full restart on hardware failure.
- **Using the test set during development.** Reported numbers become meaningless.
- **Evaluating on the same data used to select the best checkpoint.** The checkpoint selection process itself is a form of fitting; the reported metric will be optimistic.
- **Not saving a rolling checkpoint.** A long training run with only infrequent evaluation checkpoints can lose many hours of progress to a hardware failure between saves.
- **Reporting the final checkpoint's loss** rather than the best checkpoint's loss. For fine-tuning, these are often different.

---

## Quick Decision Checklist

``` text
1. Create a fixed validation set before training begins. Do not change it.
2. Log training loss every step; compute validation loss every 5–10% of steps.
3. Save a full checkpoint (weights + optimizer + scheduler + RNG) at each eval.
4. Also save a rolling checkpoint every ~500 steps to limit hardware-failure loss.
5. Track and separately save the best-validation-loss checkpoint.
6. Use early stopping for fine-tuning; omit it for large-scale pretraining.
7. Reserve the test set for one final evaluation after all decisions are made.
8. Log hyperparameters and git commit hash with each run.
```

---

*See also: [Selecting a Learning Rate](selecting-a-learning-rate.md) — schedule length must be planned to match the total token budget, which checkpointing lets you verify mid-run. [Regularization Strategy](regularization-strategy.md) — the train/val loss gap visible in checkpoint logs is the primary signal for overfitting.*
