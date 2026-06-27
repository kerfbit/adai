# Selecting a Learning Rate

*ADAI Training — Single-Point Lesson*

---

## The Core Idea

The learning rate $\eta$ controls how large a step the optimizer takes along the loss gradient at each update:

$$\theta \leftarrow \theta - \eta \cdot \nabla_\theta \mathcal{L}$$

Getting this value right is one of the highest-leverage decisions in training. Too large and the loss diverges or oscillates; too small and training stalls or converges to a poor minimum.

---

## What "Too High" and "Too Low" Look Like

| Symptom | Likely Cause |
| --- | --- |
| Loss spikes or NaN after a few steps | $\eta$ too high |
| Loss decreases very slowly or plateaus early | $\eta$ too low |
| Loss oscillates around a value without converging | $\eta$ slightly too high |
| Validation loss diverges from training loss quickly | $\eta$ too high or no schedule |
| Training is stable but final quality is poor | $\eta$ too low — under-explored loss surface |

---

## The Learning Rate Range Test

Before committing to a fixed value, run a **range test** (Leslie Smith, 2017):

1. Start with a very small $\eta$ (e.g., $10^{-7}$).
2. Increase it exponentially over a short run (a few hundred steps).
3. Plot loss vs. $\eta$.
4. Choose a value **just before** the loss begins to rise sharply — typically one order of magnitude below the minimum-loss point.

This is the most reliable empirical method for finding a good starting value.

---

## Common Starting Points by Optimizer

| Optimizer | Typical Good Range |
| --- | --- |
| SGD (no momentum) | $10^{-2}$ – $10^{-1}$ |
| SGD + Momentum | $10^{-3}$ – $10^{-2}$ |
| Adam / AdamW | $10^{-4}$ – $3\times10^{-4}$ |
| AdaGrad | $10^{-2}$ – $10^{-1}$ |

For transformer models with AdamW, **$3\times10^{-4}$** is widely used as a baseline, with warmup for the first 1–5% of training steps.

---

## Warmup

Starting with a full learning rate from step 0 can destabilize early training when weights are randomly initialized. A **linear warmup** ramps $\eta$ from 0 to its peak over $T_{warm}$ steps:

$$\eta_t = \eta_{max} \cdot \frac{t}{T_{warm}}, \quad t < T_{warm}$$

A warmup of **1 000 – 4 000 steps** is standard for transformer pretraining. For fine-tuning on a small dataset, 50–200 steps is usually sufficient.

---

## Decay Schedules

After warmup, a **decay schedule** gradually reduces $\eta$ to avoid overshooting late in training.

**Cosine decay** (most common for transformers):

$$\eta_t = \eta_{min} + \frac{1}{2}(\eta_{max} - \eta_{min})\left(1 + \cos\left(\frac{\pi \cdot t}{T_{total}}\right)\right)$$

**Step decay** (simpler, common in CNNs):
Reduce $\eta$ by a fixed factor (e.g., $\times 0.1$) at predetermined epochs.

**No decay**: Acceptable for very short fine-tuning runs where training ends before the schedule would matter.

---

## Key Rules of Thumb

- **Scale $\eta$ with batch size**: if you double the batch size, try doubling $\eta$ (linear scaling rule). Test carefully — this breaks down at very large batches.
- **Always use warmup** when training transformers from scratch.
- **Prefer cosine decay** over step decay for smoother convergence.
- **Monitor gradient norms**: if they grow unbounded, reduce $\eta$ or add gradient clipping (typically `max_norm = 1.0`).
- When resuming a checkpoint, **restart the schedule from the checkpoint step**, not from zero.

---

## Quick Decision Checklist

``` text
1. Run a learning rate range test (or start at 3e-4 for AdamW).
2. Add warmup (1 000 steps for pretraining, 100 steps for fine-tuning).
3. Add cosine decay to the end of the run.
4. Check loss curve after the first epoch — adjust by 2–5× if needed.
5. If loss spikes: halve η. If loss barely moves after warmup: double η.
```

---

*See also: [Architecting Attention](architecting-attention.md) § Chapter 8 — Optimization Stability.*
