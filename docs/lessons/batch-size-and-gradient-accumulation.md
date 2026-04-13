# Batch Size and Gradient Accumulation

*ADAI Training — Single-Point Lesson*

---

## The Core Idea

The **batch size** determines how many training examples contribute to a single gradient update. It affects gradient noise, convergence speed, memory consumption, and — when changed — requires a corresponding adjustment to the learning rate.

**Gradient accumulation** is a technique that decouples the logical batch size from the physical batch size that fits in GPU memory. It lets you train with a large effective batch size on hardware that cannot hold it all at once.

---

## What the Batch Size Controls

Each optimizer step computes the gradient as an average over a mini-batch:

$$g = \frac{1}{B} \sum_{i=1}^{B} \nabla_\theta \mathcal{L}(x_i, \theta)$$

where $B$ is the batch size.

**Small $B$** — Noisy gradients. Each update carries high variance. This noise can act as implicit regularization and help escape sharp minima, but it slows convergence and can destabilize training with sensitive optimizers.

**Large $B$** — Smooth gradients. Updates are more accurate estimates of the true gradient. Convergence per step is faster, but training can settle into sharper minima that generalize less well (the "large-batch generalization gap").

There is no universally correct batch size. The optimal value depends on model size, dataset size, hardware, and how it is paired with the learning rate.

---

## The Linear Scaling Rule

When you increase batch size by a factor of $k$, scale the learning rate by the same factor:

$$\eta_{new} = k \cdot \eta_{base}$$

**Why it works**: a larger batch reduces gradient variance. To maintain the same effective step size through parameter space per unit of data seen, the learning rate must grow proportionally.

**Where it breaks down**:

- At very large batch sizes (beyond a problem-specific critical batch size), the gradient noise is so low that further scaling the learning rate causes instability rather than improvement.
- During warmup: apply the rule only after warmup completes. Early training is sensitive to large steps regardless of batch size.

A practical upper bound: the linear rule is reliable up to roughly $B \leq 8{,}192$ tokens for most transformer training runs. Beyond that, test empirically.

---

## Effective Batch Size

The **effective batch size** is the total number of tokens (or samples) that inform one optimizer step, regardless of how the computation is split:

$$B_{eff} = B_{micro} \times N_{GPUs} \times N_{accum}$$

where:

- $B_{micro}$ — samples per forward pass on one device
- $N_{GPUs}$ — number of data-parallel devices
- $N_{accum}$ — gradient accumulation steps

All three terms are interchangeable from the optimizer's perspective. Only $B_{eff}$ matters for the learning rate scaling rule.

---

## Gradient Accumulation

Gradient accumulation runs $N_{accum}$ forward-backward passes without updating the optimizer, then sums the accumulated gradients before stepping.

``` python
for step in range(N_accum):
    loss = model(batch[step]) / N_accum   # scale loss to average, not sum
    loss.backward()                        # accumulate gradients

optimizer.step()
optimizer.zero_grad()
```

The division by $N_{accum}$ is critical. Omitting it is equivalent to multiplying the learning rate by $N_{accum}$, which will destabilize training.

**When to use it**:

- Hardware memory is the bottleneck and you cannot fit the desired $B_{micro}$.
- You want to match the effective batch size of a published model without having equivalent hardware.
- You are fine-tuning a large model where even a single-sample batch saturates GPU memory.

**Cost**: gradient accumulation does not reduce total compute — it runs the same FLOPs split across more steps. It does reduce peak memory because only one micro-batch is resident at a time.

---

## Batch Size and Memory

Memory consumption scales with batch size. For transformers, the dominant cost is the **activation memory** required to store intermediate values for backpropagation:

$$\text{Activation memory} \approx B_{micro} \times L \times d_{model} \times \text{bytes per element}$$

where $L$ is sequence length and $d_{model}$ is the hidden dimension.

**Gradient checkpointing** (activation recomputation) trades compute for memory: activations are discarded during the forward pass and recomputed during backpropagation. This typically reduces activation memory by $\sim$4–8× at the cost of $\sim$30–40% extra compute. Use it when batch size is memory-limited.

---

## Choosing an Effective Batch Size

| Training regime | Typical $B_{eff}$ (tokens) |
| --- | --- |
| Small model fine-tuning | 32 K – 256 K |
| Medium pretraining | 256 K – 1 M |
| Large-scale pretraining (GPT-3 class) | 1 M – 4 M |

These are starting points, not rules. The right value is the largest batch size where:

1. The linear scaling rule still holds for your learning rate, and
2. You are not seeing a measurable generalization gap on validation loss.

A practical approach: fix the learning rate, double the batch size, and check whether validation loss improves, stays flat, or degrades. Stop doubling when validation loss stops improving.

---

## Batch Size and Training Steps

For a fixed dataset size $D$ (tokens), increasing $B_{eff}$ reduces the number of optimizer steps:

$$N_{steps} = \frac{D}{B_{eff}}$$

Fewer steps means less time on the clock, but each step is more expensive and the model sees the same total data. Do not confuse wall-clock speedup with equivalent training — the optimizer dynamics change.

If you increase $B_{eff}$ to reduce training time, also increase $\eta$ per the linear scaling rule and verify loss curves are equivalent before committing to the full run.

---

## Common Mistakes

- **Forgetting to divide loss by $N_{accum}$** when implementing gradient accumulation manually. Gradients will be $N_{accum}\times$ too large.
- **Changing batch size without adjusting learning rate.** This is effectively a silent learning rate change.
- **Using gradient accumulation as a substitute for larger hardware** when the model would genuinely benefit from a physically larger batch (e.g., contrastive learning, where in-batch negatives matter).
- **Assuming a larger batch always trains faster.** Beyond the critical batch size, convergence per token degrades and you pay more compute for the same result.

---

## Quick Decision Checklist

``` text
1. Set B_micro to the largest value that fits in GPU memory without OOM.
2. Decide B_eff based on model scale (use table above as a starting point).
3. Compute N_accum = B_eff / (B_micro × N_GPUs).
4. Apply the linear scaling rule: η = η_base × (B_eff / B_base).
5. Keep warmup token count fixed — do not scale it with batch size.
6. If memory is the bottleneck: try gradient checkpointing before reducing B_micro.
7. Validate: run a short training and compare loss curves to a known-good baseline.
```

---

*See also: [Selecting a Learning Rate](selecting-a-learning-rate.md) — batch size and learning rate must be adjusted together.*
