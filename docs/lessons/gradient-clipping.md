# Gradient Clipping

*ADAI Training — Single-Point Lesson*

---

## The Core Idea

During backpropagation, the gradient of the loss with respect to any parameter can occasionally become very large — orders of magnitude larger than a typical step. A single such spike can corrupt model weights irreversibly, producing a loss explosion that no learning rate adjustment can recover from.

**Gradient clipping** caps the magnitude of the gradient before each optimizer step, preventing catastrophic updates while leaving normal steps unaffected.

---

## What Causes Gradient Spikes

- **Outlier training examples**: a rare document with extreme token statistics produces an unusually large loss and therefore a large gradient.
- **Model instability early in training**: before the optimizer has found a reasonable region of the loss surface, gradients can be erratic.
- **Long sequences and deep networks**: gradients flow through many multiplicative operations. Slight misalignments in weight magnitude compound across layers.
- **Low-precision arithmetic**: fp16 has a narrow dynamic range (max representable value ~65 504). Gradients that exceed this overflow to infinity or NaN, propagating failure through the entire parameter tensor.

---

## Global Norm Clipping

The standard method is **global norm clipping** (also called gradient norm clipping). Rather than clipping each parameter's gradient independently, it rescales all gradients together to preserve their relative direction:

**Step 1**: Compute the global gradient norm across all parameters:

$$\|g\| = \sqrt{\sum_{i} \|g_i\|^2}$$

**Step 2**: If $\|g\| > \tau$, rescale every gradient:

$$g_i \leftarrow g_i \cdot \frac{\tau}{\|g\|}$$

where $\tau$ is the clipping threshold (commonly `max_norm`).

If $\|g\| \leq \tau$, no rescaling occurs — the step proceeds normally.

This is preferable to per-parameter clipping because it does not distort the direction of the update, only its magnitude.

---

## Choosing the Threshold $\tau$

The canonical default for transformer training is:

$$\tau = 1.0$$

This value has become standard across GPT, BERT, T5, and Llama training runs and is a safe starting point for almost any transformer. It rarely needs tuning.

**Signs the threshold is too high**: you still see occasional loss spikes; gradient norms regularly exceed 10–20.

**Signs the threshold is too low**: training is stable but slow; the gradient norm is at the cap on nearly every step (check this by logging the pre-clip norm). This indicates the optimizer is systematically constrained, not just catching outliers.

As a guideline: clipping should activate on fewer than ~5% of steps in healthy training. If it fires constantly, something else is wrong — investigate learning rate, weight initialization, or data quality.

---

## Monitoring the Gradient Norm

Logging the pre-clip gradient norm is one of the most informative training diagnostics:

| Observation | Likely Meaning |
| --- | --- |
| Norm is stable, rarely hits $\tau$ | Training is healthy |
| Norm spikes sharply then recovers | Outlier batch; clipping handled it correctly |
| Norm grows steadily over training | Learning rate may be too high; model diverging |
| Norm hits $\tau$ on nearly every step | Threshold too low, or LR too high |
| Norm is NaN or Inf | fp16 overflow; underflow in loss scaling; or a bug |

Always log `grad_norm` during training. It is often the earliest signal of an instability that will manifest in the loss curve much later.

---

## Gradient Clipping with Mixed Precision

In fp16 training, gradients are computed in low precision and must be scaled up before the backward pass to avoid underflow (very small gradients flushing to zero). A **loss scaler** multiplies the loss by a large factor $S$ before backprop, then divides the gradients by $S$ before clipping and the optimizer step.

The correct operation order is:

``` text
1. scaler.scale(loss).backward()   # compute scaled gradients
2. scaler.unscale_(optimizer)      # divide by S → restore true gradient magnitudes
3. clip_grad_norm_(params, max_norm=1.0)   # clip AFTER unscaling
4. scaler.step(optimizer)
5. scaler.update()
```

Clipping **before** unscaling clips against the inflated gradient magnitudes and is incorrect — the effective threshold becomes $\tau / S$, which is nearly zero.

With **bf16**, loss scaling is not required (bf16 has a wider dynamic range than fp16), so the operation order is simply: backward → clip → step.

---

## Value Clipping vs. Norm Clipping

A simpler but less commonly used variant is **value clipping**, which clips each gradient component independently:

$$g_i \leftarrow \text{clip}(g_i, -\tau, +\tau)$$

This distorts the gradient direction (different components are scaled by different factors) and is generally avoided for transformer training. It is sometimes used in reinforcement learning (PPO) for separate reasons. Use global norm clipping by default.

---

## Common Mistakes

- **Clipping before unscaling in fp16 training.** This silently produces near-zero effective gradients and causes training to stall.
- **Not logging the gradient norm.** Without it, gradient explosions are only visible after the loss curve has already diverged.
- **Setting $\tau$ very small** (e.g., 0.1) hoping to stabilize an unstable run. This treats the symptom; the instability will persist. Investigate learning rate and initialization instead.
- **Omitting clipping entirely** on the assumption that the optimizer handles it. Adam/AdamW do not clip; they only normalize per-parameter learning rates based on gradient history. A single spike can still corrupt weights.

---

## Quick Decision Checklist

``` text
1. Set max_norm = 1.0. This is correct for nearly all transformer runs.
2. Log the pre-clip gradient norm every step.
3. If using fp16: ensure clipping happens AFTER loss scaler unscaling.
4. If using bf16: no loss scaler needed; backward → clip → step.
5. Monitor the fraction of steps where clipping activates.
   Target: < 5% of steps during stable mid-training.
6. If the norm is consistently at the cap: lower the learning rate first,
   then consider reducing max_norm as a last resort.
```

---

*See also: [Selecting a Learning Rate](selecting-a-learning-rate.md) — gradient spikes are often a symptom of a learning rate that is too high. [Batch Size and Gradient Accumulation](batch-size-and-gradient-accumulation.md) — when accumulating gradients, clip after the full accumulation, not after each micro-batch.*
