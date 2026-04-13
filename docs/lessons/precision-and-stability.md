# Precision and Training Stability

*ADAI Training — Single-Point Lesson*

---

## The Core Idea

Modern transformer training rarely uses 32-bit floating point (fp32) throughout. Mixed-precision training performs the forward and backward passes in a lower-precision format to reduce memory consumption and increase throughput, while keeping a master copy of weights in higher precision for the optimizer step.

The choice of low-precision format — and how it is managed — is one of the most consequential decisions for training stability. Getting it wrong produces silent numerical errors, NaN losses, or models that degrade compared to their fp32 equivalents with no obvious cause.

---

## Floating-Point Formats

| Format | Exponent bits | Mantissa bits | Max value | Notes |
| --- | --- | --- | --- | --- |
| fp32 | 8 | 23 | ~3.4 × 10³⁸ | Full precision; baseline |
| fp16 | 5 | 10 | ~65 504 | Narrow range; needs loss scaling |
| bf16 | 8 | 7 | ~3.4 × 10³⁸ | Same range as fp32; less precision |
| tf32 | 8 | 10 | ~3.4 × 10³⁸ | NVIDIA Ampere+; automatic |

**bf16 is preferred** for transformer training on hardware that supports it (Ampere GPUs and later, TPUs). Its exponent range matches fp32, which eliminates overflow and the need for loss scaling. The reduced mantissa precision (7 bits vs. 23) has no measurable effect on final model quality for training.

**fp16** requires active management. Its maximum representable value is ~65 504 — gradients and activations routinely exceed this, producing infinity or NaN. Loss scaling is the standard mitigation.

---

## Mixed-Precision Training

The standard mixed-precision workflow maintains two copies of the weights:

1. **Low-precision weights** (fp16 or bf16) used for forward and backward passes.
2. **fp32 master weights** used for the optimizer step.

The accumulation of small gradient updates requires fp32 precision to avoid rounding them to zero. Over millions of steps, the rounding error in a pure fp16 optimizer would diverge from the fp32 equivalent.

``` text
Forward pass   → fp16/bf16
Backward pass  → fp16/bf16 gradients
Optimizer step → gradients cast to fp32, update fp32 master weights
Weight sync    → fp32 master weights cast back to fp16/bf16 for next forward pass
```

Most frameworks (PyTorch `torch.cuda.amp`, JAX) handle this automatically. The key is knowing what is happening so you can diagnose when it breaks.

---

## Loss Scaling (fp16 Only)

Backpropagation through an fp16 network can produce gradients so small they underflow to zero — particularly in the early layers of deep networks. Loss scaling compensates by multiplying the loss by a large scalar $S$ before the backward pass, then dividing the resulting gradients by $S$ before the optimizer step.

$$\mathcal{L}_{scaled} = S \cdot \mathcal{L}$$

$$g_{true} = \frac{g_{scaled}}{S}$$

**Dynamic loss scaling** adjusts $S$ automatically:

- If any gradient is infinite or NaN, skip the optimizer step and reduce $S$ (e.g., halve it).
- If a fixed number of consecutive steps complete without overflow, increase $S$ (e.g., multiply by 2).

A healthy training run with dynamic loss scaling will see occasional skipped steps early on, then stabilize. Frequent skipped steps throughout training indicate the learning rate is too high or the model is architecturally unstable.

**With bf16, loss scaling is unnecessary.** The wider exponent range prevents gradient underflow under normal conditions.

---

## Numerical Stability Failure Modes

| Symptom | Likely Cause |
| --- | --- |
| Loss is NaN at step 0 | Weight initialization too large; fp16 overflow |
| Loss spikes to NaN mid-training | Gradient overflow; loss scale too high or LR too high |
| Loss scale drops to 1 and stays there | Persistent gradient overflow; architectural instability |
| Loss converges but to a worse value than fp32 | Precision error accumulating in optimizer; check master weight dtype |
| Attention logits overflow | Sequence length too long; QK dot products exceed fp16 range |
| Identical loss curves with subtly different weights | Determinism broken; check for non-deterministic ops |

---

## Attention and Softmax Stability

The attention score computation:

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

is a common source of numerical instability. The dot product $QK^T$ grows in magnitude with sequence length and embedding dimension. In fp16, large logits overflow before the softmax is applied.

Mitigations:

- The $\frac{1}{\sqrt{d_k}}$ scaling is essential — never omit it.
- Compute softmax in fp32 even when the rest of the forward pass is in fp16 (most frameworks do this automatically via `softmax_in_fp32`).
- **FlashAttention** rewrites the attention computation to avoid materializing the full $N \times N$ attention matrix, reducing both memory and numerical exposure.

---

## tf32 on NVIDIA Ampere and Later

TensorFloat-32 (tf32) is a format used internally by NVIDIA's Tensor Cores on Ampere (A100) and later GPUs. It has the exponent range of fp32 with the mantissa precision of fp16. It is applied transparently to fp32 matrix multiplications and convolutions — no code changes required.

The effect: fp32 training on Ampere hardware is ~8× faster than on Volta without any precision loss visible in practice. tf32 is enabled by default in PyTorch. There is no reason to disable it.

---

## Determinism

By default, GPU operations are non-deterministic. The same model, same data, and same seed will produce slightly different results across runs due to non-deterministic floating-point reduction order in parallel operations.

To enforce determinism (at a performance cost):

```python
torch.use_deterministic_algorithms(True)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
```

**When to use**: ablation studies where you need to isolate the effect of a single change. Running two identical configurations and comparing loss curves is only meaningful if the runs are deterministic.

**When not to use**: production training runs where throughput matters and minor non-determinism is acceptable. The performance penalty can be 10–30%.

---

## Precision Checklist by Hardware

| Hardware | Recommended format | Loss scaling needed |
| --- | --- | --- |
| NVIDIA Ampere / Hopper (A100, H100) | bf16 | No |
| NVIDIA Volta / Turing (V100, T4) | fp16 | Yes |
| NVIDIA older than Volta | fp32 | No |
| Google TPU v3 / v4 | bf16 | No |
| AMD Instinct MI250 / MI300 | bf16 | No |
| Apple Silicon (MPS) | fp32 / bf16 (check support) | No |

---

## Common Mistakes

- **Using fp16 on Ampere hardware instead of bf16.** bf16 is strictly better on this hardware in almost every way — same throughput, more stable, no loss scaling bookkeeping.
- **Performing the optimizer step in fp16.** Small gradient updates round to zero. Always keep master weights and optimizer state in fp32.
- **Not casting softmax to fp32.** Attention logit overflow is silent in fp16 — the softmax outputs look plausible but shift the learned attention distribution.
- **Disabling dynamic loss scaling manually** to avoid skipped steps. Skipped steps are the safety valve; disabling it transfers the overflow into the weights themselves.
- **Assuming bf16 and fp32 loss curves are identical.** They are close but not numerically identical. When comparing runs, use the same precision throughout.

---

## Quick Decision Checklist

``` text
1. Ampere+ GPU or TPU → use bf16; no loss scaling required.
2. Volta/Turing GPU → use fp16 with dynamic loss scaling.
3. Keep optimizer state and master weights in fp32 regardless of forward-pass format.
4. Verify softmax is computed in fp32 (check framework defaults).
5. Monitor the loss scale value — a scale that stays near 1 signals instability.
6. Count skipped optimizer steps — occasional early skips are fine; persistent
   skipping means the LR or architecture needs attention.
7. For ablations requiring exact comparison: enable deterministic mode and
   accept the throughput cost.
```

---

*See also: [Gradient Clipping](gradient-clipping.md) — fp16 overflow and gradient spikes interact; clipping after loss-scaler unscaling is critical. [Selecting a Learning Rate](selecting-a-learning-rate.md) — a learning rate that is stable in fp32 may still overflow in fp16 without loss scaling.*
