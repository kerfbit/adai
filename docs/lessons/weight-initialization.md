# Weight Initialization

*ADAI Training — Single-Point Lesson*

---

## The Core Idea

Before training begins, every parameter must be assigned a starting value. These initial values determine how signals propagate through the network at step zero — and a poor start can waste the entire warmup period recovering from a broken signal regime, or cause training to never converge at all.

The goal of weight initialization is to ensure that **activations and gradients are neither vanishingly small nor explosively large** as they pass through each layer at the start of training.

---

## Why Initialization Matters

Consider a network of $L$ layers. If each layer scales its input by a factor $c$, the output magnitude scales as $c^L$. For $L = 24$ layers:

- $c = 0.9 \Rightarrow 0.9^{24} \approx 0.08$ — activations vanish
- $c = 1.1 \Rightarrow 1.1^{24} \approx 9.8$ — activations explode
- $c = 1.0 \Rightarrow 1.0^{24} = 1.0$ — activations are stable

Good initialization keeps $c \approx 1$ at every layer, so gradients flow cleanly from the output back to the input without vanishing or exploding before the optimizer ever takes a step.

---

## Xavier / Glorot Initialization

Designed for layers followed by symmetric activations (tanh, no activation). Draws weights from a distribution scaled to preserve variance across layers:

$$W \sim \mathcal{U}\left[-\sqrt{\frac{6}{n_{in} + n_{out}}},\ \sqrt{\frac{6}{n_{in} + n_{out}}}\right]$$

or equivalently as a normal:

$$W \sim \mathcal{N}\left(0,\ \frac{2}{n_{in} + n_{out}}\right)$$

where $n_{in}$ is the number of input features and $n_{out}$ the number of output features for the layer.

**Use for**: linear projection layers without a ReLU activation following them — in transformers, this applies to query/key/value projection matrices and the output projection.

---

## He / Kaiming Initialization

Designed for layers followed by ReLU (and its variants: LeakyReLU, GELU). ReLU zeros half its inputs on average, halving the effective variance. He initialization compensates:

$$W \sim \mathcal{N}\left(0,\ \frac{2}{n_{in}}\right)$$

**Use for**: any linear layer whose output passes through a GELU or ReLU — in transformers, this is the feed-forward network's first linear layer.

---

## Transformer-Specific Conventions

Standard Xavier/He are a starting point, but transformers have additional conventions that matter at scale.

### Embedding Layer

Embeddings are typically initialized from a narrow normal distribution:

$$E \sim \mathcal{N}(0,\ \sigma^2), \quad \sigma = \frac{1}{\sqrt{d_{model}}}$$

A smaller standard deviation prevents embeddings from dominating early activations before the model has learned meaningful representations.

### Output Projection (Residual Streams)

In a transformer with $L$ layers, the residual stream accumulates contributions from all layers. Without correction, the variance of the residual stream grows as $O(L)$. The standard fix (GPT-2 and onward) scales the output projection of each sub-layer:

$$W_{out} \sim \mathcal{N}\left(0,\ \frac{\sigma^2}{2L}\right)$$

This is sometimes called the **$1/\sqrt{2L}$ initialization**. It keeps the residual stream variance stable at initialization regardless of depth.

### Bias Terms

Set all bias vectors to **zero** at initialization. Non-zero biases introduce asymmetries before training has a chance to set them correctly.

### Layer Norm Parameters

Scale parameter $\gamma = 1$, shift parameter $\beta = 0$. This is the identity transform at initialization, ensuring layer norm does not distort activations before training begins.

---

## What Happens With Bad Initialization

| Symptom | Likely Cause |
| --- | --- |
| Loss is NaN or very large at step 0 | Weights too large; activations overflow |
| Loss fails to decrease despite correct LR | Activations near zero; gradients vanish |
| Training eventually converges but slowly | Suboptimal init; warmup doing recovery work |
| Loss spikes after the first few steps | Init too large; interacts badly with a high LR during warmup |
| Attention weights are all uniform at step 0 | Normal — this is expected; attention learns to differentiate |

---

## Pre-Trained Weights: Fine-Tuning

When fine-tuning a pre-trained checkpoint, initialization is not a concern for existing weights — they are loaded directly. The only initialization decisions are for **new parameters** added to the architecture (e.g., adapter layers, task-specific heads, newly added token embeddings).

New parameters added to a pre-trained model should be initialized to produce **near-zero output** so they do not disturb the pre-trained signal at the start of fine-tuning. For adapter layers, this typically means initializing the down-projection to Xavier normal and the up-projection to zero, so the adapter is a no-op at step 0.

---

## Reproducibility

Initialization is stochastic. For reproducible training:

``` python
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
numpy.random.seed(seed)
```

Set seeds before constructing the model and before constructing the data loader. Different seeds produce different initializations; results will vary slightly. Report the seed in any experiment log.

---

## Common Mistakes

- **Using default PyTorch initialization without review.** PyTorch's default for `nn.Linear` is Kaiming uniform, which is reasonable for ReLU networks but is not scaled for the residual stream in transformers. Check what defaults your framework applies.
- **Forgetting the residual stream scaling** in deep transformers. Above ~12 layers, unscaled output projections cause the residual variance to grow noticeably and degrade early training stability.
- **Initializing new adapter weights too large** during fine-tuning. This corrupts the pre-trained representations in the first optimizer steps.
- **Using the same seed for both model initialization and data shuffling.** If both use the same RNG, changing the model architecture also changes the data order, making ablations uninterpretable.

---

## Quick Decision Checklist

``` text
1. Linear layers before GELU/ReLU → He (Kaiming) normal.
2. Linear layers without activation (Q, K, V, output projections) → Xavier normal.
3. Embedding layers → N(0, 1/sqrt(d_model)).
4. Residual output projections → scale by 1/sqrt(2L).
5. All biases → zero.
6. Layer norm: γ = 1, β = 0.
7. New layers added to a pre-trained model → initialize to produce zero output.
8. Record the random seed in the experiment log.
```

---

*See also: [Selecting a Learning Rate](selecting-a-learning-rate.md) — warmup partially compensates for imperfect initialization, but cannot fix a severely broken starting point. [Gradient Clipping](gradient-clipping.md) — initialization errors that produce large activations will also produce large gradients at step 0.*
