# `Activation` — Source File Reference

- **Files:** [`src/Activation.hpp`](../../../../src/Activation.hpp), [`src/Activation.cpp`](../../../../src/Activation.cpp)
- **Library:** `adai_core` (listed in `src/CMakeLists.txt`'s `add_library(adai_core STATIC ...)`)
- **Status tag:** `@adai-status: stable`, `@adai-version: 1.0.0`, `@adai-reviewed: 2026-09-10`
- **Tests:** [`tests/activation_test.cpp`](../../../../tests/activation_test.cpp) → binary `activationTests` (ctest name `ActivationTests`)
- **Last traced against the code:** 2026-10-07

> This document is a code-traced reference: every "where it's used" claim below points at a real
> call site found in `src/` at the time of writing. It supersedes the older
> `docs/development/api/core/activation.md` (now deleted; "Activation Class - Technical Context
> Documentation"), whose still-accurate background material has been merged in here; see
> [§12](#12-merge-notes-what-changed-from-the-old-document) for what was corrected or dropped.

---

## Contents

1. [What this file is](#1-what-this-file-is)
2. [Conventions you must know before calling anything](#2-conventions-you-must-know-before-calling-anything)
3. [Private constants](#3-private-constants)
4. [Function-by-function reference](#4-function-by-function-reference)
5. [CPU vs. GPU: where the "real" hot path runs](#5-cpu-vs-gpu-where-the-real-hot-path-runs)
6. [Testing](#6-testing)
7. [Usage patterns (correct forward/backward pairings)](#7-usage-patterns-correct-forwardbackward-pairings)
8. [Choosing an activation](#8-choosing-an-activation)
9. [Best practices in this codebase](#9-best-practices-in-this-codebase)
10. [Possible future work](#10-possible-future-work)
11. [Known gaps and gotchas (summary)](#11-known-gaps-and-gotchas-summary)
12. [Merge notes](#12-merge-notes-what-changed-from-the-old-document)

---

## 1. What this file is

`Activation` is a stateless, static-only class holding the **CPU reference implementations** of
the non-linearities used by the transformer, plus their derivatives. Its shape is:

```cpp
class Activation {
   public:
    static Matrix some_activation(const Matrix& input);
    static Matrix some_activation_derivative(const Matrix& input_or_output);
    // ...
   private:
    static constexpr float GELU_COEF = 0.044715f;
    static constexpr float SQRT_2_OVER_PI = 0.7978845608f;
};
```

Every function:

- takes one or two `Matrix` arguments by `const&`,
- allocates and returns a **new** `Matrix` of the same shape (never modifies its input — there are
  no in-place variants),
- touches no global or member state,

so all of them are pure functions and safe to call concurrently from multiple threads. No
instance of `Activation` is ever created.

There are 14 public functions: 7 forward activations and 7 matching derivatives.

| Forward | Derivative | Used in production `src/`? |
|---|---|---|
| `softmax` | `softmax_derivative` | **Yes** — attention (MHA, CrossAttention), LM head. Derivative: tests only (see §4.2) |
| `gelu` | `gelu_derivative` | **Yes** — every `FeedForward` layer, forward and backward |
| `relu` | `relu_derivative` | Tests only |
| `leaky_relu` | `leaky_relu_derivative` | Tests only |
| `sigmoid` | `sigmoid_derivative` | Tests only (also called internally by `swish`/`swish_derivative`) |
| `tanh` | `tanh_derivative` | Tests only |
| `swish` | `swish_derivative` | Tests only |

So in practice the file's production importance rests on **two** functions — `softmax` and the
`gelu`/`gelu_derivative` pair — and everything else is a tested utility library kept available for
experimentation (e.g. swapping the FFN non-linearity) without needing new code.

### Why it matters

Without non-linearities, any stack of linear layers collapses into a single linear map, no matter
how deep it is:

```text
f(x) = W₃(W₂(W₁x)) = (W₃W₂W₁)x = Wx
```

With a non-linearity σ between them, `f(x) = W₃·σ(W₂·σ(W₁x))` can represent genuinely non-linear
functions — the universal approximation theorem says a network with at least one hidden layer, a
non-linear activation, and enough units can approximate any continuous function on a compact set.
Concretely in this codebase:

- **`gelu`** is the only non-linearity inside each `FeedForward` sub-layer — i.e. the sole source
  of per-token non-linear capacity in every encoder and decoder block on the CPU path.
- **`softmax`** turns raw `Q·Kᵀ` scores into attention weights for every self-attention and
  cross-attention head, and turns LM-head logits into a next-token probability distribution.

A numerical bug here would therefore silently corrupt *every* forward pass and *every* gradient on
the CPU path. That's why the file is tagged `stable` and why its tests include numerical gradient
checks rather than only spot values.

---

## 2. Conventions you must know before calling anything

### 2.1 What each derivative expects as input

The derivatives do **not** share one input convention. This is the single easiest way to misuse
the class:

| Derivative | Pass this | Why |
|---|---|---|
| `gelu_derivative(x)` | **pre-activation** input `x` | GELU′ can't be recovered from GELU(x) cheaply |
| `relu_derivative(x)` | **pre-activation** input `x` | needs the sign of `x` |
| `leaky_relu_derivative(x, α)` | **pre-activation** input `x` (and the **same α** as forward) | needs the sign of `x` |
| `swish_derivative(x)` | **pre-activation** input `x` | recomputes `sigmoid(x)` internally |
| `sigmoid_derivative(y)` | **output** `y = sigmoid(x)` | σ′ = y(1−y) is cheapest from the output; no need to keep `x` |
| `tanh_derivative(y)` | **output** `y = tanh(x)` | tanh′ = 1−y² is cheapest from the output; no need to keep `x` |
| `softmax_derivative(y, g)` | **output** `y = softmax(x)` **and** upstream gradient `g` | see below |

Passing a pre-activation value to `sigmoid_derivative`/`tanh_derivative` (or vice-versa) does not
throw — it just returns wrong gradients. The practical consequence is that a layer must cache
whichever of input/output its activation's derivative needs during the forward pass (see §9.1).

### 2.2 What each derivative returns

- The six **element-wise** derivatives return the local Jacobian diagonal `f′(x)`. The caller must
  still multiply it element-wise by the upstream gradient:
  `grad_in = grad_out.hadamard(Activation::gelu_derivative(x))` (exactly what `FeedForward::backward`
  does).
- **`softmax_derivative` is different**: because softmax's Jacobian is not diagonal, it returns the
  finished vector–Jacobian product — the gradient with respect to softmax's *input* — already
  combined with `grad_output`. Do **not** Hadamard it with `grad_output` again.

### 2.3 Row semantics

Only `softmax`/`softmax_derivative` are row-wise (each row is an independent distribution). All
other functions are purely element-wise and shape-agnostic.

### 2.4 Cost and allocation

Every call allocates a fresh `rows × cols` result. Nothing here is vectorized or threaded
explicitly; the hot-path GPU equivalents live in `src/gpu/` (§5).

| Function | Per-element work | Notes |
|---|---|---|
| `relu` / `relu_derivative` | 1 comparison | cheapest |
| `leaky_relu` / `_derivative` | 1 comparison + 1 multiply | |
| `sigmoid` | 1 `std::exp` + a branch + ~2 ops | |
| `sigmoid_derivative` / `tanh_derivative` | 2 ops on the cached output | no transcendental |
| `tanh` | 1 `std::tanh` | |
| `gelu` / `gelu_derivative` | 1 `std::tanh` + ~8–12 mults/adds | derivative reuses `tanh` for `sech²` |
| `swish` / `swish_derivative` | full `sigmoid` pass + 1–3 ops | allocates **two** matrices (calls `sigmoid` first) |
| `softmax` | 1 `std::exp` + 1 divide, plus a row max and row sum | 3 passes over each row |
| `softmax_derivative` | 2 multiplies + 1 subtract, plus a row dot product | O(cols) per row, not O(cols²) |

For an `m × n` input all of them are O(m·n).

---

## 3. Private constants

```cpp
static constexpr float GELU_COEF      = 0.044715f;      // cubic-term coefficient
static constexpr float SQRT_2_OVER_PI = 0.7978845608f;  // √(2/π)
```

These are the constants of the Hendrycks & Gimpel tanh approximation of GELU. The **same literal
values are hardcoded independently** in the CUDA kernels (`src/gpu/MatrixGPU.cu`, the `case 3`
branch of the element-wise activation device function and `gelu_backward_kernel`) and the SYCL
kernels (`src/gpu/sycl/`). They are not shared via a header — if you ever change the GELU variant
here (e.g. to exact `erf` GELU), you must change both GPU backends too, or CPU and GPU training
will silently diverge and checkpoints trained on one path will behave differently on the other.

---

## 4. Function-by-function reference

### 4.1 `softmax`

```cpp
static Matrix softmax(const Matrix& input);
```

**What it does.** For each row independently:

1. find the row max `m`,
2. compute `e_j = exp(x_j − m)` and their sum `S`,
3. divide: `y_j = e_j / S`.

```text
softmax(x)_j = exp(x_j − max(x)) / Σ_k exp(x_k − max(x))
```

Each output row is a probability distribution (non-negative, sums to 1).

**Why the max subtraction matters.** `exp(x)` overflows `float` at x ≈ 88.7. Attention scores and
LM logits routinely exceed that during early or unstable training. Subtracting the row max makes
the largest exponent exactly `exp(0) = 1`, so the sum is ≥ 1 and can never be zero or infinite; the
result is mathematically identical (softmax is invariant to adding a constant to a whole row).

**Properties.**
- Smooth and differentiable everywhere.
- Preserves ordering within a row (argmax of output = argmax of input), which is why greedy
  decoding can skip it entirely.
- Output values are in `[0, 1]` in `float` — mathematically `(0, 1)`, but entries far below the
  row max underflow to exactly `0.0f`. That's exactly how the `-1e9f` attention mask works.
- Saturates when one logit dominates: the output approaches one-hot and the gradient through it
  approaches zero.

**Where it's used (production):**

| Call site | Context |
|---|---|
| `src/MultiHeadAttention.cpp` — `MultiHeadAttention::forward_with_cache()` (≈ line 350) | Per-head attention weights during KV-cached incremental decoding: `weights_h = softmax(scale · Q_h·K_hᵀ + mask)`; weights are cached per head (`cached_head_weights_`) for `backward()` |
| `src/CrossAttention.cpp` — `CrossAttention::forward()` (≈ line 146) | Per-head decoder→encoder attention weights |
| `src/CrossAttention.cpp` — `CrossAttention::forward_with_scores()` (≈ line 261) | Same, after adding an external `score_bias` to each head's scores |
| `src/CrossAttention.cpp` — `CrossAttention::forward_with_cache()` (≈ line 388) | Same, for cached incremental decoding |
| `src/LanguageModelHead.cpp` — `LanguageModelHead::get_probabilities()` (≈ line 65) | Wraps a single logit vector into a `1 × vocab` matrix, softmaxes it, unwraps it. Currently only exercised by `tests/languagemodelhead_test.cpp` |

All four attention call sites follow the same TD-059 per-head pattern: slice each head's `d_k`
columns, compute `Q_h·K_hᵀ`, scale by `1/√d_k`, set masked positions to `-1e9f`, then call
`Activation::softmax`. The mask uses `-1e9f` rather than `-∞` deliberately: with `-∞`, a fully
masked row would compute `exp(-∞ − (-∞)) = exp(NaN)` and poison the whole row; with `-1e9f` the
max subtraction keeps everything finite.

**Where softmax is *not* routed through this function.** Several hot paths inline their own
identical max-subtracted softmax instead of calling `Activation::softmax`:

- `MultiHeadAttention::forward_parallel()` (the main, non-cached self-attention path) — in-place
  row loop around line 181, to avoid an extra allocation per head;
- `TextGenerator::softmax()` (`src/TextGenerator.cpp`) — operates on `std::vector<float>` for
  sampling;
- `EncoderDecoderModel.cpp` loss/gradient loops (≈ lines 123, 157) — fused with cross-entropy;
- `RLHFTrainer.cpp` policy-gradient loops (≈ lines 120, 220);
- every GPU path, via `GPUMatrix::softmax_rows_inplace()` (§5).

They are mathematically the same algorithm. If you ever change softmax semantics (e.g. add
temperature or a different masking convention), grep for `max_logit`/`max_score` as well as
`Activation::softmax`.

**Edge cases.**
- A matrix with `cols == 0` is undefined behaviour: the max-finding step reads `input(i, 0)`
  unconditionally.
- A row containing `NaN` produces an all-`NaN` row (no sanitizing).
- A row containing `+∞` produces `NaN` (`∞ − ∞`).

**Tests:** `ActivationSoftmaxTest.{BasicSoftmax, MultipleBatches, NumericalStability, UniformInput}`
check that rows sum to 1, rows are independent, large logits don't overflow, and equal logits give
a uniform distribution; `ActivationIntegrationTest.{ClassificationWithSoftmax,
AttentionScoresWithSoftmax}` exercise it in classifier- and attention-shaped scenarios.

---

### 4.2 `softmax_derivative`

```cpp
static Matrix softmax_derivative(const Matrix& output, const Matrix& grad_output);
```

**What it does.** Computes the vector–Jacobian product through a row-wise softmax. The full
Jacobian of one row is

```text
∂y_i/∂x_j = y_i · (δ_ij − y_j)          i.e.  J = diag(y) − y yᵀ
```

but materializing it would cost O(cols²) per row. Multiplying it by `g` collapses to:

```text
for each row i:
    s_i            = Σ_k  y(i,k) · g(i,k)
    grad_in(i, j)  = y(i,j) · ( g(i,j) − s_i )
```

where `y = output` (the softmax result) and `g = grad_output` — O(cols) per row.

> The header comment says "For cross-entropy loss". That's misleading: the formula is the
> **general** softmax backward and is correct for any upstream gradient, not only cross-entropy.
> (For softmax + cross-entropy specifically, the combined gradient simplifies further to
> `y − one_hot(target)`, which is what `EncoderDecoderModel.cpp` uses directly — see §7.2.)

**Why it's important.** Attention backward needs exactly this operation once per head per layer.

**Where it's used.** Only in `tests/activation_test.cpp`. The production attention backward passes
**inline the identical formula** rather than calling this function:

- `MultiHeadAttention::backward()` — `src/MultiHeadAttention.cpp` ≈ lines 437–446 (comment spells
  out the same formula);
- `CrossAttention::backward()` — same pattern (≈ line 471);
- GPU: `GPUMatrix::softmax_backward()` → `matrix_softmax_backward_gpu`.

This makes `softmax_derivative` the **reference oracle** for those inline loops: if you're
debugging an attention-gradient mismatch, comparing the inline result to
`Activation::softmax_derivative(weights_h, grad_weights_h)` is the quickest sanity check.

**Contract.** `output` and `grad_output` must have identical shapes; this is **not** checked — a
smaller `grad_output` reads out of bounds.

---

### 4.3 `gelu`

```cpp
static Matrix gelu(const Matrix& input);
```

**What it does.** Exact GELU is `x · Φ(x)`, where Φ is the standard normal CDF — i.e. the input
scaled by the probability that a standard Gaussian falls below it. Computing Φ needs `erf`, so this
implementation uses the tanh approximation, element-wise:

```text
GELU(x) ≈ 0.5 · x · (1 + tanh( √(2/π) · (x + 0.044715 · x³) ))
```

Measured against exact `erf`-based GELU over x ∈ [−8, 8], the approximation's maximum absolute
error is ≈ 4.7 × 10⁻⁴ — negligible for training. It's the same variant used by the original
BERT/GPT-2 code, which is why it's the de-facto transformer default.

**Shape.** Behaves like identity for large positive `x` and like 0 for large negative `x`, with a
smooth transition around 0 and a small negative dip: minimum ≈ −0.170 at x ≈ −0.75, so the output
range is ≈ (−0.17, ∞). Unlike ReLU it has no kink at 0 and no hard zero-gradient region, so units
don't "die".

| Property | GELU | ReLU |
|---|---|---|
| Smoothness | smooth everywhere | kink at 0 |
| Gradient for negative inputs | small but non-zero | exactly 0 |
| Per-element cost | 1 `tanh` + ~10 ops | 1 comparison |
| Typical use | transformer FFNs (this project) | CNNs, cheap baselines |

**Where it's used — the main production consumer of this file:**

`FeedForward::forward()` — `src/FeedForward.cpp` ≈ line 85:

```cpp
Matrix hidden = input * W1 (+ b1);        // [seq, d_ff]
cached_hidden = hidden;                   // pre-activation, kept for backward
Matrix hidden_activated = Activation::gelu(hidden);
cached_hidden_activated = hidden_activated;
if (activation_hook_) activation_hook_(cached_hidden_activated);   // TD-013
Matrix output = hidden_activated * W2 (+ b2);
```

Every `EncoderBlock` and `DecoderBlock` owns a `FeedForward`, so this runs once per block per
forward pass on the CPU path — in training (`ChatbotTrainer`/`IncrementalTrainer`) and in CPU
inference (`chatbot_api_server` when no GPU is available).

**Two things depend on its exact output:**

1. **Weight initialization.** `FeedForward`'s constructor (`src/FeedForward.cpp` ≈ lines 26–46)
   uses He-style init sized for GELU: `W1 ~ N(0, √(2/d_model))`, `W2 ~ N(0, √(2/d_ff))`, zero biases.
   Swapping the activation without revisiting the init would change the variance profile of every
   layer.
2. **Activation-saturation metrics (TD-013).** The post-GELU matrix is passed to
   `FeedForward::activation_hook_`, which `ChatbotTrainer` installs on every encoder and decoder
   block (`src/ChatbotTrainer.cpp` ≈ lines 1029–1033) to compute `activation_saturation_ratio`
   (fraction of near-zero units), reported through
   `TrainingMetricsService::update_activation_saturation()` to `metrics_api_server` and the ops
   dashboard. A rising ratio is an early warning that the FFN is going inactive.

**GPU counterpart:** `FeedForward::gpu_forward` calls
`apply_activation_inplace(ActivationType::GELU)` (`src/FeedForward.cpp` ≈ line 465), which runs the
same formula in a CUDA/SYCL kernel (§5).

**Tests:** `ActivationGELUTest.{BasicGELU, PositiveValues, Smooth, GELUDerivative}` — zero → zero,
positive ≈ identity / negative ≈ 0 asymptotics, smoothness, and a central finite-difference
gradient check against `gelu_derivative`.

---

### 4.4 `gelu_derivative`

```cpp
static Matrix gelu_derivative(const Matrix& input);   // input = PRE-activation x
```

**What it does.** Analytic derivative of the tanh approximation above, via the product and chain
rules. With `u = √(2/π)·(x + 0.044715·x³)`:

```text
du/dx    = √(2/π) · (1 + 3 · 0.044715 · x²)
GELU′(x) = 0.5 · (1 + tanh u)  +  0.5 · x · sech²(u) · du/dx
```

implemented with `sech²(u) = 1 − tanh²(u)` so only one `std::tanh` call is needed per element (the
local variable holding `du/dx` is confusingly named `tanh_derivative` in the source). Note this is
the derivative of the *approximation* the forward pass actually uses, not of exact GELU — which is
what you want, since gradients must match the function that was evaluated.

**Where it's used:** `FeedForward::backward()` — `src/FeedForward.cpp` ≈ line 129:

```cpp
Matrix grad_hidden_activated = grad_output * W2.transpose();
Matrix gelu_grad   = Activation::gelu_derivative(cached_hidden);   // pre-activation!
Matrix grad_hidden = grad_hidden_activated.hadamard(gelu_grad);
```

This is why `FeedForward::forward()` caches `cached_hidden` (pre-activation) in addition to
`cached_hidden_activated`: the GELU derivative can't be recovered from GELU's output.

**Why it matters.** Every gradient flowing back through an FFN sub-layer — and therefore into W1,
b1, and everything upstream (attention, embeddings) — passes through this function on the CPU
path. An error here wouldn't crash; it would make training quietly worse or diverge.

**GPU counterpart:** `GPUMatrix::gelu_backward()` → `gelu_backward_kernel` (CUDA,
`src/gpu/MatrixGPU.cu` ≈ line 652) and the SYCL equivalent; both comment the identical formula.

---

### 4.5 `relu` / `relu_derivative`

```cpp
static Matrix relu(const Matrix& input);              // max(0, x)
static Matrix relu_derivative(const Matrix& input);   // input = PRE-activation; 1 if x > 0 else 0
```

**Behaviour.** Standard rectifier. The derivative is mathematically undefined at `x == 0`; this
implementation defines it as **0** (the comparison is strict `> 0.0f`).

**Properties.**
- Cheapest possible non-linearity (one comparison); gradient is exactly 1 for positive inputs, so
  no vanishing through the active units.
- Produces sparse activations (many exact zeros).
- Not zero-centered, unbounded above, non-smooth at 0.
- **Dead neurons:** a unit whose pre-activation is negative for every input gets zero gradient
  forever and stops learning. This is the main reason the transformer uses GELU instead.

**Where it's used.** Tests only. A `RELU` variant also exists in the GPU `ActivationType` enum but
no production layer selects it.

**Why keep it.** It's the cheapest baseline non-linearity for experiments (e.g. comparing FFN
activations), and its tests (`ReLUSparsity`, `GELUvsReLU`) document the behaviour that motivated
GELU.

---

### 4.6 `leaky_relu` / `leaky_relu_derivative`

```cpp
static Matrix leaky_relu(const Matrix& input, float alpha = 0.01f);
static Matrix leaky_relu_derivative(const Matrix& input, float alpha = 0.01f);   // PRE-activation
```

**Behaviour.** `x` for `x > 0`, otherwise `alpha·x`. Derivative is `1` for `x > 0`, otherwise
`alpha`. Always pass the **same `alpha`** to both functions — nothing ties them together.

Common `alpha` values: `0.01` (default, very small leak), `0.1` (more aggressive), `0.2`–`0.25`
(typical starting values when α is learned, as in PReLU — not supported here; α is a fixed
argument).

**Properties.** Nearly as cheap as ReLU, and the non-zero negative slope means units can't die. The
cost is one extra hyperparameter.

> Small doc/code mismatch: the header documents the derivative as "alpha if x < 0, else 1", which
> would give 1 at `x == 0`. The implementation uses `x > 0 ? 1 : alpha`, so at exactly 0 it
> returns **alpha**. Either convention is valid (it's a subgradient point), but the tests encode
> the implementation's behaviour.

**Where it's used.** Tests only (including `NoDeadNeurons` and `LeakyReLUvsReLU`, which show it
keeps gradient alive on negative inputs). No GPU equivalent exists.

---

### 4.7 `sigmoid` / `sigmoid_derivative`

```cpp
static Matrix sigmoid(const Matrix& input);
static Matrix sigmoid_derivative(const Matrix& output);   // output = sigmoid(x), NOT x
```

**Behaviour.** `σ(x) = 1 / (1 + e^{−x})`, squashing to `(0, 1)`, implemented in the numerically
stable branched form:

- `x ≥ 0`: `1 / (1 + exp(−x))` — `exp(−x) ≤ 1`, no overflow;
- `x < 0`: `exp(x) / (1 + exp(x))` — `exp(x) < 1`, no overflow.

The naive form would compute `exp(−x)` for very negative `x` → `+∞`, which happens to give the
right limit (0) in IEEE floats, but the branched form avoids the `∞` intermediate entirely and
keeps precision for large-magnitude inputs.

The derivative uses `σ′ = σ(1 − σ)` from the already-computed output, so a caller needs to keep
only the output, not the input.

**Gradient characteristics.** σ′ is always positive, peaks at **0.25** at x = 0, and is
effectively 0 for |x| ≳ 4–5. Stacking sigmoids multiplies factors ≤ 0.25, which is the classic
vanishing-gradient problem; it's also not zero-centered. That's why it's only appropriate for gates
and binary outputs, not hidden layers.

**Where it's used.** Tests, and internally by `swish`/`swish_derivative`. Not called by any
production layer. (The GPU `SIGMOID` activation kernel uses the naive one-branch formula — fine
in practice for the same IEEE reason above, but not bit-identical in the extreme tails.)

**Note on gating elsewhere.** `DecoderBlock.cpp` computes its own world-model/hippocampal gates
with raw `std::tanh` rather than through this class (see `gate_tanh`/`gate_h_tanh`), so changes
here don't affect those gates.

---

### 4.8 `tanh` / `tanh_derivative`

```cpp
static Matrix tanh(const Matrix& input);              // std::tanh element-wise
static Matrix tanh_derivative(const Matrix& output);  // output = tanh(x), NOT x
```

**Behaviour.** Thin element-wise wrapper over `std::tanh`; output in `(−1, 1)`. Derivative
`1 − y²` from the output.

```text
tanh(x) = (eˣ − e⁻ˣ) / (eˣ + e⁻ˣ) = 2·σ(2x) − 1
```

**Gradient characteristics.** Zero-centered (unlike sigmoid), and its gradient peaks at **1.0** at
x = 0 (4× sigmoid's), but it still saturates to ≈ 0 for |x| ≳ 3, so deep tanh stacks also suffer
vanishing gradients.

| | `tanh` | `sigmoid` |
|---|---|---|
| Output range | (−1, 1) | (0, 1) |
| Zero-centered | yes | no |
| Max gradient | 1.0 | 0.25 |

**Where it's used.** Tests only. Inside this class, GELU calls `std::tanh` directly rather than
`Activation::tanh` (avoids allocating a matrix per element).

---

### 4.9 `swish` / `swish_derivative`

```cpp
static Matrix swish(const Matrix& input);             // x · σ(x)   (a.k.a. SiLU)
static Matrix swish_derivative(const Matrix& input);  // PRE-activation x
```

**Behaviour.** `swish(x) = x·σ(x)`; derivative

```text
swish′(x) = σ(x) + x·σ(x)·(1 − σ(x))  =  σ(x) · (1 + x·(1 − σ(x)))
```

(the first term from σ, the second from the product rule). Both functions call
`Activation::sigmoid(input)` first, so they inherit its numerical stability and cost one extra
matrix allocation.

**Shape.** Self-gated: the input gates itself through its own sigmoid. Smooth and non-monotonic,
like GELU — ≈ identity for large positive x, → 0 for large negative x, with a minimum of
≈ −0.278 at x ≈ −1.278. Swish was popularized by an automated activation-function search
(Ramachandran et al., 2017); the same function had earlier been named SiLU (sigmoid-weighted
linear unit).

**Where it's used.** Tests only. It's the natural building block if the FFN is ever moved to a
SwiGLU-style gated variant (common in modern LLMs), which is the main reason to keep it here.

---

## 5. CPU vs. GPU: where the "real" hot path runs

When `GPUManager::is_available()` and the model is on the GPU path, **none of the functions in
this file run** — the GPU layers use device kernels exposed through `src/gpu/MatrixGPU.hpp`
(CUDA) or `src/gpu/sycl/MatrixGPU_SYCL.hpp` (SYCL):

| `Activation` (CPU) | GPU equivalent | Used by |
|---|---|---|
| `gelu` | `GPUMatrix::apply_activation_inplace(ActivationType::GELU)` → `matrix_apply_activation_gpu` | `FeedForward::gpu_forward` |
| `gelu_derivative` (+ Hadamard) | `GPUMatrix::gelu_backward(dout)` → `matrix_gelu_backward_gpu` (fused with the upstream multiply) | `FeedForward::gpu_backward` |
| `softmax` | `GPUMatrix::softmax_rows_inplace()` → `matrix_softmax_rows_gpu` | `MultiHeadAttention::gpu_forward*`, `CrossAttention::gpu_forward*` |
| `softmax_derivative` | `GPUMatrix::softmax_backward(dout)` → `matrix_softmax_backward_gpu` | `MultiHeadAttention`/`CrossAttention::gpu_backward` |
| `relu`/`sigmoid`/`tanh` | `ActivationType::RELU/SIGMOID/TANH` | (no production caller) |

Two important consequences:

1. **This file is the specification the GPU kernels must match.** The CPU path is what the unit
   tests check with finite differences; the GPU kernels were written to reproduce it. When
   CPU-vs-GPU results disagree, start by checking constants and formulas against this file.
2. **CPU performance work here only helps CPU-only hosts** (or the CPU fallback in
   `chatbot_api_server`). Training on `ai-machine` runs the SYCL kernels instead.

---

## 6. Testing

`tests/activation_test.cpp` (40 `TEST` cases, built as `activationTests`, linked against
`adai_core` + `GTest::gtest_main`):

```bash
cmake --build --preset=debug --target activationTests
./build/debug/tests/activationTests
./build/debug/tests/activationTests --gtest_filter="*GELU*"
```

Highlights:

- **Numerical gradient checks** for `gelu`, `sigmoid`, `tanh`, `swish`: a central finite
  difference is compared to the analytic derivative — the strongest guarantee in the suite that
  forward/derivative pairs stay consistent. The helper at the top of the file:

  ```cpp
  float numerical_derivative(Matrix& input, int i, int j,
                             std::function<Matrix(const Matrix&)> activation) {
      float epsilon = 1e-5f;
      float orig = input(i, j);
      input(i, j) = orig + epsilon;  Matrix out_plus  = activation(input);
      input(i, j) = orig - epsilon;  Matrix out_minus = activation(input);
      input(i, j) = orig;
      return (out_plus(i, j) - out_minus(i, j)) / (2.0f * epsilon);
  }
  ```

  Tolerances are deliberately loose because `ε = 1e-5` in `float` is close to the precision floor:
  `5e-3` for sigmoid and tanh, `1e-2` for GELU and swish. `GELUDerivative` also uses **fixed**
  inputs in [−1, 1] instead of random ones — larger random values occasionally exceeded the
  tolerance on CI.
- Value/limit tests for every function (`Basic*`, `ExtremeValues`, `AllNegative`/`AllPositive`,
  `DifferentAlpha`, `NoDeadNeurons`, `SelfGating`).
- Softmax distribution properties and large-value stability (`ActivationSoftmaxTest.*`).
- Scenario tests mimicking a FFN forward/backward (`ForwardBackwardGELU`) and an attention-weight
  computation (`AttentionScoresWithSoftmax`), plus side-by-side comparisons
  (`ActivationComparisonTest.*`) and edge cases (`SingleElement`, `ZeroInput`, `LargeMatrix`).

`tests/feedforward_test.cpp`, `tests/multiheadattention_test.cpp`,
`tests/languagemodelhead_test.cpp`, and `tests/integration_test.cpp` also include this header and
exercise it indirectly through the layers above. `tests/test_base.hpp` includes it so every test
built on the shared base has it available. `src/encoder.hpp` includes it as part of its umbrella
include set (it does not call it directly).

There are no performance benchmarks for this file.

---

## 7. Usage patterns (correct forward/backward pairings)

### 7.1 Element-wise activation inside a layer (the `FeedForward` pattern)

```cpp
// forward
cached_pre = x * W + b;                         // keep what the derivative needs
Matrix y   = Activation::gelu(cached_pre);

// backward
Matrix grad_pre = grad_y.hadamard(Activation::gelu_derivative(cached_pre));
Matrix grad_W   = cached_input.transpose() * grad_pre;
Matrix grad_x   = grad_pre * W.transpose();
```

For `sigmoid`/`tanh`, cache `y` instead of `cached_pre` and pass `y` to the derivative (§2.1).

### 7.2 Softmax + cross-entropy: do **not** chain `softmax_derivative`

With cross-entropy loss `L = −log y_target`, the gradient with respect to the **logits** is simply

```text
∂L/∂logits = softmax(logits) − one_hot(target)
```

That expression *already includes* the softmax backward. Applying `softmax_derivative` to it again
would differentiate through softmax twice and produce wrong gradients. This is how
`EncoderDecoderModel.cpp` does it (≈ line 174: "Gradient: softmax - one_hot(target)").

Use `softmax_derivative(y, g)` only when `g` is the gradient with respect to softmax's **output** —
e.g. attention weights, where `g = grad_out · Vᵀ` comes from the subsequent `weights · V` product.

### 7.3 Sigmoid + binary cross-entropy: same rule

For BCE on `y = σ(z)`, `∂L/∂z = y − target` already includes the sigmoid derivative. Use
`sigmoid_derivative(y).hadamard(g)` only when `g` is a gradient with respect to `y` itself (e.g. a
sigmoid gate whose output feeds further computation).

### 7.4 Attention weights

See the TD-059 per-head pattern in §4.1. The project's attention layers do **not** softmax one
`d_model`-wide score matrix; each head's `[*, d_k]` slice is scored, scaled, masked, and softmaxed
independently.

---

## 8. Choosing an activation

General guidance (with what this project actually does in **bold**):

| Role | Typical choice | Notes |
|---|---|---|
| Transformer FFN hidden layer | **GELU** | Smooth, no dead units; what every `FeedForward` uses |
| Gated FFN variant (SwiGLU) | Swish | Not implemented; `swish` is the building block |
| Cheap baseline / CNN-style hidden layer | ReLU, Leaky ReLU | Leaky if dead units are a problem |
| Gate (0–1 mixing weight) | Sigmoid | `DecoderBlock` uses raw `std::tanh` for its own gates instead |
| Zero-centered bounded output | Tanh | |
| Multi-class distribution (attention, next token) | **Softmax** | |
| Binary probability output | Sigmoid | |
| Unbounded regression output | none (linear) | |

Gradient behaviour at a glance:

| Activation | Can units die? | Vanishing-gradient risk |
|---|---|---|
| ReLU | yes (zero gradient for x ≤ 0) | low for active units |
| Leaky ReLU | no (α gradient) | low |
| GELU, Swish | no (smooth, non-zero almost everywhere) | low |
| Sigmoid | no, but max gradient 0.25 | high in deep stacks |
| Tanh | no, max gradient 1.0 | moderate–high in deep stacks |

Historically the field moved sigmoid → tanh → ReLU (≈2011) → Leaky ReLU (≈2013) → GELU/Swish
(2016–2017), each step mostly addressing gradient flow.

---

## 9. Best practices in this codebase

### 9.1 Cache what the derivative needs during forward

`FeedForward` keeps `cached_input` (for `W1`'s gradient), `cached_hidden` (pre-GELU, for
`gelu_derivative`), and `cached_hidden_activated` (post-GELU, for `W2`'s gradient). A layer using
sigmoid/tanh would cache the output instead. Get this wrong and the gradients are silently wrong
(§2.1).

### 9.2 Gradient clipping is norm-based, not per-value

Clipping happens after backward, outside this file: each layer exposes
`get_gradient_norm()`/`clip_gradients(float max_norm)` (e.g. `FeedForward`, `MultiHeadAttention`),
and `ChatbotTrainer` drives it from `gradient_clip_norm` (default 1.0) with an optional adaptive
mode (`adaptive_gradient_clip`, bounded by `gradient_clip_min`/`gradient_clip_max`). Don't add
element-wise value clipping inside activation code — it would distort gradient directions.

### 9.3 Monitor activations through the existing hook, not ad-hoc prints

Use `FeedForward::set_activation_hook()` (TD-013), which already feeds
`activation_saturation_ratio` into the metrics pipeline (§4.3). If you need extra diagnostics, log
through `adai::Logger::warn/info` — library code must never write to `std::cout`/`std::cerr`.

### 9.4 Initialize for the activation you use

He initialization (`std = √(2/fan_in)`) suits ReLU-family and GELU; Xavier
(`std = √(2/(fan_in + fan_out))`) suits tanh/sigmoid. `FeedForward` uses He for both weight
matrices (§4.3). If the FFN activation is ever changed, revisit its constructor too.

---

## 10. Possible future work

None of these are tracked as TD items; they're options, not commitments.

- **More activations:** Mish (`x·tanh(softplus(x))`), ELU, Softplus (`log(1 + eˣ)`), hard
  sigmoid/tanh approximations, and GLU-family gated units (SwiGLU/GeGLU — would need a
  `FeedForward` change, not just a new function here).
- **CPU performance:** SIMD vectorization of the element-wise loops; in-place variants to avoid
  the per-call allocation (MHA's `forward_parallel` already inlines an in-place softmax for this
  reason); fused activation + derivative. The GPU `gelu_backward` kernel is already fused with
  the upstream multiply.
- **Robustness:** an explicit shape check in `softmax_derivative`, a `cols == 0` guard in
  `softmax`, and fixing the two misleading header comments (§11).
- **Single source for GELU constants** shared by the CPU, CUDA, and SYCL code (§3).
- **Benchmarks:** no performance benchmarks exist for this file.

(Batch-norm fusion, sometimes listed as an activation optimization, isn't applicable here — the
transformer uses `LayerNorm`, which sits before the sub-layer, not between the linear map and the
activation.)

---

## 11. Known gaps and gotchas (summary)

Items with a TD are tracked in [TECHNICAL_DEBT.md](../../guides/TECHNICAL_DEBT.md) and tagged in the code with
`TODO: See TD-NNN`.

| Item | Impact | Notes | Tracked as |
|---|---|---|---|
| Inconsistent derivative input convention (§2.1) | Silent wrong gradients if misused | By design (cheapest form for each); documented per function | — (by design, not filed) |
| `softmax_derivative` header says "for cross-entropy" | Misleading doc | Formula is the general VJP; for CE use `y − one_hot` directly (§7.2) | [TD-215](../../guides/TECHNICAL_DEBT.md#td-215-activation-has-misleading-header-comments-and-unchecked-inputs) |
| `leaky_relu_derivative` header vs. code at `x == 0` | Doc mismatch only | Code returns `alpha` at 0 | [TD-215](../../guides/TECHNICAL_DEBT.md#td-215-activation-has-misleading-header-comments-and-unchecked-inputs) |
| `softmax` with `cols == 0` | UB (reads `input(i,0)`) | No production caller passes empty rows | [TD-215](../../guides/TECHNICAL_DEBT.md#td-215-activation-has-misleading-header-comments-and-unchecked-inputs) |
| No shape check in `softmax_derivative` | OOB read on mismatched shapes | Callers must pass equal shapes | [TD-215](../../guides/TECHNICAL_DEBT.md#td-215-activation-has-misleading-header-comments-and-unchecked-inputs) |
| Softmax logic duplicated inline (MHA `forward_parallel`, `TextGenerator`, `EncoderDecoderModel`, `RLHFTrainer`, attention backward, GPU) | Changes must be made in several places | Grep `max_logit`/`max_score` too | [TD-216](../../guides/TECHNICAL_DEBT.md#td-216-softmax-and-gelu-math-is-duplicated-across-cpu-and-gpu-code) |
| GELU constants duplicated in CUDA and SYCL kernels | CPU/GPU drift if only one is changed | No shared header | [TD-216](../../guides/TECHNICAL_DEBT.md#td-216-softmax-and-gelu-math-is-duplicated-across-cpu-and-gpu-code) |
| Every call allocates a new `Matrix` | Extra allocation per layer per step on CPU | Why MHA's hot path inlines softmax | — (context for [TD-216](../../guides/TECHNICAL_DEBT.md#td-216-softmax-and-gelu-math-is-duplicated-across-cpu-and-gpu-code)) |

---

## 12. Merge notes: what changed from the old document

This file absorbed `docs/development/api/core/activation.md`. Where the two disagreed, this file's
code-traced details won. Specifically:

**Corrected (old content was wrong or out of date):**
- *"Classification Output Layer" / "Binary Classification" examples* computed `probs − targets`
  and then passed it through `softmax_derivative`/`sigmoid_derivative`, which applies the
  activation's derivative twice. Replaced by the correct pairings in §7.2–7.3.
- *Softmax derivative described as "optimized for cross-entropy"* — it's the general VJP (§4.2).
- *"Integration with Encoder — Current Usage in LLMEncoder (from encoder.cpp)"* — the FFN lives in
  `src/FeedForward.cpp`, not `encoder.cpp`; the real code is quoted in §4.3–4.4.
- *Attention example softmaxing one full-width score matrix* — pre-TD-059; attention is now
  genuinely per-head (§4.1, §7.4).
- *Gradient-check example asserting `error < 1e-4f`* with a `std::function<Matrix(Matrix)>`
  helper — the real helper takes `const Matrix&`, and the real tolerances are `5e-3`/`1e-2` (§6).
- *Sigmoid's negative branch "prevents underflow"* — it prevents an `exp` overflow (§4.7).
- *"GELU — bounded coefficients / fixed-point friendly constants"* as a stability feature — not
  meaningful; dropped. The real concern with those constants is CPU/GPU duplication (§3).
- *"Tanh approximation is 3–5× faster, error < 0.1%"* — unverified speed claim dropped; error
  replaced with a measured max absolute error of ≈ 4.7 × 10⁻⁴ (§4.3).
- *Softmax output range "(0, 1)"* — `[0, 1]` in `float` (§4.1).
- *"Testing Additions" listed as future work* — unit tests, gradient checks, and stability tests
  all exist now; only benchmarks remain (§6, §10).
- *Best-practice examples* using per-value gradient clipping and `std::cerr` warnings — replaced
  with the project's actual norm-based clipping and TD-013 hook / `adai::Logger` (§9).
- *Summary claims "Used throughout LLMEncoder"* — replaced by the per-function usage table (§1).

**Dropped (illustrative only, no counterpart in this codebase):** the LSTM gates example, the
star-rated performance table (replaced by the concrete per-element cost table in §2.4), and the
CNN/EfficientNet use-case lists.

**Kept and merged:** exact GELU definition, GELU/ReLU comparison, shape facts (GELU and Swish
minima — re-verified numerically), ReLU/Leaky ReLU properties and α values, sigmoid/tanh gradient
characteristics and comparison table, swish derivative factorization and history, full softmax
Jacobian, non-linearity necessity / universal approximation background, selection guide, and the
future-enhancement list (updated in §10).
