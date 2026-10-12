# `FeedForward` — Source File Reference

- **Files:** [`src/FeedForward.hpp`](../../../../src/FeedForward.hpp), [`src/FeedForward.cpp`](../../../../src/FeedForward.cpp)
- **Namespace:** global (no namespace)
- **Built into:** `adai_feedforward`; used by `EncoderBlock`, `DecoderBlock` and the LeJEPA `Predictor`
- **Status tag:** `@adai-status: stable`, `@adai-version: 1.0.1`, `@adai-reviewed: 2026-09-18` (both files)
- **Tests:** [`tests/feedforward_test.cpp`](../../../../tests/feedforward_test.cpp) → `feedforwardTests` (62 tests, 4 of them GPU-only)
- **Last traced against the code:** 2026-10-12

> Code-traced reference. Behaviour marked **verified** was confirmed by running the real class
> (linked against the debug libraries) in a scratch program.

---

## 1. What this file is

The **position-wise feed-forward network** of every transformer layer: two linear maps with a GELU
between them, applied to each row (token) independently.

```text
x [seq × d_model] ─→ x·W1 + b1 ─→ GELU ─→ ·W2 + b2 ─→ [seq × d_model]
                     [d_model × d_ff]      [d_ff × d_model]
```

GELU is the tanh approximation from [`Activation`](Activation.md) on the CPU, and the backend's own
kernel on the GPU (the CPU/GPU formula duplication is TD-216). There's no dropout, no gating (no
SwiGLU), and no LoRA on the FFN (`EncoderDecoderModel`'s LoRA covers attention only).

### Why it matters

It holds about two-thirds of each block's parameters (`2·d_model·d_ff + d_ff + d_model`; with
`d_ff = 4·d_model`, that's 8d² against attention's 4d²). It appears once in every encoder block,
once in every decoder block, and as the whole of the LeJEPA world model's `Predictor`.

---

## 2. API

### Construction

| Member | Behaviour |
|---|---|
| `FeedForward(d_model, d_ff)` | `W1 ~ N(0, √(2/d_model))`, `W2 ~ N(0, √(2/d_ff))` (He initialization), biases zero, gradients zero. **No argument checks**: `FeedForward(0, 4)` constructs with an infinite scale (verified). `Predictor` validates its own arguments first; the blocks don't |
| Initialization RNG | A fresh `std::mt19937` seeded from **`std::random_device`** in every constructor. Two instances get different weights (verified), and no seed can be set: there's no global seed anywhere in `src/` (`Matrix`'s random init also uses `random_device`). Training runs and bug reproductions therefore can't be replayed from initialization |
| `learning_rate` | Public, 0.001. Used only by the no-optimizer path. `EncoderBlock`/`DecoderBlock` overwrite it with their own rate before every `update_weights()` (TD-357) |

**Output scale (verified).** With unit-variance input (what a `LayerNorm` produces), the freshly
initialized FFN's output has a standard deviation of about **1.3–1.4** (1.38 at d=64, 1.32 at d=256,
1.37 at d=512, with `d_ff = 4·d_model`). In a Pre-LN stack each block adds this to the residual stream
unscaled, so the stream's variance grows with depth from the first step. GPT-2-style models scale
residual-branch output weights by `1/√(2·num_layers)` to keep this in check. This is relevant to the
open question of residual growth in Pre-LN training on ai-machine, though it isn't a bug.

### Forward and backward

| Method | Behaviour |
|---|---|
| `forward(input)` | Throws `invalid_argument` unless `input.cols == d_model`. **Always** caches `input`, the pre-GELU hidden layer and the post-GELU activation (`2·seq·d_ff + seq·d_model` floats) for `backward()`, including at inference. Then fires `activation_hook_` with the post-GELU matrix |
| `backward(grad_output)` | Checks the shape against the last `forward()`. **Accumulates** into `W1_grad`/`W2_grad`/`b1_grad`/`b2_grad` (TD-195) and returns ∂L/∂input. **Verified correct:** input gradient within 2.3×10⁻⁴ relative error of finite differences, and one element of each of `W1`, `W2`, `b1`, `b2` within 6.5×10⁻⁵. Calling it before any `forward()` throws a misleading *"Matrix dimensions must match for addition"* (the empty cache gives the wrong shape) rather than a clear "no forward pass" message |

### Gradients and optimization

| Method | Behaviour |
|---|---|
| `set_optimizer(opt)` | Stores the pointer and calls `register_parameters()`, which adds four parameter groups. Calling it twice registers them twice (TD-297). `nullptr` reverts to plain gradient descent, but the groups stay registered in the old optimizer |
| `update_weights()` | With an optimizer: **`optimizer->step()`**, which steps every group in that (shared) optimizer, not just this layer's (TD-297). Without one: `W -= learning_rate · grad` (verified to descend), biases by hand. Either way, it then zeroes the gradients |
| `zero_grad()` | Zeroes all four gradient matrices |
| `get_gradient_norm()` / `clip_gradients(max)` | L2 norm over all four; clipping scales all four uniformly (correct, unlike `EncoderBlock`'s, which skips `LayerNorm`, TD-356). Only `EncoderBlock::clip_gradients()` calls it, and that has no caller |

### Activation hooks (training diagnostics, TD-013)

| Member | Behaviour |
|---|---|
| `set_activation_hook(fn)` / `clear_activation_hook()` | CPU: `fn(const Matrix&)` receives the post-GELU activations on every `forward()`. `ChatbotTrainer` installs a saturation hook (the fraction of `|a| < 0.01`) on every block's FFN for the epoch and clears it afterwards |
| `set_gpu_activation_stats_hook(fn)` / `clear_…` (GPU) | GPU: `fn(float)` receives the same fraction, computed on the device by `GPUMatrix::count_below_threshold(0.01f)`. That reduction returns to the host, so it **synchronizes the queue once per FFN per forward** while the hook is installed |

### Persistence

| Method | Behaviour |
|---|---|
| `save_weights(file)` | Two ints (`d_model`, `d_ff`), then `W1`, `W2`, `b1`, `b2` as raw floats, one `write()` per float. Write errors aren't checked. Prints to **`std::cout`** |
| `load_weights(file)` | Throws on a dimension mismatch. **Reads aren't checked**: a file cut off after 10 floats loaded without an exception, filled the start of `W1` and **silently kept the old values for everything else** (verified). Prints to `std::cout`. Same class as TD-298/TD-345/TD-357 |
| SafeTensors accessors | `get_W1()`…`get_b2()` and `set_W1()`…`set_b2()` for `ModelSerializer`. **The setters don't check shape**: `set_W1(3×3)` was accepted, and the next `forward()` threw a matrix-multiply error (verified). They also don't invalidate the GPU mirror (below) |
| `print_config(name)` | Parameter counts and memory to `std::cout`. Its `int` arithmetic would overflow for very large dimensions. No production caller |

### GPU (`ADAI_ENABLE_GPU`)

| Method | Behaviour |
|---|---|
| `gpu_upload_weights()` | Creates `GPUState` on first use and copies all four weights to the device |
| `gpu_download_grads()` | **Adds** the device gradients into the host gradients (consistent with CPU accumulation) |
| `gpu_zero_grads()` | Zeroes the device gradients |
| `gpu_forward(input)` | Uploads the weights on first use. Resizes the activation caches when the sequence shape changes (it checks rows **and** columns; a rows-only check crashed single-token decode on ai-machine). Then `input·W1 + b1`, GELU in place, the stats hook, and `·W2 + b2`. **Doesn't check `input.cols`** |
| `gpu_backward(dout)` | Mirrors `backward()` on the device, accumulating into `dW*`/`db*`. **Dereferences `gpu_` without a check**, so calling it before any `gpu_forward()` is a null-pointer crash |

> **Caller contract: the GPU mirror isn't kept in sync.** `gpu_forward()` uploads only when no mirror
> exists. After an optimizer step, `load_weights()` or any `set_W*()`, the device keeps the old
> weights until someone calls `gpu_upload_weights()`. `EncoderDecoderModel` handles this
> (`gpu_init_training()`/`gpu_sync_weights()` re-upload every layer), and the server loads its
> checkpoint before the first GPU call. A future hot model reload would need to re-sync explicitly.

---

## 3. Who calls what

| Caller | Uses |
|---|---|
| `EncoderBlock` / `DecoderBlock` | `forward()`/`backward()`, `update_weights()` (after overwriting `learning_rate`), `zero_grad()`, the gradient norm, `set_optimizer()`, `save_weights()`/`load_weights()` as `<block>_feedforward.bin`, the GPU methods |
| `Predictor` (LeJEPA) | The whole network: `forward()`, `backward()`, `update_weights()`, `zero_grad()`, `set_optimizer()` |
| `ModelSerializer` | The `get_*`/`set_*` accessors, via `get_feed_forward()` |
| `ChatbotTrainer` | The CPU or GPU saturation hook on every block's FFN, per epoch |

---

## 4. Tests

`feedforward_test.cpp` has 62 tests:
- construction across sizes and expansion ratios;
- forward (shapes, single token, long sequences, determinism, zero/negative/large inputs);
- backward (shapes, non-zero gradients, accumulation, varying lengths);
- `zero_grad()`, gradient norm and clipping;
- plain updates, and optimizer integration (set, null, switch, LR scheduling, comparison with SGD);
- save/load (round trip, after training, dimension mismatch, missing file, repeated);
- a training loop and a residual-connection pattern;
- CPU hooks (firing, shape, clearing, replacing, saturation ratios) and their GPU equivalents.

**Not covered:**
- gradients compared against finite differences (the tests check shapes and non-zero values; the probe above did the numerical check);
- truncated weight files;
- setter shape checks;
- argument validation in the constructor.

---

## 5. Known gaps and gotchas (summary)

Each new item is tracked in [TECHNICAL_DEBT.md](../../guides/TECHNICAL_DEBT.md) and tagged in the code with
`TODO: See TD-NNN`.

| Item | Impact | Basis | Tracked as |
|---|---|---|---|
| Weight init seeded from `std::random_device`, with no way to set a seed; no global seed exists (§2) | Training runs and bug reproductions can't be replayed from initialization | **Verified** | [TD-368](../../guides/TECHNICAL_DEBT.md#td-368-weight-initialization-cant-be-seeded-or-reproduced) |
| `load_weights()` doesn't check reads: a truncated file loads partially, keeps the old weights for the rest, and doesn't throw; writes unchecked; `std::cout` in save/load/`print_config()` (§2) | A damaged checkpoint loads as a silent mix of old and new weights | **Verified** | [TD-369](../../guides/TECHNICAL_DEBT.md#td-369-feedforwardload_weights-loads-truncated-files-as-a-silent-mix-of-old-and-new-weights); `std::cout` under [TD-370](../../guides/TECHNICAL_DEBT.md#td-370-feedforward-argument-checks-gpu-null-dereference-and-console-output) |
| `set_W*()`/`set_b*()` don't check shape; the constructor doesn't validate dimensions; `gpu_forward()` doesn't check `input.cols`; `gpu_backward()` before `gpu_forward()` dereferences null; `backward()` before `forward()` gives a misleading error (§2) | Late or confusing failures, or a crash on misuse | **Verified** (setters, constructor, `backward()`), by inspection (GPU) | [TD-370](../../guides/TECHNICAL_DEBT.md#td-370-feedforward-argument-checks-gpu-null-dereference-and-console-output) |
| Unscaled residual-branch output (std ≈ 1.35 for unit input) (§2) | Residual-stream variance grows with depth in the Pre-LN stack; a design choice, not a bug | **Verified** (measurement) | [TD-371](../../guides/TECHNICAL_DEBT.md#td-371-residual-branch-output-isnt-scaled-for-depth) (owner decision) |
| `update_weights()` steps a shared optimizer; double registration | Already tracked | — | TD-297 |
| CPU/GPU GELU formula duplication | Already tracked | — | TD-216 |

---

## 6. Merge notes

This file replaces `docs/development/api/transformer/feed-forward.md` (1,850 lines). Where they
disagree, this file wins:

- **Initialization** was called "Xavier/He". It's He (`√(2/fan_in)`); Xavier would use
  `√(2/(fan_in + fan_out))`. The old doc didn't mention that the RNG is unseeded.
- **Performance.** The benchmark tables (for example, "forward 20–30 ms at d=512, seq=128") had no
  source or date and are dropped. The parameter counts are kept (§1).
- **Pre-Norm vs Post-Norm.** The old doc presented both as options. Every block in this codebase is
  Pre-LN.
- **Gotchas.** It didn't mention the unchecked loads, the unchecked setters, the GPU null dereference,
  the GPU-mirror contract or TD-297.

**Kept from the old doc:**
- the forward and backward equations (now verified numerically);
- gradient accumulation semantics;
- the save layout;
- the activation-hook (TD-013) feature notes.

**Dropped:** generic transformer tutorials, "best practices", issue/solution essays, and speculative
"advanced topics" (custom activations, dropout, layer freezing, mixed precision), none of which are
implemented.
