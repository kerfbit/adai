# `LoRA` — Source File Reference

- **File:** [`src/LoRA.hpp`](../../../../src/LoRA.hpp) (header-only, 493 lines)
- **Namespace:** global (`LoRAAdapter`, `LoRAConfig`, `calculate_lora_params_attention()`, `print_lora_statistics()`)
- **Built into:** header-only; used by `MultiHeadAttention` and `CrossAttention` (`enable_lora()`), and through them `EncoderDecoderModel::enable_lora()`/`register_lora_parameters()`/`merge_lora()` (TD-038)
- **Status tag:** `@adai-status: experimental` (re-tagged from `beta` on October 14, 2026: no production entry point), `@adai-version: 0.8.1`, `@adai-reviewed: 2026-10-14`
- **Tests:** in `phase5_test.cpp`, `multiheadattention_test.cpp`, `crossattention_test.cpp`, `encoderdecoder_test.cpp`
- **Last traced against the code:** 2026-10-14

> Code-traced reference. Behaviour marked **verified** was confirmed with a scratch program.

---

## 1. What this file is

**Low-Rank Adaptation** for the attention projections. A frozen base projection `W` gets a trainable
update `ΔW = (α/r) · Aᵀ·Bᵀ`, with `A: [r × in]` and `B: [out × r]`, so only `r·(in + out)` parameters
train:

```text
LoRAAdapter::forward(x, base_out) = base_out + (α/r) · x·Aᵀ·Bᵀ
```

`B` starts at **zero**, so enabling LoRA on a trained model changes nothing until the adapters train.
The attention classes apply one adapter per enabled projection (Q/K/V/O) to the projection's output,
and `merge_lora()` folds `ΔW` into the base weights.

### Who uses it

**Nothing in production.** No binary, config key or HTTP endpoint calls `enable_lora()`. The trainer
never fine-tunes with LoRA, and adapters aren't saved with checkpoints (TD-302). It's a library feature
exercised only by tests.

---

## 2. API

### `LoRAAdapter`

| Member | Behaviour |
|---|---|
| `LoRAAdapter(in, out, rank, alpha = -1)` | Rejects `rank ≤ 0` or `rank > min(in, out)`; `α` defaults to `rank` (scale 1). `A ~ U(−1/√r, 1/√r)` from the **C `rand()`** (process-global, unseeded so identical every run, not thread-safe, and not the project's other RNGs); `B = 0`; zero gradients |
| `forward(x, base_out)` | Returns `base_out + scale·x·Aᵀ·Bᵀ`. The parameter is named `W_output` but is the base layer's **output**, not a weight. No shape checks beyond `Matrix`'s own |
| `backward(x, grad)` | Accumulates `grad_B += scale·gradᵀ·(x·Aᵀ)` and `grad_A += scale·Bᵀ·gradᵀ·x`, and returns `scale·grad·B·A`. **Verified correct**: input gradient within 1.7×10⁻⁴ of finite differences |
| `register_parameters(optimizer)` | Adds `A`/`B` with their gradients |
| `update(lr)` | Plain SGD. **Doesn't zero the gradients** (unlike every layer's `update_weights()`), so a caller that forgets `zero_grad()` keeps applying stale gradients |
| `zero_grad()` | Zeroes both |
| `merge_with_base(W)` | `W [in × out] + scale·Aᵀ·Bᵀ`; checks `W`'s shape |
| `num_parameters()`, `get_rank()`, `get_alpha()`, `get_A()`, `get_B()` | Accessors |
| `save(path)` / `load(path)` | `in`, `out`, `rank`, `α`, then `A` and `B`. **`load()` replaces the adapter's own dimensions, rank and α with the file's, without validation**: loading a 64×64 rank-8 file into a 6×5 rank-2 adapter succeeded silently (now rank 8, `A` 8×64), and the next `forward()` failed with a matrix-size error (verified). A corrupt or negative rank isn't rejected either, and reads and writes aren't checked |

### `LoRAConfig` and helpers

`rank = 8`, `alpha = 16`, `apply_to_query/key/value/output = true`, `apply_to_ffn = false`,
`dropout = 0`. **`dropout` is never applied anywhere**, and `apply_to_ffn` is "not applicable" per the
attention classes' own docs, since no `FeedForward` LoRA exists. `reduction_ratio()` divides by the adapter
parameter count, so it's infinite when every `apply_to_*` is false. `calculate_lora_params_attention()`
and `print_lora_statistics()` (stdout) count only attention projections. Neither has a production
caller.

---

## 3. Known gaps and gotchas (summary)

Each new item is tracked in [TECHNICAL_DEBT.md](../../guides/TECHNICAL_DEBT.md) and tagged in the code with
`TODO: See TD-NNN`.

| Item | Impact | Basis | Tracked as |
|---|---|---|---|
| `load()` silently adopts the file's dimensions, rank and α; no validation or read checks (§2) | A mismatched adapter file corrupts the adapter and fails later with a confusing error | **Verified** | [TD-432](../../guides/TECHNICAL_DEBT.md#td-432-loraadapterload-silently-adopts-the-files-shape) |
| `A` initialized with the global C `rand()`: unseeded, not thread-safe, separate from other RNGs (§2) | Identical init every run; races if adapters are created concurrently | By inspection | [TD-433](../../guides/TECHNICAL_DEBT.md#td-433-lora-initialization-update-and-config-gaps) |
| `update()` doesn't zero gradients; `LoRAConfig::dropout` and `apply_to_ffn` are unused; `reduction_ratio()` divides by zero; `forward()`'s parameter misnamed (§2) | Misleading API surface | By inspection | [TD-433](../../guides/TECHNICAL_DEBT.md#td-433-lora-initialization-update-and-config-gaps) |
| No production entry point; adapters not persisted with checkpoints; not counted in gradient norms | Already tracked | — | TD-302 (and TD-038 for the feature itself) |

---

## 4. Merge notes

This file replaces `docs/development/api/advanced/lora.md` (610 lines). Where they disagree, this file
wins:
- **Nonexistent methods.** It documented `trainable_parameters()` and `parameter_reduction()`; the real
  method is `num_parameters()`.
- **"Used in production".** It presented LoRA as such and showed a `dropout` option; there's no
  production path and `dropout` isn't implemented.

Kept: the mathematical formulation, the zero-initialized `B`, the save/load layout and the merging
semantics.
