# `LayerNorm` — Source File Reference

- **Files:** [`src/LayerNorm.hpp`](../../../../src/LayerNorm.hpp), [`src/LayerNorm.cpp`](../../../../src/LayerNorm.cpp)
- **Namespace:** global
- **Built into:** `adai_layers`; used by `EncoderBlock`/`DecoderBlock` (pre-norms), `LLMEncoder`/`LLMDecoder` (final norms), `LeJEPAEncoder`; serialized by `ModelSerializer`
- **Status tag:** `@adai-status: stable`, `@adai-version: 1.0.1`, `@adai-reviewed: 2026-09-18` (both files)
- **Tests:** [`tests/layernorm_test.cpp`](../../../../tests/layernorm_test.cpp) (48 tests)
- **Last traced against the code:** 2026-10-14

> Code-traced reference. Behaviour marked **verified** was confirmed by running the real class in scratch
> programs (including an AddressSanitizer build).

---

## 1. What this file is

**Layer normalization** over each row (token) independently:

```text
μ = mean(x_i),  σ² = mean((x_i − μ)²),  x̂ = (x_i − μ) / √(σ² + ε),  y = γ ⊙ x̂ + β
```

`γ` (`gamma`, initialized to 1) and `β` (`beta`, initialized to 0) are learnable `[1 × dim]` vectors;
ε defaults to 1e-5. It's used as the **pre-norm** before every attention and feed-forward sub-layer
(Pre-LN, so it doesn't bound the residual stream; see FeedForward.md's TD-371 note) and as the final
norm after each stack.

---

## 2. API

| Member | Behaviour |
|---|---|
| `LayerNorm(dim, eps = 1e-5)` | γ = 1, β = 0, zero gradients, `learning_rate = 0.001` |
| `forward(input)` | Per-row mean and variance (population, two-pass), caches the input, x̂, mean and variance. **Doesn't check `input.cols == dim`**: a wider input throws `std::out_of_range("Matrix index out of bounds")` from `Matrix::operator()` (verified), not a dimension error |
| `backward(grad)` | Accumulates γ/β gradients (TD-195) and returns ∂L/∂x via the standard mean/variance chain rule. **Verified correct**: 4.6×10⁻⁵ max relative error against finite differences. Doesn't check `grad` against the cached forward: a row mismatch (or a call before any `forward()`) also surfaces as `out_of_range`, verified |
| `set_optimizer()`, `register_parameters()`, `update_weights()`, `zero_grad()` | As in the other layers (shared-optimizer stepping, TD-297) |
| `get_gamma()`, `get_beta()`, `get_epsilon()`, `get_dim()` | Accessors |
| `set_gamma(m)`, `set_beta(m)` | On a shape mismatch they **print to `std::cerr` and return without changing anything**: `set_gamma(1×5)` on a dim-4 norm left γ at 1 with no exception (verified). `ModelSerializer`'s SafeTensors loader validates sizes first (`require()`), so this is latent there, but a caller can't tell the set failed. They also don't refresh the GPU mirror |
| `print_config(name)` | Prints to stdout. No production caller |
| `save_weights(file)` / `load_weights(file)` | `dim`, `eps`, then γ and β. `load_weights()` checks `dim`, **reads ε but discards it** (the instance's ε is kept), and **doesn't check its reads**: a truncated file leaves the old values for the rest. Both print to `std::cout`; save doesn't check its writes |
| GPU (`ADAI_ENABLE_GPU`) | `gpu_upload_weights()`, `gpu_download_grads()` (adds to the host gradients), `gpu_zero_grads()`, `gpu_forward()` (uploads if absent or the width changed; caches normed/mean/rstd), `gpu_backward()` (**dereferences `gpu_` unchecked**). The kernels are in MatrixGPU_cu.md / MatrixGPU_SYCL_cpp.md, where the forward kernel's **TD-384 race was verified** and the backward math (TD-061 fix) checks out |

---

## 3. Who calls what

| Caller | Uses |
|---|---|
| `EncoderBlock` (`norm1`, `norm2`), `DecoderBlock` (`norm1`–`norm3`, plus the gated paths' norms) | Forward/backward, updates, legacy save/load, GPU path |
| `LLMEncoder::final_norm`, `LLMDecoder::final_norm` | Same. The decoder's final norm isn't updated by `LLMDecoder::update_weights()` (TD-346) |
| `ModelSerializer` | `get_gamma()`/`get_beta()` on save; `set_gamma()`/`set_beta()` on load |

---

## 4. Tests

`layernorm_test.cpp` covers construction, normalization statistics (zero mean and unit variance per
row, γ/β application), backward shapes and gradient flow, accumulation, updates with and without an
optimizer, `set_gamma()`/`set_beta()`, and save/load.

**Not covered:** wrong-width input, truncated files.

---

## 5. Known gaps and gotchas (summary)

Each new item is tracked in [TECHNICAL_DEBT.md](../../guides/TECHNICAL_DEBT.md) and tagged in the code with
`TODO: See TD-NNN`.

| Item | Impact | Basis | Tracked as |
|---|---|---|---|
| `set_gamma()`/`set_beta()` ignore a shape mismatch with only a `std::cerr` line (§2) | A failed parameter load is silent for any caller that doesn't pre-validate | **Verified** | [TD-426](../../guides/TECHNICAL_DEBT.md#td-426-layernorm-silent-setters-missing-shape-checks-and-unchecked-loads) |
| No width check in `forward()`, no row check in `backward()`; both surface as a generic `out_of_range` (§2) | Confusing errors on misuse | **Verified** | [TD-426](../../guides/TECHNICAL_DEBT.md#td-426-layernorm-silent-setters-missing-shape-checks-and-unchecked-loads) |
| `load_weights()` unchecked reads, file ε ignored; `std::cout` in save/load/`print_config()`; `gpu_backward()` null dereference; setters don't refresh the GPU mirror (§2) | Partial loads keep stale values; console noise | By inspection | [TD-426](../../guides/TECHNICAL_DEBT.md#td-426-layernorm-silent-setters-missing-shape-checks-and-unchecked-loads) |
| GPU forward-kernel race; decoder final norm never updated; shared-optimizer stepping | Already tracked | — | TD-384, TD-346, TD-297 |

---

## 6. Merge notes

This file replaces `docs/development/api/transformer/layer-norm.md` (1,683 lines). Kept: the formulation, the
per-row (token-wise) semantics, the γ/β initialization and the persistence layout. Dropped: generic
tutorial material. Added: the backward pass verified numerically, the setter behaviour, and the
GPU-kernel status.

## Addendum (October 26, 2026): checked checkpoint I/O (TD-423)

`save_weights()`/`load_weights()` (TD-426's read/write item) use `adai::binio::Reader`/`Writer` (`src/BinaryIO.{hpp,cpp}`).
- **Loading.** A file that is short, empty or longer than this layer's shapes throws `std::runtime_error`. The whole file is read
  into temporaries first, so a failed load leaves the layer unchanged.
- **Saving.** A failed write or flush (a full disk) throws instead of leaving a short file.

Tests: `EncoderDecoderModelTest.EveryCheckpointFileIsCheckedOnLoad`.
