# `LayerNorm` — Source File Reference

- **Files:** [`src/LayerNorm.hpp`](../../../../src/LayerNorm.hpp) (255 lines), [`src/LayerNorm.cpp`](../../../../src/LayerNorm.cpp) (379 lines)
- **Namespace:** global
- **Built into:** `adai_layers`; used by `EncoderBlock`/`DecoderBlock` (pre-norms), `LLMEncoder`/`LLMDecoder` (final norms), `LeJEPAEncoder`; serialized by `ModelSerializer`
- **Status tag:** `@adai-status: stable`, `@adai-version: 1.2.0` (`.hpp`) / `1.3.0` (`.cpp`), `@adai-reviewed: 2026-11-13`
- **Tests:** [`tests/layernorm_test.cpp`](../../../../tests/layernorm_test.cpp) (54 tests), plus `EncoderDecoderModelTest.EveryCheckpointFileIsCheckedOnLoad`
- **Last traced against the code:** 2026-11-13

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
| `LayerNorm(dim, eps = 1e-5)` | γ = 1, β = 0, zero gradients, `learning_rate = 0.001`. Throws `std::invalid_argument` for `dim < 1` or an ε that isn't a positive finite number (TD-426) |
| `forward(input)` | Per-row mean and variance (population, two-pass), caches the input, x̂, mean and variance. Throws `std::invalid_argument` unless `input.cols == dim` (TD-426; it used to surface as `Matrix::operator()`'s generic `out_of_range`) |
| `backward(grad)` | Accumulates γ/β gradients (TD-195) and returns ∂L/∂x via the standard mean/variance chain rule. **Verified correct**: 4.6×10⁻⁵ max relative error against finite differences. Throws `std::logic_error` before any `forward()` and `std::invalid_argument` unless `grad` has the forward input's shape (TD-426) |
| `set_optimizer()`, `register_parameters()`, `update_weights()`, `zero_grad()` | As in the other layers (shared-optimizer stepping, TD-297) |
| `get_gamma()`, `get_beta()`, `get_epsilon()`, `get_dim()` | Accessors |
| `set_gamma(m)`, `set_beta(m)` | Throw `std::invalid_argument` on any shape but `[1, dim]`, changing nothing (TD-426; they used to print to `std::cerr` and return). On success they also upload the new values to the device mirror when there is one (private `refresh_gpu_mirror()`) |
| `print_config(name)` | One `adai::Logger::info` line: dim, ε, learning rate, γ and β ranges. No production caller |
| `save_weights(file)` / `load_weights(file)` | `dim`, `eps`, then γ and β, through `adai::binio` (TD-423, below). `load_weights()` checks `dim`, reads the whole file before changing anything, **adopts the file's ε** (TD-426; it used to be read and discarded; a non-positive or non-finite one throws `std::runtime_error`) and refreshes the device mirror. Both log at debug level |
| GPU (`ADAI_ENABLE_GPU`) | `gpu_upload_weights()`, `gpu_collect_parameters()`/`gpu_download_weights()` (TD-468's device optimizer), `gpu_download_grads()` (adds to the host gradients), `gpu_zero_grads()`, `gpu_forward()` (uploads if absent or the width changed; caches normed/mean/rstd), `gpu_backward()` (throws `std::logic_error` before any `gpu_forward()` and `std::invalid_argument` unless `dout` matches the cached forward's shape; it used to dereference a null `gpu_`). The kernels are in MatrixGPU_cu.md / MatrixGPU_SYCL_cpp.md; the forward kernel's TD-384 race is fixed and the backward math (TD-061 fix) checks out |

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
optimizer, `set_gamma()`/`set_beta()`, and save/load. `LayerNormValidationTest` (TD-426, each failing on
the old code): bad dimension and ε, setters throwing on a wrong shape and keeping the old values,
`forward()` rejecting the wrong width, `backward()` needing a forward of the same shape, a load
adopting the file's ε, and a load rejecting a bad ε without changing anything. Truncated files:
`EncoderDecoderModelTest.EveryCheckpointFileIsCheckedOnLoad`.

**Not covered:** `gpu_backward()`'s checks and the device-mirror refresh (GPU builds only).

---

## 5. Known gaps and gotchas (summary)

Each new item is tracked in [TECHNICAL_DEBT.md](../../guides/TECHNICAL_DEBT.md) and tagged in the code with
`TODO: See TD-NNN`.

| Item | Impact | Basis | Tracked as |
|---|---|---|---|
| Silent setters, missing shape checks, unchecked loads, discarded file ε, console output, `gpu_backward()` null dereference, stale device mirror | **Fixed in code** (loads October 26, the rest November 13, 2026) | **Verified** by tests | [TD-426](../../archive/TECHNICAL_DEBT_RESOLVED.md#td-426-layernorm-silent-setters-missing-shape-checks-and-unchecked-loads) |
| GPU forward-kernel race; decoder final norm never updated; shared-optimizer stepping | Fixed in code | — | TD-384, TD-346, TD-297 |

No open items in this file.

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

## Addendum (November 13, 2026): argument and shape checks (TD-426)

TD-426's remaining items, now reflected in §2: the constructor, setters, `forward()`, `backward()` and `gpu_backward()` throw on bad
arguments, a wrong shape or a missing forward instead of printing or failing inside `Matrix`; the setters and `load_weights()`
refresh the device mirror; `load_weights()` adopts the file's ε; logging goes through `adai::Logger`.
