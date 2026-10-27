# `gpu/MatrixGPU.hpp` — Source File Reference

- **File:** [`src/gpu/MatrixGPU.hpp`](../../../../src/gpu/MatrixGPU.hpp) (header-only, 573 lines)
- **Namespace:** `adai::gpu` (`ActivationType`, the `matrix_*_gpu` declarations, `GPUMatrix`, `GPUKVCache`, `GPUDecoderKVCache`)
- **Built into:** everything compiled with `ADAI_ENABLE_GPU`; implementations in [MatrixGPU_cu.md](MatrixGPU_cu.md) (CUDA) and `sycl/MatrixGPU_SYCL.cpp` (SYCL)
- **Status tag:** `@adai-status: beta`, `@adai-version: 0.10.0`, `@adai-reviewed: 2026-09-14`
- **Tests:** [`tests/matrixgpu_td003_test.cpp`](../../../../tests/matrixgpu_td003_test.cpp) → `matrixgpuTd003Tests` (13 tests, all skip without a GPU); GPU tests inside `feedforward_test.cpp` and `multiheadattention_test.cpp`; **no tests for `GPUKVCache`/`GPUDecoderKVCache`**
- **Last traced against the code:** 2026-10-12

> Code-traced reference. No GPU device is available here (the CUDA test build skips all 13
> `matrixgpuTd003Tests`), so everything below is **by inspection**.

---

## 1. What this file is

The **GPU math API** that the model layers program against. It has three parts:

| Part | Where defined | Purpose |
|---|---|---|
| `ActivationType` + the `matrix_*_gpu` free functions | CUDA: declared here, implemented in `MatrixGPU.cu`. SYCL: this header includes `sycl/MatrixGPU_SYCL.hpp` instead, which declares the same API | Raw device-pointer kernels |
| `GPUMatrix` | CUDA: here. SYCL: `sycl/MatrixGPU_SYCL.hpp` | A move-only, row-major, device-resident matrix built on `GPUMemory<float>` with on-device operators |
| `GPUKVCache`, `GPUDecoderKVCache` (TD-050) | Here, **outside** the backend split, so one definition serves both backends | Pre-allocated device KV caches for incremental GPU decoding |

### Why it matters

Every layer's GPU path (`FeedForward`, `MultiHeadAttention`, `CrossAttention`, `LayerNorm`,
`LanguageModelHead`, `EncoderBlock`/`DecoderBlock`, `LLMEncoder`/`LLMDecoder`) works in `GPUMatrix`.
GPU training (`EncoderDecoderModel::gpu_train_step`) and GPU generation
(`gpu_generate_response*`, the TD-033 serving path) run entirely through it.

---

## 2. `GPUMatrix` (CUDA definition)

| Member | Behaviour |
|---|---|
| `GPUMatrix(r, c)` | Allocates `r × c` floats via `GPUMemory` (counted against the budget). **The memory isn't initialized** (`cudaMalloc` doesn't zero), unlike CPU `Matrix`, so anything that accumulates into a fresh `GPUMatrix` must `zero()` it first. The layers' gradient buffers rely on `gpu_zero_grads()`, which `ChatbotTrainer` calls before the first backward. `r × c` is an unchecked `int` product |
| Move-only | Copy is deleted; `copy()` makes an explicit device-to-device copy |
| `device_ptr()`, `size()` | Raw pointer and `rows × cols` |
| `upload(ptr, n)` / `download(ptr, n)` | Via `GPUMemory`. Throws if `n > size()`; **a short upload leaves the rest uninitialized** |
| `operator*` | cuBLAS GEMM; **checks** `cols == other.rows` |
| `operator+`, `hadamard` | **Check** shapes |
| `operator-` | Checks shapes; computes `a + (−1)·b` through a temporary (two launches, one extra allocation) |
| `scale`, `transpose`, `apply_activation_inplace`, `sum` | No shape concerns. `sum()` synchronizes (TD-385) |
| `zero`, `add_inplace(other)` | `cudaMemsetAsync`; in-place add. **`add_inplace` doesn't check that sizes match**, so a smaller `other` reads past its end and a larger one is silently truncated |
| `softmax_rows_inplace`, `softmax_backward(dout)`, `gelu_backward(dout)` | Kernels from `MatrixGPU.cu` (softmax is subject to the TD-384 race). **The two backward ops don't check `dout`'s shape** |
| `add_row_bias(bias)`, `sum_rows()`, `masked_fill_inplace(mask, v)` | **No shape checks** on `bias` (must be `1 × cols`) or `mask` (must be `rows × cols`) |
| `layer_norm(γ, β, eps, &normed, &mean, &rstd)` / `layer_norm_backward(...)` | Write into caller-provided outputs; **no shape checks** on any of them (`dγ`/`dβ` accumulate) |
| `count_below_threshold(t)` | Fraction of `|x| < t`; divides by `size()`, so an empty matrix gives `NaN` |
| `row_entropy_avg()` | Allocates a per-row buffer every call, then `matrix_sum_gpu` (synchronizes) |

The unchecked operations write and read device memory by raw size, so a shape bug in a caller shows
up as corrupted values or a device fault, not as the `invalid_argument` the checked operators throw.

---

## 3. `GPUKVCache` and `GPUDecoderKVCache` (TD-050)

| Member | Behaviour |
|---|---|
| `GPUKVCache(max_seq_length, d_model)` | Pre-allocates `keys_` and `values_` as `[max_seq_length × d_model]` once. The positivity check runs **after** the members are constructed, so a bad size fails inside `GPUMemory` (a huge `size_t` from a negative product) with a budget or allocation error instead of the intended message |
| `append(k, v)` | Checks shapes, then **throws `out_of_range` when `current_length + k.rows > max_seq_length`**; otherwise copies the rows into place (one device-to-device copy each) |
| `get_keys()` / `get_values()` | Materialize the populated prefix as a **fresh** `GPUMatrix` (allocate and copy) on every call, so every decode step copies the whole cache once per layer for K and once for V. That's O(n²) bytes over a generation |
| `clear()`, `is_empty()`, `size()`, `capacity()` | — |
| `GPUDecoderKVCache(num_layers, max_seq_length, d_model)` | One self-attention and one cross-attention `GPUKVCache` per layer, **both capped at `max_seq_length`**. `self_attention_cache(i)`/`cross_attention_cache(i)` aren't bounds-checked. `clear()`, `clear_self_attention()`, `is_empty()`, `current_length()` |

> **GPU generation fails where CPU generation degrades (by inspection, definite).**
> `EncoderDecoderModel::gpu_generate_response()` and `gpu_generate_response_with_strategy()` build a
> `GPUDecoderKVCache(decoder_layers, max_seq_length, d_model)`. On the first decode step,
> `CrossAttention`'s cached GPU path appends the **whole encoder output** to the cross-attention
> cache, but encoder input is never truncated to `max_seq_length` (TD-365). So any chat input longer
> than `max_seq_length` tokens makes `append()` throw `out_of_range`, and the request fails on a GPU
> server. The CPU path for the same request completes, with positions past the limit lacking
> positional encoding (TD-365, verified). The self-attention cache has the same hard cap, but
> `TextGenerator` stops after `max_length` (= `max_seq_length`) generated tokens, so a `BOS` prompt plus
> generation fills the cache exactly to capacity without exceeding it; only a future larger
> `max_length` would make that path throw.

---

## 4. Who calls what

| Caller | Uses |
|---|---|
| All layer `gpu_*` paths | `GPUMatrix` operators, LayerNorm, softmax, bias, masking, transfers |
| `MultiHeadAttention`, `CrossAttention` | Per-head slicing via `matrix_copy_device_to_device_gpu` (TD-059); `GPUKVCache` in the cached paths |
| `DecoderBlock::gpu_forward_with_cache`, `LLMDecoder::gpu_decode_step` | `GPUDecoderKVCache` |
| `EncoderDecoderModel` | Owns `gpu_logits_`/targets; builds a `GPUDecoderKVCache` per generation |
| `Matrix::to_gpu()`/`from_gpu()` | Conversion between host `Matrix` and `GPUMatrix` |

---

## 5. Tests

`matrixgpu_td003_test.cpp` (13 tests, all skipped without a GPU) covers the diagnostics reductions,
device copies, offset downloads, `add_inplace`, softmax, masked fill and row sums.
`feedforward_test.cpp` and `multiheadattention_test.cpp` have GPU-only tests of the hooks.

**Not covered:** `GPUKVCache`/`GPUDecoderKVCache` (no tests at all), LayerNorm through
`GPUMatrix`, shape-mismatch behaviour, and over-capacity generation.

---

## 6. Known gaps and gotchas (summary)

Each new item is tracked in [TECHNICAL_DEBT.md](../../guides/TECHNICAL_DEBT.md) and tagged in the code with
`TODO: See TD-NNN`.

| Item | Impact | Basis | Tracked as |
|---|---|---|---|
| `GPUDecoderKVCache` caps the cross-attention cache at `max_seq_length`, but encoder input isn't truncated (TD-365), so over-length chat input throws `out_of_range` on GPU (§3) | GPU serving fails requests the CPU path answers | By inspection (definite) | [TD-387](../../guides/TECHNICAL_DEBT.md#td-387-gpu-generation-fails-on-input-longer-than-max_seq_length) |
| `get_keys()`/`get_values()` allocate and copy the whole populated cache every step (§3) | O(n²) device copies and allocations per generation; undercuts TD-050's purpose | By inspection | [TD-388](../../guides/TECHNICAL_DEBT.md#td-388-gpukvcache-copies-the-whole-cache-every-decode-step) |
| `add_inplace`, `softmax_backward`, `gelu_backward`, `add_row_bias`, `masked_fill_inplace`, `layer_norm`, `layer_norm_backward` don't check shapes; `GPUMatrix` memory is uninitialized; `count_below_threshold()` on empty gives `NaN`; `GPUKVCache` validates after allocating; cache layer indices unchecked (§2–3) | Caller shape bugs become silent device memory corruption or faults | By inspection | [TD-389](../../guides/TECHNICAL_DEBT.md#td-389-gpumatrix-shape-checks-uninitialized-memory-and-untested-gpu-kv-caches) |
| No tests for the GPU KV caches (§5) | TD-050's GPU cache is unverified even on hardware-equipped hosts | By inspection | [TD-389](../../guides/TECHNICAL_DEBT.md#td-389-gpumatrix-shape-checks-uninitialized-memory-and-untested-gpu-kv-caches) |
| Kernel race, `sum()` sync | Already tracked | — | TD-384, TD-385 |

## Addendum (October 26, 2026): cached attention reads the KV cache in place (TD-388)

`GPUKVCache` exposes its populated prefix as a view (`keys_data()`, `values_data()`, `leading_dimension()`) instead of copying it on
every `get_keys()`/`get_values()`. The new `matrix_gemm_strided_gpu()` is a row-major GEMM with transpose flags, alpha and leading
dimensions (`cublasSgemm` / oneMKL `gemm`). The cached self- and cross-attention paths use it to compute each head straight from the
cache and to write each head's output into its own columns, with no prefix copies, per-row slices or scatter. A 100-token decode on
the oneAPI CPU device went from 6.6 s to 1.4 s. Tests: `MatrixGPUTD388Test.StridedGemmMatchesHostReference`,
`EncoderDecoderGPUTest.CachedGpuDecodeMatchesCpuOverALongGeneration`.
