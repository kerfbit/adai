# `gpu/sycl/MatrixGPU_SYCL.hpp` — Source File Reference

- **File:** [`src/gpu/sycl/MatrixGPU_SYCL.hpp`](../../../../src/gpu/sycl/MatrixGPU_SYCL.hpp) (header-only, 241 lines)
- **Namespace:** `adai::gpu` (`ActivationType`, the `matrix_*_gpu` declarations, the SYCL `GPUMatrix`)
- **Built into:** SYCL builds; included by `gpu/MatrixGPU.hpp` in place of its CUDA section. Kernels in [MatrixGPU_SYCL_cpp.md](MatrixGPU_SYCL_cpp.md); queue and memory in [GPUUtils_SYCL.md](GPUUtils_SYCL.md)
- **CUDA counterpart:** the CUDA section of [MatrixGPU.md](MatrixGPU.md) (`gpu/MatrixGPU.hpp`), which also holds the shared `GPUKVCache`/`GPUDecoderKVCache`
- **Status tag:** `@adai-status: beta` ("capped by TD-050"), `@adai-version: 0.9.0`, `@adai-reviewed: 2026-09-10`
- **Last traced against the code:** 2026-10-12

> Code-traced reference. The `GPUMatrix` here, and the shared `GPUKVCache`/`GPUDecoderKVCache`
> built on it, were **run** on the oneAPI CPU device through the scratch harness described in
> [MatrixGPU_SYCL_cpp.md](MatrixGPU_SYCL_cpp.md) (a copy of the backend whose `GPUManager` also
> accepts CPU devices).

---

## 1. What this file is

The SYCL half of the GPU math API:
- `ActivationType` and the `matrix_*_gpu` declarations, matching `MatrixGPU.hpp`'s CUDA block line
  for line;
- a **second, hand-maintained copy of `GPUMatrix`**.

`gpu/MatrixGPU.hpp` includes this file under `ADAI_GPU_BACKEND_SYCL` and then defines the
backend-agnostic `GPUKVCache`/`GPUDecoderKVCache` after the backend split, so on ai-machine every
layer's GPU path uses the `GPUMatrix` defined here.

### Differences from the CUDA `GPUMatrix`

| Member | CUDA | SYCL (this file) |
|---|---|---|
| `copy()` | Async device copy | `q.memcpy(...).wait()`, an **unnecessary full-queue drain** on an in-order queue, against CLAUDE.md's "no `.wait()` on intermediate work" rule (and the TD-390 bubbles) |
| `zero()` | `cudaMemsetAsync` | `q.memset` (async) |
| Everything else | — | The same code, member for member (operators, LayerNorm, softmax, bias, diagnostics) |

Only those two members actually depend on the backend, and both have backend-agnostic primitives
already (`matrix_copy_device_to_device_gpu`, plus a trivial zero kernel). Yet the class is
duplicated in full, so every `GPUMatrix` change has to be made twice, and a missed one makes the two
backends silently diverge. `GPUKVCache` already shows the alternative: one definition after the
backend split, built only on the shared API.

---

## 2. Verification on the CPU device

| Check | Result |
|---|---|
| `operator-` (via `−1 ×` and add), `copy()` + `zero()` (source left intact) | Correct |
| A fresh `GPUMatrix(2, 4)` | **Uninitialized**: first value `6.6×10⁻³⁷` (TD-389) |
| `GPUMatrix(0, 8)` | **Throws** `"sycl::malloc_device failed: allocation returned null"` (TD-392) |
| `big(4×4).add_inplace(small(2×2))` | **Accepted with no error**, reading past `small`'s buffer (TD-389) |
| `GPUKVCache(8, 4)`: two `append()`s (3 + 2 rows), `get_keys()`/`get_values()`, `clear()` | Correct: rows land at the right offsets, `size()` = 5, then 0, capacity kept. **This is the first time the GPU KV cache has been run anywhere** (TD-389 noted it had no tests) |
| `GPUDecoderKVCache(2, max_seq_length = 8, 4)`: append a 12-row encoder output to a cross-attention cache | **Throws** `"GPUKVCache::append: exceeds max_seq_length capacity"`, confirming TD-387's failure mode for over-length input |

The kernels underneath were verified in [MatrixGPU_SYCL_cpp.md](MatrixGPU_SYCL_cpp.md) §2, apart
from the TD-384 and TD-393 failures recorded there.

---

## 3. Who calls what

The same as the CUDA `GPUMatrix` ([MatrixGPU.md](MatrixGPU.md) §4): every layer's GPU path,
`EncoderDecoderModel`'s GPU training and generation, and `Matrix::to_gpu()`/`from_gpu()`. Under SYCL
the `GPUKVCache` from `MatrixGPU.hpp` wraps this class.

---

## 4. Known gaps and gotchas (summary)

Each new item is tracked in [TECHNICAL_DEBT.md](../../guides/TECHNICAL_DEBT.md) and tagged in the code with
`TODO: See TD-NNN`.

| Item | Impact | Basis | Tracked as |
|---|---|---|---|
| `GPUMatrix` is duplicated in full between the CUDA and SYCL headers, though only `copy()`/`zero()` differ (§1) | Every change must be made twice; drift between backends is silent | By inspection | [TD-395](../../guides/TECHNICAL_DEBT.md#td-395-gpumatrix-is-duplicated-between-the-cuda-and-sycl-headers) |
| SYCL `copy()` does a `.wait()` (§1) | An extra full-queue drain wherever a matrix is copied | By inspection | [TD-395](../../guides/TECHNICAL_DEBT.md#td-395-gpumatrix-is-duplicated-between-the-cuda-and-sycl-headers) |
| Unchecked `add_inplace` and other operations; uninitialized memory; zero-size throws (§2) | As in the CUDA class | **Verified** on the CPU device (TD-389, TD-392) | [TD-389](../../guides/TECHNICAL_DEBT.md#td-389-gpumatrix-shape-checks-uninitialized-memory-and-untested-gpu-kv-caches) (amended), [TD-392](../../guides/TECHNICAL_DEBT.md#td-392-sycl-gpumemory0-throws-where-cuda-succeeds) |
| Over-length encoder output throws from the cross-attention cache (§2) | GPU serving fails over-length requests | **Verified** at the cache level (TD-387) | [TD-387](../../guides/TECHNICAL_DEBT.md#td-387-gpu-generation-fails-on-input-longer-than-max_seq_length) (amended) |
| Header status note says "capped by TD-050", but TD-050 is mostly resolved; the GPU KV cache it added now has a first run | Stale status note | By inspection | [TD-395](../../guides/TECHNICAL_DEBT.md#td-395-gpumatrix-is-duplicated-between-the-cuda-and-sycl-headers) |
