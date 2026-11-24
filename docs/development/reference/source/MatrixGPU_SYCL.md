# `gpu/sycl/MatrixGPU_SYCL.hpp` — Source File Reference

- **File:** [`src/gpu/sycl/MatrixGPU_SYCL.hpp`](../../../../src/gpu/sycl/MatrixGPU_SYCL.hpp) (header-only, 133 lines)
- **Namespace:** `adai::gpu` (`ActivationType` and the `matrix_*_gpu` declarations; `GPUMatrix` is in `MatrixGPU.hpp` since TD-395)
- **Built into:** SYCL builds; included by `gpu/MatrixGPU.hpp` in place of its CUDA section. Kernels in [MatrixGPU_SYCL_cpp.md](MatrixGPU_SYCL_cpp.md); queue and memory in [GPUUtils_SYCL.md](GPUUtils_SYCL.md)
- **CUDA counterpart:** the CUDA section of [MatrixGPU.md](MatrixGPU.md) (`gpu/MatrixGPU.hpp`), which also holds the shared `GPUMatrix` (TD-395) and `GPUKVCache`/`GPUDecoderKVCache`
- **Status tag:** `@adai-status: beta` (capped by TD-389), `@adai-version: 0.21.0`, `@adai-reviewed: 2026-11-23` (TD-395: `GPUMatrix` moved to `MatrixGPU.hpp`, `matrix_zero_gpu()` declared)
- **Last traced against the code:** 2026-11-23 (TD-395)

> Code-traced reference. The `GPUMatrix` here, and the shared `GPUKVCache`/`GPUDecoderKVCache`
> built on it, were **run** on the oneAPI CPU device through the scratch harness described in
> [MatrixGPU_SYCL_cpp.md](MatrixGPU_SYCL_cpp.md) (a copy of the backend whose `GPUManager` also
> accepts CPU devices).

---

## 1. What this file is

The SYCL half of the GPU math API: `ActivationType` and the `matrix_*_gpu` declarations, matching
`MatrixGPU.hpp`'s CUDA block line for line (including TD-395's `matrix_zero_gpu()`).

`gpu/MatrixGPU.hpp` includes this file under `ADAI_GPU_BACKEND_SYCL` and then, after the backend
split, defines `GPUMatrix` and `GPUKVCache`/`GPUDecoderKVCache` once for both backends
([MatrixGPU.md](MatrixGPU.md) §2–§3). Until November 23, 2026 (TD-395) this file held a second,
hand-maintained copy of `GPUMatrix` that differed only in `copy()` (which drained the queue) and
`zero()`; see the addendum below.

---

## 2. Verification on the CPU device

| Check | Result |
|---|---|
| `operator-` (via `−1 ×` and add), `copy()` + `zero()` (source left intact) | Correct |
| A fresh `GPUMatrix(2, 4)` | **Uninitialized**: first value `6.6×10⁻³⁷` (TD-389) |
| `GPUMatrix(0, 8)` | Threw `"sycl::malloc_device failed: allocation returned null"` (TD-392). **Fixed November 18, 2026:** an empty matrix, checked in `EmptyGPUMemoryIsValid` on the CPU device |
| `big(4×4).add_inplace(small(2×2))` | **Accepted with no error**, reading past `small`'s buffer (TD-389) |
| `GPUKVCache(8, 4)`: two `append()`s (3 + 2 rows), `get_keys()`/`get_values()`, `clear()` | Correct: rows land at the right offsets, `size()` = 5, then 0, capacity kept. **This is the first time the GPU KV cache has been run anywhere** (TD-389 noted it had no tests) |
| `GPUDecoderKVCache(2, max_seq_length = 8, 4)`: append a 12-row encoder output to a cross-attention cache | **Throws** `"GPUKVCache::append: exceeds max_seq_length capacity"`, confirming TD-387's failure mode for over-length input |

The kernels underneath were verified in [MatrixGPU_SYCL_cpp.md](MatrixGPU_SYCL_cpp.md) §2, apart
from the TD-384 and TD-393 failures recorded there.

---

## 3. Who calls what

Everything that uses the GPU math API includes `MatrixGPU.hpp`, never this file directly; see
[MatrixGPU.md](MatrixGPU.md) §4.

---

## 4. Known gaps and gotchas (summary)

Each new item is tracked in [TECHNICAL_DEBT.md](../../guides/TECHNICAL_DEBT.md) and tagged in the code with
`TODO: See TD-NNN`.

| Item | Impact | Basis | Tracked as |
|---|---|---|---|
| ~~`GPUMatrix` is duplicated in full between the CUDA and SYCL headers, though only `copy()`/`zero()` differ~~ **Fixed November 23, 2026**: one definition in `MatrixGPU.hpp` | — | Tested on the CPU device | [TD-395](../../archive/TECHNICAL_DEBT_RESOLVED.md#td-395-gpumatrix-is-duplicated-between-the-cuda-and-sycl-headers) |
| ~~SYCL `copy()` does a `.wait()`~~ **Fixed November 23, 2026**: queued with no wait | — | Tested on the CPU device | [TD-395](../../archive/TECHNICAL_DEBT_RESOLVED.md#td-395-gpumatrix-is-duplicated-between-the-cuda-and-sycl-headers) |
| Unchecked `add_inplace` and other operations; uninitialized memory (§2). Zero-size throws, fixed November 18, 2026 (TD-392) | As in the CUDA class | **Verified** on the CPU device (TD-389, TD-392) | [TD-389](../../guides/TECHNICAL_DEBT.md#td-389-gpumatrix-shape-checks-uninitialized-memory-and-untested-gpu-kv-caches) (amended), [TD-392](../../archive/TECHNICAL_DEBT_RESOLVED.md#td-392-sycl-gpumemory0-throws-where-cuda-succeeds) |
| Over-length encoder output throws from the cross-attention cache (§2) | GPU serving fails over-length requests | **Verified** at the cache level (TD-387) | [TD-387](../../guides/TECHNICAL_DEBT.md#td-387-gpu-generation-fails-on-input-longer-than-max_seq_length) (amended) |
| ~~Header status note says "capped by TD-050", though TD-050 is resolved~~ **Fixed November 23, 2026** | — | — | [TD-395](../../archive/TECHNICAL_DEBT_RESOLVED.md#td-395-gpumatrix-is-duplicated-between-the-cuda-and-sycl-headers) |

## Addendum (November 4, 2026): `matrix_token_correct_gpu()` (TD-276)

New kernel for validation token accuracy: one work-group of 256 per row finds the row's argmax (lowest index on ties, matching the
CPU's first maximum) by a strided scan and a tree reduction in local memory, and writes 1 when it equals the row's target (a target
outside `[0, vocab_size)` never matches). The per-row flags are summed with `matrix_sum_gpu()`, so the call downloads one number.
Same algorithm in CUDA and SYCL. Tested by `MatrixGPUTD003Test.TokenCorrectCountMatchesHostArgmax` (40 rows of 1000 with a tie
and out-of-range targets), run on the oneAPI CPU device; the CUDA version is compile-checked only.

## Addendum (November 13, 2026): `copy()` and `zero()` traced (TD-574)

`GPUMatrix::copy()` and `zero()`, the two operations that use the queue directly, call `GPUManager::trace_kernel()`
(`gpu_matrix_copy`, `gpu_matrix_zero`), like every kernel entry point; see [GPUDebugSync.md](GPUDebugSync.md).

## Addendum (November 14, 2026): the wait timeout (TD-574)

`GPUMatrix::copy()`'s wait goes through `GPUManager::wait_for()`, so it gives up after `GPU_WAIT_TIMEOUT_SECONDS` on a GPU that
stopped running work, like every other wait in the SYCL backend ([GPUUtils_SYCL.md](GPUUtils_SYCL.md)).

## Addendum (November 23, 2026): `GPUMatrix` moved out (TD-395)

This file no longer defines `GPUMatrix`; the single definition is in `MatrixGPU.hpp`, after the backend split. Its `copy()` now
calls `matrix_copy_device_to_device_gpu()` (traced as `matrix_copy_device_to_device_gpu`) and no longer waits, so the two addenda
above describe the removed copy: the `gpu_matrix_copy`/`gpu_matrix_zero` trace names are gone (`zero()` is traced as
`matrix_zero_gpu`), and `copy()` has no wait to time out.
