# `Matrix` — Source File Reference

- **Files:** [`src/Matrix.hpp`](../../../../src/Matrix.hpp), [`src/Matrix.cpp`](../../../../src/Matrix.cpp) (1,349 lines together); SIMD detection in [`src/MatrixSIMD.hpp`](../../../../src/MatrixSIMD.hpp) ([MatrixSIMD.md](MatrixSIMD.md))
- **Namespace:** global
- **Built into:** `adai_core`; the numeric type of every layer, optimizer, trainer and inference engine
- **Status tag:** `@adai-status: beta` ("capped by TD-033"), `Matrix.cpp` `@adai-version: 0.9.2`, `@adai-reviewed: 2026-10-17`; `Matrix.hpp` `0.9.0`, `2026-09-10`
- **Tests:** [`tests/matrix_test.cpp`](../../../../tests/matrix_test.cpp) (60 tests), [`tests/matrix_simd_test.cpp`](../../../../tests/matrix_simd_test.cpp) (20), [`tests/matrixgpu_td003_test.cpp`](../../../../tests/matrixgpu_td003_test.cpp) (13, GPU helper kernels)
- **GPU kernels:** [MatrixGPU.md](MatrixGPU.md), [MatrixGPU_cu.md](MatrixGPU_cu.md), [MatrixGPU_SYCL_cpp.md](MatrixGPU_SYCL_cpp.md)
- **Last traced against the code:** 2026-10-19

> Code-traced reference. Behaviour marked **verified** was confirmed with scratch programs compiled
> against the real `Matrix.cpp` (AVX2/FMA + OpenMP, and the portable `-march=x86-64` configuration).

---

## 1. What this file is

A dense, row-major `float` matrix:

```cpp
std::vector<std::vector<float>> data;   // one heap allocation per row
int rows, cols;                          // all three are public and mutable
```

Every operation returns a new `Matrix` (no views, no in-place arithmetic except `apply_gradients()`,
`fill()`, `randomize()` and `set_row()`/`set_col()`). Rows are contiguous, but the matrix as a whole
isn't, so every BLAS or GPU call must first pack it into a flat buffer and unpack the result.

Because `data`, `rows` and `cols` are public, nothing keeps them consistent: assigning `rows` or
resizing a row leaves a matrix that the operations index out of range. `is_valid()` checks this, but
nothing calls it on the hot path.

---

## 2. API

### Construction and access

| Member | Behaviour |
|---|---|
| `Matrix()` | 0×0 |
| `Matrix(r, c)` | Zero-filled. No check for negative sizes: `std::vector` throws `length_error`/`bad_alloc` instead of a dimension error |
| `Matrix(vector<vector<float>>)` | Copies; throws `invalid_argument` on ragged rows |
| Copy (defaulted), move (leaves the source 0×0) | — |
| `operator()(i, j)` (const and non-const) | **Bounds-checked** (`out_of_range("Matrix index out of bounds")`) and defined out-of-line in `Matrix.cpp`. Without LTO it can't be inlined: 5.2M reads took **16.7 ms** through `operator()` against **6.9 ms** through `data[i][j]` (2.4×, **verified**). Hot loops that use it (about 80 sites in `MultiHeadAttention.cpp`) pay this cost |
| `get_row()`, `get_col()`, `set_row()`, `set_col()` | Bounds- and size-checked copies |

### Arithmetic

| Member | Behaviour |
|---|---|
| `operator*` | Matrix product; §3 |
| `operator+`, `operator-`, `hadamard()`, `scale()` | Shape-checked (`invalid_argument`), AVX2 or NEON row by row, else scalar |
| `transpose()` | OpenMP when `rows·cols > 10,000` |
| `apply_gradients(g, lr)` | In place, `W −= lr·g` with FMA; OpenMP over rows above 10,000 elements. Used by the layers' SGD fallbacks |
| `fill(v)`, `sum()`, `mean()` | `sum()` accumulates in `float` (8 AVX2 lanes plus a scalar tail): summing 16.8M copies of 1e-4 gave **1714.48** against the exact 1677.72, **a 2.2% error** (**verified**). Only one caller outside `Matrix.cpp` uses `sum()`/`mean()`, so the practical impact is small; `mean()` of an empty matrix is 0 |
| `randomize(scale)` | `N(0, scale)` from a **function-static** `random_device`-seeded `mt19937`: unseedable (TD-368) and **not thread-safe** (two threads initializing layers at once race on the generator) |
| `reshape(r, c)` | Row-major copy; checks the element count only |
| `is_valid()`, `print()` | `print()` writes to `std::cout` (debug only; no production caller) |

### GPU interop (`ADAI_ENABLE_GPU`)

| Member | Behaviour |
|---|---|
| `gpu_available()`, `gpu_initialize()`, `gpu_try_initialize()` (probe, then initialize, catching exceptions), `gpu_cleanup()`, `gpu_info()` | Thin forwards to `GPUManager`; CPU-only stubs otherwise |
| `multiply_gpu()`, `add_gpu()`, `transpose_gpu()`, `scale_gpu()`, `hadamard_gpu()` | **Allocate, upload, compute, download and free on every call** (TD-033). They **fall back silently to the CPU** when no GPU is available; they don't throw. Only `multiply_gpu()` has a production caller, through `operator*`'s auto-dispatch; the others are used by tests |
| `to_gpu()` / `from_gpu()` | Persistent `GPUMatrix` upload/download (TD-003). `to_gpu()` throws `runtime_error` without a GPU. `from_gpu()` is used by `LLMEncoder`'s and `Decoder`'s GPU backward paths |

A zero-row operand reaching a SYCL `GPUMemory` throws (TD-392).

---

## 3. `operator*` dispatch

Checked once per call, in this order (the first match wins):

| # | Path | Condition | Notes |
|---|---|---|---|
| 1 | GPU (`multiply_gpu()`) | `GPUManager::is_available() && implicit_gpu_dispatch_allowed() && cols ≥ 32 && other.cols ≥ 32` | Per-call round trip (TD-033); ignores `rows`, so a 1-row decode step still round-trips. A thread holding `adai::gpu::ScopedCpuOnly` stays on the CPU (TD-382, October 17, 2026); other threads still share one stream |
| 2 | BLAS `cblas_sgemm` | `ADAI_ENABLE_BLAS` and every dimension ≥ 256 | Packs and unpacks. **Inactive in the local build**: CMake finds no `cblas.h` |
| 3 | AVX2 (+FMA) ikj | `__AVX2__` at compile time | OpenMP `schedule(dynamic, 16) if (rows > 64)` |
| 4 | NEON ikj | AArch64 | Same OpenMP gate |
| 5 | OpenMP ijk | `ADAI_ENABLE_OPENMP` only | `if (rows > 64 && other.cols > 64)`; strided reads of `other` |
| 6 | Scalar ijk | — | — |

**Correctness (verified):** every CPU path agrees with a double-precision reference within 1.1×10⁻⁵
relative error, for sizes up to 300×260×270, including widths that aren't multiples of 8.

### Two performance problems (verified)

**The OpenMP gate is on `rows` only, so short sequences run on one thread.** Training samples and
decode steps usually have ≤ 64 rows (tokens), so the weight-matrix products they do (`d_model × d_model`,
`d_model × d_ff`, `d_model × vocab`) never parallelize, however large `cols` and `other.cols` are.
Measured on this 8-core machine (AVX2 build, best of several runs):

| Shape (rows × inner × cols) | Current gate | Work-based gate (`rows·cols·other.cols > 2¹⁸`, static schedule) |
|---|---|---|
| 64 × 512 × 512 (attention projection) | 2.6–3.4 ms | 0.7–1.0 ms |
| 64 × 512 × 2048 (FFN) | 15.6–18.9 ms | 7.2–8.1 ms |
| 300 × 512 × 2048 (already parallel) | 17.3–17.9 ms | 14.2–15.5 ms |
| 8 × 512 × 8000 (LM head, decode) | 12.2–12.7 ms | 2.0–2.5 ms |

That's 2–6× on the shapes CPU training and inference actually use. Callers that already run inside
an OpenMP region (`MultiHeadAttention`'s per-head loop) get a serialized inner region under the default
`OMP_MAX_ACTIVE_LEVELS`, so a work-based gate doesn't oversubscribe.

**Portable builds lose the SIMD paths entirely.** `ADAI_SIMD_AVX2` comes from the compiler's
`__AVX2__`, and the `portable` preset (`-march=x86-64`, deliberately no `-mavx2`) leaves it undefined,
so `operator*` falls to path 5: about **7–10× slower** than the AVX2 build on the same shapes
(64×512×2048: 162 ms against 16 ms; 8×512×8000: 93 ms against 12.7 ms). The CLAUDE.md preset table
recommends `portable` for "binaries deployed to other machines". That's harmless for the daemons
(`metrics_api_server`, `mns_server`, `registry_server`) but costly for a CPU `chatbot_api_server` or
trainer. The usual fix is runtime dispatch: compile the AVX2 kernel with `__attribute__((target("avx2,fma")))`
and select it with the existing `adai::simd` CPUID helpers in `MatrixSIMD.hpp`.

---

## 4. Thread safety

- `const` operations are safe to call concurrently on the same matrix; mutation isn't (no locking,
  as for `std::vector`).
- `randomize()` races on its static generator (§2).
- The GPU auto-dispatch runs from any thread on the shared stream. Since TD-382's fix (October 17, 2026),
  a thread holding `adai::gpu::ScopedCpuOnly` stays on the CPU, and the CUDA budget counter and cuBLAS
  handle are locked.
- The block comment at the top of `Matrix.cpp` documents ThreadSanitizer reports inside the OpenMP
  regions as a GCC-libgomp + TSan false positive (libgomp isn't TSan-instrumented); that analysis
  still matches the code.

---

## 5. Tests

- `matrix_test.cpp` (58): construction, access and bounds, every arithmetic operation and its shape
  checks, transpose, reshape, row/column access, `fill`/`sum`/`mean`, `apply_gradients`, `is_valid`.
- `matrix_simd_test.cpp` (20): each SIMD path (add, subtract, Hadamard, scale, `apply_gradients`,
  `sum`, multiply) against a scalar reference over parameterized shapes, a large BLAS-sized multiply,
  edge cases, and the CPUID helpers (compile-time AVX2 matches runtime).
- `matrixgpu_td003_test.cpp` (13): the persistent-residency GPU helper kernels from `MatrixGPU.hpp`
  (threshold counting, row entropy, device copies, in-place add, softmax, masked fill, row sums),
  not `Matrix` itself.

**Not covered:** `sum()` accuracy on large inputs; the BLAS path specifically (its test runs whichever path is compiled in); multiply
performance or threading thresholds; concurrent `randomize()`.

---

## 6. Known gaps and gotchas (summary)

Each new item is tracked in [TECHNICAL_DEBT.md](../../guides/TECHNICAL_DEBT.md) and tagged in the code with
`TODO: See TD-NNN`.

| Item | Impact | Basis | Tracked as |
|---|---|---|---|
| `operator*`'s OpenMP gate is `rows > 64`, so ≤ 64-row products (training samples, decode steps) run single-threaded (§3) | 2–6× slower CPU matmuls on the common shapes | **Verified** (timing) | [TD-434](../../guides/TECHNICAL_DEBT.md#td-434-matrixoperator-runs-short-products-single-threaded-and-portable-builds-have-no-simd) |
| Portable builds have no SIMD matmul; AVX2 is compile-time only (§3) | 7–10× slower `operator*` in `portable` binaries | **Verified** (timing) | [TD-434](../../guides/TECHNICAL_DEBT.md#td-434-matrixoperator-runs-short-products-single-threaded-and-portable-builds-have-no-simd) |
| `operator()` is out-of-line and bounds-checked (§2) | 2.4× slower element access in hot loops that use it | **Verified** (timing) | [TD-435](../../guides/TECHNICAL_DEBT.md#td-435-matrix-accessor-cost-sum-accuracy-randomize-race-and-api-hygiene) |
| `sum()` accumulates in `float` (§2) | 2.2% error at 16.8M elements; few callers | **Verified** | [TD-435](../../guides/TECHNICAL_DEBT.md#td-435-matrix-accessor-cost-sum-accuracy-randomize-race-and-api-hygiene) |
| `randomize()` static RNG isn't thread-safe (§2) | Data race if layers initialize concurrently | By inspection | [TD-435](../../guides/TECHNICAL_DEBT.md#td-435-matrix-accessor-cost-sum-accuracy-randomize-race-and-api-hygiene) |
| Public mutable `data`/`rows`/`cols`; negative sizes give `length_error`; `print()` to stdout; `*_gpu()` wrappers unused outside tests (§1, §2) | API hygiene | By inspection | [TD-435](../../guides/TECHNICAL_DEBT.md#td-435-matrix-accessor-cost-sum-accuracy-randomize-race-and-api-hygiene) |
| Per-op GPU round trips; GPU dispatch ignores `rows`; unseedable `randomize()`; SYCL zero-size buffers | Already tracked | — | TD-033, TD-368, TD-392 (the shared-stream race, TD-382, was fixed October 17, 2026) |

---

## 7. Merge notes

This file replaces `docs/development/api/core/matrix.md` (1,315 lines). Where they disagree, this file
wins:
- **GPU error handling.** It said the `*_gpu()` methods throw `runtime_error` without a GPU; they fall
  back to the CPU silently. Only `to_gpu()` throws.
- **SIMD macros.** It said `ADAI_SIMD_AVX2`/`FMA` follow `ENABLE_SIMD=ON`; they follow the compiler's
  `__AVX2__`/`__FMA__`, which `PORTABLE_BUILD` deliberately suppresses.
- **"Memory pooling".** Its example claimed `temp = compute_something()` "reuses same memory"; move
  assignment replaces the storage.
- **BLAS "default ON".** The option is on, but the path only activates when `cblas.h` is found, which it
  isn't in the local build.

Kept: the storage layout, the constructor and operation semantics, the exception types and the
build-option table (corrected). Dropped: generic tutorial material (integration patterns, debugging
tips, mathematical foundations, complexity tables, future-enhancement wish lists).

---

## Addendum (October 20, 2026): tiled, runtime-dispatched multiply (TD-434)

`operator*`'s CPU path (after the cuBLAS and BLAS checks) now computes tiles of one row × 256 output columns, run in parallel
(`schedule(static)`) when the product has more than 2¹⁸ multiply-adds. Products with few rows use every core. Each output element
sums over k in order, so results are bit-identical whatever the thread count. The tile kernel is chosen once per process:
- `avx2_fma`: x86 GCC/Clang builds, compiled with `target("avx2,fma")` and used when
  `adai::simd::cpu_supports_avx2_fma()` is true, so `portable` builds get it too;
- `neon`: ARM;
- `generic`: a contiguous loop the compiler vectorizes.

`Matrix::cpu_matmul_kernel()` returns the name, and `ADAI_MATMUL_KERNEL=generic` (set before the first multiply) forces the generic
kernel. 64×512×2048 went from 15.4 to 2.7 ms native and from 161 to 2.6 ms portable. Tests: `matrix_simd_test.cpp`'s
`MatrixTiledMulTest`, `ResultDoesNotDependOnThreadCount`, `KernelMatchesTheCpuAndTheOverride`, and the
`MatrixSIMDTestsGenericKernel` ctest entry.
