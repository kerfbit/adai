# `MatrixSIMD` — Source File Reference

- **File:** [`src/MatrixSIMD.hpp`](../../../../src/MatrixSIMD.hpp) (header-only, 137 lines)
- **Namespace:** `adai::simd`
- **Built into:** header-only; included by `Matrix.hpp`, so every translation unit that uses `Matrix` sees it. Only `Matrix.cpp` uses its macros and helpers
- **Status tag:** `@adai-status: stable`, `@adai-version: 1.0.0`, `@adai-reviewed: 2026-09-10`
- **Tests:** [`tests/matrix_simd_test.cpp`](../../../../tests/matrix_simd_test.cpp) (`MatrixSIMDCPUDetect` suite; the rest of the file tests `Matrix`'s SIMD paths, see [Matrix.md](Matrix.md))
- **Build wiring:** `ENABLE_SIMD` in [`src/CMakeLists.txt`](../../../../src/CMakeLists.txt); release flags in the top-level `CMakeLists.txt`
- **Last traced against the code:** 2026-10-14

> Code-traced reference. Behaviour marked **verified** was confirmed by compiling and timing the real
> `Matrix.cpp` under each preset's compiler and flags.

---

## 1. What this file is

Three small pieces that let `Matrix.cpp` choose SIMD code:

| Piece | What it does |
|---|---|
| **Compile-time macros** | `ADAI_SIMD_AVX2` when the compiler defines `__AVX2__` (plus `ADAI_SIMD_FMA` with `__FMA__`); otherwise `ADAI_SIMD_SSE4` with `__SSE4_1__`; `ADAI_SIMD_NEON` with `__ARM_NEON`. Includes `<immintrin.h>`/`<smmintrin.h>`/`<arm_neon.h>` accordingly |
| **Runtime detection** | `cpuid_query()` (MSVC `__cpuidex`, GCC/Clang `__get_cpuid_count`, zeros on failure); `has_avx2()` (CPUID leaf 7 EBX bit 5, after checking the maximum leaf ≥ 7); `has_fma()` (leaf 1 ECX bit 12). Both return `false` off x86 |
| **Reduction helpers** | `hsum256(__m256)` (two `hadd`s) and `hsum128(float32x4_t)` (`vaddvq_f32`), used by `Matrix::sum()` |

The selection is **entirely compile-time**: `Matrix.cpp` uses `#if defined(ADAI_SIMD_AVX2)` and so on.
`has_avx2()`/`has_fma()` have **no production caller** (only the test suite calls them), so nothing
checks at startup that the CPU can run what was compiled in.

---

## 2. Where the macros are actually defined

Because the macros follow the compiler's own `__AVX2__`, what a build gets depends on its flags:

| Preset | Compiler | Flags on `Matrix.cpp` | `ADAI_SIMD_AVX2` in `Matrix.cpp` |
|---|---|---|---|
| `debug`, `relwithdebinfo`, sanitizers | GCC/Clang | `-mavx2 -mfma` (`ENABLE_SIMD`, `PRIVATE` to `adai_core`) | Yes |
| `release`, `ci`, `gpu` (CUDA) | GCC/Clang | `-march=native` (top-level), plus `-mavx2 -mfma` | Yes, if the build host has AVX2 |
| `portable` | GCC/Clang | `-march=x86-64` (deliberately no AVX2) | No (TD-434) |
| **`sycl`** (ai-machine) | **`icpx`** | **none**: `-O3 -DNDEBUG -fsycl -fiopenmp` only | **No** |

### The SYCL build never gets SIMD (verified)

Both the top-level release flags and the `ENABLE_SIMD` block match the compiler ID against
`"GNU|Clang"`. oneAPI's `icpx` reports `IntelLLVM`, so **neither applies**. The configured
`build/sycl` compiles `Matrix.cpp` with `-O3 -DNDEBUG -std=gnu++17 -fsycl -fiopenmp`, and `icpx`'s default
target defines `__SSE2__` but not `__AVX2__` (checked with `icpx -dM -E`). So the trainer on ai-machine
runs every CPU-side `Matrix` operation on the non-SIMD paths. Timing the real `Matrix.cpp` built with
`icpx` (this machine, 8 threads):

| Shape | `icpx` as configured | `icpx -march=native` |
|---|---|---|
| 64 × 512 × 512 | 24.8 ms | 3.2 ms |
| 64 × 512 × 2048 | 136.9 ms | 16.7 ms |
| 300 × 512 × 2048 | 101.4 ms | 17.1 ms |
| 8 × 512 × 8000 | 89.8 ms | 12.9 ms |

That's 6–8× on the multiply. On ai-machine, `operator*` sends most products to the GPU (`cols ≥ 32 &&
other.cols ≥ 32`), so the multiply penalty applies to what stays on the CPU: products under that
threshold, any work while the GPU is unavailable or has fallen back, and the element-wise operations
(`+`, `-`, `hadamard`, `scale`, `apply_gradients`, `sum`), which get only the compiler's SSE2
auto-vectorization. How much of a pass that is hasn't been measured on ai-machine.

### Macros differ between translation units

`-mavx2 -mfma` is `PRIVATE` to `adai_core`, so in `debug`-style builds `ADAI_SIMD_AVX2` is defined in
`Matrix.cpp` but **not** in any other file that includes `Matrix.hpp`. Today that's harmless (no other
file tests the macros, and `Matrix`'s layout doesn't depend on them), but:
- the old `api/core/matrix.md` said the macros are available "in any translation unit that includes
  `Matrix.hpp`", which isn't true;
- the test `MatrixSIMDCPUDetect.CompileTimeAVX2MatchesRuntime` is wrapped in `#if
  defined(ADAI_SIMD_AVX2)` inside the test file, which isn't compiled with `-mavx2`, so **it never
  compiles in the debug build** (verified: `--gtest_list_tests` shows only the two "ReturnsBool" tests).
  The one test meant to compare the compile-time choice with the CPU never runs.

---

## 3. Other gaps (by inspection)

- **No startup check.** A `release`/`gpu` binary built with `-march=native`, or any `debug` build, run on a
  CPU without the build host's features dies with `SIGILL` ("Illegal instruction") the first time a SIMD
  instruction executes, with no message saying why. `has_avx2()` exists but isn't called.
- **Incomplete runtime detection.** `has_avx2()` and `has_fma()` check only the CPUID feature bits, not
  `OSXSAVE` and `XGETBV` (whether the OS saves the YMM registers). Under a hypervisor or OS that disables
  AVX state they'd report `true` for code that would fault. This matters once TD-434's runtime dispatch
  relies on them; GCC/Clang's `__builtin_cpu_supports("avx2")` handles it.
- **`ADAI_SIMD_SSE4` is unused**: nothing in the code base tests it.
- **Header weight.** `<immintrin.h>` is pulled into every file that includes `Matrix.hpp`.

---

## 4. Tests

`MatrixSIMDCPUDetect` has three tests: `has_avx2()` and `has_fma()` return a `bool` (always passes), and
`CompileTimeAVX2MatchesRuntime`, which doesn't compile in practice (§2). `hsum256()`/`hsum128()` are covered
indirectly by `Matrix::sum()`'s SIMD tests.

**Not covered:** what each preset actually compiles in; the SYCL build's flags.

---

## 5. Known gaps and gotchas (summary)

Each new item is tracked in [TECHNICAL_DEBT.md](../../guides/TECHNICAL_DEBT.md) and tagged in the code with
`TODO: See TD-NNN`.

| Item | Impact | Basis | Tracked as |
|---|---|---|---|
| `icpx` (`IntelLLVM`) matches neither `"GNU\|Clang"` check, so the `sycl` preset gets no `-march`/`-mavx2`; `Matrix.cpp` has no SIMD on ai-machine (§2) | 6–8× slower CPU multiply; scalar/SSE2 element-wise ops in the trainer's CPU-side work | **Verified** (flags and timing) | [TD-436](../../guides/TECHNICAL_DEBT.md#td-436-sycl-icpx-builds-get-no--march-or-simd-flags) |
| `CompileTimeAVX2MatchesRuntime` never compiles (macros are per-translation-unit); doc claim that they're global is wrong (§2) | The only compile-time/runtime consistency test never runs | **Verified** | [TD-437](../../guides/TECHNICAL_DEBT.md#td-437-simd-self-check-test-never-compiles-and-no-startup-cpu-check) |
| No startup CPU check; `-march=native`/`-mavx2` binaries `SIGILL` on older CPUs without explanation (§3) | Confusing crash when a binary is copied to another host | By inspection | [TD-437](../../guides/TECHNICAL_DEBT.md#td-437-simd-self-check-test-never-compiles-and-no-startup-cpu-check) |
| `has_avx2()`/`has_fma()` skip the `OSXSAVE`/`XGETBV` check; `ADAI_SIMD_SSE4` unused; `<immintrin.h>` in a widely included header (§3) | Wrong answer under AVX-disabled hypervisors once runtime dispatch exists; hygiene | By inspection | [TD-437](../../guides/TECHNICAL_DEBT.md#td-437-simd-self-check-test-never-compiles-and-no-startup-cpu-check) |
| AVX2 compile-time only, so `portable` builds have no SIMD multiply | Already tracked | — | TD-434 |
