# `gpu/GPUUtils.hpp` — Source File Reference

- **File:** [`src/gpu/GPUUtils.hpp`](../../../../src/gpu/GPUUtils.hpp) (header-only, 492 lines)
- **Namespace:** `adai::gpu` (`GPUManager`, `GPUMemory<T>`), plus the `CUDA_CHECK`/`CUBLAS_CHECK` macros
- **Built into:** every GPU-enabled target via `adai_gpu`; the no-GPU stub is used by every other build
- **Status tag:** `@adai-status: beta`, `@adai-version: 0.9.2`, `@adai-reviewed: 2026-09-13`
- **Tests:** [`tests/gpuutils_test.cpp`](../../../../tests/gpuutils_test.cpp) → `gpuutilsTests` (18 tests; 9 need a CUDA device and skip without one)
- **SYCL counterpart:** `src/gpu/sycl/GPUUtils_SYCL.hpp` (documented separately)
- **Last traced against the code:** 2026-10-12

> Code-traced reference. No CUDA device is available in this environment, so nothing here was run on
> real hardware. The CUDA test build (`build/cuda-gputest`) was run: its 9 host-side tests pass and
> the 9 device tests skip. Everything else is **by inspection**.

---

## 1. What this file is

The **GPU device and memory manager**, in three forms selected at compile time:

| Build | What this header provides |
|---|---|
| `ADAI_ENABLE_GPU` + `ADAI_GPU_BACKEND_SYCL` | Nothing of its own: it includes `sycl/GPUUtils_SYCL.hpp`, which defines the same `GPUManager`/`GPUMemory<T>` API on a SYCL queue |
| `ADAI_ENABLE_GPU` (CUDA, the default backend) | `CUDA_CHECK`/`CUBLAS_CHECK`, a CUDA `GPUManager` (one device, one stream, one cuBLAS handle, a memory budget) and the RAII `GPUMemory<T>` |
| Neither | A stub `GPUManager` whose `probe()`/`initialize()`/`is_available()` return false. It has no `GPUMemory`, `get_stream()` or `reserve_memory()`; code that needs those is compiled only under `ADAI_ENABLE_GPU` |

The rest of the codebase reaches it mostly through `Matrix`'s wrappers (`gpu_try_initialize()`,
`gpu_available()`, `gpu_cleanup()`, `gpu_info()`) and, for kernels, through `MatrixGPU.hpp`.

### Why it matters

`GPUManager::is_available()` is the global switch for GPU execution. Once `initialize()` succeeds,
**every** `Matrix::operator*` with inner dimensions of at least 32 goes to cuBLAS on this one stream,
in every thread of the process, and the layers' `gpu_*` paths allocate through `GPUMemory`. The design
goals in the header are about sharing a GPU politely with other tenants:
- a low-priority stream that other work can preempt;
- a memory budget as a fraction of the device;
- a `cleanup()` that never calls `cudaDeviceReset()`, which would kill other processes' contexts.

---

## 2. CUDA `GPUManager`

All state is `inline static`, so there's one manager per process and **no locking**.

| Member | Behaviour |
|---|---|
| `probe()` | `cudaGetDeviceCount() == cudaSuccess && n > 0`. Safe before `initialize()` |
| `probe_diagnostic()` | `"CUDA found N device(s)"`. It ignores the error code, so a driver failure reports "0 device(s)" without saying why |
| `initialize(device_id = 0, memory_fraction = 0.5, use_low_priority = true)` | Idempotent (returns true once initialized, **ignoring any new arguments**). Soft-fails (returns false) with no device. Throws `out_of_range` for a bad device, `invalid_argument` for a fraction outside (0, 1]. Then: `cudaSetDevice`, budget = **total** device memory × fraction (not free memory), a non-blocking stream at the lowest priority (or the highest when `use_low_priority` is false), and a cuBLAS handle bound to it. If a later step throws, the stream created earlier is leaked and `initialized_` stays false |
| `cleanup()` | Synchronizes and destroys the stream, destroys the cuBLAS handle, **sets `allocated_bytes_` to 0**, and clears `initialized_`. It doesn't free (or know about) live `GPUMemory` objects. Layers keep persistent `GPUState` mirrors, so after `cleanup()` and a re-`initialize()` the budget undercounts what's really allocated |
| `is_available()`, `device_count()`, `current_device()` | Plain getters |
| `set_device(d)` | Bounds-checks and calls `cudaSetDevice(d)`. Its doc says it "also resets the memory budget to the new device's capacity", but **it doesn't**. It also doesn't recreate the stream or cuBLAS handle, which stay bound to the old device, so any GPU work after switching would fail or run on the old device's stream. **No caller exists** |
| `get_cublas_handle()`, `get_stream()` | Throw before `initialize()` |
| `get_memory_limit_bytes()`, `get_used_memory_bytes()`, `get_available_memory_bytes()` | Budget getters. Tests only |
| `reserve_memory(bytes)` / `release_memory(bytes)` | Called by `GPUMemory`. Reserve throws when the budget would be exceeded (budget 0 = unlimited, which is the state before `initialize()`). Release clamps at 0. Guarded by `memory_mutex_` since TD-382's fix (it used to be an unlocked `size_t`; see §4) |
| `get_device_info(device = -1)` | Device name, compute capability, memory, SMs, threads per block, and the ADAI budget for the current device. Throws `out_of_range` for an invalid or never-initialized device. Used by `Matrix::gpu_info()` for the startup log |
| `synchronize()` | `cudaStreamSynchronize` on ADAI's stream only. Used by `EncoderDecoderModel::gpu_synchronize()` |

## 3. CUDA `GPUMemory<T>`

| Member | Behaviour |
|---|---|
| `GPUMemory(count)` | Reserves `count × sizeof(T)` in the budget, then `cudaMalloc`; on failure it releases the reservation and throws. The multiplication isn't overflow-checked |
| Destructor | `cudaFree` and release. `cudaFree` waits for the device, so it's safe with work still queued, but it stalls the CPU on every free |
| Move constructor and assignment | Transfer ownership; assignment frees the old buffer first. Not copyable |
| `copy_from_host(ptr, n)` | `cudaMemcpyAsync` on ADAI's stream; throws when `n > size()`. The comment says the host buffer must stay valid until the stream syncs. That's true for pinned memory, but every caller passes pageable `std::vector` storage, for which CUDA stages the copy before returning, so the callers' temporaries are safe |
| `copy_to_host(ptr, n)` | Async copy, then `cudaStreamSynchronize`, so it behaves as a blocking copy |

---

## 4. Who calls what

| Caller | Uses |
|---|---|
| `Matrix::gpu_try_initialize()` | `probe()` then `initialize()`, catching exceptions and falling back to CPU with a warning. Called once at startup by `incremental_trainer` and `chatbot_api_server` |
| `incremental_trainer` | Passes `GPU_DEVICE_ID`, `GPU_MEMORY_FRACTION` and `GPU_STRATEGY` (`background` → low-priority stream, `full` → high) |
| `chatbot_api_server` | Passes only the device and fraction. **`GPU_STRATEGY` is ignored**, so the server always uses the low-priority stream, even though `config.chatbot.conf` ships `GPU_STRATEGY=full` |
| `Matrix::operator*` | `multiply_gpu()` whenever `is_available()` and both inner dimensions are at least 32, **from any thread** |
| Layers' `gpu_*` methods, `MatrixGPU` | `GPUMemory`, `get_stream()`, `get_cublas_handle()` |

**Concurrency.** The manager has no locks. `chatbot_api_server` serializes model work under
`model_mutex_`, but `Matrix::operator*` dispatches to the GPU from any thread that multiplies large
enough matrices. One example: `ChatbotTrainer`'s async BLEU/ROUGE thread is described as "CPU-only"
because it calls `generate_response()` on a cloned model, but that model's `Matrix` products still
go to cuBLAS on the shared stream, concurrently with the training thread. The shared
`allocated_bytes_` counter and cuBLAS handle are then used from two threads without synchronization.
The budget counter can lose updates, and NVIDIA recommends against sharing one cuBLAS handle across
concurrently running host threads.

**Fixed in code October 17, 2026 (TD-382):**
- The budget counter and limit are now guarded by `memory_mutex_`, as in the SYCL manager.
- cuBLAS calls hold `GPUManager::cublas_mutex()`. They only enqueue work, so the lock is brief.
- A thread can opt out of `Matrix::operator*`'s implicit GPU dispatch with `adai::gpu::ScopedCpuOnly`,
  which is per-thread and nestable and is defined in every build. The async BLEU/ROUGE thread now
  holds one, so it really is CPU-only.

Explicit GPU calls (`GPUMatrix`, `gpu_forward()`) don't check the scope. Other threads that multiply
large matrices still share the stream, but no longer race on the counter or the handle.

---

## 5. Tests

`gpuutils_test.cpp` (18 tests): `probe()`/`initialize()` soft-fail, `set_device()` bounds,
`get_device_info()` validation, the budget getters, `reserve_memory()`/`release_memory()` arithmetic,
and (device only) `GPUMemory` allocation, round trip, budget tracking, move semantics and range checks.
In this environment the CUDA build passes 9 and skips 9.

**Not covered:** `cleanup()` followed by re-initialization, `set_device()` on a multi-GPU host, concurrent allocation, or the
`GPU_STRATEGY` wiring.

---

## 6. Known gaps and gotchas (summary)

Each new item is tracked in [TECHNICAL_DEBT.md](../../guides/TECHNICAL_DEBT.md) and tagged in the code with
`TODO: See TD-NNN`.

| Item | Impact | Basis | Tracked as |
|---|---|---|---|
| `chatbot_api_server` ignores `GPU_STRATEGY` (always low priority) although its config ships `full` (§4) | The documented serving GPU mode doesn't take effect | By inspection (definite) | [TD-381](../../guides/TECHNICAL_DEBT.md#td-381-chatbot_api_server-ignores-gpu_strategy) |
| No locking: `allocated_bytes_` and the cuBLAS handle are shared by every thread, and `Matrix::operator*` uses the GPU from any thread, including the "CPU-only" async quality thread (§2, §4) | Lost budget updates; unsupported concurrent cuBLAS handle use during training | By inspection; **fixed in code October 17, 2026** | [TD-382](../../guides/TECHNICAL_DEBT.md#td-382-gpu-work-from-any-thread-shares-unsynchronized-gpumanager-state) |
| `set_device()` doesn't rebind the stream or cuBLAS handle and doesn't reset the budget, contrary to its doc; no caller (§2) | A latent trap for multi-GPU use | By inspection | [TD-383](../../guides/TECHNICAL_DEBT.md#td-383-gpumanager-lifecycle-gaps-set_device-cleanup-initialize) |
| `cleanup()` zeroes the budget while `GPUMemory` buffers still exist; `initialize()` ignores new arguments once initialized and leaks the stream on partial failure; `probe_diagnostic()` hides the error; unchecked `count × sizeof(T)` (§2–3) | Wrong accounting after re-init; confusing diagnostics | By inspection | [TD-383](../../guides/TECHNICAL_DEBT.md#td-383-gpumanager-lifecycle-gaps-set_device-cleanup-initialize) |
| Nothing runs on a real CUDA device in any environment used so far | All device paths unverified | — | — (needs a CUDA host) |
