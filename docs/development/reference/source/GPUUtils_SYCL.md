# `gpu/sycl/GPUUtils_SYCL.hpp` — Source File Reference

- **File:** [`src/gpu/sycl/GPUUtils_SYCL.hpp`](../../../../src/gpu/sycl/GPUUtils_SYCL.hpp) (header-only, 420 lines)
- **Namespace:** `adai::gpu` (`GPUManager`, `GPUMemory<T>`): the same public API as the CUDA versions in [GPUUtils.md](GPUUtils.md)
- **Built into:** SYCL builds (`ENABLE_SYCL`, the `sycl` preset, `icpx`), through `gpu/GPUUtils.hpp`. **This is the backend ai-machine trains on** (Intel Arc)
- **Status tag:** `@adai-status: beta`, `@adai-version: 0.6.3`, `@adai-reviewed: 2026-09-13`
- **Tests:** shared with CUDA, [`tests/gpuutils_test.cpp`](../../../../tests/gpuutils_test.cpp) (19 tests; 9 skip without a GPU, as they do in `build/sycl-gputest` here)
- **Last traced against the code:** 2026-10-19

> Code-traced reference. There's no Intel GPU here, and `GPUManager` deliberately ignores the oneAPI
> CPU device this machine does have. The SYCL behaviours marked **verified** were measured with a
> standalone `icpx -fsycl` program that reproduces the exact patterns on an in-order CPU queue.

---

## 1. What this file is

The SYCL implementation of the **GPU device and memory manager**:
- one in-order `sycl::queue` on the selected GPU, with a priority hint from `GPU_STRATEGY`;
- a memory budget (a fraction of the device's global memory);
- `GPUMemory<T>`, an RAII wrapper over USM `sycl::malloc_device`.

Every SYCL kernel in `MatrixGPU_SYCL.cpp` and every `GPUMatrix` uses this queue.
CLAUDE.md's rule ("**do not add `.wait()` to intermediate kernels**; the in-order queue enforces
ordering") is about this queue.

### Differences from the CUDA manager

| Aspect | CUDA ([GPUUtils.md](GPUUtils.md)) | SYCL (this file) |
|---|---|---|
| Ordering | Non-blocking stream | **In-order queue** |
| Budget counters | Guarded by `memory_mutex_` since TD-382's fix (October 17, 2026) | Guarded by `memory_mutex_` (a ThreadSanitizer-confirmed race fixed earlier) |
| Freeing | `cudaFree` (synchronous) | **Deferred**: a `host_task` on the queue calls `sycl::free` and releases the budget |
| Uploads/downloads | Async upload; download synchronizes the stream | **Both** `.wait()` on the queue (full drain) |
| `set_device()` | Doesn't rebuild anything (TD-383) | Recreates the queue (but see §2) |
| Zero-size allocation | `cudaMalloc(0)` succeeds | Throws (see §3) |
| Device list | CUDA devices | **Every GPU on every SYCL platform** (see §2) |

---

## 2. `GPUManager`

| Member | Behaviour |
|---|---|
| `enumerate_gpu_devices()` (private) | Collects `device_type::gpu` devices from **all** platforms. With both the Level Zero and OpenCL runtimes installed (the usual oneAPI setup on Intel Arc), **the same physical GPU appears once per platform**, so `device_count()` is 2 for one card and `GPU_DEVICE_ID=0` means "whichever platform enumerates first" rather than a chosen backend. Nothing logs which backend was picked. `ONEAPI_DEVICE_SELECTOR` can restrict it externally |
| `make_queue(dev, low)` (private) | In-order queue with `priority_low`/`priority_high` (oneAPI extension; a hint that backends may ignore) |
| `probe()` / `probe_diagnostic()` | Any GPU found? The diagnostic lists every platform and device with a Level Zero / render-group hint, so it's much more useful than CUDA's |
| `initialize(id, fraction, low_priority)` | Like CUDA: idempotent (new arguments ignored), soft-fails with no GPU, validates `id` and `fraction`, budget = global memory × fraction. Also remembers `low_priority` for `set_device()` |
| `cleanup()` | Waits for the queue, deletes it, zeroes `allocated_bytes_`. Live `GPUMemory` buffers are then **never freed**: their destructors see `!is_available()` and skip the free (TD-096). The comment accepts that, since the runtime reclaims them with the context |
| `set_device(d)` | Waits, deletes and recreates the queue on device `d`. **Existing allocations were made against the old queue's context and are later freed with `sycl::free(ptr, new_queue)`**, which is undefined when the device or context differs. The budget isn't recomputed for the new device's memory. No caller |
| `get_queue()` | Throws before `initialize()` |
| Budget: `get_*_bytes()`, `reserve_memory()`, `release_memory()` | All under `memory_mutex_`. `reserve_memory()` first tries; on failure it **drains the queue** (so pending deferred frees run) and retries before throwing |
| `get_device_info(d)` | Name, global memory, compute units, max work-group size, and the ADAI budget. Throws `out_of_range` for an invalid device |
| `synchronize()` | `queue_->wait()` |

---

## 3. `GPUMemory<T>`

| Member | Behaviour |
|---|---|
| `GPUMemory(count)` | Reserves the budget, then `sycl::malloc_device<T>(count, queue)`, releasing the reservation on failure. **`count == 0` throws**: `malloc_device(0)` returns `nullptr` (verified), which this treats as an allocation failure. CUDA's version succeeds for zero, so a zero-row `GPUMatrix` (for example an empty sequence) throws only on SYCL |
| Destructor, move assignment | `defer_free(ptr, bytes)`: submits a `host_task` on the in-order queue that calls `sycl::free` and `release_memory`. This is correct: USM `free` isn't ordered against queued kernels, so freeing directly could race them |
| `copy_from_host`, `copy_to_host` | `queue.memcpy(...).wait()`. **Each waits for everything already queued**, so every upload (weights, embeddings per call, targets per step) is a full pipeline drain |

> **Deferred frees serialize the queue with the host (verified on a CPU queue).** In an in-order
> queue a `host_task` can't start until all earlier commands finish, and nothing after it starts
> until the host callback returns. So every `GPUMemory` destruction inserts a device→host→device
> round trip. On this machine's oneAPI CPU device, the same pattern
> (allocate, small kernel, `host_task` free) took **1.35 ms per iteration**, against 0.014 ms for the
> kernel alone with a reused buffer and 0.051 ms for a blocking `wait()` + `free`. That's about 26×
> slower than simply blocking. GPU timings will differ, but the shape is the same: each free costs a
> host round trip with the device idle. `GPUMatrix` temporaries are created and destroyed many times
> per layer in every forward and backward pass (`MultiHeadAttention.cpp` and `CrossAttention.cpp`
> alone have about 50 `GPUMatrix` uses each), so ai-machine's training steps likely spend much of
> their time in these bubbles. The `ChatbotTrainer` comments about `allocated_bytes_` "racing ahead
> of the deferred host_task frees" (hence its periodic `gpu_synchronize()`) are a symptom of the
> same mechanism. A buffer pool or caching allocator (reuse freed blocks on the same in-order queue
> without a `host_task`) would remove both the bubbles and the accounting lag.

---

## 4. Who calls what

The same callers as the CUDA manager ([GPUUtils.md](GPUUtils.md) §4). `incremental_trainer` passes
`GPU_STRATEGY` (→ `priority_low`/`priority_high`); `chatbot_api_server` doesn't (TD-381). Every
SYCL kernel wrapper and `GPUMatrix` operation calls `get_queue()`.

---

## 5. Known gaps and gotchas (summary)

Each new item is tracked in [TECHNICAL_DEBT.md](../../guides/TECHNICAL_DEBT.md) and tagged in the code with
`TODO: See TD-NNN`.

| Item | Impact | Basis | Tracked as |
|---|---|---|---|
| Deferred `host_task` frees put a device→host→device bubble on the in-order queue for every `GPUMemory` destruction, and every upload/download drains the queue (§3) | Likely a large share of SYCL training and generation time on ai-machine; budget accounting lags (hence `ChatbotTrainer`'s workaround syncs) | **Verified** on a CPU queue (1.35 ms vs 0.051 ms per op); GPU magnitude unmeasured | [TD-390](../../guides/TECHNICAL_DEBT.md#td-390-sycl-deferred-frees-and-blocking-copies-stall-the-in-order-queue) |
| GPUs enumerated across all platforms, so one Arc card appears twice (Level Zero and OpenCL); `GPU_DEVICE_ID` picks by platform order and the chosen backend isn't logged (§2) | The trainer may run on the OpenCL backend instead of Level Zero without anyone knowing | By inspection | [TD-391](../../guides/TECHNICAL_DEBT.md#td-391-sycl-lists-the-same-gpu-once-per-platform-and-doesnt-log-the-backend) |
| `GPUMemory(0)` throws on SYCL but not CUDA (§3) | Zero-size `GPUMatrix` (empty sequences) fails only on SYCL | **Verified** (`malloc_device(0)` → `nullptr`) | [TD-392](../../guides/TECHNICAL_DEBT.md#td-392-sycl-gpumemory0-throws-where-cuda-succeeds) |
| `set_device()` frees old-context allocations with the new queue and doesn't recompute the budget; `cleanup()` leaks live buffers by design and zeroes the counter; `initialize()` ignores new arguments when already initialized (§2) | Latent traps for multi-GPU and re-initialization | By inspection | [TD-383](../../guides/TECHNICAL_DEBT.md#td-383-gpumanager-lifecycle-gaps-set_device-cleanup-initialize) (amended) |
| Unlike CUDA, the budget is locked; the queue itself is thread-safe for submission | TD-382 is milder on SYCL; the shared fix (`ScopedCpuOnly` for the async quality thread) covers it | By inspection; **fixed in code October 17, 2026** | [TD-382](../../guides/TECHNICAL_DEBT.md#td-382-gpu-work-from-any-thread-shares-unsynchronized-gpumanager-state) (amended) |
| Nothing has run on a real Intel GPU in the development environment | — | — | — |

## Addendum (October 19, 2026): CPU-device opt-in (TD-394)

`enumerate_gpu_devices()` lists every device type, not just GPUs, when `ADAI_SYCL_ALLOW_CPU_DEVICE=1` is set. That lets a SYCL build run its real kernels on oneAPI's OpenCL CPU device on a machine with no GPU. It is off by default, so production never falls back to the CPU device silently. It replaced the scratch copy of this header that earlier kernel checks used. TD-468's tests and the trainer suites run this way in `build/sycl-cpudev`, configured with the current `icpx`; the older `build/sycl-gputest` no longer links against oneAPI 2026.2.

---

## Addendum (October 20, 2026): caching allocator (TD-390)

`GPUMemory` no longer frees through a `host_task`. It allocates with `GPUManager::allocate_block(bytes)` and releases with
`release_block(ptr, bytes, generation)`, which keeps the block in a size-bucketed pool (`pool_size()`: 512-byte steps up to 1 MiB,
then an eighth of the next power of two). Reuse needs no wait on the in-order queue: a reused block's earlier commands were
submitted before its next owner's.
- **Budget:** live plus cached bytes count against the budget. A refused allocation waits for the queue, returns the pool to USM
  (`trim_pool_locked()`) and retries.
- **Trimming:** `cleanup()` and `set_device()` trim too, and a generation counter keeps blocks from an old queue out of the new pool.
- **Getters:** `get_used_memory_bytes()` reports live blocks at their pool size, with no lag after a release.
  `get_cached_memory_bytes()` reports the pool.
- **Budget-only calls:** `reserve_memory()`/`release_memory()` still adjust only the budget.
- **Copies:** `copy_from_host()`/`copy_to_host()` still wait.

On the oneAPI CPU device, a temporary (allocate + kernel + free) went from 1.25 ms to 0.0145 ms, and a small training pass from 69 s
to 38 s. Tests: `gpuutils_test.cpp`'s SYCL-only `ReleasedBlockIsReusedAndCountedAtOnce`, `ReusingABlockKeepsQueueOrder`,
`BudgetPressureReturnsThePoolToTheDevice`, `BlockFromBeforeCleanupIsNotPooled` and `GPUManagerPoolSize.RoundsToSharedSizes`.

