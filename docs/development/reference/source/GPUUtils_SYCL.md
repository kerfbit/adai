# `gpu/sycl/GPUUtils_SYCL.hpp` — Source File Reference

- **File:** [`src/gpu/sycl/GPUUtils_SYCL.hpp`](../../../../src/gpu/sycl/GPUUtils_SYCL.hpp) (header-only, 676 lines)
- **Namespace:** `adai::gpu` (`GPUManager`, `GPUMemory<T>`): the same public API as the CUDA versions in [GPUUtils.md](GPUUtils.md)
- **Built into:** SYCL builds (`ENABLE_SYCL`, the `sycl` preset, `icpx`), through `gpu/GPUUtils.hpp`. **This is the backend ai-machine trains on** (Intel Arc)
- **Status tag:** `@adai-status: beta` (capped by TD-383), `@adai-version: 0.13.0`, `@adai-reviewed: 2026-11-22` (TD-590 transfer counts)
- **Tests:** shared with CUDA, [`tests/gpuutils_test.cpp`](../../../../tests/gpuutils_test.cpp) (28 tests in a SYCL build, all run on the oneAPI CPU device with `ADAI_SYCL_ALLOW_CPU_DEVICE=1` (one skips with a device present), including TD-392's `EmptyGPUMemoryIsValid`; the two SYCL-only priority tests, `QueueCarriesTheStrategysPriority` and `DefaultPriorityIsNormal`, run on the oneAPI CPU device; 9 skip without a GPU, as they do in `build/sycl-gputest` here); the wait timeout in [`tests/gpu_debug_sync_test.cpp`](../../../../tests/gpu_debug_sync_test.cpp) (`AWaitOnStuckWorkTimesOutAndTheDeviceCountsAsLost`, on the oneAPI CPU device)
- **Last traced against the code:** 2026-11-14 (the wait timeout, TD-574); `make_queue()`, `initialize()` and §4 re-traced 2026-11-15 (`GPU_STRATEGY=normal`) and 2026-11-17 (`normal` the default); §3's zero-size case 2026-11-18 (TD-392)

> Code-traced reference. There's no Intel GPU here, and `GPUManager` deliberately ignores the oneAPI
> CPU device this machine does have. The SYCL behaviours marked **verified** were measured with a
> standalone `icpx -fsycl` program that reproduces the exact patterns on an in-order CPU queue.

---

## 1. What this file is

The SYCL implementation of the **GPU device and memory manager**:
- one in-order `sycl::queue` on the selected GPU, with a priority hint from `GPU_STRATEGY` (`normal`: none);
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
| `make_queue(dev, priority)` (private) | In-order queue with `priority_low` (`LOW`), no priority property (`NORMAL`) or `priority_high` (`HIGH`) (oneAPI extension; a hint that backends may ignore). `gpuutilsTests`' `QueueCarriesTheStrategysPriority` checks each on the oneAPI CPU device |
| `probe()` / `probe_diagnostic()` | Any GPU found? The diagnostic lists every platform and device with a Level Zero / render-group hint, so it's much more useful than CUDA's |
| `initialize(id, fraction, priority = QueuePriority::NORMAL)` | Like CUDA: idempotent (new arguments ignored), soft-fails with no GPU, validates `id` and `fraction`, budget = global memory × fraction. Also remembers `priority` (`priority_`, `NORMAL` until then) for `set_device()` |
| `cleanup()` | Waits for the queue, deletes it, zeroes `allocated_bytes_`. After a timed-out wait (TD-574) it neither waits nor deletes the stuck queue, and leaves it to process exit. Live `GPUMemory` buffers are then **never freed**: their destructors see `!is_available()` and skip the free (TD-096). The comment accepts that, since the runtime reclaims them with the context |
| `set_device(d)` | Waits, deletes and recreates the queue on device `d`. **Existing allocations were made against the old queue's context and are later freed with `sycl::free(ptr, new_queue)`**, which is undefined when the device or context differs. The budget isn't recomputed for the new device's memory. No caller |
| `get_queue()` | Throws before `initialize()` |
| Budget: `get_*_bytes()`, `reserve_memory()`, `release_memory()` | All under `memory_mutex_`. `reserve_memory()` first tries; on failure it **drains the queue** (so pending deferred frees run) and retries before throwing |
| `get_device_info(d)` | Name, global memory, compute units, max work-group size, and the ADAI budget. Throws `out_of_range` for an invalid device |
| `synchronize()` | `wait_for_queue(false)`: waits for the queue, with the timeout |
| `wait_for(event, throw_async = true)` | TD-574: waits for one event like `wait_and_throw()` (or `wait()`), but polls its status against `gpu_wait_timeout()` ([GPUDebugSync.md](GPUDebugSync.md); 60 s by default, 0 = no limit): it yields for the first 200 µs, then sleeps 10 µs doubling to 1 ms per poll. On the deadline it marks the queue stuck, records it for the process (`note_gpu_work_stuck()`, which the binaries check at exit), and throws `std::runtime_error` naming `GPU_WAIT_TIMEOUT_SECONDS`; on a stuck queue it throws at once. Every blocking wait in the SYCL backend goes through it: `GPUMemory`'s copies, `matrix_download_gpu()`, `GPUMatrix::copy()`, the diagnostics' host copies, `device_usable()` |
| `wait_for_queue(throw_async = true)` | TD-574: `wait_for()` on a barrier (`ext_oneapi_submit_barrier()`), so it waits for everything queued |
| `queue_stuck()`, `clear_stuck_for_testing()`, `stuck_message()` | TD-574: the stuck mark, set by a timed-out wait and cleared by a new queue (`initialize()`, `set_device()`); `clear_stuck_for_testing()` also clears the process flag ([GPUDebugSync.md](GPUDebugSync.md)), which a new queue doesn't. While it is set `device_usable()` is false, the pool trim drops its cache without freeing, and `cleanup()` leaves the queue |
| `trace_kernel(name)` | TD-574: records `name` as the latest GPU work; first waits for the queue (`wait_for_queue()`) when `wait_before_gpu_operation()` says so: always with `ADAI_GPU_SYNC=kernel`, before every Nth call with `kernel:N` or `GPU_QUEUE_DEPTH=N`. Called by every kernel entry point and by `GPUMemory`'s copies |
| `debug_sync_point(phase)` | TD-574: records the training phase; with `ADAI_GPU_SYNC=step` or `kernel` waits for the queue first (`wait_for_queue()`) |

---

## 3. `GPUMemory<T>`

| Member | Behaviour |
|---|---|
| `GPUMemory(count)` | Takes a block of `count × sizeof(T)` bytes from `GPUManager::allocate_block()` (TD-390's pool, see its addendum). `count == 0` gives an empty buffer with a null pointer and no budget, as CUDA's `cudaMalloc(0)` does (TD-392, fixed November 18, 2026: `malloc_device(0)` returns `nullptr`, which `allocate_block()` read as a failed allocation, so a zero-row `GPUMatrix` threw only on SYCL) |
| Destructor, move assignment | Returns the block to `GPUManager`'s pool (TD-390, see its addendum; this used to be a `host_task` free) |
| `copy_from_host`, `copy_to_host` | `GPUManager::wait_for(queue.memcpy(...), false)`; a zero count does nothing (TD-392). **Each waits for everything already queued**, so every upload and download drains the pipeline; since TD-574 the wait times out (§2) |

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
`GPU_STRATEGY` (`background` → `priority_low`, `normal` → no hint, `full` → `priority_high`, through `adai::gpu_queue_priority()`), and so does `chatbot_api_server` since November 10, 2026 (TD-381); `normal` was added November 15, 2026, since on the Arc B60 the GuC resets low-priority contexts (TD-574), and is the default since November 17, 2026. Every
SYCL kernel wrapper and `GPUMatrix` operation calls `get_queue()`.

---

## 5. Known gaps and gotchas (summary)

Each new item is tracked in [TECHNICAL_DEBT.md](../../guides/TECHNICAL_DEBT.md) and tagged in the code with
`TODO: See TD-NNN`.

| Item | Impact | Basis | Tracked as |
|---|---|---|---|
| Deferred `host_task` frees put a device→host→device bubble on the in-order queue for every `GPUMemory` destruction, and every upload/download drains the queue (§3) | Likely a large share of SYCL training and generation time on ai-machine; budget accounting lags (hence `ChatbotTrainer`'s workaround syncs) | **Verified** on a CPU queue (1.35 ms vs 0.051 ms per op); GPU magnitude unmeasured | [TD-390](../../guides/TECHNICAL_DEBT.md#td-390-sycl-deferred-frees-and-blocking-copies-stall-the-in-order-queue) |
| GPUs enumerated across all platforms, so one Arc card appears twice (Level Zero and OpenCL); `GPU_DEVICE_ID` picks by platform order and the chosen backend isn't logged (§2) | The trainer may run on the OpenCL backend instead of Level Zero without anyone knowing | By inspection; **fixed in code November 9, 2026** (see the addendum below) | [TD-391](../../archive/TECHNICAL_DEBT_RESOLVED.md#td-391-sycl-lists-the-same-gpu-once-per-platform-and-doesnt-log-the-backend) |
| `GPUMemory(0)` threw on SYCL but not CUDA (§3) | — | **Fixed November 18, 2026** (verified on the oneAPI CPU device) | [TD-392](../../archive/TECHNICAL_DEBT_RESOLVED.md#td-392-sycl-gpumemory0-throws-where-cuda-succeeds) |
| `set_device()` frees old-context allocations with the new queue and doesn't recompute the budget; `cleanup()` leaks live buffers by design and zeroes the counter; `initialize()` ignores new arguments when already initialized (§2) | Latent traps for multi-GPU and re-initialization | By inspection | [TD-383](../../guides/TECHNICAL_DEBT.md#td-383-gpumanager-lifecycle-gaps-set_device-cleanup-initialize) (amended) |
| Unlike CUDA, the budget is locked; the queue itself is thread-safe for submission | TD-382 is milder on SYCL; the shared fix (`ScopedCpuOnly` for the async quality thread) covers it | By inspection; **fixed in code October 17, 2026** | [TD-382](../../archive/TECHNICAL_DEBT_RESOLVED.md#td-382-gpu-work-from-any-thread-shares-unsynchronized-gpumanager-state) (amended) |
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

## TD-391 addendum (November 9, 2026)

`enumerate_gpu_devices()` now passes what the platforms list through `select_devices()` (`src/gpu/DeviceSelection.hpp`). Level Zero is numbered first and OpenCL last, and a listing is
dropped when a better backend has an unmatched device of the same vendor and name. `GPU_DEVICE_ID` indexes this list, so 0 is the Level Zero device whichever platform enumerates first.
`get_device_info()` adds a `Backend:` line, logged at INFO when `Matrix::gpu_initialize()`/`gpu_try_initialize()` succeed. Check it with `grep Backend:` in a pass's log.

## Addendum (November 10, 2026): `device_usable()` (TD-566)

`GPUManager::device_usable()` reports whether the device still runs work: one small allocation outside the pool and the budget, a
fill, a copy back and a wait. It returns false before `initialize()` and after a lost or reset device (and always in the no-GPU
stub). `ChatbotTrainer` calls it only after a sample fails, to tell a lost device from a bad sample.

## Addendum (November 13, 2026): engine-reset diagnostics (TD-574)

`GPUManager::trace_kernel()` and `debug_sync_point()` (table in §2) connect the kernels and the trainer to
[GPUDebugSync.md](GPUDebugSync.md). `GPUMemory::copy_from_host()`/`copy_to_host()` trace themselves as `upload`/`download`.
`device_usable()` uses the queue directly and isn't traced, so the probe after a failure doesn't overwrite the record of what failed.

## Addendum (November 14, 2026): the wait timeout (TD-574)

In ai-machine's third bisection round some waits never returned (the queue bound's, then the decode benchmark's): three passes
and one benchmark run sat until the runner killed them. The fourth round's GPU crash dumps showed why: each such pass had an
engine reset within a second of its last progress, which the oneAPI runtime never reported, so the events it waited on never
completed. Every
blocking wait in this backend now goes through `GPUManager::wait_for()` (§2), which gives up after `GPU_WAIT_TIMEOUT_SECONDS`
(60 s by default), marks the queue stuck and throws. The trainer's device-loss check (`device_usable()`, TD-566) is then false at
once, so the pass fails as on a lost device instead of hanging; later waits throw at once; and `cleanup()` leaves the stuck queue
to process exit rather than waiting on it. A wait inside the SYCL or Level Zero runtime itself isn't covered; its teardown at exit
blocked on the stuck work on the Level Zero v1 adapter, so `incremental_trainer` and `chatbot_api_server` end without it once a
wait has timed out (`gpu_work_stuck()`, `exit_without_gpu_teardown()`).

## Addendum (November 22, 2026): transfers are counted (TD-590)

`GPUMemory::copy_from_host()`/`copy_to_host()` count an upload or a download through `src/gpu/TransferCounter.hpp`
([GPUUtils.md](GPUUtils.md) has the counter); so do `matrix_download_gpu()`, the host-kernel diagnostics' `host_copy_of()`/
`device_copy_of()` and the batch helpers' pointer-array copies in `MatrixGPU_SYCL.cpp`. Device-to-device copies don't.
