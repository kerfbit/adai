# `GPUDebugSync` — Source File Reference

- **Files:** [`src/gpu/GPUDebugSync.hpp`](../../../../src/gpu/GPUDebugSync.hpp) (113 lines), [`src/gpu/GPUDebugSync.cpp`](../../../../src/gpu/GPUDebugSync.cpp) (290 lines)
- **Namespace:** `adai::gpu`
- **Built into:** `adai_gpu` (CUDA and SYCL builds); compiled directly into `gpuDebugSyncTests` in builds without a GPU
- **Used by:** `incremental_trainer` and `chatbot_api_server`, which set the queue bound from `GPU_QUEUE_DEPTH` ([Config.md](Config.md)), the SYCL kernels ([MatrixGPU_SYCL_cpp.md](MatrixGPU_SYCL_cpp.md)), `GPUManager::trace_kernel()`/`debug_sync_point()` ([GPUUtils_SYCL.md](GPUUtils_SYCL.md), [GPUUtils.md](GPUUtils.md)), `ChatbotTrainer` ([ChatbotTrainer.md](ChatbotTrainer.md)), `gpu_encode_soak`, and the GPU test package's engine-reset bisection
- **Status tag:** `@adai-status: experimental`, `@adai-version: 0.2.0`, `@adai-reviewed: 2026-11-14` (hpp and cpp)
- **Tests:** [`tests/gpu_debug_sync_test.cpp`](../../../../tests/gpu_debug_sync_test.cpp) → `gpuDebugSyncTests` (11 in every build; 4 more in a SYCL build with a device or `ADAI_SYCL_ALLOW_CPU_DEVICE=1`)
- **Origin:** TD-574 (November 13, 2026); the queue bound November 14, 2026
- **Last traced against the code:** 2026-11-14

> Code-traced reference, written with the files.

---

## 1. What this file is

The GPU queue bound, and diagnostics for finding which GPU work the Arc B60's engine resets interrupt
(TD-574).

**The queue bound (November 14, 2026).** `GPUManager::trace_kernel()` waits for the queue before
every Nth GPU operation (kernel, upload or download), so at most N are queued: N = 8 by default
(`kDefaultGpuQueueDepth`), 0 = unbounded, the behaviour before. The two ai-machine rounds had no
device loss in 567 s on the Arc when the queue was waited on at least every 8 operations, against
about one a minute without (`kernel:64` and `kernel:512` still reset); at 8 a training pass took
about 5% longer. The bound starts from the `GPU_QUEUE_DEPTH` environment variable (a malformed value
is the default), and `incremental_trainer` and `chatbot_api_server` set it again from their
`ServiceConfig::gpu_queue_depth` (`GPU_QUEUE_DEPTH` in the config file or the environment, validated
0–1,000,000) after the GPU initializes, logging `[GPU] Queue depth: …`. An `ADAI_GPU_SYNC=kernel` or
`kernel:N` diagnostic takes precedence over it; `step` keeps it between its phase waits. Only the
SYCL backend counts its operations, so it does nothing on CUDA. It isn't a diagnostic, so
`gpu_debug_summary()` doesn't list it.

**The diagnostics.** The resets
appear only when training runs its own SYCL kernels in long un-waited chains. The LeJEPA pass,
which sends the GPU only one oneMKL multiply at a time and waits for it, ran 220 passes without one.
These settings let one test run take the suspects out one at a time and name the work that failed.
They are all off by default, and they are for test runs only: the host versions move data across
PCIe, against the rule that GPU training keeps its math on the device.

| Variable | Values | Effect |
|---|---|---|
| `ADAI_GPU_SYNC` | `off` (default), `step`, `kernel`, `kernel:N` | `step`: `GPUManager::debug_sync_point()` waits for the queue where each training phase starts (`forward`, `backward`, `optimizer`, `validation`), so a failure is attributed to the phase before. `kernel`: also `GPUManager::trace_kernel()` waits before every kernel, upload and download, so a failure is attributed to the one kernel that ran. `kernel:N` (N from 1 to 1,000,000): waits before every Nth of those operations only, so at most N are queued; `GPUDebugSettings::sync_every` |
| `ADAI_GPU_HOST_KERNELS` | comma-separated: `embedding`, `optimizer`, `dropout` | Those kernel families run on the host instead: download the inputs, compute with the same formulas, upload the results, waiting for every copy. `embedding` covers the gather and the scatter-add (float atomics); `optimizer` the update and `matrix_sum_squares_accumulate_gpu()` (float atomics); `dropout` the device mask |
| `ADAI_GPU_TRACE_FILE` | a path | Holds one fixed-width record, `<seq> <phase> <kernel>`, rewritten in place at every kernel and phase. A pass killed in a hang leaves the work it was on |

Values are case-insensitive. An unknown mode or family is ignored and listed in
`GPUDebugSettings::problems`, which `gpu_debug_summary()` reports.

## 2. API

- `parse_gpu_debug_settings(sync, host_kernels, trace_file)`: pure parser (null = unset), for tests.
- `gpu_debug_settings()`: the process's settings, read from the environment on first use; the trace
  file is opened then. `set_gpu_debug_settings_for_testing()` replaces them and resets the record.
- `host_kernel(family)`: whether a family runs on the host.
- `kDefaultGpuQueueDepth` (8), `parse_gpu_queue_depth(value)` (0–1,000,000, else the default),
  `set_gpu_queue_depth(depth)` / `gpu_queue_depth()`: the queue bound, an atomic in the process state,
  first read from `GPU_QUEUE_DEPTH` in the environment.
- `wait_before_gpu_operation()`: whether the next operation waits first: always under `kernel`, before operations 1, N+1, 2N+1, …
  under `kernel:N` (counted by the GPU work number), otherwise before operations 1, D+1, 2D+1, … for
  the queue depth D, never when it is 0. `GPUManager::trace_kernel()` asks it.
- `note_gpu_kernel(name)` / `note_gpu_phase(name)`: record the latest kernel or phase (string
  literals; stored as pointers in atomics) and rewrite the trace record. Every SYCL kernel entry
  point calls `note_gpu_kernel()` through `GPUManager::trace_kernel()` whatever the settings, so a
  lost device can always report the last work queued.
- `note_host_kernel_run()` / `host_kernel_runs()`: every host version counts its calls, so a run shows that
  `ADAI_GPU_HOST_KERNELS` took effect; `ChatbotTrainer::train()` logs the count at the end of a pass.
- `describe_last_gpu_work()`: `kernel <k> in phase <p> (GPU work #<n>)` plus what it means in the
  current mode: with `kernel`, the queue was drained before that kernel, so it is the one that
  failed; with `kernel:N`, the failure is in it or one of the operations queued since the last wait; with `step`, the phase is exact and the kernel is the last it queued; with `off`, the failure is
  in it or one of the operations queued since the last wait under the queue bound, or, unbounded, it
  is the last work queued, which the GPU may not have reached. `ChatbotTrainer::stop_if_device_lost()` logs
  it as `Last GPU work: …`.
- `gpu_debug_summary()`: one line of the active settings, empty when nothing is set;
  `ChatbotTrainer::train()` and `gpu_encode_soak` log it.

It is plain C++ with no SYCL or CUDA types, compiled into `adai_gpu`, which sits below `adai_core`;
so it can't use `adai::Logger`, and its callers log.

## 3. The GPU test package's bisection

**Third round (package 0.6.0, November 14, 2026).** After the bound became the default, the
bisection compares it with the unbounded queue: `control` (`GPU_QUEUE_DEPTH=0`) and `default` (the
shipped 8), each twice per round, `depth_16` and `depth_32`, `sync_step`, and `encode_soak` without
and with the bound (`encode_soak_default`). Every variant except `default`, `depth_N` and
`encode_soak_default` runs with `GPU_QUEUE_DEPTH=0`, so it stays comparable with the first two
rounds; the variants they settled still run when named in `BISECT_VARIANTS`. The rest of the package
(test binaries, decode benchmark, soaks) runs with the bound.

**First two rounds.** `run_gpu_tests.sh --bisect-only` (or the full run) ran its variants round-robin, each pass with its
own `ADAI_GPU_TRACE_FILE`: `control` and `sync_kernel` (twice per round), `sync_step`, `sync_every_8`,
`sync_every_64`, `sync_every_512` (`kernel:N`), `full_priority` (`GPU_STRATEGY=full`), `no_dropout`
(`DROPOUT=0`), `host_embedding`, `host_optimizer`, `host_all`, and `encode_soak` (`gpu_encode_soak`:
`LeJEPAEncoder::gpu_encode()` for 10 s, forward kernels only, a wait every 64 encodes); with
`--engine-tuning`, also `long_preempt` and `long_timeslice` (the compute engine's preempt timeout and
time slice raised in sysfs for those passes only). Every step also copies the xe driver's crash dump,
if one was written, to `coredumps/`. The report gives each variant's seconds on the GPU and device
losses per minute.

The first run (November 13, 2026) had no device loss under `sync_kernel` and losses in every other
variant, including `encode_soak`; see TD-574 in TECHNICAL_DEBT.md for the numbers and the crash dumps'
common signature. The second run (November 14, 2026) had no device loss under `sync_kernel`, `sync_every_8` or `sync_step`
(0 in 378 s, about one a minute for the others), losses under `sync_every_64` and `sync_every_512`, and resets in every GPU decode
attempt, so a wait at least every 8 operations is what stops them. See the package's
[README](../../../../scripts/sycl_gpu_tests/README.md).

How to read it: if the resets stop under `sync_kernel` or `sync_step`, the long un-waited chains
matter; if they stop under one `host_*` variant, that family's kernels are involved; if
`encode_soak` resets, forward kernels alone are enough. `sync_kernel`'s `Last GPU work` names the
kernel whenever a reset still happens.

## 4. Tests

| Test | Covers |
|---|---|
| `UnsetMeansEverythingOff`, `ReadsModesAndFamiliesInAnyCase`, `UnknownValuesAreIgnoredAndReported`, `ReadsKernelEveryN` | Parsing, including `kernel:N` and its bad values |
| `KernelEveryNWaitsBeforeEveryNthOperation` | `wait_before_gpu_operation()` under `kernel:3`, `kernel` and `step` (unbounded); the summary and description |
| `DefaultQueueDepthWaitsBeforeEveryEighthOperation` | With nothing set, a wait before operations 1, 9, 17; the summary stays empty |
| `ParsesGpuQueueDepth` | Unset and malformed values give 8; 0 and padded numbers are read |
| `QueueDepthZeroNeverWaits` | Depth 0 never waits; depth 3 waits every third |
| `KernelDiagnosticOverridesTheQueueDepth` | `kernel:2` wins over depth 0 and 4; `step` keeps the bound |
| `RecordsLastWorkAndKeepsOneTraceRecord` | The record, its description (bounded and unbounded), one fixed-width trace record with no older text showing through, the summary |
| `SummaryIsEmptyWhenNothingIsSet` | |
| `GPUHostKernelTest.*` (SYCL) | Each family's host version against the device kernel, with `host_kernel_runs()` checked so the host path really ran (embedding gather and scatter-add with repeated ids, dropout, the optimizer update under all four rules and the gradient norm), and `kernel` mode naming the kernel and phase |

## 5. Known gaps

| Gap | Notes |
|---|---|
| CUDA kernels don't call `trace_kernel()` | Only `debug_sync_point()` exists on the CUDA `GPUManager`; the investigation is on the SYCL backend, and the queue bound does nothing on CUDA |
| The bound is a workaround | It keeps the Arc's queue short enough that the GuC's reset of the long-running compute context doesn't fire; the cause is in the driver or firmware (TD-574's upstream report). Its cost on a real model is still to be measured on ai-machine |
| A malformed `GPU_QUEUE_DEPTH` in the environment is silent here | The library takes the default; the two binaries' `ServiceConfig` warns about it and `validate()` refuses an out-of-range value |
| `kernel` mode changes the timing it measures | Waiting before every kernel may itself make the resets go away; that is a result too (the chains matter) |
| The LeJEPA pass has no phase sync points | It isn't where the resets happen |
