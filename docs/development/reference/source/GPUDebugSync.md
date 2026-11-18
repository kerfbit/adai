# `GPUDebugSync` — Source File Reference

- **Files:** [`src/gpu/GPUDebugSync.hpp`](../../../../src/gpu/GPUDebugSync.hpp) (148 lines), [`src/gpu/GPUDebugSync.cpp`](../../../../src/gpu/GPUDebugSync.cpp) (335 lines)
- **Namespace:** `adai::gpu`
- **Built into:** `adai_gpu` (CUDA and SYCL builds); compiled directly into `gpuDebugSyncTests` in builds without a GPU
- **Used by:** `incremental_trainer` and `chatbot_api_server`, which set the queue bound and the wait timeout from `GPU_QUEUE_DEPTH` and `GPU_WAIT_TIMEOUT_SECONDS` ([Config.md](Config.md)), the SYCL kernels ([MatrixGPU_SYCL_cpp.md](MatrixGPU_SYCL_cpp.md)), `GPUManager::trace_kernel()`/`debug_sync_point()`/`wait_for()` ([GPUUtils_SYCL.md](GPUUtils_SYCL.md), [GPUUtils.md](GPUUtils.md)), `ChatbotTrainer` ([ChatbotTrainer.md](ChatbotTrainer.md)), `gpu_encode_soak`, and the GPU test package's engine-reset bisection
- **Status tag:** `@adai-status: experimental`, `@adai-version: 0.3.0`, `@adai-reviewed: 2026-11-14` (hpp and cpp; the November 15 `keep_gpu_teardown_requested()` is in the same unreleased change)
- **Tests:** [`tests/gpu_debug_sync_test.cpp`](../../../../tests/gpu_debug_sync_test.cpp) → `gpuDebugSyncTests` (14 in every build; 5 more in a SYCL build with a device or `ADAI_SYCL_ALLOW_CPU_DEVICE=1`)
- **Origin:** TD-574 (November 13, 2026); the queue bound, the wait timeout, the stuck flag and the fast exit November 14, 2026; `ADAI_GPU_KEEP_TEARDOWN` November 15, 2026
- **Last traced against the code:** 2026-11-15

> Code-traced reference, written with the files.

---

## 1. What this file is

The GPU queue bound, the wait timeout, and diagnostics for finding which GPU work the Arc B60's engine resets interrupt
(TD-574).

**The queue bound (November 14, 2026).** `GPUManager::trace_kernel()` waits for the queue before
every Nth GPU operation (kernel, upload or download), so at most N are queued: N = 8 by default
(`kDefaultGpuQueueDepth`), 0 = unbounded. Over the second to fifth ai-machine rounds a bound of 8 cut
the Arc's engine resets without stopping them: 4 failed training passes in 120 against 18 unbounded
over the third to fifth rounds, and 6 forward-only soak passes in 40 against 18 over the fourth and
fifth (section 3). After the third round it was opt-in for a few hours, because some bounded passes seemed to hang
in its own wait; the fourth round's GPU crash dumps showed those were engine resets the oneAPI runtime
never reported, which happen unbounded too, and the wait timeout ends them. At 8 a 3-second training
pass took about 0.5 s longer; the forward-only soak was no slower. The bound starts from the `GPU_QUEUE_DEPTH` environment variable (a malformed value is the default),
and `incremental_trainer` and `chatbot_api_server` set it again from their
`ServiceConfig::gpu_queue_depth` (`GPU_QUEUE_DEPTH` in the config file or the environment, validated
0–1,000,000) after the GPU initializes, logging `[GPU] Queue depth: …`. An `ADAI_GPU_SYNC=kernel` or
`kernel:N` diagnostic takes precedence over it; `step` keeps it between its phase waits. Only the
SYCL backend counts its operations, so it does nothing on CUDA. It isn't a diagnostic, so
`gpu_debug_summary()` doesn't list it.

**The wait timeout (November 14, 2026).** `gpu_wait_timeout()` is how long the SYCL backend's
`GPUManager::wait_for()` lets one wait take, the bound's and every download's, upload's and probe's:
60 s by default (`kDefaultGpuWaitTimeoutSeconds`), 0 = as long as it takes. After it the queue is
marked stuck and the wait throws, so an engine reset the runtime never reports fails the pass like a
lost device instead of hanging it ([GPUUtils_SYCL.md](GPUUtils_SYCL.md)). The timeout also records
the stuck work for the process (`note_gpu_work_stuck()`), and `incremental_trainer` and
`chatbot_api_server` check `gpu_work_stuck()` when `main()` is done: if set, they flush their logs and
end with `exit_without_gpu_teardown()` (`std::_Exit` with the same exit code), because on the Level
Zero v1 adapter the oneAPI runtime's own teardown blocked on the stuck work at exit (3 of 3 times).
With `ADAI_GPU_KEEP_TEARDOWN=1` they log that the work is stuck and return normally instead, running
the teardown, so the GPU test package can take a stack of that hang for the report to Intel
(`keep_gpu_teardown_requested()`). The timeout starts from the
`GPU_WAIT_TIMEOUT_SECONDS` environment variable (0–86,400, else the default), and the two binaries set
it again from `ServiceConfig::gpu_wait_timeout_seconds`, logging it on the same line as the depth.

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
- `kDefaultGpuWaitTimeoutSeconds` (60), `parse_gpu_wait_timeout(value)` (0–86,400, else the default),
  `set_gpu_wait_timeout(seconds)` / `gpu_wait_timeout()`: the wait timeout, likewise, first read from
  `GPU_WAIT_TIMEOUT_SECONDS`.
- `note_gpu_work_stuck()` / `gpu_work_stuck()` / `clear_gpu_work_stuck_for_testing()`: a
  process-lifetime flag the SYCL `GPUManager::wait_for()` sets when a wait times out (a new queue
  doesn't clear it). `exit_without_gpu_teardown(code)` (`[[noreturn]]`) flushes the C streams and
  calls `std::_Exit(code)`, running no static destructors or `atexit` handlers; callers flush the
  logger and the C++ streams first. `keep_gpu_teardown_requested()`: whether the environment has
  `ADAI_GPU_KEEP_TEARDOWN` exactly `1` (read on each call), a diagnostic that makes the two binaries
  skip the fast exit.
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

**Third round (package 0.6.0, run November 14, 2026).** It compared the bound with the unbounded
queue: `control` (`GPU_QUEUE_DEPTH=0`) and `default` (then 8), each twice per round, `depth_16` and
`depth_32`, `sync_step`, and `encode_soak` without and with the bound (`encode_soak_default`).
Results: `control` 6 device losses in 40 passes (2.50/min), `default` 1 in 40 (0.43/min), `depth_16`
2 in 20, `depth_32` 3 in 20, `sync_step` 0 in 20, `encode_soak` 10 in 20, `encode_soak_default` 1 loss
and 3 hangs in 20. The Level Zero soak, at 8, lost the device in 6 of 20 v2 passes, three of them
within 30 s. Every bounded hang, and the decode benchmark's hung retry, waited in the bound's own wait
(GPU work #149208, #126032, #321032: multiples of 8) with no engine reset until the process was
killed at the step's time limit. So the bound became opt-in, and the wait timeout was added.

**Fourth round (package 0.8.0, run November 14, 2026).** The same list, renamed: `default` is now `depth_8` and
`encode_soak_default` `encode_soak_depth_8` (a variant ending in `depth_N` runs with
`GPU_QUEUE_DEPTH=N`, every other one with 0). Every pass uses the shipped 60 s wait timeout, and a pass
whose wait gave up (its log names `GPU_WAIT_TIMEOUT_SECONDS`) is reported as `stuck`, in its own
column. The first two rounds' variants still run when named in `BISECT_VARIANTS`. Results (failed =
device lost or stuck): `control` 7 of 40 (1.34/min), `depth_8` 3 of 40 (0.56/min), `depth_16` 1 of 20,
`depth_32` 3 of 20, `sync_step` 1 of 20, `encode_soak` 12 of 20, `encode_soak_depth_8` 1 of 20; no pass
hung. Every stuck pass, at any depth including 0, had an engine reset and a GPU crash dump within a
second of its last progress, though its waits never completed: the runtime didn't report the reset.
On the Level Zero v1 adapter all three stuck passes then hung at exit, inside the runtime's teardown,
which is why the binaries now exit without it. Since then the rest of the package runs at the default
bound of 8; the bisection's variants still set their own depth.

**Fifth round (run November 14, 2026, with `--engine-tuning`).** `control` 5 of 40, `depth_8` 0 of 40,
`depth_16` 2 of 20, `depth_32` 0 of 20, `sync_step` 2 of 20, `encode_soak` 6 of 20,
`encode_soak_depth_8` 5 of 20, `long_preempt` (5 s preempt timeout) 1 of 20, `long_timeslice` (100 ms
time slice) 0 of 20. 29 of the 31 failures in the run were stuck waits. In each failed training pass
the reset came 0.68–0.90 s after the last log line, and 5.07 s after it under `long_preempt`: a longer
preempt timeout only delays the reset. Every stuck pass exited within 30 ms of giving up through
`exit_without_gpu_teardown()`, the four on Level Zero v1 included, and `gpuDebugSyncTests` passed on
the Arc in 1 s.

**Sixth round (run November 15, 2026).** For the report to Intel: `control` (twice),
`depth_8`, `encode_soak`, the standalone reproducer (`sycl_engine_reset_repro`: `repro_mkl`,
`repro_sycl`, `repro_mkl_wait_8`), `ur_debug` (the Level Zero adapter's debug log) and `v1_teardown`
(Level Zero v1 with `ADAI_GPU_KEEP_TEARDOWN=1`); with `--engine-tuning`, `long_timeslice` twice and
`no_preempt` (preempt timeout 0) if the driver accepts it. The runner takes `gdb` stacks of a step's
processes after an engine reset and after a wait gave up. Results: the reproducer failed 31 of 40 passes
unbounded and 1 of 20 with a wait every 8 operations, with the training passes' dump; `control` 2 of 40,
`depth_8` 0 of 20, `long_timeslice` 4 of 40; the driver refused preempt timeout 0. A process stuck on an
unreported reset sits in `wait_for()`'s own poll, so its stack shows nothing of the runtime's;
`UR_LOG_LEVEL_ZERO=level:debug` logs only initialization. Details in TD-574 and the report's observation 7.

**Seventh round (run November 15, 2026).** The sixth round's variants again (the run used the package built before the
next round's changes), after updating ai-machine to the newest compute runtime (26.40.40196.15) and GuC (70.83.0) with debug
symbols: no change, 57 of 220 passes failed against 52 of 220, with the same dump and timing. The symbols name the reproducer's
exit-time wait (`L0::DriverHandle::~DriverHandle` freeing the USM pool, waiting for a task count the reset context never reaches)
and the event query a polled unreported reset sits in (`queryCounterBasedEventStatus`). Details in TD-574 and the report's
observations 4 and 8.

**Eighth round (packaged November 15, 2026).** `ur_debug` gives way to `repro_sycl_blocking` (the reproducer
with `--blocking-wait`: plain `sycl::event::wait()` calls, a watchdog thread ending the process after the
timeout, so a stuck process's stack shows the runtime's wait) and `repro_sycl_trace` (`SYCL_UR_TRACE=2`, the
last 64 MiB of a failed pass's call trace in `ur_trace/`); stacks are capped at two per variant. The round
also runs `repro_sycl_normal` and `repro_mkl_normal` (the reproducer on a normal-priority queue), and with
`--kernel-debug` `repro_sycl_drm_debug` (`drm.debug` 0x2 for its passes only, restored after each and at exit;
a failed pass's kernel log in `kernel_debug/`). Every run now saves `host_details.txt` and the kernel log since
boot. `xe_debug.sh` raises the GuC log level (a module option, from the next boot) and `mainline_kernel.sh`
boots Ubuntu's mainline kernel once, for the same round on a current upstream kernel.

**Eighth round, first part (run November 15, 2026, GuC log level 3).** The variants the seventh round also ran failed
6 of 160 against 52 of 160 (the reproducer 1 of 40 against 34 of 40), while `repro_sycl_blocking` failed 11 of 20, one
of them stuck 60 s with no reset at all; the v1 exit hang recurred and was stacked (the v2 teardown wait, never ending).
Two runner fixes followed: `--kernel-debug` reads `/sys/module/drm/parameters/debug` (root-only) through sudo, which
it hadn't, so the `drm.debug` variant was always left out; and the stack watcher takes a `stalled` stack of a
reproducer whose log has been quiet for `STACK_AFTER_STALL` (20) seconds with no dump and no give-up, and no
`waiting` stack after a reproducer's reported loss (its exit). Next: the same run at the default GuC log level, then
the mainline kernel. ai-machine has Secure Boot on, so `mainline_kernel.sh` now signs the mainline kernel with a
code-signing key of its own and enrolls it as a MOK (confirmed at the console at the next boot); `remove` offers to
take the key off again.

**Eighth round, second part (run November 15–16, 2026, GuC log level 1).** The same variants failed 42 of 160 again, so level 3's
drop was the level. The reproducer with no priority hint failed 0 of 40 against 31 of 40 with `priority_low`, the queue
`GPU_STRATEGY=background` (the default) makes; `full` makes `priority_high`, and `normal` (added after this run) sets no hint. With `drm.debug`
0x2, every reported reset followed a submission the kernel refused (`q->ops->reset_status(q)`), every unreported one a process only
polling its event; blocking waits were reported 20 of 20. Next: training passes at `full` and with no hint (`GPU_STRATEGY=normal`,
added for it: the runner's `normal_priority`, `encode_soak_normal_priority` and `encode_soak_full_priority`; `gpu_encode_soak` and
`gpu_decode_bench` read `GPU_STRATEGY` from the environment), then the mainline kernel.

**Priority run (November 16, 2026, GuC log level 1).** Training (`control` against `normal_priority` and `full_priority`), the
forward-only soak and the reproducer, 20 rounds: at `GPU_STRATEGY=normal` 0 of 60 passes failed, at `full` 0 of 40, at the default
`background` 34 of 60 (training 5, soak 11, reproducer 18; p ≈ 1.5·10⁻¹³ against `normal`), each failure one ccs engine reset. The
default is `normal` since November 17, 2026; the runner's bisection keeps its other variants at `background`, comparable with
the earlier rounds. Next: the mainline kernel.

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
| `QueueWaitsBeforeEveryEighthByDefaultAndDepthZeroIsUnbounded` | The default (8) waits before operations 1, 9, 17 and the summary stays empty; depth 0 never waits |
| `StuckWorkIsRecordedAndTheExitSkipsTeardown` | The stuck flag sets and clears; `exit_without_gpu_teardown(3)` in a death test exits with 3, keeps what was written to stderr, and runs no `atexit` handler |
| `KeepTeardownOnlyForExactlyOne` | `keep_gpu_teardown_requested()` is true only for `ADAI_GPU_KEEP_TEARDOWN=1`: unset, empty, `0`, `yes` and `1 ` are false; the variable is restored after |
| `ParsesGpuQueueDepth` | Unset and malformed values give 8; 0 and padded numbers are read |
| `ParsesGpuWaitTimeout` | Unset and malformed values (including 86,401 and `5s`) give 60; 0, 86,400 and padded numbers are read; the setter |
| `QueueDepthZeroNeverWaits` | Depth 0 never waits; depth 3 waits every third |
| `KernelDiagnosticOverridesTheQueueDepth` | `kernel:2` wins over depth 0 and 4; `step` keeps the bound |
| `RecordsLastWorkAndKeepsOneTraceRecord` | The record, its description (bounded and unbounded), one fixed-width trace record with no older text showing through, the summary |
| `SummaryIsEmptyWhenNothingIsSet` | |
| `GPUHostKernelTest.*` (SYCL) | Each family's host version against the device kernel, with `host_kernel_runs()` checked so the host path really ran (embedding gather and scatter-add with repeated ids, dropout, the optimizer update under all four rules and the gradient norm), and `kernel` mode naming the kernel and phase; `AWaitOnStuckWorkTimesOutAndTheDeviceCountsAsLost`: a host task waiting on a host flag holds the in-order queue, standing in for a stuck GPU, and with a 1 s timeout the wait throws after 1–10 s, the queue is stuck, `gpu_work_stuck()` is set, `device_usable()` is false and a later download throws at once naming `GPU_WAIT_TIMEOUT_SECONDS`; released, the device is usable again. (Until November 14, 2026 a device kernel spinning on a `malloc_shared` flag held it; on the Arc the kernel never saw the host's write and hung the test for its hour.) |

## 5. Known gaps

| Gap | Notes |
|---|---|
| CUDA kernels don't call `trace_kernel()` | Only `debug_sync_point()` exists on the CUDA `GPUManager`; the investigation is on the SYCL backend, and the queue bound does nothing on CUDA |
| The bound cuts the resets, doesn't stop them | The cause is in the driver or firmware (a longer preempt timeout only delays the reset, fifth round; see the [draft report for Intel](../../../operations/guides/troubleshooting/ARC_B60_ENGINE_RESETS_INTEL_REPORT.md)) |
| The wait timeout only covers our own waits | A wait inside the SYCL or Level Zero runtime can still block. The binaries skip the runtime's teardown at exit once work is stuck, but destructors that run before `main()` returns (`run_main()`'s locals) and other binaries (`gpu_encode_soak`, `gpu_decode_bench`, the tests) don't; the step's time limit, or `trainer_service`'s stall timer (TD-569), remains the backstop |
| A malformed `GPU_QUEUE_DEPTH` or `GPU_WAIT_TIMEOUT_SECONDS` in the environment is silent here | The library takes the default; the two binaries' `ServiceConfig` warns about it and `validate()` refuses an out-of-range value |
| `kernel` mode changes the timing it measures | Waiting before every kernel may itself make the resets go away; that is a result too (the chains matter) |
| The LeJEPA pass has no phase sync points | It isn't where the resets happen |
