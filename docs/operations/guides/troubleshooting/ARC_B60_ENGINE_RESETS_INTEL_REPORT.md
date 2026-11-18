# Arc Pro B60: compute engine resets, and resets the runtime never reports — report for Intel

Status: final (November 18, 2026). The resets are gone on Ubuntu's 7.0 kernel (observation 11), so they are not filed with
the xe driver: a newer kernel already fixes them. What remains is the runtime's side (observations 3 and 4), filed with
intel/compute-runtime as [intel/compute-runtime#1057](https://github.com/intel/compute-runtime/issues/1057) on November 18, 2026 (from the kerfbit account; text in
[ARC_B60_COMPUTE_RUNTIME_ISSUE.md](ARC_B60_COMPUTE_RUNTIME_ISSUE.md)).

From the investigation (TD-574): eight rounds of tests on ai-machine (November 13–14, 2026 local time; the fifth
ran 06:27–07:22 UTC, the sixth 09:08–10:04 UTC, the seventh 10:48–11:50 UTC and the first part of the eighth 22:22–23:19 UTC on
November 15, its second part 23:55–01:16 UTC into November 16, a run comparing queue priorities 22:14–22:59 UTC on November 16, and one on Ubuntu's 7.0 kernel 01:40–01:57 UTC on November 18; the seventh after updating the compute runtime and GuC to the newest
available, the eighth's first part with the GuC log at level 3 and its second back at the driver's default, level 1; clock times
below are UTC). It records what we
observed, kept separate from what we infer. Where to file: the xe driver (drm/xe, freedesktop.org GitLab) for the resets, and
[intel/compute-runtime](https://github.com/intel/compute-runtime) for the resets the runtime never reports. Before filing, see
[What Intel will ask for](#what-intel-will-ask-for).

## Summary

1. **A SYCL compute workload made of many small kernels and copies on one in-order queue makes the GuC reset the compute engine**, in
   about 1 pass in 6 of a 3-second training pass, and in every attempt of a small inference benchmark. Every reset has the same
   signature: `Reason: LR job cleanup`, a GuC-captured reset of the process's long-running compute context, preempt timeout 640 ms,
   time slice 1 ms, all EU rows busy, the command streamer at `MI_STORE_DATA_IMM` (`IPEHR 0x10400002`), no engine error
   (`RING_ESR`/`RING_EIR` clear), no page fault or CAT error. No kernel in the workload runs longer than a few milliseconds.
   **A standalone reproducer using only SYCL (optionally oneMKL) resets the engine the same way**, in 65 of 80 passes of 15 s
   (observation 7).
2. **In most resets, the Level Zero runtime never tells the process** (110 of 172 failed passes over the fourth to seventh rounds:
   18 of 32, 29 of 31, 30 of 52 and 33 of 57; 32 of 80 in the eighth round's second part). The engine reset is in the kernel log and the devcoredump within a second, but the process's waits on its events never
   complete and no error is returned, on both the v1 and v2 adapters. Without a timeout of our own the process waits forever (once for
   6.5 hours, before we knew what it was).
3. **On the v1 adapter, a process that gave up on such a wait hung at exit**, inside the runtime's teardown (3 of 3 in the fourth
   round, and the one time it could happen since, in the eighth; our processes now skip the teardown in that state). Its stack is
   the same wait for a task count the reset context never reaches that delays a v2 process's exit (observation 4).
4. **Bounding the queue reduces the resets but doesn't end them.** Waiting for the queue before every 8th operation: 4 training passes
   failed of 160 against 26 of 200 unbounded over five rounds; waiting after every operation has never had one (0 in 60 passes). The
   reproducer with the same bound failed 2 of 40 against 30 of 40 without, though it also does about an eighth as much work in the time.
5. **Raising the preempt timeout doesn't stop the resets; it delays them by the same amount.** With the compute engine's
   `preempt_timeout_us` at 5 s instead of 640 ms, a pass still reset, 5.07 s after its last progress instead of 0.65–0.93 s
   (observation 6).
6. **The newest compute runtime and GuC change nothing.** With compute runtime 26.40.40196.15 (IGC 2.42.14), Level Zero loader
   1.34.0 and GuC 70.83.0, the same variants failed 57 of 220 passes against 52 of 220 with 26.31 and GuC 70.72.1 (p ≈ 0.66), with
   the same dump signature, the same timing and as many resets unreported (33 of 57; observation 8).
7. **A more verbose GuC log made the polled workloads reset far less, but not the same workload with blocking waits.** With
   `xe.guc_log_level=3` instead of the driver's 1, and nothing else changed but a reboot, the variants the seventh round also ran
   failed 6 of 160 passes against 52 of 160 (p ≈ 6·10⁻¹²), the reproducer 1 of 40 against 34 of 40. The same reproducer waiting
   with plain `sycl::event::wait()` calls instead of polling failed 11 of 20 in the same run (polling: 0 of 20, p ≈ 0.0002), and one
   of those waits stalled for 60 s with no engine reset at all (observation 9). Back at level 1 after another reboot, the same
   variants failed 42 of 160 again (p ≈ 10⁻⁸ against level 3; p ≈ 0.27 against the seventh round), so the level, not the reboot,
   made the difference.
8. **Whether a reset is reported depends on what the process does when it happens.** With `drm.debug` on, every reported loss (12
   of 12) followed the process submitting more work after the reset, which the kernel refused (`xe_exec_ioctl`: `Ioctl argument check
   failed at drivers/gpu/drm/xe/xe_exec.c:147: q->ops->reset_status(q)`); in every unreported one (6 of 6) the process made no call
   into the kernel after the reset: it was polling its event, whose counter the reset context never writes. With blocking waits all
   20 of 20 failures at level 1 were reported (polled: 7 of 19, p ≈ 10⁻⁵). The kernel logs nothing from the driver before the reset
   (observation 3).
9. **Queue priority decides it.** The same reproducer on a normal-priority queue (`--normal-priority`) failed 0 of 40 passes at level
   1 while the low-priority one failed 31 of 40 in the same run (p ≈ 4·10⁻¹⁴), and 0 of 80 against 32 of 80 over both parts of the
   eighth round, doing more work per pass (about 9,000 steps in 15 s; observation 10). Training and the forward-only soak behave
   the same: at normal or high priority none of 100 passes was reset, against 34 of 60 at low priority in the same run (p ≈
   10⁻¹³).
10. **Ubuntu's 7.0 kernel doesn't reset at all.** On `7.0.0-43-generic` (upstream 7.0.14), with the same GuC, compute runtime and
   binaries, none of 260 passes failed and the kernel logged no engine reset, including 0 of 60 low-priority reproducer passes and
   0 of 160 in the variants that failed 65 of 160 times on `6.17.0-1025-oem` the run before (observation 11). Low priority still
   runs slower there, so the hint still takes effect.

Our reading (an inference, not stated in the dumps): the GuC asks the long-running context to yield, the context never completes the
preemption, however long it is given, and when the preempt timeout runs out the GuC resets the engine. The fifth round's timing argues
that the context has stopped making progress before the request, rather than being slow to preempt because of a deep chain of
queued work: we would expect a context that was merely slow to finish within 5 s. The sixth and seventh rounds' 19 failed training
passes, six of them with a 100 ms time slice, fit the same timing (0.65–0.93 s). The eighth round adds two things that fit it: one
process's work stopped for 60 s with no reset until the process exited, so the work can stop with nothing asking it to yield; and
the rate depends strongly on the GuC log level, so on the GuC's own timing, which points at the scheduling of the context rather than
at any one kernel (observation 9). Its second part and the priority run narrow it further: only low-priority contexts are reset, in our training too (observation 10). We
read the unreported resets as a gap in the runtime: a polled counter-based event never checks the queue's reset status, so the
process learns of the reset only if it calls into the kernel again (observation 3).

## Environment

| | |
|---|---|
| GPU | Intel Arc Pro B60 (Battlemage), PCI ID `8086:e211`, rev 0x00, GT0 IP 20.1.0 (one card, no display attached) |
| Host | AMD Ryzen 7 5800XT, Ubuntu 24.04 |
| Kernel | `6.17.0-1025-oem` (Ubuntu) in the eight rounds and the priority run; `7.0.0-43-generic` (Ubuntu, upstream 7.0.14) for observation 11; `xe` driver, `force_execlist=N`; command line `intel_iommu=on iommu=pt pci=realloc` |
| GuC | `xe/bmg_guc_70.bin` 70.72.1 for the first six rounds (the driver wants 70.49.4; with 70.49.4 the resets were the same, November 11); 70.83.0 for the seventh |
| Compute runtime | first six rounds: `libze-intel-gpu1` and `intel-opencl-icd` 26.31.39395.14-1~24.04~ppa1, `libigc2` 2.40.13+ds1-1~24.04, `libigdgmm12` 22.10.1, Level Zero loader 1.32.0 (driver reports 1.17.39395+14). Seventh: 26.40.40196.15-1~24.04~ppa1, `libigc2` 2.42.14-1~24.04, `libigdgmm12` 22.10.2, loader 1.34.0 (driver reports 1.18.40196+15), with debug symbols (the Intel graphics PPA's newest on November 15) |
| oneAPI | DPC++ 2026.2.0 (compiler and runtime, `intel-oneapi-runtime-dpcpp-cpp` 2026.2.0-863), oneMKL 2026.2.1 |
| Level Zero adapter | the default v2 (`Unified Runtime over Level-Zero V2`) and v1 (`SYCL_UR_USE_LEVEL_ZERO_V2=0`), both affected |
| GuC log level | `xe.guc_log_level` 1 (the driver's default) in the first seven rounds; 3 in the eighth round's first part (`options xe guc_log_level=3`, the dumps' `Log level: 3`) |
| Compute engine settings | `job_timeout_ms` 5000; the context in every dump: timeout infinite, time slice 1000 µs, preempt timeout 640000 µs |
| Other GPU users | none (checked before every run) |

## The workload

A small encoder–decoder transformer (`d_model` 64, 16 tokens, 18 training samples, 10 epochs, about 3 s per pass) trained entirely on
the device through SYCL: our own kernels (softmax, LayerNorm and cross-entropy with work-group reductions, an embedding gather and a
float-atomic scatter-add, dropout masks, an Adam update with float-atomic norms), oneMKL GEMMs, and `memcpy` uploads and downloads,
all on one in-order `sycl::queue` (low priority by default; normal priority too). The host waits only for one scalar per sample (the
loss) and for downloads; between them the queue holds long chains, tens of thousands of operations per pass. Each pass is a fresh
process. A forward-only variant (`gpu_encode_soak`: no backward, no atomics, no optimizer) and a decode benchmark (`gpu_decode_bench`)
show the same resets, and so does a standalone program with no model at all (observation 7). A pass whose only GPU work is one oneMKL GEMM at a time with a wait after each (our LeJEPA pass) has never had
one: 0 in about 300 passes.

## How to reproduce

On the B60 with the environment above, with no other GPU users:

```
icpx -fsycl -O2 SyclEngineResetRepro.cpp -o sycl_engine_reset_repro
# or, for the oneMKL multiply: icpx -fsycl -qmkl -O2 -DREPRO_WITH_MKL SyclEngineResetRepro.cpp -o sycl_engine_reset_repro
for i in $(seq 20); do ./sycl_engine_reset_repro --seconds 15; echo "exit $?"; done
journalctl -k | grep -E 'Engine reset|exec queue reset'
ls /sys/class/devcoredump/
```

Options: `--seconds N` (default 60; our runs used 15), `--layers N` (default 24), `--mkl`, `--wait-every N` (wait for the queue before every Nth
operation), `--normal-priority`, `--timeout N` (default 60), `--blocking-wait`. It prints a UTC-timestamped progress line each second.
Exit codes: 0 ran the whole time, 2 a wait didn't finish within the timeout (an unreported reset), 3 the runtime reported an error,
4 no GPU.

- **Expected:** every pass exits 0; a workload of millisecond kernels on an otherwise idle engine isn't reset.
- **Actual:** most passes end with `GT0: Engine reset: engine_class=ccs … guc_id=…` and a devcoredump within the first 14 s (65 of 80
  passes over two rounds). About half exit 3 with `OUT_OF_RESOURCES` or `OUT_OF_DEVICE_MEMORY`; the rest exit 2, the runtime never
  having reported the reset. Without the timeout the process would wait forever.
- **Also expected:** after a reset, every pending and later wait on the context's events fails with `ZE_RESULT_ERROR_DEVICE_LOST`
  (`UR_RESULT_ERROR_DEVICE_LOST`), and exit doesn't wait for work the reset discarded.

What we can attach: the reproducer source, the devcoredumps (about 400 KB gzipped each), the kernel log around each reset, the gdb
stacks (observations 3 and 4) and the per-pass logs.

## Observations

### 1. Engine resets: one signature

All 315 devcoredumps saved during the third to eighth rounds (and 30 earlier ones) agree, the reproducer's 148 included (the exceptions
are the preempt timeout, 5000000 µs in the dump of the pass run with it raised, observation 6; the time slice, 100000 µs in the passes
run with it raised; the GuC version, 70.83.0 from the seventh round; and the GuC log level, 3 in the eighth's first part):

```
Reason: LR job cleanup, guc_id=8
GuC version: 70.72.1 (wanted 70.49.4)
**** Contexts ****
  Name: ccs8   Class: 5   Logical mask: 0x1   Width: 1
  Timeout: 9223372036854775807 (ms)   Timeslice: 1000 (us)   Preempt timeout: 640000 (us)
  Start seqno == Seqno (memory)       Schedule State: 0x250   Flags: 0x20
**** HW Engines ****  ccs0, Capture_source: GuC, Coverage: full-capture
  IPEHR: 0x10400002 (MI_STORE_DATA_IMM)   RING_ESR: 0   RING_EIR: 0
  ROW_INSTDONE[*]: 0x00000000   RING_EXECLIST_STATUS: 0x000004004000209a
```

The kernel log for each: `GT0: Engine reset: engine_class=ccs, logical_mask: 0x1, guc_id=8`, then the devcoredump. Never a page
fault, CAT error or `job timedout`. The LRC head is near the end of the 16 KiB ring and the tail near its start (the ring has wrapped)
in every dump; the seqnos differ. The dumps are available (gzip, about 400 KB each).

### 2. Rates

Per variant, fourth to seventh rounds (package `e7b5f0c1`, v2 adapter, round-robin so the variants share the same stretch of time;
the seventh on the newest compute runtime and GuC, observation 8). "Failed" counts both a reported loss and a reset the runtime never
reported (observation 3):

| Waits for the queue | Training passes failed, 4th round | 5th | 6th | 7th | Forward-only soak failed, 4th | 5th | 6th | 7th |
|---|---|---|---|---|---|---|---|---|
| never (except one scalar per sample) | 7 of 40 | 5 of 40 | 2 of 40 | 6 of 40 | 12 of 20 | 6 of 20 | 12 of 20 | 11 of 20 |
| before every 8th operation | 3 of 40 | 0 of 40 | 0 of 20 | 0 of 20 | 1 of 20 | 5 of 20 | | |
| before every 16th | 1 of 20 | 2 of 20 | | | | | | |
| before every 32nd | 3 of 20 | 0 of 20 | | | | | | |
| at each phase (forward, backward, optimizer, validation) | 1 of 20 | 2 of 20 | | | | | | |
| after every operation (November 13–14) | 0 of 60 | | | | | | | |

The eighth round's first part, with the GuC log at level 3 (observation 9): unbounded 0 of 40, a wait before every 8th operation 0
of 20, the forward-only soak 4 of 20.

Pooled over the third to seventh rounds, a wait before every 8th operation: 4 failed of 160 against 26 of 200 (Fisher p ≈
0.0002). Forward-only, over the fourth and fifth: 6 of 40 against 18 of 40 (p ≈ 0.007), though in the fifth round alone the bound
made no difference (5 of 20 against 6 of 20). The rate of one variant varies from round to round by more than the difference between
variants, so we only trust pooled comparisons. Waiting before every 64th or 512th operation did not help (second round). Running our
atomic, dropout or optimizer kernels on the host instead did not help either (November 13–14), so no one kernel family is needed.

Engine settings (fifth round, no queue bound, the setting changed through sysfs for each of the variant's passes and restored after):

| Compute engine setting | Training passes failed |
|---|---|
| unchanged (`preempt_timeout_us` 640000, `timeslice_duration_us` 1000) | 5 of 40 |
| `preempt_timeout_us` 5000000 | 1 of 20 |
| `timeslice_duration_us` 100000 | 0 of 20; sixth round 4 of 40 (control 2 of 40); seventh 2 of 40 (control 6 of 40) |

Neither differs from the control (p ≈ 0.65 for the preempt timeout; the time slice over three rounds 6 of 100 against 13 of 120, p ≈
0.24). `preempt_timeout_us` 0 (no preempt timeout) was refused by the driver, so we couldn't test it.

Adapters (same training pass, alternating, three rounds; the fourth and fifth with a wait before every 8th operation): v2 failed 18 of
60, v1 8 of 60 (p ≈ 0.045). In the fifth round alone: v2 3 of 20, v1 4 of 20. In the sixth (unbounded): v1 1 of 20, v2 3 of 60;
in the seventh: v1 0 of 20, v2 9 of 60.

### 3. Resets the runtime never reports

In 18 of the 32 failed passes of the fourth round (training, decode and seed runs), and 29 of the 31 of the fifth, the process got no
error. The fifth round's only reported losses were the two of the variant that waits at each phase (`UR_RESULT_ERROR_OUT_OF_RESOURCES`,
and for the first time `UR_RESULT_ERROR_OUT_OF_DEVICE_MEMORY`, each within 1 s of the reset). In the sixth, 30 of 52 went
unreported: the forward-only soak 12 of 12 and the unbounded control 2 of 2, but only 1 of the 4 with a 100 ms time slice and 15 of
the reproducer's 32 (observation 7). In the seventh, on the newest runtime, 33 of 57: the forward-only soak again 11 of 11, the
reproducer 20 of 35, the training passes 2 of 11 (the unbounded control 1 of 6). A reported loss arrives at the reset itself: the
reproducer's error is timestamped within 1 ms of the kernel's `Engine reset` line in all 17 of its reported failures in the sixth
round, and in the seventh within 8 ms in 12 of 15 and 1.0 s after in the other 3; a training pass's error comes 0–0.36 s after it.
The seventh round's training passes reported `UR_RESULT_ERROR_DEVICE_LOST` (7) or `OUT_OF_DEVICE_MEMORY` (2); the fifth and
sixth rounds' seven reported training failures were `OUT_OF_RESOURCES` (3), `DEVICE_LOST` (2) and `OUT_OF_DEVICE_MEMORY` (2).

A stack taken while the reproducer polled an unreported reset's event (seventh round, `bisect_repro_mkl_02`, with the runtime's debug
symbols) shows the query reading the event's counter and finding it short:

```
L0::EventImp<unsigned long>::queryCounterBasedEventStatus (blockOnMiss=false)  level_zero/core/source/event/event_impl.inl
L0::EventImp<unsigned long>::queryStatus
ur::level_zero::urEventGetInfo                    libur_adapter_level_zero_v2.so.0
sycl::detail::event_impl::get_info<info::event::command_execution_status>
main
```

The event is a counter-based one, so its status comes from a counter in memory that the reset context will never write. We don't
know whether the runtime is meant to check the queue's reset status on that path. We don't know what decides whether a reset is
reported:

| | Reported (`UR_RESULT_ERROR_DEVICE_LOST`, `OUT_OF_RESOURCES` or `OUT_OF_DEVICE_MEMORY`) | Not reported |
|---|---|---|
| devcoredump, `Engine reset` in the kernel log | within 1 s of the pass's last progress | within 1 s of the pass's last progress |
| what the process sees | the next wait or copy fails, within 0–1 s | every wait (`sycl::event::wait`, polling `info::event_command_status`, a `memcpy` to the host) stays pending; we stopped it after 60 s |
| `exec queue reset detected` in the kernel log | 3.6–5.4 s after the reset (seventh round: 3.6–7.9 s for training passes, 7.0–10.5 s and again 4.5 s later for the reproducer) | only after the process gave up (62–63 s) or never |

In the eighth round's first part (GuC log level 3), 6 of 17 went unreported: the forward-only soak 4 of 4, the v1 pass 1 of 1, and
1 of the blocking-wait reproducer's 11, which is different again: no reset came while it waited (observation 9). In its second part
(level 1), 32 of 80: the soak 5 of 5, the unbounded training passes 3 of 6, the reproducer 12 of 19 (SYCL) and 6 of 12 (oneMKL), the
`drm.debug` reproducer 6 of 18, and the blocking-wait reproducer 0 of 20. Each reported reproducer failure came within 2 ms of the
`Engine reset` line, except one `drm.debug` pass (1.0 s) and half the blocking ones (10 of 20 at 0.358 s, the other 10 within 1 ms).

**What decides it (eighth round, second part).** The `repro_sycl_drm_debug` variant runs the reproducer with `drm.debug` 0x2
(driver messages) for its passes and keeps the kernel log of each failed one (`kernel_debug/`). It failed 18 of 20, and the logs
split exactly by outcome:

| | Reported (12) | Not reported (6) |
|---|---|---|
| driver messages before `Engine reset … guc_id=8` | none (only the start-up `xe_vm_create`/`vm_bind` argument checks) | none |
| after the reset | `xe_exec_ioctl: Ioctl argument check failed at drivers/gpu/drm/xe/xe_exec.c:147: q->ops->reset_status(q)`, 14–16 times in 12 s, then `exec queue reset detected` twice | nothing but the devcoredump |
| what the process was doing | submitting: its next submission was refused within 2 ms | waiting in `queryCounterBasedEventStatus` (stack above, again in this run) |

So the kernel knows the queue was reset and refuses further submissions on it, and the runtime turns that refusal into an error,
but a process that is only waiting on an event never asks. Its own wait keeps reading a counter the reset context will never write.
Blocking waits, which in our understanding wait in the kernel rather than polling memory, were reported every time at level 1 (20 of
20, against 7 of 19 polled). We think the polled path should check the queue's reset status, as the refused submission does (an
inference about the runtime, which we haven't read).

Example, pass `bisect_control_04` (pid 179894): last progress at 04:43:18.55, `Engine reset … guc_id=8` and the devcoredump at
04:43:19.32 naming pid 179894, then nothing until our 60 s timeout at 04:44:18.67. It happens with no queue bound as well as with one,
and on both adapters. Before we added a timeout, the same state hung processes for 15 minutes to 6.5 hours, with the main thread in
`sycl::event::wait → urEventWait → libze_intel_gpu.so`
([ONEAPI_SYCL_DRIVER_SEGFAULT.md](ONEAPI_SYCL_DRIVER_SEGFAULT.md) describes the August 2026 occurrence).

### 4. Teardown after a reset

After giving up on such a wait, a process on the v1 adapter released its work and then hung during exit until killed (3 of 3 times, 900
s each); on the v2 adapter it exited normally (15 of 15). We don't yet have a stack for it.

In the fifth round the process skipped the runtime's teardown after such a wait (`std::_Exit` once its work is released, our
workaround), and every v1 pass in that state exited at once (4 of 4, within 30 ms of giving up, 62–63 s into the pass). So the hang is in
the teardown that runs after `main` returns (static destructors or `atexit` handlers of the SYCL or Level Zero libraries), not in
anything our code does on the way out.

The sixth round kept the teardown on v1 (`ADAI_GPU_KEEP_TEARDOWN=1`) to take a stack of the hang, but its only v1 failure was a
reported loss, whose teardown finished normally, so the hang wasn't seen. On the v2 adapter, the reproducer, which returns from `main`
normally after a reported loss, spent about 10 s in exit each time: the stacks show the main thread in `exit → _dl_fini →` a
destructor of `libze_intel_gpu.so.1` (11 frames, no symbols) `→ sched_yield`, spinning. Two `exec queue reset detected` lines appear in
the kernel log 4 and 8 s after the reset, and the process exits after the second. For example `bisect_repro_sycl_01`: the error and
the reset at 09:09:46.48, the stack at 09:09:51.75, `exec queue reset detected` at 09:09:50.53 and 09:09:55.66, then exit.

The seventh round, on the newest runtime with its debug symbols, names the destructor (`bisect_repro_mkl_01`, failed with
`OUT_OF_RESOURCES` after 112,251 operations; stack 6 s after the reset). Freeing the USM pool at exit waits for the engine to reach a
task count the reset context will never reach:

```
NEO::WaitUtils::waitFunction → sched_yield               (timeElapsedSinceWaitStarted=71024)
NEO::CommandStreamReceiver::baseWaitFunction             (taskCountToWait=112300)
NEO::DrmMemoryOperationsHandler::evictUnusedAllocationsImpl
NEO::BufferObject::exec / evictUnusedAllocations
NEO::DrmCommandStreamReceiver<Xe2HpgCoreFamily>::flush   (flushPipeControl, flushSmallTask)
NEO::CommandStreamReceiver::baseWaitFunction             (taskCountToWait=112301)
NEO::MemoryManager::waitForEnginesCompletion
NEO::MemoryManager::freeGraphicsMemory
NEO::SVMAllocsManager::freeSVMAlloc
NEO::UsmMemAllocPool::cleanup → UsmMemAllocPoolsManager::cleanup → UsmMemAllocPoolsFacade::cleanup
L0::DriverHandle::~DriverHandle                          level_zero/core/source/driver/driver_handle.cpp:235
_dl_fini → exit
```

The outer wait flushes a pipe control first, and that submission in turn waits, to evict unused allocations, for the task count
before it. On this runtime the reproducer took 12.5–16 s to exit after a reported loss (14 of 15 passes; the other was held by the
debugger taking this stack), against about 10 s before, with the two `exec queue reset
detected` lines 7.0–10.5 s and 11.5–15.5 s after the reset. The v1 adapter had no failure in the seventh round, so the v1 exit hang
still had no stack.

The eighth round's first part had one: `bisect_v1_teardown_15` (v1, the runtime's teardown kept). The engine reset at
23:03:25.92 and wasn't reported; the process gave up at its 60 s limit, released its work, and then hung at exit until our 240 s
limit killed it. Its stack, 10 s after it gave up, is the same teardown wait as v2's above (`~DriverHandle → … →
UsmMemAllocPool::cleanup → freeGraphicsMemory → waitForEnginesCompletion → baseWaitFunction` (task count 71334) `→
flushPipeControl → … → evictUnusedAllocationsImpl → baseWaitFunction` (task count 5094) `→ sched_yield`). Two things differ from
v2: the kernel logged no `exec queue reset detected` line at all after the reset (on v2 the process exits after the second one), and
when the process was killed, the copy engine was reset as well (`Engine reset: engine_class=bcs, logical_mask: 0x1, guc_id=10` at
23:07:23.11, no devcoredump). So on v1 the same wait never ends.

### 5. A long kernel alone is not reset

A single-work-item kernel spinning on a `malloc_shared` flag ran for an hour with no engine reset, and the host's write to the flag
never ended it. When the process was killed, the engine reset (`03:40:22`, `guc_id=8`), with no devcoredump. This is what we'd expect
for a long-running context with an infinite job timeout and nothing else asking for the engine, and is why we think the resets above
need a preemption request. (Our test now holds the queue with a host task instead, which passes on this card in 1 s.)

### 6. The reset follows the stall by the preempt timeout

From the last progress line a training pass logged to the engine reset in the kernel log (every failed training and seed pass of
the fifth to seventh rounds):

| `preempt_timeout_us` | Passes | Last progress → `Engine reset` |
|---|---|---|
| 640000 (default) | 19 (fifth round) + 4 (sixth) + 9 (seventh, newest runtime) | 0.68–0.90 s; seventh 0.65–0.83 s |
| 640000, time slice 100 ms | 4 (sixth) + 2 (seventh) | 0.83–0.93 s |
| 5000000 | 1 | 5.07 s |

Example, pass `bisect_long_preempt_06`: last progress at 07:02:21.54, `Engine reset … guc_id=8` and the devcoredump at 07:02:26.60,
whose context shows `Preempt timeout: 5000000`; the process got no error and gave up at its 60 s timeout. Our training passes log
several lines a second, so the last line comes some tenths of a second before the work stops. (The forward-only soak and decode
benchmark log nothing while they run, so they can't be timed this way.)

The reset therefore comes one preempt timeout after the work stopped, whatever that timeout is, and raising it from 640 ms to 5 s
did not let the context finish being preempted. The GPU dumps can't tell us what asked for the preemption: nothing else was using
the GPU.

### 7. A standalone reproducer

`benchmarks/SyclEngineResetRepro.cpp` (one file, SYCL and optionally oneMKL; `icpx -fsycl -qmkl -O2 -DREPRO_WITH_MKL
SyclEngineResetRepro.cpp`) repeats the training pass's GPU pattern with no model and no project code: on one low-priority in-order
queue, per step an upload, 24 "layers" of a 16×64 by 64×64 matrix multiply (a SYCL kernel, or oneMKL `gemm`), a LayerNorm and a softmax with
work-group reductions, a GELU, a float-atomic sum of squares and a device-to-device copy, then a one-float download the host waits for
(polling, with a 60 s limit). It runs about 340 steps (50,000 operations) a second. 15 s per pass; the seventh round on the newest
runtime and GuC:

| Reproducer | Failed, 6th round | Reported | Unreported | Failures per GPU minute | Failed, 7th | Reported | Unreported | Per GPU minute |
|---|---|---|---|---|---|---|---|---|
| oneMKL multiply | 14 of 20 | 9 | 5 | 1.43 | 16 of 20 | 9 | 7 | 1.28 |
| SYCL kernel multiply | 17 of 20 | 8 | 9 | 1.34 | 18 of 20 | 6 | 12 | 1.17 |
| oneMKL, a wait before every 8th operation | 1 of 20 | 0 | 1 | 0.16 | 1 of 20 | 0 | 1 | 0.16 |

Every failure left the same devcoredump as the training passes' (`LR job cleanup`, `IPEHR 0x10400002`). Failures came anywhere from
the first to the 14th second of a pass. The bound helps here too (over both rounds 2 of 40 against 30 of 40 for oneMKL unbounded,
p ≈ 6·10⁻¹¹), though with a wait every 8
operations the reproducer runs about 650 steps per pass against 5,100, so per operation the difference is about 3 times (0.5 against
1.5–2.3 failures per million operations), not 9. oneMKL isn't needed: the SYCL-only multiply fails as often.

A reported failure is `UR_RESULT_ERROR_OUT_OF_RESOURCES` (12, seventh round 12) or `UR_RESULT_ERROR_OUT_OF_DEVICE_MEMORY` (5,
seventh 3) from the next wait (exit code 3); an unreported one leaves the download's event pending until the 60 s limit (exit
code 2). These runs used the default low-priority queue; at normal priority it isn't reset (observation 10).

### 8. The newest compute runtime and GuC

The seventh round repeated the sixth's variants after `update_gpu_stack.sh install --dbgsym` and a reboot: compute runtime 26.40.40196.15
(IGC 2.42.14, gmmlib 22.10.2, Level Zero loader 1.34.0) and GuC 70.83.0, same kernel. Nothing changed:

| Variant | 6th round (26.31, GuC 70.72.1) | 7th (26.40, GuC 70.83.0) |
|---|---|---|
| training, unbounded | 2 of 40 | 6 of 40 |
| training, a wait before every 8th operation | 0 of 20 | 0 of 20 |
| forward-only soak | 12 of 20 | 11 of 20 |
| reproducer, oneMKL / SYCL / oneMKL bounded | 14, 17, 1 of 20 | 16, 18, 1 of 20 |
| training with the Level Zero debug log | 1 of 20 | 3 of 20 |
| training on the v1 adapter | 1 of 20 | 0 of 20 |
| training, time slice 100 ms | 4 of 40 | 2 of 40 |
| all | 52 of 220, 30 unreported | 57 of 220, 33 unreported |

All 57 devcoredumps have the same signature (`LR job cleanup`, `IPEHR 0x10400002`, `RING_ESR` 0), now with `GuC version: 70.83.0`.
The training passes reset 0.65–0.93 s after their last progress, as before (observation 6). Two things differ, both small: the
training passes' reported errors are now mostly `UR_RESULT_ERROR_DEVICE_LOST`, and the reproducer's exit after a reported loss takes
about 13 s instead of 10 (observation 4).

### 9. A more verbose GuC log, and blocking waits

The eighth round's first part ran after `xe_debug.sh enable` (`options xe guc_log_level=3` in `/etc/modprobe.d`) and a reboot, with
the kernel, GuC 70.83.0, compute runtime 26.40 and oneAPI of the seventh round, and the same package build plus new variants:

| Variant | 7th round (GuC log level 1) | 8th, first part (level 3) | 8th, second part (level 1) |
|---|---|---|---|
| training, unbounded | 6 of 40 | 0 of 40 | 6 of 40 |
| training, a wait before every 8th operation | 0 of 20 | 0 of 20 | 0 of 20 |
| forward-only soak | 11 of 20 | 4 of 20 | 5 of 20 |
| reproducer, oneMKL multiply | 16 of 20 | 1 of 20 | 12 of 20 |
| reproducer, SYCL kernel multiply | 18 of 20 | 0 of 20 | 19 of 20 |
| reproducer, oneMKL, a wait before every 8th operation | 1 of 20 | 0 of 20 | 0 of 20 |
| training on the v1 adapter, teardown kept | 0 of 20 | 1 of 20 | 0 of 20 |
| all of the above | 52 of 160 | 6 of 160 (p ≈ 6·10⁻¹²) | 42 of 160 |
| reproducer, SYCL, blocking waits (`--blocking-wait`) | | 11 of 20 | 20 of 20 |
| reproducer, SYCL, every runtime call traced (`SYCL_UR_TRACE=2`) | | 0 of 20 | 0 of 20 |
| reproducer, SYCL and oneMKL, normal priority (`--normal-priority`) | | 0 of 20, 0 of 20 | 0 of 20, 0 of 20 |
| reproducer, SYCL, `drm.debug` 0x2 | | | 18 of 20 |

The reproducer alone: 1 of 40 against 34 of 40 (p ≈ 5·10⁻¹⁵); the unbounded training passes 0 of 40 against 6 of 40 (p ≈ 0.03); the
soak 4 of 20 against 11 of 20 (p ≈ 0.05). All 16 dumps have the usual signature (`LR job cleanup`, `IPEHR 0x10400002`) with `Log
level: 3`. As far as we know the level changes only how much the GuC writes to its log, and so how long it takes over each scheduling
event; we read the drop as the resets depending on the GuC's timing (an inference). The second part, after `xe_debug.sh disable` and
another reboot, ran the same build at level 1 (`host_details.txt`: `xe.guc_log_level: 1`; every one of its 80 dumps says `Log level:
1`): 42 of 160 against level 3's 6 (p ≈ 10⁻⁸), and no different from the seventh round's 52 (p ≈ 0.27). So the level, not the
reboot, made the difference.

**Blocking waits.** The reproducer with plain `sycl::event::wait()` calls for its per-step download failed 11 of 20 in the same run
as the polled one's 0 of 20 (p ≈ 0.0002). It also runs faster (about 8,200 steps in 15 s against 5,300), but that doesn't explain
the difference. Ten of the 11 were reported (`OUT_OF_RESOURCES` 5, `DEVICE_LOST` 4, `OUT_OF_DEVICE_MEMORY` 1), each within a
millisecond of the reset. The eleventh, `bisect_repro_sycl_blocking_08`, is the first failure we have seen without a reset: its
progress stopped after 22:41:32.80 (about 1,016,000 operations), the wait didn't return, and the kernel logged nothing for 60 s,
until our watchdog ended the process (`std::_Exit`, 22:42:33.654); 13 ms later the kernel logged `Engine reset: engine_class=ccs,
logical_mask: 0x1, guc_id=8`, with no devcoredump, the way the spinning kernel's engine was reset when its process was killed
(observation 5). So here the context stopped making progress, and nothing asked it to yield for 60 s. At level 1 blocking waits
failed 20 of 20 against level 3's 11 of 20 (p ≈ 0.001), all 20 reported (observation 3), none stuck.

The normal-priority reproducers (0 of 40, and about 9,800 steps a pass) and the traced one (0 of 20) failed too rarely at this level
to compare with anything; at level 1 they can be (observation 10). The traced reproducer had no failure at level 1 either (0 of 20,
about 2,350 steps a pass against the untraced one's ~5,300 when it survives), so tracing every call slows it out of the problem and
gives us no trace of a failure.

### 10. Queue priority

Our workloads all create their queue with `sycl::ext::oneapi::property::queue::priority_low` (the codebase's default
`GPU_STRATEGY=background`, and the reproducer without `--normal-priority`). The reproducer with `--normal-priority`, the only
difference, in the same round-robin as the low-priority one:

| Reproducer | 8th, first part (level 3) | 8th, second part (level 1) |
|---|---|---|
| SYCL kernel multiply, low priority | 0 of 20 | 19 of 20 |
| SYCL kernel multiply, normal priority | 0 of 20 | 0 of 20 |
| oneMKL multiply, low priority | 1 of 20 | 12 of 20 |
| oneMKL multiply, normal priority | 0 of 20 | 0 of 20 |

At level 1, 0 of 40 against 31 of 40 (p ≈ 4·10⁻¹⁴). A normal-priority pass also does more work: about 9,100 (SYCL) and 8,100
(oneMKL) steps in its 15 s, against about 5,300 for a low-priority pass that survives. So more work at normal priority is never reset,
less at low priority nearly always is.

The priority run (November 16, level 1, 20 rounds) repeated it with our own training and forward-only soak, at no priority hint
(`GPU_STRATEGY=normal`, the queue's only property `in_order`) and at `priority_high` (`GPU_STRATEGY=full`):

| Workload | Low priority | No hint | High priority |
|---|---|---|---|
| Training pass (unbounded queue) | 5 of 20 | 0 of 20 | 0 of 20 |
| Forward-only soak | 11 of 20 | 0 of 20 | 0 of 20 |
| Reproducer, SYCL kernel multiply | 18 of 20 | 0 of 20 | |
| All | 34 of 60 | 0 of 60 | 0 of 40 |

No hint against low priority: p ≈ 1.5·10⁻¹³. Each of the 34 failed passes has one `Engine reset: engine_class=ccs` in the kernel log
and one devcoredump, all with the signature of observation 1; the passes at no hint or high priority logged no kernel event. The
low-priority rates match the earlier level-1 rounds. A training pass is short (about 1 s of GPU work), so its own 0 of 20 says less
than the soak's and the reproducer's. Since the GuC schedules contexts by priority, we read this as the reset coming from how the GuC
schedules or preempts a low-priority long-running context (an inference).

A full run on November 17 (level 1, the same reproducer variants) repeated the reproducer comparison: 0 of 40 with no hint against
35 of 40 at low priority. A fourth run later that day: 0 of 40 against 32 of 40. Over the four level-1 runs: 0 of 140 against
116 of 140.

That fourth run ran everything else with no priority hint: our GPU unit tests, a forward-only decode benchmark at two model sizes
(about 7.5 minutes of GPU work), training passes, and 40 passes of a 10-epoch training soak. None was reset; the run just before it,
the same workloads at low priority, lost the device in all 4 decode-benchmark attempts and 6 of the 40 soak passes. A low-priority
context was also slower on an otherwise idle GPU: the decode benchmark's cached path took 7.7 ms per token at low priority against
2.6 ms with no hint.

### 11. A newer kernel

On November 18, 2026, ai-machine booted Ubuntu's `7.0.0-43-generic` (`7.0.0-43.43~24.04.1`, upstream 7.0.14, Secure Boot on) once and
ran the eighth round's bisection there (`--bisect-only --kernel-debug`, 01:40–01:57 UTC), with the same package, GuC 70.83.0, compute
runtime and oneAPI runtime as the run before it on `6.17.0-1025-oem` (November 17, 23:07–00:21 UTC). We could not use Ubuntu's mainline
build (7.3.5): it isn't signed for Secure Boot, and enrolling our own key needs the console, which this host doesn't have.

| Variant (low priority unless named) | 7.0.0-43 | 6.17.0-1025-oem |
|---|---|---|
| training pass, unbounded (`control`) | 0 of 40 | 2 of 40 |
| training pass, wait every 8th operation | 0 of 20 | 1 of 20 |
| forward-only soak | 0 of 20 | 9 of 20 |
| reproducer, oneMKL GEMM | 0 of 20 | 12 of 20 |
| reproducer, SYCL kernel, polled waits | 0 of 20 | 20 of 20 |
| reproducer, blocking waits | 0 of 20 | 20 of 20 |
| training pass on the v1 adapter with its teardown | 0 of 20 | 1 exit hang |
| reproducer with `drm.debug` 0x2 | 0 of 20 | not run (31 of 40 failed in the eighth round's second part) |
| reproducer waiting every 8th operation, traced, or at normal priority (both kinds) | 0 of 80 | 0 of 80 |
| kernel log: engine resets, job timeouts, faults | 0 | 130 |

The low-priority reproducer logs `low priority` on 7.0 too, and does a median 6,597 steps in 15 s against 9,740 at normal priority
(on 6.17, 5,491 against 9,783 in the passes that finished). So the GuC still schedules the context at low priority; it is no longer
reset. With the GuC and the user-space stack unchanged, the difference is in the kernel's xe driver between 6.17.0-1025-oem and 7.0.14.
We have not bisected which change.

A full test run on the same boot a few hours later (unit tests, the decode benchmark, training passes, both soaks and the bisection
again) also had no failure and no engine reset.

## What Intel will ask for

- [x] A standalone reproducer, with its reset rate: `benchmarks/SyclEngineResetRepro.cpp` fails 65 of 80 passes of 15 s on the
  B60 over two rounds (observation 7).
- [x] The rate with the compute engine's `preempt_timeout_us` raised and with `timeslice_duration_us` raised (`run_gpu_tests.sh
  --engine-tuning`, fifth round): a 5 s preempt timeout still reset, 5 s after the stall (observation 6); a 100 ms time slice had
  0 of 20, 4 of 40 in the sixth round and 2 of 40 in the seventh: no effect. The driver refused `preempt_timeout_us` 0.
- [ ] A stack of a process waiting on an unreported reset, and of the v1 exit hang. Partly done:
  - The v2 teardown after a reported loss is named (observation 4): `L0::DriverHandle::~DriverHandle` frees the USM pool and waits
    in `CommandStreamReceiver::baseWaitFunction` for a task count the reset context never reaches.
  - A process polling an unreported reset's event is in `L0::EventImp::queryCounterBasedEventStatus` (observation 3). The stack
    of a plain `sycl::event::wait` (the August stack, in `urEventWait → libze_intel_gpu.so`, without symbols) is still to take:
    at level 1 every blocking-wait failure was reported (20 of 20), so none waited long enough for a stack. That fits observation 3:
    a blocking wait learns of the reset.
  - The v1 exit hang recurred in the eighth round, and its stack is the v2 teardown wait, never ending (observation 4).
  - The eighth round's blocking-wait reproducer got stuck once, but with no reset to trigger our stack, and the variant's two
    stacks had gone to reported losses (their exits, as in observation 4). The runner now takes a stack of a reproducer whose
    progress stops with no reset, and none of a reproducer's exit after a reported loss.
- [x] The result with the newest compute runtime and GuC available: compute runtime 26.40.40196.15 with IGC 2.42.14 and gmmlib 22.10.2
  (Intel graphics PPA for Ubuntu 24.04) and GuC 70.83.0 (linux-firmware `f51d6d83`), seventh round: no change (observation 8).
- [x] The result on a newer kernel (we run Ubuntu's `6.17.0-1025-oem`): Ubuntu's 7.0.0-43 (7.0.14) has no resets at all
  (observation 11). Not the current mainline: its build isn't signed for Secure Boot, and this host has no console to enroll a key.
- [x] The reproducer at normal priority (`--normal-priority`): 0 of 40 against the low-priority reproducer's 31 of 40 at GuC log
  level 1 (eighth round, second part; observation 10). Training and the forward-only soak at no hint and at high priority: 0 of
  100 against 34 of 60 at low priority (priority run, November 16; observation 10).
- [x] A GuC log at a higher level and a `drm.debug` kernel log around one reset: the eighth round's first part's 16 dumps carry a
  level-3 GuC log, and that level changed the rate (observation 9); its second part kept the kernel log of 18 failed passes with
  `drm.debug` 0x2 (observation 3). A level-3 dump of a reset at level 1's rate we can't have, since level 3 suppresses them.
- [x] Host details the xe template asks for (eighth round, `host_details.txt` and `kernel_boot_log.txt.gz`): resizable BAR on,
  BAR 2 32 GB; the card's own switch (`8086:e2ff`) links to the board at 16 GT/s x8 (it can do 32 GT/s; the board's PCIe 4.0 slot
  can't), the GPU behind it at the switch's internal 2.5 GT/s x1; board MSI B550M PRO-VDH WIFI (MS-7C95), BIOS 2.N0 of
  09/02/2025; ASRock card (subsystem `1849:6023`); `intel_iommu=on iommu=pt pci=realloc` on an AMD host; SR-IOV capable, no
  VFs.
- [ ] A Level Zero log around one unreported reset. `UR_LOG_LEVEL_ZERO=level:debug` (the `ur_debug` variant) logs only
  initialization (95 lines, nothing per call), in the seventh round too, and its failures there were all reported.
  `repro_sycl_trace` runs the reproducer with `SYCL_UR_TRACE=2` instead, keeping the last 64 MiB of each failed pass's call trace;
  in the eighth round's first part it had no failure (0 of 20, at GuC log level 3).

## Workarounds in this codebase

- `GPU_QUEUE_DEPTH` (default 8): wait for the queue before every 8th GPU operation.
- `GPU_WAIT_TIMEOUT_SECONDS` (default 60): every SYCL wait polls its event and gives up, so an unreported reset fails the pass like a
  lost device; the process then exits without running the runtime's teardown.
- `trainer_service` restarts the pass; `TRAINER_PASS_STALL_MINUTES` stops one that makes no progress.
- `GPU_STRATEGY=normal` (a queue with no priority hint, as the reproducer's `--normal-priority`) or `full` (`priority_high`):
  neither was reset in the priority run (observation 10). `normal` is the default since November 17, 2026 (it was `background`,
  `priority_low`).

Details, raw numbers and history: TD-574 in [TECHNICAL_DEBT.md](../../../development/guides/TECHNICAL_DEBT.md);
the settings in [GPUDebugSync.md](../../../development/reference/source/GPUDebugSync.md).
