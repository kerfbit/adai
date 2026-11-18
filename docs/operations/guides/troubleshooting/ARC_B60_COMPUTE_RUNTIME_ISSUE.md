After a GPU engine reset, a polled counter-based event never completes and process exit waits for the reset context (Arc Pro B60, xe)

<!-- Issue text for intel/compute-runtime (TD-574). Line 1 is the title; the rest is the body. The full investigation is
ARC_B60_ENGINE_RESETS_INTEL_REPORT.md next to this file. The reproducer's source is appended to the body when filed. Filed November 18, 2026 as https://github.com/intel/compute-runtime/issues/1057. -->

### Summary

When the GuC resets the compute engine under a Level Zero (SYCL) process on an Arc Pro B60 with the xe driver, the runtime
often never reports the reset to the process:

1. **A process that polls a counter-based event never learns of the reset.** Its event stays pending forever and no error is
   returned. In 110 of 172 failed runs the process got no error; without a timeout of its own it waits indefinitely (once
   6.5 hours). The kernel knows: the reset is in the kernel log and a devcoredump within a second, and if the process submits
   again, `xe_exec_ioctl` refuses (`xe_exec.c:147: q->ops->reset_status(q)`) and the runtime turns that into
   `UR_RESULT_ERROR_DEVICE_LOST`/`OUT_OF_RESOURCES`. A process that only polls never makes that call.
2. **Process exit waits for work the reset discarded.** `L0::DriverHandle::~DriverHandle` frees the USM pool and waits in
   `CommandStreamReceiver::baseWaitFunction` for a task count the reset context will never reach. On the v2 adapter that
   delays exit by 10–16 s (until the kernel's `exec queue reset detected`); on the v1 adapter it never ends (3 of 3, killed
   after 900 s, and once more later after 240 s).

**Expected:** after an engine reset, every pending and later wait on the reset queue's events fails with
`ZE_RESULT_ERROR_DEVICE_LOST`, polled or blocking, and teardown doesn't wait for work the reset discarded.

The resets themselves came from Ubuntu's `6.17.0-1025-oem` kernel and are gone on `7.0.0-43-generic` (upstream 7.0.14: 0 of
500 test passes reset, against 65 of 160 in the same variants on 6.17, with the same GuC, runtime and binaries). This issue is only about how the runtime
handles a reset when one happens.

### Environment

| | |
|---|---|
| GPU | Intel Arc Pro B60 (Battlemage, `8086:e211`), one card, no display |
| Host | AMD Ryzen 7 5800XT, Ubuntu 24.04, Secure Boot on |
| Kernel | `6.17.0-1025-oem` (Ubuntu) where the resets happen; `force_execlist=N`; `intel_iommu=on iommu=pt pci=realloc` |
| GuC | 70.72.1 and 70.83.0 (same behaviour) |
| Compute runtime | 26.31.39395.14 (IGC 2.40.13) and 26.40.40196.15 (IGC 2.42.14, Level Zero loader 1.34.0), Intel graphics PPA for 24.04 (same behaviour) |
| oneAPI | DPC++ 2026.2.0, oneMKL 2026.2.1 |
| Adapters | Unified Runtime Level Zero v2 (default) and v1 (`SYCL_UR_USE_LEVEL_ZERO_V2=0`) |

### Reproducer

The source is below: SYCL only (optionally oneMKL), no other code. One in-order queue with
`sycl::ext::oneapi::property::queue::priority_low` (that is what makes the GuC reset it on 6.17: with no priority hint, 0 of 80
passes reset against 32 of 80), and per step an upload, 24 small matrix multiplies, LayerNorm and softmax with work-group
reductions, a float-atomic reduction, a device-to-device copy, and a one-float download the host waits for.

```
icpx -fsycl -O2 SyclEngineResetRepro.cpp -o sycl_engine_reset_repro
for i in $(seq 20); do ./sycl_engine_reset_repro --seconds 15; echo "exit $?"; done
journalctl -k | grep -E 'Engine reset|exec queue reset'
```

Exit codes: 0 ran the whole time, 2 a wait didn't finish within `--timeout` (default 60 s; an unreported reset), 3 the runtime
reported an error, 4 no GPU. `--blocking-wait` uses plain `sycl::event::wait()` instead of polling
`info::event::command_execution_status`. On `6.17.0-1025-oem` at GuC log level 1, about 80% of 15-second passes end in an engine
reset (`GT0: Engine reset: engine_class=ccs`, `Reason: LR job cleanup`); of those, the polled ones are reported or not about
half and half, and the blocking ones are always reported (20 of 20, against 7 of 19 polled).

### What reported and unreported resets differ in

With `drm.debug=0x2` around 18 failed passes:

| | Reported (12) | Not reported (6) |
|---|---|---|
| driver messages before `Engine reset … guc_id=8` | none | none |
| after the reset | `xe_exec_ioctl: Ioctl argument check failed at drivers/gpu/drm/xe/xe_exec.c:147: q->ops->reset_status(q)` 14–16 times, then `exec queue reset detected` twice | only the devcoredump |
| the process | submitting: its next submission was refused within 2 ms | polling its event |

Stack of a process polling an unreported reset's event (26.40 with debug symbols):

```
L0::EventImp<unsigned long>::queryCounterBasedEventStatus (blockOnMiss=false)  level_zero/core/source/event/event_impl.inl
L0::EventImp<unsigned long>::queryStatus
ur::level_zero::urEventGetInfo                    libur_adapter_level_zero_v2.so.0
sycl::detail::event_impl::get_info<info::event::command_execution_status>
main
```

The event's status comes from a counter in memory that the reset context will never write, and nothing on this path checks
the queue's reset status. (Our inference: that check is what's missing; we haven't read the runtime's code.)

Stack at exit after a reported reset (v2; the v1 exit hang has the same stack, never ending):

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

On v1 the kernel logged no `exec queue reset detected` line at all after the reset, and when the stuck process was killed the
copy engine was reset as well (`Engine reset: engine_class=bcs … guc_id=10`, no devcoredump).

### Our workaround

Every wait polls with a 60 s limit, treats a timeout as a lost device, and exits with `std::_Exit` (skipping the runtime's
teardown) once a wait has timed out. We can provide devcoredumps (about 400 KB gzipped each), kernel logs and per-pass logs.
