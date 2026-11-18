# ADAI SYCL GPU test package

Runs this repository's GPU tests, the TD-171 stacked-multiply benchmark, the TD-050 decode benchmark, seeded training passes, the
LeJEPA world-model pass (TD-572), a Level Zero soak and an engine-reset bisection (TD-574) on a host with an Intel GPU (ai-machine), and packs the results into one archive to
copy back. Built by `scripts/package_sycl_gpu_tests.sh`. `MANIFEST.txt` records the commit and the
oneAPI release it was built with (`oneapi:`).

## What it checks

| Step | Checks |
|---|---|
| Device check | A SYCL GPU is found, and which backend it uses (TD-391) |
| Test binaries | Every test that runs GPU kernels: the GPU primitives and reductions (TD-385), the device optimizer, embeddings, dropout masks, gated paths, GPU training against the CPU (TD-358), the device-loss check (TD-566), and GPU generation and serving |
| Benchmark | Stacked against separate multiplies at the model's real shapes (TD-171) |
| Decode benchmark | TD-050/TD-033: greedy decoding with the GPU KV cache, checked token by token against the GPU full recompute it replaced (a choice may differ only at a near tie), then timed against it, the CPU cache, and the CPU with each large multiply sent to the GPU on its own (`chatbot_api_server`'s path before TD-033), for 50 and 100 tokens, at a small shape and at `config.trainer.conf`'s 768/12/3072, 8+24 layers (`gpu_decode_bench`). A lost device is retried once. Results in `decode_benchmark.txt` |
| Training passes | Two `SEED=7` passes and one `SEED=8` pass on the GPU: whether a seed replays (TD-368) |
| LeJEPA world-model pass | `incremental_trainer --objective=lejepa` on the GPU (TD-572): a pass trains and checkpoints, a resume finds nothing pending (exit 3), a re-queued file trains again from the checkpoint, a SIGTERM checkpoints and gives the texts back, and a resume continues from that checkpoint. Then a soak of fresh passes on Level Zero v2 (the default), where a real engine reset should reach TD-566's check: the pass stops at once, saves nothing and releases its texts. Results in `lejepa_report.txt` |
| Level Zero soak | Back-to-back passes alternating the default Level Zero v2 adapter with `SYCL_UR_USE_LEVEL_ZERO_V2=0`, with the kernel's GPU messages per pass: whether either one avoids the engine resets that end in `DEVICE_LOST`, and whether a device loss stops the pass at once (TD-566) |
| Engine-reset bisection | TD-574: evidence for the report to Intel ([ARC_B60_ENGINE_RESETS_INTEL_REPORT.md](../../docs/operations/guides/troubleshooting/ARC_B60_ENGINE_RESETS_INTEL_REPORT.md)). Variants run round-robin (the eighth round's list): `control` (`GPU_QUEUE_DEPTH=0`, the unbounded queue, twice per round) and `depth_8`; `encode_soak`, forward kernels only (`gpu_encode_soak`); `repro_mkl`, `repro_sycl` and `repro_mkl_wait_8`, the standalone reproducer (`sycl_engine_reset_repro`, below) for 15 s with oneMKL or SYCL-kernel multiplies, the last waiting before every 8th operation; `repro_sycl_blocking`, `repro_sycl` with plain `sycl::event::wait()` calls (`--blocking-wait`), so the stack of a process stuck on an unreported reset shows the runtime's wait; `repro_sycl_trace`, `repro_sycl` with every Unified Runtime call traced (`SYCL_UR_TRACE=2`; only the last 64 MiB are kept, in memory, and go to `ur_trace/` for a failed pass); `v1_teardown`, `control` on the Level Zero v1 adapter with `ADAI_GPU_KEEP_TEARDOWN=1`, so a stuck pass runs the runtime's teardown and its hang at exit can be stacked (limit 240 s, reported as `hung`); and `repro_sycl_normal` and `repro_mkl_normal`, the reproducer on a normal-priority queue (`--normal-priority`; every earlier reproducer pass used a low-priority one). `--kernel-debug` (sudo) adds `repro_sycl_drm_debug`, `repro_sycl` with the xe driver's DRM debug messages (`drm.debug` 0x2) in the kernel log for its passes only; a failed pass's kernel log goes to `kernel_debug/`. A variant ending in `depth_N` runs with `GPU_QUEUE_DEPTH=N`, every other one unbounded, so it stays comparable with the earlier rounds. The sixth round's `ur_debug` (`control` with the Level Zero adapter's debug log, `UR_LOG_LEVEL_ZERO`, which logs only initialization; a failed pass's last 64 MiB go to `ur_debug/`) and the fourth and fifth rounds' `depth_16`, `depth_32`, `sync_step` (`ADAI_GPU_SYNC=step`) and `encode_soak_depth_N` still run when named in `BISECT_VARIANTS`. A pass whose wait gave up on stuck GPU work (`GPU_WAIT_TIMEOUT_SECONDS`, 60 s) is reported as `stuck`; one killed at the step's time limit as `hung`. The variants settled by them (`sync_kernel`, `sync_every_N`, `full_priority`, `no_dropout`, `host_embedding`, `host_optimizer`, `host_all`) still run when named in `BISECT_VARIANTS`, and so do the priority variants `normal_priority` (`GPU_STRATEGY=normal`) and `encode_soak_normal_priority`/`encode_soak_full_priority`, a space-separated list that replaces the default one. Every pass keeps its last GPU work in `gpu_trace.txt`, so a hung pass leaves it too. `--engine-tuning` adds the sysfs variants described below, unbounded. Results in `bisect_report.txt`, with each variant's seconds on the GPU and losses per minute |

Every step runs with what `incremental_trainer`, `chatbot_api_server` and every other binary ship with: a wait for the GPU queue
before every 8th operation (`GPU_QUEUE_DEPTH=8`) and a 60 s limit on each wait for GPU work (`GPU_WAIT_TIMEOUT_SECONDS=60`), after
which the work counts as stuck and the pass fails like one that lost the device. Set either in the environment to run the whole
package with other values; the bisection's variants set their own bound. (After the third round the bound was opt-in for a few
hours on November 14, 2026, because some bounded passes seemed to hang in its wait; the fourth round's GPU crash dumps showed
those were engine resets the runtime never reported, which happen unbounded too, so 8 is the default again.)

Every step also copies any crash dump the xe driver wrote (`/sys/class/drm/card*/device/devcoredump/data`,
one per engine reset) to `coredumps/<step>_<card>.txt` and frees it, so the next reset can write one;
`coredumps/index.txt` lists them. Needs sudo.

While a step runs, the runner watches for an engine reset (a new GPU crash dump) and for the step's
log saying a wait gave up. 5 s after a reset whose process is still running (a reset the runtime
didn't report), and 10 s after a wait gave up in a process that still hasn't exited (the v1 hang at
exit), it takes every thread's stack of the step's own processes with `gdb` (through sudo, since
ptrace may otherwise reach only descendants): `stacks/<step>.waiting.txt` and
`stacks/<step>.exit.txt`. A reproducer whose progress lines stop for 20 s with no reset and no give-up
(its work stuck with nothing resetting it) gets `stacks/<step>.stalled.txt`; one that reported a loss
gets no `waiting` stack (that would be its exit, already stacked). At most two of each per variant and 30
per run. `prepare_host.sh` offers to install `gdb`.

**`bin/sycl_engine_reset_repro`** (source in `src/SyclEngineResetRepro.cpp`) is a standalone
reproducer for the report to Intel: SYCL and oneMKL only, none of this project's code. On one
in-order low-priority queue it repeats our training pass's pattern without the model: an upload,
then per layer a 16x64 by 64x64 multiply, LayerNorm and softmax with work-group reductions, GELU,
a float-atomic sum of squares and a device-to-device copy, then a one-float download the host waits
for. Each wait gives up after 60 s; with `--blocking-wait` each wait is a plain `sycl::event::wait()`
and a watchdog thread ends the process after the timeout instead. `--seconds N`, `--layers N`,
`--wait-every N`, `--mkl`, `--normal-priority`, `--timeout N`, `--blocking-wait`; exit 0 ok, 2 stuck, 3 device lost, 4 no GPU. It prints a
timestamped progress line every second. To build it elsewhere: `icpx -fsycl -qmkl -O2
-DREPRO_WITH_MKL SyclEngineResetRepro.cpp -o sycl_engine_reset_repro` (without `-qmkl
-DREPRO_WITH_MKL` it builds without oneMKL, and `--mkl` is refused).

## Running it

Run as a user in the `render` group who can use sudo (on ai-machine, `adai`).

```bash
tar -xzf adai-sycl-gpu-tests-*.tar.gz
cd adai-sycl-gpu-tests-*/
./prepare_host.sh
./run_gpu_tests.sh
./restore_host.sh
```

**`prepare_host.sh`** (sudo; asks before each change, `--yes` to accept the upgrade and service stops):

1. Upgrades the `intel-oneapi-runtime-*` packages from Intel's apt repository to at least the build's
   oneAPI release. It refuses if apt would change any other package, so the GPU driver and kernel
   stay as they are. Afterwards it checks that the deployed binaries in `/opt/adai/bin` still load
   the same libraries (the deployed `chatbot_api_server` uses `/opt/adai/lib`), and sets aside any
   new linker configuration if they don't.
2. Lists every process with the GPU open (with its VRAM use), GPU programs found by name even when
   idle (Ollama, llama.cpp, vLLM, Qwen runs, ADAI's own services...), matching systemd services and
   Docker containers given the GPU, and offers to stop each. It stops services and containers through
   systemd and Docker; any other process gets SIGTERM only after its own confirmation, and is never
   killed. A report goes to `host_state/` and into the results.

**`run_gpu_tests.sh`** loads the oneAPI runtime only from `/opt/intel/oneapi/redist/lib` and refuses
to start when that runtime is older than the build, mixes directories, or another process has the
GPU open (`--allow-runtime-mismatch`, `--allow-shared-gpu` override). It uses sudo only to read the
kernel log, the GuC firmware version and the driver's crash dumps (freeing each after copying it), to
see other users' processes, to take stacks of a step's stuck processes with `gdb` and, only with
`--engine-tuning`, to change two compute engine settings, and only with `--kernel-debug`, to change
`drm.debug`; it stops nothing.
`--quick` takes a few minutes. `--lejepa-only` runs just the device check and the LeJEPA passes,
which is what TD-572 needed (`--lejepa-soak-passes N`, default 20, sets the soak's length).
`--bisect-only` runs just the device check and the engine-reset bisection, which is what TD-574 needs
(`--bisect-passes N` rounds, default 20, about an hour and a half with `--engine-tuning`; `--skip-bisect` leaves
it out). `--engine-tuning` (needs sudo) adds `long_timeslice` (twice per round), which raises the
compute engine's time slice to 100 ms in sysfs for its own passes, and `no_preempt`, which sets
its preempt timeout to 0 (none) if the driver accepts 0 (it's tried once at the start; refused,
the variant is left out). The fifth round's `long_preempt` (preempt timeout 5 s, which only delayed
each reset by as much) runs when named in `BISECT_VARIANTS` with `--engine-tuning`. Values are
capped at the driver's maxima; the original values are restored after every such pass and when
the script exits, and recorded, with the minima and maxima, in `engine_tuning.txt`. Each step has a time limit; if the driver hangs a step it is reported
as timed out, and if the GPU stops running work after a device loss the soak stops and asks for a
reboot.

**The eighth round (TD-574: what the report to Intel still lacks)** runs in three parts, each one boot:

1. On the stock kernel, with a more verbose GuC log: `./xe_debug.sh enable`, reboot, `./prepare_host.sh`, then
   `./run_gpu_tests.sh --bisect-only --kernel-debug`. Run November 15, 2026: the GuC log level cut the polled
   resets to a few, so part 2 is the baseline it needs.
2. On the stock kernel at the driver's default GuC log level: `./xe_debug.sh disable`, reboot, `./prepare_host.sh`,
   then `./run_gpu_tests.sh --bisect-only --kernel-debug` again. That gives the normal-priority and blocking-wait
   rates at the usual level, the plain-wait stack, the call trace, the `drm.debug` kernel logs and whether part 1's
   drop came from the log level. Run November 15–16, 2026: it did (42 of 160 failed again); the reproducer with no
   priority hint was never reset (0 of 40 against 31 of 40).
   Then, still at level 1, training at the other priorities: `BISECT_VARIANTS="control normal_priority full_priority
   encode_soak encode_soak_normal_priority encode_soak_full_priority repro_sycl repro_sycl_normal"
   ./run_gpu_tests.sh --bisect-only` (`normal_priority` is `control` with `GPU_STRATEGY=normal`, no priority hint;
   `full_priority` with `GPU_STRATEGY=full`, `priority_high`). Run November 16, 2026: nothing at `normal` or `full`
   was reset (0 of 100 passes over training, the soak and the reproducer) against 34 of 60 at low priority, so
   `normal` is the default since November 17. The bisection still runs every variant without a priority in its name
   at `GPU_STRATEGY=background`, as in the earlier rounds; the unit tests and benchmarks now run at `normal`.
3. On the newest upstream kernel, also at the default level: `./mainline_kernel.sh install`, `sudo reboot` (it boots
   once), `./prepare_host.sh`, then `./run_gpu_tests.sh --bisect-only --kernel-debug`; then `sudo reboot` back to the
   stock kernel. With Secure Boot on, `install` asks for a one-time password, and the reboot needs someone at the
   console: MokManager's blue screen waits 10 s for a key press, then Enroll MOK, Continue, Yes, the password, Reboot.
   ai-machine has no console, so it ran this part on Ubuntu's Canonical-signed 7.0.0-43 instead (November 18, 2026:
   `sudo grub-reboot "Advanced options for Ubuntu>Ubuntu, with Linux 7.0.0-43-generic"`, `sudo reboot`, then the
   same two steps): 0 of 260 passes failed and no engine reset, against 65 of 160 in the same variants on 6.17.

Afterwards `./mainline_kernel.sh remove` puts the host back. Each part takes about an hour.

**`guc_firmware.sh status | latest | install | install-latest | uninstall`** (sudo) checks or changes the Arc B60's GuC
firmware. Ubuntu 24.04's newest `linux-firmware` has GuC 70.44.1, older than the 70.49.4 ai-machine's
xe driver asks for. `install` downloads GuC 70.72.1 from a pinned upstream linux-firmware commit,
checks its SHA-256 and header version, puts it in `/lib/firmware/updates/xe/` (searched before
Ubuntu's firmware, untouched by package upgrades) and rebuilds the initramfs; reboot afterwards.
`uninstall` removes only that file. `latest` looks up the newest Battlemage GuC in upstream
linux-firmware (from its log of `xe/bmg_guc_70.bin`) against the one the driver loaded;
`install-latest` installs that one the same way, checking the header's version against the
commit's title instead of a pinned SHA-256 (70.83.0 on November 15, 2026). `install` stays at
70.72.1, so it also takes the firmware back. Run the soak before and after to compare.

Result on ai-machine (November 11, 2026): with 70.72.1 loaded, engine resets were as frequent as with 70.44.1 (8 in 40 soak
passes, against about 7), so the firmware isn't the cause. 70.72.1 is still installed: it is at least the version the driver
asks for and made nothing worse. See TD-566 in `docs/development/guides/TECHNICAL_DEBT.md`.

**`update_gpu_stack.sh [check] | install [--yes] [--dbgsym] [--skip-guc] [--allow-shared-gpu] | rollback`**
(sudo) updates the compute runtime and the GuC firmware to the newest available, for TD-574's report to
Intel. `check` (the default) changes nothing: it shows the installed runtime packages against what the
Intel graphics PPA (`ppa:kobuk-team/intel-graphics`, where ai-machine's runtime comes from) has, and the
loaded GuC against the newest upstream. `install` refuses while another process has the GPU open, keeps
a copy of every package it will replace in `host_state/runtime_backup/` (from apt's cache, or rebuilt
with `dpkg-repack`; the PPA keeps only its newest versions), then upgrades only the runtime packages
(`libze-intel-gpu1`, `libze1`, `intel-opencl-icd`, `intel-ocloc`, `libigc2`, `libigdfcl2`,
`libigdgmm12`), plus their installed development packages (`libze-dev` and the like, pinned to their library's version), and refuses if apt would change anything else. `--dbgsym` also installs their debug
symbols from the PPA's debug archive, so the stacks `run_gpu_tests.sh` takes name the runtime's
functions. It then checks which libraries load and that the GPU still runs work, and offers
`guc_firmware.sh install-latest` if a newer GuC exists (reboot afterwards). `rollback` reinstalls the
kept packages. On November 15, 2026 the PPA had compute runtime 26.40.40196.15, IGC 2.42.14 and gmmlib
22.10.2 for Ubuntu 24.04 (ai-machine: 26.31.39395.14, 2.40.13, 22.10.1), and upstream had GuC 70.83.0
(ai-machine: 70.72.1).

**`coredumps.sh status | install | uninstall [--days N]`** (sudo, once per host) makes ai-machine keep its crash
dumps (TD-574). By default both kinds are lost: systemd-coredump deletes process core dumps after 3 days and
the adai services run with `LimitCORE=0`, and the xe driver keeps one GPU crash dump at a time in
`/sys/class/devcoredump` for at most 5 minutes, writing none for later resets while it holds one. `install`
keeps process core dumps N days (default 30, at most 20 GB), turns core dumps on for every `adai*` service
(from its next restart) and for your logins (from the next one), and adds a udev rule that saves each GPU
dump to `/var/lib/adai/gpu-coredumps/` (gzipped, N days) the moment it is written, then frees it, so every
reset leaves one. If process core dumps don't reach systemd-coredump (apport's handler), it offers to install
`systemd-coredump` and disable apport. `uninstall` removes all of it and leaves the saved dumps.
`run_gpu_tests.sh` then copies the GPU dumps saved during a run into `coredumps/` and lists the run's process
core dumps in `process_coredumps.txt` (they stay on the host: `coredumpctl dump <PID> -o core.<PID>`).

**`xe_debug.sh [status] | enable [--guc-log-level N] | disable`** (sudo) sets the xe driver's GuC log level,
which the GuC log in every GPU crash dump is written at (TD-574's report to Intel). The driver loads with 1;
`enable` writes `options xe guc_log_level=N` (default 3, at most 5) to `/etc/modprobe.d/adai-xe-debug.conf`
and rebuilds the initramfs if xe is in it; `disable` removes the file. Both take effect at the next boot.
A more verbose log costs the GuC time, so compare rates only within one run.

**`mainline_kernel.sh [check] | install [VERSION] | boot-once [VERSION] | remove`** (sudo) runs the tests on a
current upstream kernel, which the xe developers ask for. `install` downloads Ubuntu's mainline build
(kernel.ubuntu.com/mainline; default the newest release that isn't a release candidate), checks it against
the release's `CHECKSUMS`, installs its image and modules packages (refusing if apt would change anything
else), and sets GRUB to boot the running stock kernel by default and the new one at the next boot only
(`GRUB_DEFAULT=saved`, `grub-reboot`; `/etc/default/grub` is kept in `host_state/`). If the new kernel
doesn't come up, a power cycle boots the stock one. With Secure Boot on (the builds are unsigned) it needs
`sbsigntool`: it creates a code-signing key in `/var/lib/adai-mainline-mok` (root only), signs the kernel
image with it, and queues the key with `mokutil --import`, which asks for a one-time password that
MokManager asks for again at the next boot, at the console. The kernel's modules need nothing (the build
signs them with a key built into that kernel). If the enrollment is missed, shim refuses the upstream kernel
and the boot stops there until a power cycle, which boots the stock kernel (the one-time entry is used up);
`boot-once` queues it again. **So on a headless host don't use it with Secure Boot on:** before queuing the key
or arming a kernel that needs it, the script says so, lists the installed Canonical-signed kernels newer than
the running one (they boot with no key; ai-machine ran TD-574's kernel test on Ubuntu's 7.0.0-43 that way,
November 18, 2026, with `sudo grub-reboot "Advanced options for Ubuntu>Ubuntu, with Linux 7.0.0-43-generic"`),
and asks whether someone is at the console. A mainline package built for a newer Ubuntu
passes `run-parts` both `/etc/kernel/X.d` and `/usr/share/kernel/X.d`, which 24.04's `run-parts` (debianutils 5.17)
refuses with `missing operand` (v7.3.5 on November 17, 2026); `install` rebuilds such a package in
`host_state/mainline/VERSION/fitted/` with only `/etc/kernel/X.d` in those calls (the kernel and modules unchanged), and
first offers to purge packages a failed attempt left half-installed. `boot-once` arms it for one more
boot; `remove` removes it, puts `/etc/default/grub` back, and offers to take the key off the firmware's
list (`mokutil --delete`, confirmed in MokManager at the next boot) and delete it.

**`restore_host.sh`** starts the services and containers `prepare_host.sh` stopped. The runtime
upgrade stays.

A GPU process stopped in the middle of GPU work can make the next one hang early (reported on the
B60). If something was busy when it was stopped, reboot before running the tests; services that start
at boot then need `prepare_host.sh` again.

## What to send back

The script ends by printing the path of `results/adai-gpu-results-<host>-<time>.tar.gz`. Copy that
file back. It holds `summary.txt`, `lejepa_report.txt`, `soak_report.txt`, `bisect_report.txt`, the driver's
crash dumps (`coredumps/`), the stacks of stuck processes (`stacks/`), the Unified Runtime call traces of failed
`repro_sycl_trace` passes (`ur_trace/`), the kernel logs of failed `repro_sycl_drm_debug` passes (`kernel_debug/`), the Level Zero debug logs of failed `ur_debug` passes (`ur_debug/`), the list of process core dumps (`process_coredumps.txt`), every step's log, the test XML, the environment
(packages, firmware, GPU users, `ldd`, the preparation report), the host details the xe bug template asks for
(`host_details.txt`: the GPU's `lspci -vvv` with resizable BAR and link, its PCI bridges' links, `dmidecode`'s board
and firmware, the driver's debugfs `info`, `drm.debug` and the GuC log level), the kernel log for the run and the
whole kernel log since boot (`kernel_boot_log.txt.gz`); trained
weights are left out.
