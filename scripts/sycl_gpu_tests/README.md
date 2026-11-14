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
| Engine-reset bisection | TD-574: whether the shipped GPU queue bound stops the Arc engine resets. Training passes in variants run round-robin (third round, after the bound became the default): `control` (`GPU_QUEUE_DEPTH=0`, the unbounded queue of the first two rounds) and `default` (the shipped `GPU_QUEUE_DEPTH=8`), each twice per round; `depth_16` and `depth_32` (where the bound stops working); `sync_step` (`ADAI_GPU_SYNC=step`); and `encode_soak` and `encode_soak_default`, forward kernels only (`gpu_encode_soak`) without and with the bound. Every variant except `default`, `depth_N` and `encode_soak_default` runs unbounded, so it stays comparable with the earlier rounds. The variants settled by them (`sync_kernel`, `sync_every_N`, `full_priority`, `no_dropout`, `host_embedding`, `host_optimizer`, `host_all`) still run when named in `BISECT_VARIANTS`, a space-separated list that replaces the default one. Every pass keeps its last GPU work in `gpu_trace.txt`, so a hung pass leaves it too. `--engine-tuning` adds the two sysfs variants described below, unbounded. Results in `bisect_report.txt`, with each variant's seconds on the GPU and losses per minute |

Every step except the bisection's unbounded variants runs with the GPU queue bound `incremental_trainer`,
`chatbot_api_server` and every other binary ship with since TD-574 (`GPU_QUEUE_DEPTH=8`, a wait for
the GPU before every 8th operation). Run the whole package with `GPU_QUEUE_DEPTH=0` in the
environment to see the unbounded behaviour instead. The Level Zero and LeJEPA soaks check how a lost
device is handled (TD-566), so with the bound they may see few or no losses to check.

Every step also copies any crash dump the xe driver wrote (`/sys/class/drm/card*/device/devcoredump/data`,
one per engine reset) to `coredumps/<step>_<card>.txt` and frees it, so the next reset can write one;
`coredumps/index.txt` lists them. Needs sudo.

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
see other users' processes and, only with `--engine-tuning`, to change two compute engine settings; it
stops nothing.
`--quick` takes a few minutes. `--lejepa-only` runs just the device check and the LeJEPA passes,
which is what TD-572 needed (`--lejepa-soak-passes N`, default 20, sets the soak's length).
`--bisect-only` runs just the device check and the engine-reset bisection, which is what TD-574 needs
(`--bisect-passes N` rounds, default 20, about 30 minutes without hangs; `--skip-bisect` leaves
it out). `--engine-tuning` (needs sudo) adds `long_preempt` and `long_timeslice`, which raise the
compute engine's preempt timeout to 5 s or its time slice to 100 ms in sysfs for their own passes,
capped at the driver's maxima; the original values are restored after every such pass and when the
script exits, and recorded in `engine_tuning.txt`. Each step has a time limit; if the driver hangs a step it is reported
as timed out, and if the GPU stops running work after a device loss the soak stops and asks for a
reboot.

**`guc_firmware.sh status | install | uninstall`** (sudo) checks or changes the Arc B60's GuC
firmware. Ubuntu 24.04's newest `linux-firmware` has GuC 70.44.1, older than the 70.49.4 ai-machine's
xe driver asks for. `install` downloads GuC 70.72.1 from a pinned upstream linux-firmware commit,
checks its SHA-256 and header version, puts it in `/lib/firmware/updates/xe/` (searched before
Ubuntu's firmware, untouched by package upgrades) and rebuilds the initramfs; reboot afterwards.
`uninstall` removes only that file. Run the soak before and after to compare.

Result on ai-machine (November 11, 2026): with 70.72.1 loaded, engine resets were as frequent as with 70.44.1 (8 in 40 soak
passes, against about 7), so the firmware isn't the cause. 70.72.1 is still installed: it is at least the version the driver
asks for and made nothing worse. See TD-566 in `docs/development/guides/TECHNICAL_DEBT.md`.

**`restore_host.sh`** starts the services and containers `prepare_host.sh` stopped. The runtime
upgrade stays.

A GPU process stopped in the middle of GPU work can make the next one hang early (reported on the
B60). If something was busy when it was stopped, reboot before running the tests; services that start
at boot then need `prepare_host.sh` again.

## What to send back

The script ends by printing the path of `results/adai-gpu-results-<host>-<time>.tar.gz`. Copy that
file back. It holds `summary.txt`, `lejepa_report.txt`, `soak_report.txt`, `bisect_report.txt`, the driver's
crash dumps (`coredumps/`), every step's log, the test XML, the environment
(packages, firmware, GPU users, `ldd`, the preparation report) and the kernel log for the run; trained
weights are left out.
