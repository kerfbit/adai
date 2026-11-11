# ADAI SYCL GPU test package

Runs this repository's GPU tests, the TD-171 stacked-multiply benchmark, seeded training passes and a
Level Zero soak on a host with an Intel GPU (ai-machine), and packs the results into one archive to
copy back. Built by `scripts/package_sycl_gpu_tests.sh`. `MANIFEST.txt` records the commit and the
oneAPI release it was built with (`oneapi:`).

## What it checks

| Step | Checks |
|---|---|
| Device check | A SYCL GPU is found, and which backend it uses (TD-391) |
| Test binaries | Every test that runs GPU kernels: the GPU primitives and reductions (TD-385), the device optimizer, embeddings, dropout masks, gated paths, GPU training against the CPU (TD-358), the device-loss check (TD-566), and GPU generation and serving |
| Benchmark | Stacked against separate multiplies at the model's real shapes (TD-171) |
| Training passes | Two `SEED=7` passes and one `SEED=8` pass on the GPU: whether a seed replays (TD-368) |
| Level Zero soak | Back-to-back passes alternating the default Level Zero v2 adapter with `SYCL_UR_USE_LEVEL_ZERO_V2=0`, with the kernel's GPU messages per pass: whether either one avoids the engine resets that end in `DEVICE_LOST`, and whether a device loss stops the pass at once (TD-566) |

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
kernel log and the GuC firmware version and to see other users' processes; it stops nothing.
`--quick` takes a few minutes. Each step has a time limit; if the driver hangs a step it is reported
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
file back. It holds `summary.txt`, `soak_report.txt`, every step's log, the test XML, the environment
(packages, firmware, GPU users, `ldd`, the preparation report) and the kernel log for the run; trained
weights are left out.
