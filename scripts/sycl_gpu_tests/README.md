# ADAI SYCL GPU test package

Runs this repository's GPU tests, the TD-171 stacked-multiply benchmark and two small seeded training
passes on a host with an Intel GPU (ai-machine), and packs the results into one archive to copy back.
Built by `scripts/package_sycl_gpu_tests.sh`; `MANIFEST.txt` records the commit and oneAPI version.

## What it checks

| Step | Checks |
|---|---|
| Device check | A SYCL GPU is found, and which backend it uses (TD-391) |
| Test binaries | Every test that runs GPU kernels: the GPU primitives and reductions (TD-385), the device optimizer, embeddings, dropout masks, gated paths, GPU training against the CPU (TD-358), and GPU generation and serving |
| Benchmark | Stacked against separate multiplies at the model's real shapes (TD-171) |
| Training passes | Two `SEED=7` passes and one `SEED=8` pass on the GPU: whether a seed replays byte for byte (TD-368) |

## Running it

It needs the oneAPI runtime at least as new as the build's (see `MANIFEST.txt`), the GPU driver, and
membership of the `render` group. It stops nothing and changes nothing outside this directory. A
running `trainer_service` shares the GPU, which skews the benchmark; pause it first if you can
(`curl -s -X POST http://127.0.0.1:8084/admin/pause`, and `/admin/resume` afterwards).

```bash
tar -xzf adai-sycl-gpu-tests-*.tar.gz
cd adai-sycl-gpu-tests-*/
./run_gpu_tests.sh
```

`./run_gpu_tests.sh --quick` takes a few minutes: the GPU-specific binaries, a 3-repeat benchmark
and one seed pair. Each step has a time limit. If the driver hangs a step, the step is reported as
timed out; a process stuck in the driver may need a reboot to clear.

## What to send back

The script ends by printing the path of `results/adai-gpu-results-<host>-<time>.tar.gz`. Copy that
file back. It holds `summary.txt`, every step's log, the test XML, the environment (`sycl-ls`,
packages, `ldd`) and the tail of `dmesg`; trained weights are left out.
