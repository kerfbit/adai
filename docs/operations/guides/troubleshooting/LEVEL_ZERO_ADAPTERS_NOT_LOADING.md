# The GPU Runs Through OpenCL: Level Zero Adapters Not Loading

**Seen:** ai-machine (Intel Arc Pro B60), November 18-22, 2026, after an `apt upgrade` of Intel's oneAPI packages.

## Symptom

Nothing fails. Every SYCL binary (`incremental_trainer`, `chatbot_api_server`, the GPU test package) finds the GPU and runs, but the
startup line names OpenCL:

```
[GPU] Device 0: Intel(R) Arc(TM) Pro B60 Graphics
  Backend: opencl (Intel(R) OpenCL Graphics (discrete))
```

instead of `Backend: level_zero (Intel(R) oneAPI Unified Runtime over Level-Zero V2)`. GPU work is slower (the decode benchmark's GPU
paths ran 1.5-2× slower), and results measure OpenCL, not the Level Zero path every GPU decision in this project was made on. Our
device selection prefers Level Zero (TD-391), so OpenCL here means Level Zero offered no device at all.

## Cause

The SYCL runtime opens its backend adapters (`libur_adapter_level_zero.so.0`, `libur_adapter_level_zero_v2.so.0`, from
`/opt/intel/oneapi/redist/lib`) itself at startup and quietly skips any that fails to load. The Level Zero adapters need
`libumf.so.1` (Intel's Unified Memory Framework). Intel's 2026.2 packages moved it out of `redist/lib` into the versioned
`intel-oneapi-umf-1.2` package (`/opt/intel/oneapi/umf/1.2/lib`), which no linker configuration names. `ldd` of our own binaries
shows nothing wrong, since only the adapters need it. The Level Zero GPU driver (`libze-intel-gpu1`) was fine throughout.

## Diagnosis

```bash
LD_LIBRARY_PATH=/opt/intel/oneapi/redist/lib SYCL_UR_TRACE=1 ./bin/gpuutilsTests \
    --gtest_filter=GPUManagerTest.ProbeDiagnosticIsNonEmpty 2>&1 | grep -E 'level_zero'
```

A broken setup prints `failed to load adapter '…libur_adapter_level_zero.so.0' with error: libumf.so.1: cannot open shared object
file`; a working one prints `loaded adapter … libur_adapter_level_zero.so.0`. Directly:

```bash
for a in level_zero level_zero_v2; do ldd /opt/intel/oneapi/redist/lib/libur_adapter_$a.so.0 | grep -E 'umf|not found'; done
```

## Fix

Put the library's directory on the linker's path:

```bash
echo /opt/intel/oneapi/umf/latest/lib | sudo tee /etc/ld.so.conf.d/adai-oneapi-adapters.conf && sudo ldconfig
```

`umf/latest` follows a later UMF upgrade (ai-machine's own fix names `umf/1.2/lib`, in `intel-oneapi-umf.conf`). The GPU test
package's `prepare_host.sh` now checks the adapters, finds a missing library in a oneAPI component's `latest/lib`, and offers this
fix; `run_gpu_tests.sh` warns when the adapters can't load and fails a run in which any pass logged a backend other than Level Zero.
Restart the deployed services afterwards so they pick up Level Zero.

See TD-590's entry in [TECHNICAL_DEBT_RESOLVED.md](../../../development/archive/TECHNICAL_DEBT_RESOLVED.md#td-590-nothing-checks-that-gpu-training-stays-on-the-device)
for the run that found it and the rerun on Level Zero.
