# GPU Acceleration Module

GPU-accelerated matrix operations with two mutually exclusive backends, chosen at configure time:
**CUDA** (NVIDIA, `ENABLE_GPU`) and **SYCL** (Intel oneAPI, `ENABLE_SYCL`). Without either, a stub
`GPUManager` reports no GPU and everything runs on the CPU.

Code-traced references (behaviour, callers and known issues) live in
[`docs/development/reference/source/`](../../docs/development/reference/source/):

| File | What it is | Reference |
| --- | --- | --- |
| `GPUUtils.hpp` | Gateway: CUDA `GPUManager` (device, stream, cuBLAS handle, memory budget) and RAII `GPUMemory<T>`; includes `sycl/GPUUtils_SYCL.hpp` under SYCL; a stub without GPU | [GPUUtils.md](../../docs/development/reference/source/GPUUtils.md) |
| `MatrixGPU.hpp` | Gateway: the `matrix_*_gpu` kernel API, the device-resident `GPUMatrix`, and the backend-agnostic `GPUKVCache`/`GPUDecoderKVCache`; includes `sycl/MatrixGPU_SYCL.hpp` under SYCL | [MatrixGPU.md](../../docs/development/reference/source/MatrixGPU.md) |
| `MatrixGPU.cu` | CUDA kernels and cuBLAS wrappers | [MatrixGPU_cu.md](../../docs/development/reference/source/MatrixGPU_cu.md) |
| `sycl/GPUUtils_SYCL.hpp` | SYCL `GPUManager` (in-order queue) and USM-based `GPUMemory<T>` | [GPUUtils_SYCL.md](../../docs/development/reference/source/GPUUtils_SYCL.md) |
| `sycl/MatrixGPU_SYCL.hpp` | SYCL kernel declarations and `GPUMatrix` | — |
| `sycl/MatrixGPU_SYCL.cpp` | SYCL kernels (oneMKL for GEMM) | — |

## How it's used

The GPU is enabled at process startup by `Matrix::gpu_try_initialize()`, called by
`incremental_trainer` and `chatbot_api_server` when `GPU_ENABLED=true`, using `GPU_DEVICE_ID`,
`GPU_MEMORY_FRACTION` and (trainer only, TD-381) `GPU_STRATEGY`. Once initialized:

- `Matrix::operator*` sends any product whose inner dimensions are both at least 32 to the GPU
  (`multiply_gpu()`), from any thread (see TD-382).
- Each layer has a persistent GPU path (`gpu_upload_weights()`, `gpu_forward()`,
  `gpu_backward()`, …) that works in `GPUMatrix` and keeps weights and activations on the device.
- `EncoderDecoderModel` uses those paths for GPU training (`gpu_forward()`/`gpu_backward()`) and for
  GPU generation (`gpu_generate_response*()`, with `GPUDecoderKVCache`).

```cpp
#include "Matrix.hpp"

if (Matrix::gpu_try_initialize(/*device_id=*/0, /*memory_fraction=*/0.5f)) {
    auto A_gpu = A.to_gpu();              // one upload
    auto C_gpu = A_gpu * B.to_gpu();      // on-device GEMM
    Matrix C   = Matrix::from_gpu(C_gpu); // one download
}
Matrix::gpu_cleanup();
```

`examples/GPUExample.cpp` (target `gpu_example`) compares CPU and GPU timings for multiply, add and
Hadamard products, and reports status when no GPU is present.

## Building

GPU support is off by default, and only one backend can be enabled per build.

```bash
# NVIDIA (CUDA). The preset builds only incremental_trainer.
cmake --preset gpu && cmake --build --preset gpu

# Intel (SYCL / oneAPI). The preset builds only incremental_trainer.
source /opt/intel/oneapi/setvars.sh
cmake --preset sycl && cmake --build --preset sycl
```

To build other targets (for example `chatbot_api_server` or the tests) with a GPU backend, configure
manually with `-DENABLE_GPU=ON` or `-DENABLE_SYCL=ON -DCMAKE_CXX_COMPILER=icpx`. See
[building.md](../../docs/development/guides/building/building.md).

## Requirements

**CUDA:** CUDA Toolkit **11.2 or later** (the batch helpers use `cudaMallocAsync`), an NVIDIA GPU of
compute capability 6.0+ (the default `CMAKE_CUDA_ARCHITECTURES` is 60–86), and matching drivers.
`libcudart` is linked statically and `libcublas` dynamically.

**SYCL:** the Intel oneAPI Base Toolkit (`icpx`, oneMKL), an Intel GPU with the compute runtime
(Level Zero / OpenCL), and access to `/dev/dri/renderD*`. The SYCL `GPUManager` uses GPU devices only;
CPU SYCL devices are ignored.

## Operations

- GEMM (cuBLAS / oneMKL), transpose
- Element-wise add, subtract, Hadamard product, scalar multiply
- Activations (ReLU, sigmoid, tanh, GELU) and GELU backward
- Row softmax and its backward, LayerNorm forward/backward, row-bias add, row sums, masked fill
- Cross-entropy loss and gradient
- Sum and diagnostics reductions (activation saturation, attention entropy)
- GPU KV caches for incremental decoding

`matrix_batch_add_gpu`/`matrix_batch_multiply_gpu` exist but have no callers (TD-386).

## Known limitations

- **No code in this module has run on real GPU hardware in the development environment.** The
  CUDA and SYCL builds compile, and GPU-gated tests skip without a device. ai-machine runs the SYCL
  backend in production training.
- Open items are tracked in [TECHNICAL_DEBT.md](../../docs/development/guides/TECHNICAL_DEBT.md),
  notably TD-384 (a shared-memory race in the softmax, LayerNorm and cross-entropy kernels, in both
  backends), TD-358 (GPU training feeds unshifted targets), TD-382 (no locking across threads) and
  TD-387 (over-length input fails GPU generation).
