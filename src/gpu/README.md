# GPU Acceleration Module

This directory contains GPU-accelerated matrix operations with two mutually
exclusive backends: **CUDA** (NVIDIA) and **SYCL** (Intel oneAPI).

## Files

### Gateway headers (backend-agnostic)

- **GPUUtils.hpp** - Routes to CUDA or SYCL implementation based on build config
- **MatrixGPU.hpp** - Routes to CUDA or SYCL implementation based on build config

### CUDA backend

- **MatrixGPU.cu** - CUDA kernel implementations

### Intel SYCL backend (`sycl/`)

- **sycl/GPUUtils_SYCL.hpp** - SYCL GPUManager and GPUMemory (USM-based)
- **sycl/MatrixGPU_SYCL.hpp** - SYCL GPU matrix operation declarations and GPUMatrix class
- **sycl/MatrixGPU_SYCL.cpp** - SYCL kernel implementations (oneMKL for GEMM)

## Building

GPU support is **optional** and disabled by default. Only one backend can be
enabled per build.

### NVIDIA GPU (CUDA)

```bash
cmake --preset gpu          # or: cmake -DENABLE_GPU=ON ..
cmake --build --preset gpu
```

### Intel GPU (SYCL / oneAPI)

```bash
source /opt/intel/oneapi/setvars.sh
cmake --preset sycl         # or: cmake -DENABLE_SYCL=ON -DCMAKE_CXX_COMPILER=icpx ..
cmake --build --preset sycl
```

### Disable GPU Support (default)

```bash
cmake -DENABLE_GPU=OFF -DENABLE_SYCL=OFF ..
make
```

## Requirements

### CUDA backend requirements

- CUDA Toolkit 11.0 or later
- NVIDIA GPU with compute capability 6.0+
- Compatible NVIDIA drivers

### SYCL backend requirements

- Intel oneAPI Base Toolkit (icpx compiler, oneMKL, Level Zero runtime)
- Intel ARC or Data Center GPU with Xe architecture
- Intel compute runtime (`intel-opencl-icd`, `level-zero-gpu`)

## Usage

See `../GPUExample.cpp` for complete usage examples.

### Basic Usage

```cpp
#include "Matrix.hpp"

// Initialize GPU
Matrix::gpu_initialize();

// Use GPU operations
Matrix A(1000, 1000);
Matrix B(1000, 1000);
Matrix C = A.multiply_gpu(B);

// Cleanup
Matrix::gpu_cleanup();
```

## Supported Operations

- Matrix multiplication (cuBLAS-optimized)
- Element-wise addition
- Element-wise multiplication (Hadamard)
- Scalar operations
- Matrix transpose
- Activation functions (ReLU, Sigmoid, Tanh, GELU)
- Reduction operations (sum)
- Batch processing

## Performance

GPU operations are fastest on:

- Large matrices (>500x500)
- Batch operations
- Repeated computations

For small matrices (<100x100), CPU may be faster due to transfer overhead.

## Documentation

See `/docs/guides/building.md` for complete build and usage instructions.
