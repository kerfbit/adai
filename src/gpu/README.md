# GPU Acceleration Module

This directory contains CUDA implementations for GPU-accelerated matrix operations.

## Files

- **GPUUtils.hpp** - GPU device management, memory utilities, and error handling
- **MatrixGPU.hpp** - GPU matrix operation declarations
- **MatrixGPU.cu** - CUDA kernel implementations

## Building

GPU support is **optional** and disabled by default.

### Enable GPU Support
```bash
cmake -DENABLE_GPU=ON ..
make
```

### Disable GPU Support (default)
```bash
cmake -DENABLE_GPU=OFF ..
make
```

## Requirements

- CUDA Toolkit 11.0 or later
- NVIDIA GPU with compute capability 6.0+
- Compatible NVIDIA drivers

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
