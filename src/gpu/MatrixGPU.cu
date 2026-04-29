#ifdef ADAI_ENABLE_GPU

#include "MatrixGPU.hpp"
#include "GPUUtils.hpp"
#include <cuda_runtime.h>
#include <cublas_v2.h>

namespace adai {
namespace gpu {

// ============================================================================
// CUDA Kernels
// ============================================================================

/**
 * @brief Element-wise addition kernel
 */
__global__ void add_kernel(const float* a, const float* b, float* c, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        c[idx] = a[idx] + b[idx];
    }
}

/**
 * @brief Element-wise scalar addition kernel
 */
__global__ void add_scalar_kernel(const float* a, float scalar, float* c, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        c[idx] = a[idx] + scalar;
    }
}

/**
 * @brief Element-wise multiplication kernel
 */
__global__ void multiply_kernel(const float* a, const float* b, float* c, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        c[idx] = a[idx] * b[idx];
    }
}

/**
 * @brief Element-wise scalar multiplication kernel
 */
__global__ void multiply_scalar_kernel(const float* a, float scalar, float* c, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        c[idx] = a[idx] * scalar;
    }
}

/**
 * @brief Transpose kernel (optimized with shared memory)
 */
__global__ void transpose_kernel(const float* input, float* output, int rows, int cols) {
    __shared__ float tile[32][33];  // +1 to avoid bank conflicts
    
    int x = blockIdx.x * 32 + threadIdx.x;
    int y = blockIdx.y * 32 + threadIdx.y;
    
    // Load data into shared memory
    if (x < cols && y < rows) {
        tile[threadIdx.y][threadIdx.x] = input[y * cols + x];
    }
    
    __syncthreads();
    
    // Write transposed data to global memory
    x = blockIdx.y * 32 + threadIdx.x;
    y = blockIdx.x * 32 + threadIdx.y;
    
    if (x < rows && y < cols) {
        output[y * rows + x] = tile[threadIdx.x][threadIdx.y];
    }
}

/**
 * @brief Apply activation function kernel (supports multiple types)
 */
__device__ float apply_activation(float x, int activation_type) {
    switch (activation_type) {
        case 0: // ReLU
            return fmaxf(0.0f, x);
        case 1: // Sigmoid
            return 1.0f / (1.0f + expf(-x));
        case 2: // Tanh
            return tanhf(x);
        case 3: // GELU (approximation)
            return 0.5f * x * (1.0f + tanhf(0.7978845608f * (x + 0.044715f * x * x * x)));
        default:
            return x;
    }
}

__global__ void activation_kernel(float* data, int size, int activation_type) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        data[idx] = apply_activation(data[idx], activation_type);
    }
}

/**
 * @brief Sum reduction kernel
 */
__global__ void sum_kernel(const float* input, float* output, int size) {
    extern __shared__ float sdata[];
    
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    sdata[tid] = (i < size) ? input[i] : 0.0f;
    __syncthreads();
    
    // Reduction in shared memory
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    
    if (tid == 0) output[blockIdx.x] = sdata[0];
}

// ============================================================================
// MatrixGPU Implementation
// ============================================================================

void matrix_add_gpu(const float* a, const float* b, float* c, int size) {
    const int threads = 256;
    const int blocks = (size + threads - 1) / threads;
    add_kernel<<<blocks, threads, 0, GPUManager::get_stream()>>>(a, b, c, size);
    CUDA_CHECK(cudaGetLastError());
}

void matrix_add_scalar_gpu(const float* a, float scalar, float* c, int size) {
    const int threads = 256;
    const int blocks = (size + threads - 1) / threads;
    add_scalar_kernel<<<blocks, threads, 0, GPUManager::get_stream()>>>(a, scalar, c, size);
    CUDA_CHECK(cudaGetLastError());
}

void matrix_multiply_elementwise_gpu(const float* a, const float* b, float* c, int size) {
    const int threads = 256;
    const int blocks = (size + threads - 1) / threads;
    multiply_kernel<<<blocks, threads, 0, GPUManager::get_stream()>>>(a, b, c, size);
    CUDA_CHECK(cudaGetLastError());
}

void matrix_multiply_scalar_gpu(const float* a, float scalar, float* c, int size) {
    const int threads = 256;
    const int blocks = (size + threads - 1) / threads;
    multiply_scalar_kernel<<<blocks, threads, 0, GPUManager::get_stream()>>>(a, scalar, c, size);
    CUDA_CHECK(cudaGetLastError());
}

void matrix_transpose_gpu(const float* input, float* output, int rows, int cols) {
    dim3 threads(32, 32);
    dim3 blocks((cols + 31) / 32, (rows + 31) / 32);
    transpose_kernel<<<blocks, threads, 0, GPUManager::get_stream()>>>(input, output, rows, cols);
    CUDA_CHECK(cudaGetLastError());
}

void matrix_multiply_gpu(const float* a, const float* b, float* c,
                        int m, int k, int n) {
    // cuBLAS handle is already bound to the low-priority ADAI stream.
    cublasHandle_t handle = GPUManager::get_cublas_handle();

    const float alpha = 1.0f;
    const float beta  = 0.0f;

    // cuBLAS uses column-major order, so we compute: C = B * A
    CUBLAS_CHECK(cublasSgemm(handle,
                             CUBLAS_OP_N, CUBLAS_OP_N,
                             n, m, k,
                             &alpha,
                             b, n,
                             a, k,
                             &beta,
                             c, n));
}

void matrix_apply_activation_gpu(float* data, int size, ActivationType type) {
    const int threads = 256;
    const int blocks = (size + threads - 1) / threads;
    activation_kernel<<<blocks, threads, 0, GPUManager::get_stream()>>>(
        data, size, static_cast<int>(type));
    CUDA_CHECK(cudaGetLastError());
}

float matrix_sum_gpu(const float* data, int size) {
    const int threads = 256;
    const int blocks = (size + threads - 1) / threads;

    GPUMemory<float> block_sums(blocks);

    sum_kernel<<<blocks, threads, threads * sizeof(float), GPUManager::get_stream()>>>(
        data, block_sums.get(), size);
    CUDA_CHECK(cudaGetLastError());

    if (blocks == 1) {
        float result;
        block_sums.copy_to_host(&result, 1);
        return result;
    }

    // Recursive reduction for multiple blocks
    return matrix_sum_gpu(block_sums.get(), blocks);
}

// ============================================================================
// Batch Operations
// ============================================================================

void matrix_batch_add_gpu(const float** a_batch, const float** b_batch,
                         float** c_batch, int batch_size, int size) {
    for (int i = 0; i < batch_size; ++i) {
        matrix_add_gpu(a_batch[i], b_batch[i], c_batch[i], size);
    }
}

void matrix_batch_multiply_gpu(const float** a_batch, const float** b_batch,
                              float** c_batch, int batch_size,
                              int m, int k, int n) {
    // cuBLAS handle is bound to the ADAI stream via cublasSetStream in GPUManager.
    cublasHandle_t handle = GPUManager::get_cublas_handle();

    const float alpha = 1.0f;
    const float beta  = 0.0f;

    for (int i = 0; i < batch_size; ++i) {
        CUBLAS_CHECK(cublasSgemm(handle,
                                 CUBLAS_OP_N, CUBLAS_OP_N,
                                 n, m, k,
                                 &alpha,
                                 b_batch[i], n,
                                 a_batch[i], k,
                                 &beta,
                                 c_batch[i], n));
    }
}

} // namespace gpu
} // namespace adai

#endif // ADAI_ENABLE_GPU
