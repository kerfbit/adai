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
 * @brief float4 vectorized element-wise addition kernel.
 *
 * Each thread loads/stores 4 floats in a single 128-bit transaction,
 * giving ~4x the effective memory bandwidth vs. the scalar kernel.
 * @p n4 is the number of float4 elements (i.e. size / 4).
 */
__global__ void add_kernel_v4(const float4* a, const float4* b, float4* c, int n4) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n4) {
        float4 va = a[idx];
        float4 vb = b[idx];
        c[idx] = {va.x + vb.x, va.y + vb.y, va.z + vb.z, va.w + vb.w};
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
 * @brief float4 vectorized scalar addition kernel.
 */
__global__ void add_scalar_kernel_v4(const float4* a, float scalar, float4* c, int n4) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n4) {
        float4 va = a[idx];
        c[idx] = {va.x + scalar, va.y + scalar, va.z + scalar, va.w + scalar};
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
 * @brief float4 vectorized element-wise multiplication kernel.
 */
__global__ void multiply_kernel_v4(const float4* a, const float4* b, float4* c, int n4) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n4) {
        float4 va = a[idx];
        float4 vb = b[idx];
        c[idx] = {va.x * vb.x, va.y * vb.y, va.z * vb.z, va.w * vb.w};
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
 * @brief float4 vectorized scalar multiplication kernel.
 */
__global__ void multiply_scalar_kernel_v4(const float4* a, float scalar, float4* c, int n4) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n4) {
        float4 va = a[idx];
        c[idx] = {va.x * scalar, va.y * scalar, va.z * scalar, va.w * scalar};
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
 * @brief float4 vectorized in-place activation kernel.
 *
 * Applies the activation function to 4 floats per thread using a single
 * 128-bit load and store, roughly quadrupling memory throughput vs. the
 * scalar kernel for memory-bound activation sizes.
 */
__global__ void activation_kernel_v4(float4* data, int n4, int activation_type) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n4) {
        float4 v = data[idx];
        v.x = apply_activation(v.x, activation_type);
        v.y = apply_activation(v.y, activation_type);
        v.z = apply_activation(v.z, activation_type);
        v.w = apply_activation(v.w, activation_type);
        data[idx] = v;
    }
}

/**
 * @brief Intra-warp reduction using shuffle-down instructions.
 *
 * Sums @p val across all 32 threads in the calling warp and returns the
 * result in lane 0 (other lanes hold an undefined partial sum).
 * No shared memory or __syncthreads() required — the warp executes in
 * lock-step, so __shfl_down_sync() is both sufficient and faster.
 */
__device__ __forceinline__ float warp_reduce_sum(float val) {
    // Full warp mask — all 32 lanes participate.
    constexpr unsigned FULL_MASK = 0xFFFFFFFFu;
    val += __shfl_down_sync(FULL_MASK, val, 16);
    val += __shfl_down_sync(FULL_MASK, val, 8);
    val += __shfl_down_sync(FULL_MASK, val, 4);
    val += __shfl_down_sync(FULL_MASK, val, 2);
    val += __shfl_down_sync(FULL_MASK, val, 1);
    return val;
}

/**
 * @brief Sum reduction kernel using warp shuffle intrinsics (TD-003 warp opt).
 *
 * Algorithm for a 256-thread block (8 warps):
 *  1. Each thread loads one element (0.0 for out-of-bounds threads).
 *  2. warp_reduce_sum() reduces all 32 lanes within each warp using
 *     __shfl_down_sync — no shared memory, no __syncthreads().
 *  3. Lane 0 of each warp writes its warp-sum to shared memory
 *     (only 8 floats vs. the previous 256).
 *  4. A single __syncthreads() ensures all warp results are visible.
 *  5. The first warp reduces the 8 warp-sums with another
 *     warp_reduce_sum() call — again without any barrier.
 *  6. Thread 0 writes the block sum to global memory.
 *
 * Compared to the previous tree-reduction this eliminates 7 out of 8
 * __syncthreads() barriers and halves shared-memory traffic.
 *
 * @note The caller must allocate (blockDim.x / 32) * sizeof(float) bytes
 *       of dynamic shared memory (down from blockDim.x * sizeof(float)).
 */
__global__ void sum_kernel(const float* input, float* output, int size) {
    // One slot per warp in the block.
    extern __shared__ float warp_sums[];

    const unsigned int tid  = threadIdx.x;
    const unsigned int i    = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int lane = tid & 31u;          // lane within warp (0-31)
    const unsigned int wid  = tid >> 5u;          // warp index within block

    // Step 1: load element (pad with 0 for threads beyond array end).
    float val = (i < static_cast<unsigned int>(size)) ? input[i] : 0.0f;

    // Step 2: reduce within the warp — no shared memory, no barrier.
    val = warp_reduce_sum(val);

    // Step 3: lane 0 of each warp stores its partial sum.
    if (lane == 0) {
        warp_sums[wid] = val;
    }

    // Step 4: ONE barrier to make all warp results visible.
    __syncthreads();

    // Step 5: first warp reduces the (blockDim.x/32) warp sums.
    //         Threads beyond that count contribute 0.
    const unsigned int warps_per_block = blockDim.x >> 5u;
    val = (tid < warps_per_block) ? warp_sums[tid] : 0.0f;
    if (wid == 0) {
        val = warp_reduce_sum(val);
    }

    // Step 6: thread 0 writes the block result.
    if (tid == 0) {
        output[blockIdx.x] = val;
    }
}

// ============================================================================
// MatrixGPU Implementation
// ============================================================================

void matrix_add_gpu(const float* a, const float* b, float* c, int size) {
    const int threads = 256;
    const int n4   = size / 4;
    const int tail = size % 4;
    cudaStream_t s = GPUManager::get_stream();
    if (n4 > 0) {
        const int blocks4 = (n4 + threads - 1) / threads;
        add_kernel_v4<<<blocks4, threads, 0, s>>>(
            reinterpret_cast<const float4*>(a),
            reinterpret_cast<const float4*>(b),
            reinterpret_cast<float4*>(c), n4);
        CUDA_CHECK(cudaGetLastError());
    }
    if (tail > 0) {
        const int offset = n4 * 4;
        add_kernel<<<1, tail, 0, s>>>(a + offset, b + offset, c + offset, tail);
        CUDA_CHECK(cudaGetLastError());
    }
}

void matrix_add_scalar_gpu(const float* a, float scalar, float* c, int size) {
    const int threads = 256;
    const int n4   = size / 4;
    const int tail = size % 4;
    cudaStream_t s = GPUManager::get_stream();
    if (n4 > 0) {
        const int blocks4 = (n4 + threads - 1) / threads;
        add_scalar_kernel_v4<<<blocks4, threads, 0, s>>>(
            reinterpret_cast<const float4*>(a), scalar,
            reinterpret_cast<float4*>(c), n4);
        CUDA_CHECK(cudaGetLastError());
    }
    if (tail > 0) {
        const int offset = n4 * 4;
        add_scalar_kernel<<<1, tail, 0, s>>>(a + offset, scalar, c + offset, tail);
        CUDA_CHECK(cudaGetLastError());
    }
}

void matrix_multiply_elementwise_gpu(const float* a, const float* b, float* c, int size) {
    const int threads = 256;
    const int n4   = size / 4;
    const int tail = size % 4;
    cudaStream_t s = GPUManager::get_stream();
    if (n4 > 0) {
        const int blocks4 = (n4 + threads - 1) / threads;
        multiply_kernel_v4<<<blocks4, threads, 0, s>>>(
            reinterpret_cast<const float4*>(a),
            reinterpret_cast<const float4*>(b),
            reinterpret_cast<float4*>(c), n4);
        CUDA_CHECK(cudaGetLastError());
    }
    if (tail > 0) {
        const int offset = n4 * 4;
        multiply_kernel<<<1, tail, 0, s>>>(a + offset, b + offset, c + offset, tail);
        CUDA_CHECK(cudaGetLastError());
    }
}

void matrix_multiply_scalar_gpu(const float* a, float scalar, float* c, int size) {
    const int threads = 256;
    const int n4   = size / 4;
    const int tail = size % 4;
    cudaStream_t s = GPUManager::get_stream();
    if (n4 > 0) {
        const int blocks4 = (n4 + threads - 1) / threads;
        multiply_scalar_kernel_v4<<<blocks4, threads, 0, s>>>(
            reinterpret_cast<const float4*>(a), scalar,
            reinterpret_cast<float4*>(c), n4);
        CUDA_CHECK(cudaGetLastError());
    }
    if (tail > 0) {
        const int offset = n4 * 4;
        multiply_scalar_kernel<<<1, tail, 0, s>>>(a + offset, scalar, c + offset, tail);
        CUDA_CHECK(cudaGetLastError());
    }
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
    const int n4   = size / 4;
    const int tail = size % 4;
    const int act  = static_cast<int>(type);
    cudaStream_t s = GPUManager::get_stream();
    if (n4 > 0) {
        const int blocks4 = (n4 + threads - 1) / threads;
        activation_kernel_v4<<<blocks4, threads, 0, s>>>(
            reinterpret_cast<float4*>(data), n4, act);
        CUDA_CHECK(cudaGetLastError());
    }
    if (tail > 0) {
        const int offset = n4 * 4;
        activation_kernel<<<1, tail, 0, s>>>(data + offset, tail, act);
        CUDA_CHECK(cudaGetLastError());
    }
}

float matrix_sum_gpu(const float* data, int size) {
    const int threads = 256;
    const int blocks = (size + threads - 1) / threads;

    GPUMemory<float> block_sums(blocks);

    // Shared memory: one float per warp (threads/32), not one per thread.
    const int smem_bytes = (threads / 32) * static_cast<int>(sizeof(float));
    sum_kernel<<<blocks, threads, smem_bytes, GPUManager::get_stream()>>>(
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
// Batch Kernels
// ============================================================================

/**
 * @brief Fused float4 vectorized batch element-wise add.
 *
 * Grid layout: gridDim.y = batch index, gridDim.x * blockDim.x = float4 index.
 * Each thread performs one 128-bit load from A and B, adds four floats, and
 * writes one 128-bit store to C — eliminating per-batch kernel launch overhead
 * and doubling effective memory bandwidth vs. serialised scalar launches.
 *
 * @p n4  Number of float4 elements per matrix (i.e. size / 4).
 */
__global__ void batch_add_kernel_v4(const float** a_batch, const float** b_batch,
                                     float** c_batch, int n4) {
    const int b   = blockIdx.y;
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n4) {
        const float4* a4 = reinterpret_cast<const float4*>(a_batch[b]);
        const float4* b4 = reinterpret_cast<const float4*>(b_batch[b]);
        float4*       c4 = reinterpret_cast<float4*>(c_batch[b]);
        float4 va = a4[idx], vb = b4[idx];
        c4[idx] = {va.x + vb.x, va.y + vb.y, va.z + vb.z, va.w + vb.w};
    }
}

/**
 * @brief Scalar tail kernel for the size%4 remainder elements.
 *
 * Launched as a single block (1 × batch_size grid, tail threads per block).
 * @p offset  Byte offset into each matrix where the tail begins (= n4 * 4).
 */
__global__ void batch_add_tail_kernel(const float** a_batch, const float** b_batch,
                                       float** c_batch, int offset, int tail) {
    const int b   = blockIdx.y;
    const int idx = threadIdx.x;
    if (idx < tail) {
        c_batch[b][offset + idx] = a_batch[b][offset + idx] + b_batch[b][offset + idx];
    }
}

// ============================================================================
// Batch Operations
// ============================================================================

void matrix_batch_add_gpu(const float** a_batch, const float** b_batch,
                         float** c_batch, int batch_size, int size) {
    cudaStream_t stream = GPUManager::get_stream();
    const std::size_t ptr_bytes = static_cast<std::size_t>(batch_size) * sizeof(float*);

    // Stage the host-side pointer arrays (already device pointers) onto the
    // device so the kernel can index into them via blockIdx.y.
    // cudaMallocAsync / cudaFreeAsync are stream-ordered (CUDA 11.2+): the
    // alloc is live before the memcpy executes and the free runs only after
    // the last kernel consuming the arrays has finished on the device.
    const float** d_a = nullptr;
    const float** d_b = nullptr;
    float**        d_c = nullptr;

    CUDA_CHECK(cudaMallocAsync(reinterpret_cast<void**>(&d_a), ptr_bytes, stream));
    CUDA_CHECK(cudaMallocAsync(reinterpret_cast<void**>(&d_b), ptr_bytes, stream));
    CUDA_CHECK(cudaMallocAsync(reinterpret_cast<void**>(&d_c), ptr_bytes, stream));

    CUDA_CHECK(cudaMemcpyAsync(d_a, a_batch, ptr_bytes, cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(d_b, b_batch, ptr_bytes, cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(d_c, c_batch, ptr_bytes, cudaMemcpyHostToDevice, stream));

    const int threads = 256;
    const int n4   = size / 4;
    const int tail = size % 4;

    if (n4 > 0) {
        const int blocks_x = (n4 + threads - 1) / threads;
        const dim3 grid(blocks_x, batch_size);
        batch_add_kernel_v4<<<grid, threads, 0, stream>>>(d_a, d_b, d_c, n4);
        CUDA_CHECK(cudaGetLastError());
    }
    if (tail > 0) {
        const int offset = n4 * 4;
        const dim3 grid(1, batch_size);
        batch_add_tail_kernel<<<grid, tail, 0, stream>>>(d_a, d_b, d_c, offset, tail);
        CUDA_CHECK(cudaGetLastError());
    }

    CUDA_CHECK(cudaFreeAsync(d_a, stream));
    CUDA_CHECK(cudaFreeAsync(d_b, stream));
    CUDA_CHECK(cudaFreeAsync(d_c, stream));
}

void matrix_batch_multiply_gpu(const float** a_batch, const float** b_batch,
                              float** c_batch, int batch_size,
                              int m, int k, int n) {
    cublasHandle_t handle = GPUManager::get_cublas_handle();
    cudaStream_t   stream = GPUManager::get_stream();

    const float alpha = 1.0f;
    const float beta  = 0.0f;

    // cublasSgemmBatched requires device-side arrays of device pointers.
    // The caller supplies host-side arrays (a_batch, b_batch, c_batch) whose
    // elements are already valid device pointers — we only need to stage the
    // pointer arrays themselves onto the device.
    // cudaMallocAsync / cudaFreeAsync are stream-ordered (CUDA 11.2+): the
    // allocation is guaranteed to be live before the subsequent memcpy, and the
    // free executes only after cublasSgemmBatched completes on the device,
    // so no explicit host–device synchronisation is required.
    const std::size_t ptr_bytes = static_cast<std::size_t>(batch_size) * sizeof(float*);
    const float** d_a = nullptr;
    const float** d_b = nullptr;
    float**        d_c = nullptr;

    CUDA_CHECK(cudaMallocAsync(reinterpret_cast<void**>(&d_a), ptr_bytes, stream));
    CUDA_CHECK(cudaMallocAsync(reinterpret_cast<void**>(&d_b), ptr_bytes, stream));
    CUDA_CHECK(cudaMallocAsync(reinterpret_cast<void**>(&d_c), ptr_bytes, stream));

    CUDA_CHECK(cudaMemcpyAsync(d_a, a_batch, ptr_bytes, cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(d_b, b_batch, ptr_bytes, cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(d_c, c_batch, ptr_bytes, cudaMemcpyHostToDevice, stream));

    // cuBLAS uses column-major storage; our matrices are row-major.
    // By swapping A and B (computing B×A in column-major == A×B in row-major)
    // we obtain the correct result without any explicit transpose.
    CUBLAS_CHECK(cublasSgemmBatched(handle,
                                     CUBLAS_OP_N, CUBLAS_OP_N,
                                     n, m, k,
                                     &alpha,
                                     d_b, n,
                                     d_a, k,
                                     &beta,
                                     d_c, n,
                                     batch_size));

    // Stream-ordered frees: these execute after cublasSgemmBatched finishes
    // on the device, so the pointer arrays remain valid for the full GEMM.
    CUDA_CHECK(cudaFreeAsync(d_a, stream));
    CUDA_CHECK(cudaFreeAsync(d_b, stream));
    CUDA_CHECK(cudaFreeAsync(d_c, stream));
}

} // namespace gpu
} // namespace adai

#endif // ADAI_ENABLE_GPU
