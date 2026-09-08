// @adai-status: beta        (capped by TD-061 — layer_norm_bwd had an undetected math bug; most kernels have no dedicated test and need real GPU hardware to verify)
// @adai-version: 0.9.0
// @adai-reviewed: 2026-09-08

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

// ============================================================================
// TD-003 training kernels — ported from src/gpu/sycl/MatrixGPU_SYCL.cpp.
// Each kernel below is a direct translation of its SYCL counterpart (same
// one-block-per-row / shared-memory tree-reduction shape, __shared__ +
// __syncthreads() in place of sycl::local_accessor + item.barrier()), kept
// deliberately unoptimized (no warp-shuffle) to minimize risk in code that
// cannot be executed/tested on this machine. Numerics match the SYCL version
// formula-for-formula.
// ============================================================================

__global__ void add_inplace_kernel(const float* src, float* dst, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        dst[idx] += src[idx];
    }
}

void matrix_add_inplace_gpu(const float* src, float* dst, int size) {
    const int threads = 256;
    const int blocks = (size + threads - 1) / threads;
    add_inplace_kernel<<<blocks, threads, 0, GPUManager::get_stream()>>>(src, dst, size);
    CUDA_CHECK(cudaGetLastError());
}

__global__ void softmax_rows_kernel(float* data, int cols) {
    extern __shared__ float scratch[];
    const int row = blockIdx.x;
    const int lid = static_cast<int>(threadIdx.x);
    const int WG = static_cast<int>(blockDim.x);
    float* row_ptr = data + row * cols;

    // Pass 1: row max
    float my_max = -1e30f;
    for (int j = lid; j < cols; j += WG) my_max = fmaxf(my_max, row_ptr[j]);
    scratch[lid] = my_max;
    __syncthreads();
    for (int s = WG / 2; s > 0; s >>= 1) {
        if (lid < s) scratch[lid] = fmaxf(scratch[lid], scratch[lid + s]);
        __syncthreads();
    }
    const float row_max = scratch[0];

    // Pass 2: exp(x - max) and sum
    float my_sum = 0.0f;
    for (int j = lid; j < cols; j += WG) {
        float e = expf(row_ptr[j] - row_max);
        row_ptr[j] = e;
        my_sum += e;
    }
    scratch[lid] = my_sum;
    __syncthreads();
    for (int s = WG / 2; s > 0; s >>= 1) {
        if (lid < s) scratch[lid] += scratch[lid + s];
        __syncthreads();
    }
    const float row_sum = scratch[0];

    // Pass 3: normalize
    for (int j = lid; j < cols; j += WG) row_ptr[j] /= row_sum;
}

void matrix_softmax_rows_gpu(float* data, int rows, int cols) {
    constexpr int WG = 256;
    const int smem_bytes = WG * static_cast<int>(sizeof(float));
    softmax_rows_kernel<<<rows, WG, smem_bytes, GPUManager::get_stream()>>>(data, cols);
    CUDA_CHECK(cudaGetLastError());
}

__global__ void softmax_backward_kernel(const float* s, const float* dout, float* din, int cols) {
    extern __shared__ float scratch[];
    const int row = blockIdx.x;
    const int lid = static_cast<int>(threadIdx.x);
    const int WG = static_cast<int>(blockDim.x);
    const float* s_row = s + row * cols;
    const float* dout_row = dout + row * cols;
    float* din_row = din + row * cols;

    float my_dot = 0.0f;
    for (int j = lid; j < cols; j += WG) my_dot += s_row[j] * dout_row[j];
    scratch[lid] = my_dot;
    __syncthreads();
    for (int st = WG / 2; st > 0; st >>= 1) {
        if (lid < st) scratch[lid] += scratch[lid + st];
        __syncthreads();
    }
    const float dot_val = scratch[0];

    for (int j = lid; j < cols; j += WG) din_row[j] = s_row[j] * (dout_row[j] - dot_val);
}

void matrix_softmax_backward_gpu(const float* s, const float* dout, float* din,
                                  int rows, int cols) {
    constexpr int WG = 256;
    const int smem_bytes = WG * static_cast<int>(sizeof(float));
    softmax_backward_kernel<<<rows, WG, smem_bytes, GPUManager::get_stream()>>>(s, dout, din, cols);
    CUDA_CHECK(cudaGetLastError());
}

// GELU'(x) = 0.5*(1+tanh(u)) + 0.5*x*sech^2(u)*c*(1+3*0.044715*x^2)
__global__ void gelu_backward_kernel(const float* pre_act, const float* dout, float* din, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        const float x = pre_act[idx];
        constexpr float c = 0.7978845608f;
        constexpr float a = 0.044715f;
        const float u = c * (x + a * x * x * x);
        const float tanh_u = tanhf(u);
        const float sech2 = 1.0f - tanh_u * tanh_u;
        const float gelu_prime =
            0.5f * (1.0f + tanh_u) + 0.5f * x * sech2 * c * (1.0f + 3.0f * a * x * x);
        din[idx] = dout[idx] * gelu_prime;
    }
}

void matrix_gelu_backward_gpu(const float* pre_act, const float* dout, float* din, int size) {
    const int threads = 256;
    const int blocks = (size + threads - 1) / threads;
    gelu_backward_kernel<<<blocks, threads, 0, GPUManager::get_stream()>>>(pre_act, dout, din, size);
    CUDA_CHECK(cudaGetLastError());
}

__global__ void layer_norm_fwd_kernel(const float* input, float* output, float* out_normed,
                                       const float* gamma, const float* beta,
                                       float* out_mean, float* out_rstd,
                                       int cols, float eps) {
    extern __shared__ float scratch[];
    const int row = blockIdx.x;
    const int lid = static_cast<int>(threadIdx.x);
    const int WG = static_cast<int>(blockDim.x);
    const float inv_cols = 1.0f / static_cast<float>(cols);
    const float* x = input + row * cols;
    float* y = output + row * cols;
    float* xn = out_normed + row * cols;

    // Mean
    float my_sum = 0.0f;
    for (int j = lid; j < cols; j += WG) my_sum += x[j];
    scratch[lid] = my_sum;
    __syncthreads();
    for (int s = WG / 2; s > 0; s >>= 1) {
        if (lid < s) scratch[lid] += scratch[lid + s];
        __syncthreads();
    }
    const float mean = scratch[0] * inv_cols;
    if (lid == 0 && out_mean) out_mean[row] = mean;

    // Variance
    float my_var = 0.0f;
    for (int j = lid; j < cols; j += WG) {
        float d = x[j] - mean;
        my_var += d * d;
    }
    scratch[lid] = my_var;
    __syncthreads();
    for (int s = WG / 2; s > 0; s >>= 1) {
        if (lid < s) scratch[lid] += scratch[lid + s];
        __syncthreads();
    }
    const float rstd = 1.0f / sqrtf(scratch[0] * inv_cols + eps);
    if (lid == 0 && out_rstd) out_rstd[row] = rstd;

    // Normalize and write both normed (pre-affine) and output (post-affine)
    for (int j = lid; j < cols; j += WG) {
        const float n = (x[j] - mean) * rstd;
        xn[j] = n;
        y[j] = n * gamma[j] + beta[j];
    }
}

void matrix_layer_norm_fwd_gpu(const float* input, float* output, float* out_normed,
                                const float* gamma, const float* beta,
                                float* out_mean, float* out_rstd,
                                int rows, int cols, float eps) {
    constexpr int WG = 256;
    const int smem_bytes = WG * static_cast<int>(sizeof(float));
    layer_norm_fwd_kernel<<<rows, WG, smem_bytes, GPUManager::get_stream()>>>(
        input, output, out_normed, gamma, beta, out_mean, out_rstd, cols, eps);
    CUDA_CHECK(cudaGetLastError());
}

__global__ void layer_norm_bwd_dx_kernel(const float* dout, const float* input_norm,
                                          const float* gamma, const float* rstd,
                                          float* dx, int cols) {
    extern __shared__ float smem[];
    float* sc_a = smem;
    float* sc_b = smem + blockDim.x;
    const int row = blockIdx.x;
    const int lid = static_cast<int>(threadIdx.x);
    const int WG = static_cast<int>(blockDim.x);
    const float inv_N = 1.0f / static_cast<float>(cols);
    const float* dout_row = dout + row * cols;
    const float* xn_row = input_norm + row * cols;
    float* dx_row = dx + row * cols;
    const float r = rstd[row];

    // sum(d_xn * xn) and sum(d_xn)
    float my_a = 0.0f, my_b = 0.0f;
    for (int j = lid; j < cols; j += WG) {
        const float d_xn = dout_row[j] * gamma[j];
        my_a += d_xn * xn_row[j];
        my_b += d_xn;
    }
    sc_a[lid] = my_a;
    sc_b[lid] = my_b;
    __syncthreads();
    for (int s = WG / 2; s > 0; s >>= 1) {
        if (lid < s) {
            sc_a[lid] += sc_a[lid + s];
            sc_b[lid] += sc_b[lid + s];
        }
        __syncthreads();
    }
    // d_var and d_mean (via rstd)
    // TD-061 (fixed): this accumulates sum(d_xn * xn) — the ALREADY-NORMALIZED
    // value — not sum(d_xn * (x-mean)) like the (correct) CPU LayerNorm::backward()
    // in LayerNorm.cpp. Since xn = (x-mean)*rstd, using xn here needs one fewer
    // power of rstd than the CPU version's x-mean-based accumulation to reach the
    // same quantity; the original `r*r*r` copied the CPU version's power without
    // that adjustment, an extra factor of rstd verified both analytically and by
    // finite-difference check (numpy) against this exact kernel's formula structure.
    const float d_var = sc_a[0] * (-0.5f) * r * r;
    const float d_mean = sc_b[0] * (-r);

    for (int j = lid; j < cols; j += WG) {
        const float d_xn = dout_row[j] * gamma[j];
        // dx = d_xn*rstd + d_var*2*(x-mean)/N + d_mean/N; x-mean = xn/rstd
        dx_row[j] = d_xn * r + (2.0f * d_var * xn_row[j] / r + d_mean) * inv_N;
    }
}

__global__ void layer_norm_bwd_dgamma_dbeta_kernel(const float* dout, const float* input_norm,
                                                    float* dgamma, float* dbeta,
                                                    int rows, int cols) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j < cols) {
        float sg = 0.0f, sb = 0.0f;
        for (int i = 0; i < rows; ++i) {
            sg += dout[i * cols + j] * input_norm[i * cols + j];
            sb += dout[i * cols + j];
        }
        dgamma[j] += sg;
        dbeta[j] += sb;
    }
}

// dgamma/dbeta accumulate (caller zeros them before first call). 'mean' is
// unused in the gradient formula — matches the SYCL implementation, which
// takes the same parameter and also never reads it.
void matrix_layer_norm_bwd_gpu(const float* dout, const float* input_norm,
                                const float* gamma, const float* mean, const float* rstd,
                                float* dx, float* dgamma, float* dbeta,
                                int rows, int cols) {
    (void)mean;
    constexpr int WG = 256;
    cudaStream_t s = GPUManager::get_stream();

    const int smem_bytes = 2 * WG * static_cast<int>(sizeof(float));
    layer_norm_bwd_dx_kernel<<<rows, WG, smem_bytes, s>>>(dout, input_norm, gamma, rstd, dx, cols);
    CUDA_CHECK(cudaGetLastError());

    const int threads = 256;
    const int blocks = (cols + threads - 1) / threads;
    layer_norm_bwd_dgamma_dbeta_kernel<<<blocks, threads, 0, s>>>(
        dout, input_norm, dgamma, dbeta, rows, cols);
    CUDA_CHECK(cudaGetLastError());
}

__global__ void add_row_bias_kernel(const float* mat, const float* bias, float* out,
                                     int total, int cols) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total) {
        const int j = idx % cols;
        out[idx] = mat[idx] + bias[j];
    }
}

void matrix_add_row_bias_gpu(const float* mat, const float* bias, float* out,
                              int rows, int cols) {
    const int total = rows * cols;
    const int threads = 256;
    const int blocks = (total + threads - 1) / threads;
    add_row_bias_kernel<<<blocks, threads, 0, GPUManager::get_stream()>>>(mat, bias, out, total, cols);
    CUDA_CHECK(cudaGetLastError());
}

__global__ void sum_rows_kernel(const float* mat, float* out, int rows, int cols) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j < cols) {
        float sum = 0.0f;
        for (int i = 0; i < rows; ++i) sum += mat[i * cols + j];
        out[j] = sum;
    }
}

void matrix_sum_rows_gpu(const float* mat, float* out, int rows, int cols) {
    const int threads = 256;
    const int blocks = (cols + threads - 1) / threads;
    sum_rows_kernel<<<blocks, threads, 0, GPUManager::get_stream()>>>(mat, out, rows, cols);
    CUDA_CHECK(cudaGetLastError());
}

__global__ void masked_fill_kernel(float* data, const float* mask, float fill_val, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        if (mask[idx] == 0.0f) data[idx] = fill_val;
    }
}

void matrix_masked_fill_gpu(float* data, const float* mask, float fill_val, int size) {
    const int threads = 256;
    const int blocks = (size + threads - 1) / threads;
    masked_fill_kernel<<<blocks, threads, 0, GPUManager::get_stream()>>>(data, mask, fill_val, size);
    CUDA_CHECK(cudaGetLastError());
}

__global__ void cross_entropy_loss_kernel(const float* logits, const int* targets,
                                           float* losses, int vocab_size) {
    extern __shared__ float scratch[];
    const int row = blockIdx.x;
    const int lid = static_cast<int>(threadIdx.x);
    const int WG = static_cast<int>(blockDim.x);
    const float* row_ptr = logits + row * vocab_size;

    float my_max = -1e30f;
    for (int j = lid; j < vocab_size; j += WG) my_max = fmaxf(my_max, row_ptr[j]);
    scratch[lid] = my_max;
    __syncthreads();
    for (int s = WG / 2; s > 0; s >>= 1) {
        if (lid < s) scratch[lid] = fmaxf(scratch[lid], scratch[lid + s]);
        __syncthreads();
    }
    const float row_max = scratch[0];

    float my_sum = 0.0f;
    for (int j = lid; j < vocab_size; j += WG) my_sum += expf(row_ptr[j] - row_max);
    scratch[lid] = my_sum;
    __syncthreads();
    for (int s = WG / 2; s > 0; s >>= 1) {
        if (lid < s) scratch[lid] += scratch[lid + s];
        __syncthreads();
    }

    if (lid == 0) {
        const int tgt = targets[row];
        const float log_sum_exp = logf(scratch[0]) + row_max;
        losses[row] = log_sum_exp - row_ptr[tgt];
    }
}

float matrix_cross_entropy_loss_gpu(const float* logits, const int* targets,
                                     int seq_len, int vocab_size) {
    constexpr int WG = 256;
    GPUMemory<float> d_losses(seq_len);
    const int smem_bytes = WG * static_cast<int>(sizeof(float));
    cross_entropy_loss_kernel<<<seq_len, WG, smem_bytes, GPUManager::get_stream()>>>(
        logits, targets, d_losses.get(), vocab_size);
    CUDA_CHECK(cudaGetLastError());

    return matrix_sum_gpu(d_losses.get(), seq_len) / static_cast<float>(seq_len);
}

__global__ void cross_entropy_grad_kernel(const float* logits, const int* targets,
                                           float* grad, int vocab_size, float inv_seq) {
    extern __shared__ float scratch[];
    const int row = blockIdx.x;
    const int lid = static_cast<int>(threadIdx.x);
    const int WG = static_cast<int>(blockDim.x);
    const float* row_logits = logits + row * vocab_size;
    float* row_grad = grad + row * vocab_size;

    float my_max = -1e30f;
    for (int j = lid; j < vocab_size; j += WG) my_max = fmaxf(my_max, row_logits[j]);
    scratch[lid] = my_max;
    __syncthreads();
    for (int s = WG / 2; s > 0; s >>= 1) {
        if (lid < s) scratch[lid] = fmaxf(scratch[lid], scratch[lid + s]);
        __syncthreads();
    }
    const float row_max = scratch[0];

    float my_sum = 0.0f;
    for (int j = lid; j < vocab_size; j += WG) my_sum += expf(row_logits[j] - row_max);
    scratch[lid] = my_sum;
    __syncthreads();
    for (int s = WG / 2; s > 0; s >>= 1) {
        if (lid < s) scratch[lid] += scratch[lid + s];
        __syncthreads();
    }
    const float row_sum = scratch[0];

    const int tgt = targets[row];
    for (int j = lid; j < vocab_size; j += WG) {
        float soft = expf(row_logits[j] - row_max) / row_sum;
        row_grad[j] = (soft - (j == tgt ? 1.0f : 0.0f)) * inv_seq;
    }
}

void matrix_cross_entropy_grad_gpu(const float* logits, const int* targets,
                                    float* grad, int seq_len, int vocab_size) {
    constexpr int WG = 256;
    const float inv_seq = 1.0f / static_cast<float>(seq_len);
    const int smem_bytes = WG * static_cast<int>(sizeof(float));
    cross_entropy_grad_kernel<<<seq_len, WG, smem_bytes, GPUManager::get_stream()>>>(
        logits, targets, grad, vocab_size, inv_seq);
    CUDA_CHECK(cudaGetLastError());
}

// ============================================================================
// Training-diagnostics reductions (activation saturation / attention entropy)
// ============================================================================

// Count of elements with |x| < threshold — clone of sum_kernel's warp-shuffle
// reduction shape with the per-thread load replaced by a thresholded predicate.
__global__ void count_below_threshold_kernel(const float* input, float* output, int size,
                                              float threshold) {
    extern __shared__ float warp_sums[];
    const unsigned int tid = threadIdx.x;
    const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int lane = tid & 31u;
    const unsigned int wid = tid >> 5u;

    float val = (i < static_cast<unsigned int>(size) && fabsf(input[i]) < threshold) ? 1.0f : 0.0f;
    val = warp_reduce_sum(val);
    if (lane == 0) {
        warp_sums[wid] = val;
    }
    __syncthreads();

    const unsigned int warps_per_block = blockDim.x >> 5u;
    val = (tid < warps_per_block) ? warp_sums[tid] : 0.0f;
    if (wid == 0) {
        val = warp_reduce_sum(val);
    }
    if (tid == 0) {
        output[blockIdx.x] = val;
    }
}

float matrix_count_below_threshold_gpu(const float* data, int size, float threshold) {
    const int threads = 256;
    const int blocks = (size + threads - 1) / threads;
    GPUMemory<float> block_sums(blocks);

    const int smem_bytes = (threads / 32) * static_cast<int>(sizeof(float));
    count_below_threshold_kernel<<<blocks, threads, smem_bytes, GPUManager::get_stream()>>>(
        data, block_sums.get(), size, threshold);
    CUDA_CHECK(cudaGetLastError());

    if (blocks == 1) {
        float result;
        block_sums.copy_to_host(&result, 1);
        return result;
    }
    return matrix_sum_gpu(block_sums.get(), blocks);
}

// Per-row Shannon entropy -sum(p*log(p+eps)) of an already-normalized (e.g.
// post-softmax) matrix. out_row_entropy is a [rows]-sized buffer the caller
// reduces separately (via matrix_sum_gpu) to get a single scalar average.
__global__ void row_entropy_kernel(const float* data, float* out_row_entropy, int cols) {
    extern __shared__ float scratch[];
    const int row = blockIdx.x;
    const int lid = static_cast<int>(threadIdx.x);
    const int WG = static_cast<int>(blockDim.x);
    const float* row_ptr = data + row * cols;

    float my_entropy = 0.0f;
    for (int j = lid; j < cols; j += WG) {
        const float p = row_ptr[j];
        if (p > 0.0f) my_entropy -= p * logf(p + 1e-10f);
    }
    scratch[lid] = my_entropy;
    __syncthreads();
    for (int s = WG / 2; s > 0; s >>= 1) {
        if (lid < s) scratch[lid] += scratch[lid + s];
        __syncthreads();
    }
    if (lid == 0) out_row_entropy[row] = scratch[0];
}

void matrix_row_entropy_gpu(const float* data, float* out_row_entropy, int rows, int cols) {
    constexpr int WG = 256;
    const int smem_bytes = WG * static_cast<int>(sizeof(float));
    row_entropy_kernel<<<rows, WG, smem_bytes, GPUManager::get_stream()>>>(data, out_row_entropy, cols);
    CUDA_CHECK(cudaGetLastError());
}

// ============================================================================
// Backend-agnostic transfer helpers (Part D — replace direct GPUManager::get_queue()
// calls at FeedForward/MultiHeadAttention/CrossAttention/EncoderDecoderModel/
// LanguageModelHead call sites so they compile under both backends).
// ============================================================================

void matrix_copy_device_to_device_gpu(const float* src, float* dst, int count) {
    CUDA_CHECK(cudaMemcpyAsync(dst, src, static_cast<size_t>(count) * sizeof(float),
                               cudaMemcpyDeviceToDevice, GPUManager::get_stream()));
}

void matrix_download_gpu(const float* device_ptr, float* host_ptr, int count) {
    cudaStream_t s = GPUManager::get_stream();
    CUDA_CHECK(cudaMemcpyAsync(host_ptr, device_ptr, static_cast<size_t>(count) * sizeof(float),
                               cudaMemcpyDeviceToHost, s));
    CUDA_CHECK(cudaStreamSynchronize(s));
}

} // namespace gpu
} // namespace adai

#endif // ADAI_ENABLE_GPU
