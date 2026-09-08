// @adai-status: beta        (capped by TD-030 — see TECHNICAL_DEBT.md)
// @adai-version: 0.10.0
// @adai-reviewed: 2026-09-07

#ifdef ADAI_ENABLE_GPU

#include "MatrixGPU_SYCL.hpp"
#include <cmath>
#include <oneapi/mkl/blas.hpp>
#include <vector>

namespace adai {
namespace gpu {

// ============================================================================
// Element-wise Kernels
//
// Each op below is split into a vectorized sycl::vec<float,4> pass over the
// size/4 aligned prefix (one 128-bit load/store per work-item, ~4x the
// effective memory bandwidth of a scalar loop) plus a scalar pass over the
// size%4 remainder — mirrors CUDA's add_kernel_v4/add_kernel split in
// MatrixGPU.cu. USM device allocations satisfy sycl::vec<float,4>'s 16-byte
// alignment requirement, same assumption CUDA's float4 reinterpret makes of
// cudaMalloc.
// ============================================================================

void matrix_add_gpu(const float* a, const float* b, float* c, int size) {
    auto& q = GPUManager::get_queue();
    const int n4 = size / 4;
    const int tail = size % 4;
    if (n4 > 0) {
        const auto* a4 = reinterpret_cast<const sycl::vec<float, 4>*>(a);
        const auto* b4 = reinterpret_cast<const sycl::vec<float, 4>*>(b);
        auto* c4 = reinterpret_cast<sycl::vec<float, 4>*>(c);
        q.parallel_for(sycl::range<1>(n4), [=](sycl::id<1> idx) { c4[idx] = a4[idx] + b4[idx]; });
    }
    if (tail > 0) {
        const int offset = n4 * 4;
        q.parallel_for(sycl::range<1>(tail), [=](sycl::id<1> idx) {
            c[offset + idx] = a[offset + idx] + b[offset + idx];
        });
    }
}

void matrix_add_scalar_gpu(const float* a, float scalar, float* c, int size) {
    auto& q = GPUManager::get_queue();
    const int n4 = size / 4;
    const int tail = size % 4;
    if (n4 > 0) {
        const auto* a4 = reinterpret_cast<const sycl::vec<float, 4>*>(a);
        auto* c4 = reinterpret_cast<sycl::vec<float, 4>*>(c);
        q.parallel_for(sycl::range<1>(n4), [=](sycl::id<1> idx) { c4[idx] = a4[idx] + scalar; });
    }
    if (tail > 0) {
        const int offset = n4 * 4;
        q.parallel_for(sycl::range<1>(tail),
                       [=](sycl::id<1> idx) { c[offset + idx] = a[offset + idx] + scalar; });
    }
}

void matrix_multiply_elementwise_gpu(const float* a, const float* b, float* c, int size) {
    auto& q = GPUManager::get_queue();
    const int n4 = size / 4;
    const int tail = size % 4;
    if (n4 > 0) {
        const auto* a4 = reinterpret_cast<const sycl::vec<float, 4>*>(a);
        const auto* b4 = reinterpret_cast<const sycl::vec<float, 4>*>(b);
        auto* c4 = reinterpret_cast<sycl::vec<float, 4>*>(c);
        q.parallel_for(sycl::range<1>(n4), [=](sycl::id<1> idx) { c4[idx] = a4[idx] * b4[idx]; });
    }
    if (tail > 0) {
        const int offset = n4 * 4;
        q.parallel_for(sycl::range<1>(tail), [=](sycl::id<1> idx) {
            c[offset + idx] = a[offset + idx] * b[offset + idx];
        });
    }
}

void matrix_multiply_scalar_gpu(const float* a, float scalar, float* c, int size) {
    auto& q = GPUManager::get_queue();
    const int n4 = size / 4;
    const int tail = size % 4;
    if (n4 > 0) {
        const auto* a4 = reinterpret_cast<const sycl::vec<float, 4>*>(a);
        auto* c4 = reinterpret_cast<sycl::vec<float, 4>*>(c);
        q.parallel_for(sycl::range<1>(n4), [=](sycl::id<1> idx) { c4[idx] = a4[idx] * scalar; });
    }
    if (tail > 0) {
        const int offset = n4 * 4;
        q.parallel_for(sycl::range<1>(tail),
                       [=](sycl::id<1> idx) { c[offset + idx] = a[offset + idx] * scalar; });
    }
}

// ============================================================================
// Transpose Kernel (shared-memory tiled)
// ============================================================================

void matrix_transpose_gpu(const float* input, float* output, int rows, int cols) {
    constexpr int TILE = 32;
    constexpr int TILE_PAD = 33;  // +1 to avoid bank conflicts
    auto& q = GPUManager::get_queue();

    sycl::range<2> global_size(((rows + TILE - 1) / TILE) * TILE,
                               ((cols + TILE - 1) / TILE) * TILE);
    sycl::range<2> local_size(TILE, TILE);

    q.submit([&](sycl::handler& cgh) {
        sycl::local_accessor<float, 2> tile(sycl::range<2>(TILE, TILE_PAD), cgh);

        cgh.parallel_for(sycl::nd_range<2>(global_size, local_size), [=](sycl::nd_item<2> item) {
            int lx = item.get_local_id(1);
            int ly = item.get_local_id(0);

            int x = item.get_group(1) * TILE + lx;
            int y = item.get_group(0) * TILE + ly;

            if (x < cols && y < rows) {
                tile[ly][lx] = input[y * cols + x];
            }

            item.barrier(sycl::access::fence_space::local_space);

            x = item.get_group(0) * TILE + lx;
            y = item.get_group(1) * TILE + ly;

            if (x < rows && y < cols) {
                output[y * rows + x] = tile[lx][ly];
            }
        });
    });
}

// ============================================================================
// Matrix Multiplication (oneMKL GEMM)
// ============================================================================

void matrix_multiply_gpu(const float* a, const float* b, float* c, int m, int k, int n) {
    auto& q = GPUManager::get_queue();
    const float alpha = 1.0f;
    const float beta = 0.0f;

    // oneMKL uses column-major by default. Same trick as cuBLAS:
    // compute C = B * A in column-major = A * B in row-major.
    oneapi::mkl::blas::gemm(q, oneapi::mkl::transpose::nontrans, oneapi::mkl::transpose::nontrans,
                            n, m, k, alpha, b, n, a, k, beta, c, n);
}

// ============================================================================
// Activation Kernels
// ============================================================================

// Shared by the vectorized and scalar-tail passes below, same role as
// MatrixGPU.cu's __device__ apply_activation().
static inline float apply_activation_sycl(float x, int act) {
    switch (act) {
        case 0:  // ReLU
            return sycl::fmax(0.0f, x);
        case 1:  // Sigmoid
            return 1.0f / (1.0f + sycl::exp(-x));
        case 2:  // Tanh
            return sycl::tanh(x);
        case 3:  // GELU (approximation)
            return 0.5f * x * (1.0f + sycl::tanh(0.7978845608f * (x + 0.044715f * x * x * x)));
        default:
            return x;
    }
}

void matrix_apply_activation_gpu(float* data, int size, ActivationType type) {
    auto& q = GPUManager::get_queue();
    const int act = static_cast<int>(type);
    const int n4 = size / 4;
    const int tail = size % 4;

    if (n4 > 0) {
        auto* data4 = reinterpret_cast<sycl::vec<float, 4>*>(data);
        q.parallel_for(sycl::range<1>(n4), [=](sycl::id<1> idx) {
            sycl::vec<float, 4> v = data4[idx];
            v.x() = apply_activation_sycl(v.x(), act);
            v.y() = apply_activation_sycl(v.y(), act);
            v.z() = apply_activation_sycl(v.z(), act);
            v.w() = apply_activation_sycl(v.w(), act);
            data4[idx] = v;
        });
    }
    if (tail > 0) {
        const int offset = n4 * 4;
        q.parallel_for(sycl::range<1>(tail), [=](sycl::id<1> idx) {
            data[offset + idx] = apply_activation_sycl(data[offset + idx], act);
        });
    }
}

// ============================================================================
// Sum Reduction (sub-group optimized)
//
// Sub-group reduce_over_group is SYCL's portable equivalent of CUDA's
// __shfl_down_sync warp reduction (MatrixGPU.cu's warp_reduce_sum /
// TD-003 warp opt): no local memory or barrier needed within a sub-group.
// A single barrier then combines the per-sub-group partials, mirroring
// sum_kernel's 6-step shape exactly — just with reduce_over_group standing
// in for the shuffle-based warp_reduce_sum.
// ============================================================================

// Reduce val across one sub-group (portable stand-in for warp_reduce_sum).
static inline float subgroup_reduce_sum(sycl::sub_group sg, float val) {
    return sycl::reduce_over_group(sg, val, sycl::plus<float>());
}

// Reduce val across an entire work-group. Valid only in the item whose
// local_id(0) == 0; other work-items' return values are unspecified.
// `subgroup_sums` must have at least (work-group size / sub-group size)
// elements — WG_SIZE is generous for any Intel sub-group width (16 or 32).
static inline float workgroup_reduce_sum(sycl::nd_item<1> item,
                                          sycl::local_accessor<float, 1> subgroup_sums,
                                          float val) {
    auto sg = item.get_sub_group();

    // Step 1: reduce within the sub-group — no shared memory, no barrier.
    val = subgroup_reduce_sum(sg, val);

    // Step 2: lane 0 of each sub-group stores its partial sum.
    const int sg_id = static_cast<int>(sg.get_group_linear_id());
    if (sg.get_local_linear_id() == 0) {
        subgroup_sums[sg_id] = val;
    }

    // Step 3: ONE barrier to make all sub-group results visible.
    item.barrier(sycl::access::fence_space::local_space);

    // Step 4: the first sub-group reduces the per-sub-group partial sums.
    //         Lanes beyond that count contribute 0.
    const int num_subgroups =
        static_cast<int>(item.get_local_range(0) / sg.get_local_linear_range());
    const int local_id = static_cast<int>(item.get_local_id(0));
    val = (local_id < num_subgroups) ? subgroup_sums[local_id] : 0.0f;
    if (sg_id == 0) {
        val = subgroup_reduce_sum(sg, val);
    }
    return val;
}

float matrix_sum_gpu(const float* data, int size) {
    auto& q = GPUManager::get_queue();
    constexpr int WG_SIZE = 256;
    int num_groups = (size + WG_SIZE - 1) / WG_SIZE;

    GPUMemory<float> group_sums(num_groups);

    q.submit([&](sycl::handler& cgh) {
         sycl::local_accessor<float, 1> subgroup_sums(sycl::range<1>(WG_SIZE), cgh);
         float* out = group_sums.get();

         cgh.parallel_for(sycl::nd_range<1>(num_groups * WG_SIZE, WG_SIZE),
                          [=](sycl::nd_item<1> item) {
                              const int global_id = static_cast<int>(item.get_global_id(0));
                              float val = (global_id < size) ? data[global_id] : 0.0f;

                              val = workgroup_reduce_sum(item, subgroup_sums, val);

                              if (item.get_local_id(0) == 0) {
                                  out[item.get_group(0)] = val;
                              }
                          });
     }).wait();

    if (num_groups == 1) {
        float result;
        group_sums.copy_to_host(&result, 1);
        return result;
    }

    return matrix_sum_gpu(group_sums.get(), num_groups);
}

// ============================================================================
// Batch Operations
// ============================================================================

void matrix_batch_add_gpu(const float** a_batch, const float** b_batch, float** c_batch,
                          int batch_size, int size) {
    auto& q = GPUManager::get_queue();
    const size_t ptr_bytes = static_cast<size_t>(batch_size) * sizeof(float*);

    // Stage host pointer arrays onto the device
    const float** d_a = sycl::malloc_device<const float*>(batch_size, q);
    const float** d_b = sycl::malloc_device<const float*>(batch_size, q);
    float** d_c = sycl::malloc_device<float*>(batch_size, q);

    // Pointer arrays must land before the kernel dereferences them; in-order
    // queue guarantees the copies complete before the parallel_for starts.
    q.memcpy(d_a, a_batch, ptr_bytes);
    q.memcpy(d_b, b_batch, ptr_bytes);
    q.memcpy(d_c, c_batch, ptr_bytes);

    // Vectorized float4 pass over the size/4 aligned prefix + scalar tail —
    // mirrors CUDA's batch_add_kernel_v4 / batch_add_tail_kernel split.
    const int n4 = size / 4;
    const int tail = size % 4;

    if (n4 > 0) {
        q.parallel_for(sycl::range<2>(batch_size, n4), [=](sycl::id<2> idx) {
            const int b = idx[0];
            const int i = idx[1];
            const auto* a4 = reinterpret_cast<const sycl::vec<float, 4>*>(d_a[b]);
            const auto* b4 = reinterpret_cast<const sycl::vec<float, 4>*>(d_b[b]);
            auto* c4 = reinterpret_cast<sycl::vec<float, 4>*>(d_c[b]);
            c4[i] = a4[i] + b4[i];
        });
    }
    if (tail > 0) {
        const int offset = n4 * 4;
        q.parallel_for(sycl::range<2>(batch_size, tail), [=](sycl::id<2> idx) {
            const int b = idx[0];
            const int i = idx[1];
            d_c[b][offset + i] = d_a[b][offset + i] + d_b[b][offset + i];
        });
    }

    // Free after the kernels — in-order queue ensures they finish first.
    q.submit([=](sycl::handler& cgh) {
        cgh.host_task([=]() {
            sycl::free(d_a, q);
            sycl::free(d_b, q);
            sycl::free(d_c, q);
        });
    });
}

void matrix_batch_multiply_gpu(const float** a_batch, const float** b_batch, float** c_batch,
                               int batch_size, int m, int k, int n) {
    auto& q = GPUManager::get_queue();
    const float alpha = 1.0f;
    const float beta = 0.0f;

    // Stage host pointer arrays onto the device
    const size_t ptr_bytes = static_cast<size_t>(batch_size) * sizeof(float*);
    const float** d_a = sycl::malloc_device<const float*>(batch_size, q);
    const float** d_b = sycl::malloc_device<const float*>(batch_size, q);
    float** d_c = sycl::malloc_device<float*>(batch_size, q);

    // Pointer arrays must land before the GEMM dereferences them; in-order
    // queue guarantees the copies complete before gemm_batch starts.
    q.memcpy(d_a, a_batch, ptr_bytes);
    q.memcpy(d_b, b_batch, ptr_bytes);
    q.memcpy(d_c, c_batch, ptr_bytes);

    // oneMKL batched GEMM with group API: single group, all same sizes.
    // Same column-major trick: compute B*A = A*B in row-major.
    std::int64_t m64 = m, k64 = k, n64 = n;
    std::int64_t lda = k, ldb = n, ldc = n;
    std::int64_t group_count = 1;
    std::int64_t group_size = batch_size;
    oneapi::mkl::transpose trans_n = oneapi::mkl::transpose::nontrans;

    oneapi::mkl::blas::gemm_batch(
        q, &trans_n, &trans_n, &n64, &m64, &k64, &alpha, reinterpret_cast<const float**>(d_b), &ldb,
        reinterpret_cast<const float**>(d_a), &lda, &beta, d_c, &ldc, group_count, &group_size);

    // Free after the GEMM — in-order queue ensures it finishes first.
    q.submit([=](sycl::handler& cgh) {
        cgh.host_task([=]() {
            sycl::free(d_a, q);
            sycl::free(d_b, q);
            sycl::free(d_c, q);
        });
    });
}

// ============================================================================
// In-Place Accumulate (dst += src)
// ============================================================================

void matrix_add_inplace_gpu(const float* src, float* dst, int size) {
    auto& q = GPUManager::get_queue();
    q.parallel_for(sycl::range<1>(size), [=](sycl::id<1> idx) { dst[idx] += src[idx]; });
}

// ============================================================================
// Row-Wise Softmax (in-place, numerically stable)
// ============================================================================

void matrix_softmax_rows_gpu(float* data, int rows, int cols) {
    auto& q = GPUManager::get_queue();
    constexpr int WG = 256;
    q.submit([&](sycl::handler& cgh) {
        sycl::local_accessor<float, 1> scratch(sycl::range<1>(WG), cgh);
        cgh.parallel_for(sycl::nd_range<1>(static_cast<size_t>(rows) * WG, WG),
                         [=](sycl::nd_item<1> item) {
                             const int row = static_cast<int>(item.get_group(0));
                             const int lid = static_cast<int>(item.get_local_id(0));
                             float* row_ptr = data + row * cols;

                             // Pass 1: row max
                             float my_max = -1e30f;
                             for (int j = lid; j < cols; j += WG)
                                 my_max = sycl::fmax(my_max, row_ptr[j]);
                             scratch[lid] = my_max;
                             item.barrier(sycl::access::fence_space::local_space);
                             for (int s = WG / 2; s > 0; s >>= 1) {
                                 if (lid < s)
                                     scratch[lid] = sycl::fmax(scratch[lid], scratch[lid + s]);
                                 item.barrier(sycl::access::fence_space::local_space);
                             }
                             const float row_max = scratch[0];

                             // Pass 2: exp(x - max) and sum
                             float my_sum = 0.0f;
                             for (int j = lid; j < cols; j += WG) {
                                 float e = sycl::exp(row_ptr[j] - row_max);
                                 row_ptr[j] = e;
                                 my_sum += e;
                             }
                             scratch[lid] = my_sum;
                             item.barrier(sycl::access::fence_space::local_space);
                             for (int s = WG / 2; s > 0; s >>= 1) {
                                 if (lid < s)
                                     scratch[lid] += scratch[lid + s];
                                 item.barrier(sycl::access::fence_space::local_space);
                             }
                             const float row_sum = scratch[0];

                             // Pass 3: normalize
                             for (int j = lid; j < cols; j += WG)
                                 row_ptr[j] /= row_sum;
                         });
    });
}

// ============================================================================
// Softmax Backward
// d_in[i][j] = s[i][j] * (d_out[i][j] - sum_k(d_out[i][k] * s[i][k]))
// ============================================================================

void matrix_softmax_backward_gpu(const float* s, const float* dout, float* din, int rows,
                                 int cols) {
    auto& q = GPUManager::get_queue();
    constexpr int WG = 256;
    q.submit([&](sycl::handler& cgh) {
        sycl::local_accessor<float, 1> scratch(sycl::range<1>(WG), cgh);
        cgh.parallel_for(sycl::nd_range<1>(static_cast<size_t>(rows) * WG, WG),
                         [=](sycl::nd_item<1> item) {
                             const int row = static_cast<int>(item.get_group(0));
                             const int lid = static_cast<int>(item.get_local_id(0));
                             const float* s_row = s + row * cols;
                             const float* dout_row = dout + row * cols;
                             float* din_row = din + row * cols;

                             // dot(s, dout) for this row
                             float my_dot = 0.0f;
                             for (int j = lid; j < cols; j += WG)
                                 my_dot += s_row[j] * dout_row[j];
                             scratch[lid] = my_dot;
                             item.barrier(sycl::access::fence_space::local_space);
                             for (int st = WG / 2; st > 0; st >>= 1) {
                                 if (lid < st)
                                     scratch[lid] += scratch[lid + st];
                                 item.barrier(sycl::access::fence_space::local_space);
                             }
                             const float dot_val = scratch[0];

                             for (int j = lid; j < cols; j += WG)
                                 din_row[j] = s_row[j] * (dout_row[j] - dot_val);
                         });
    });
}

// ============================================================================
// GELU Backward
// GELU'(x) = 0.5*(1+tanh(u)) + 0.5*x*sech²(u)*c*(1+3*0.044715*x²)
// ============================================================================

void matrix_gelu_backward_gpu(const float* pre_act, const float* dout, float* din, int size) {
    auto& q = GPUManager::get_queue();
    q.parallel_for(sycl::range<1>(size), [=](sycl::id<1> idx) {
        const float x = pre_act[idx];
        constexpr float c = 0.7978845608f;
        constexpr float a = 0.044715f;
        const float u = c * (x + a * x * x * x);
        const float tanh_u = sycl::tanh(u);
        const float sech2 = 1.0f - tanh_u * tanh_u;
        const float gelu_prime =
            0.5f * (1.0f + tanh_u) + 0.5f * x * sech2 * c * (1.0f + 3.0f * a * x * x);
        din[idx] = dout[idx] * gelu_prime;
    });
}

// ============================================================================
// Layer Norm Forward
// Stores reciprocal std (rstd) for efficient backward pass.
// out_mean and out_rstd are flat arrays of length 'rows'.
// ============================================================================

void matrix_layer_norm_fwd_gpu(const float* input, float* output, float* out_normed,
                               const float* gamma, const float* beta, float* out_mean,
                               float* out_rstd, int rows, int cols, float eps) {
    auto& q = GPUManager::get_queue();
    constexpr int WG = 256;
    const float inv_cols = 1.0f / static_cast<float>(cols);
    q.submit([&](sycl::handler& cgh) {
        sycl::local_accessor<float, 1> scratch(sycl::range<1>(WG), cgh);
        cgh.parallel_for(sycl::nd_range<1>(static_cast<size_t>(rows) * WG, WG),
                         [=](sycl::nd_item<1> item) {
                             const int row = static_cast<int>(item.get_group(0));
                             const int lid = static_cast<int>(item.get_local_id(0));
                             const float* x = input + row * cols;
                             float* y = output + row * cols;
                             float* xn = out_normed + row * cols;

                             // Mean
                             float my_sum = 0.0f;
                             for (int j = lid; j < cols; j += WG)
                                 my_sum += x[j];
                             scratch[lid] = my_sum;
                             item.barrier(sycl::access::fence_space::local_space);
                             for (int s = WG / 2; s > 0; s >>= 1) {
                                 if (lid < s)
                                     scratch[lid] += scratch[lid + s];
                                 item.barrier(sycl::access::fence_space::local_space);
                             }
                             const float mean = scratch[0] * inv_cols;
                             if (lid == 0 && out_mean)
                                 out_mean[row] = mean;

                             // Variance
                             float my_var = 0.0f;
                             for (int j = lid; j < cols; j += WG) {
                                 float d = x[j] - mean;
                                 my_var += d * d;
                             }
                             scratch[lid] = my_var;
                             item.barrier(sycl::access::fence_space::local_space);
                             for (int s = WG / 2; s > 0; s >>= 1) {
                                 if (lid < s)
                                     scratch[lid] += scratch[lid + s];
                                 item.barrier(sycl::access::fence_space::local_space);
                             }
                             const float rstd = 1.0f / sycl::sqrt(scratch[0] * inv_cols + eps);
                             if (lid == 0 && out_rstd)
                                 out_rstd[row] = rstd;

                             // Normalize and write both normed (pre-affine) and output
                             // (post-affine)
                             for (int j = lid; j < cols; j += WG) {
                                 const float n = (x[j] - mean) * rstd;
                                 xn[j] = n;
                                 y[j] = n * gamma[j] + beta[j];
                             }
                         });
    });
}

// ============================================================================
// Layer Norm Backward
// Accumulates dgamma and dbeta (caller zeros them before first call).
// input_norm: cached (x-mean)*rstd before affine [rows, cols]
// ============================================================================

void matrix_layer_norm_bwd_gpu(const float* dout, const float* input_norm, const float* gamma,
                               const float* mean, const float* rstd, float* dx, float* dgamma,
                               float* dbeta, int rows, int cols) {
    auto& q = GPUManager::get_queue();
    constexpr int WG = 256;
    const float inv_N = 1.0f / static_cast<float>(cols);

    // Kernel 1: compute dx per row
    q.submit([&](sycl::handler& cgh) {
        sycl::local_accessor<float, 1> sc_a(sycl::range<1>(WG), cgh);
        sycl::local_accessor<float, 1> sc_b(sycl::range<1>(WG), cgh);
        cgh.parallel_for(
            sycl::nd_range<1>(static_cast<size_t>(rows) * WG, WG), [=](sycl::nd_item<1> item) {
                const int row = static_cast<int>(item.get_group(0));
                const int lid = static_cast<int>(item.get_local_id(0));
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
                item.barrier(sycl::access::fence_space::local_space);
                for (int s = WG / 2; s > 0; s >>= 1) {
                    if (lid < s) {
                        sc_a[lid] += sc_a[lid + s];
                        sc_b[lid] += sc_b[lid + s];
                    }
                    item.barrier(sycl::access::fence_space::local_space);
                }
                // d_var and d_mean (via rstd)
                const float d_var = sc_a[0] * (-0.5f) * r * r * r;
                const float d_mean = sc_b[0] * (-r);

                for (int j = lid; j < cols; j += WG) {
                    const float d_xn = dout_row[j] * gamma[j];
                    // dx = d_xn*rstd + d_var*2*(x-mean)/N + d_mean/N
                    // x-mean = xn/rstd, so 2*(x-mean) = 2*xn/rstd
                    dx_row[j] = d_xn * r + (2.0f * d_var * xn_row[j] / r + d_mean) * inv_N;
                }
            });
    });

    // Kernel 2: column-wise accumulation of dgamma and dbeta
    q.parallel_for(sycl::range<1>(cols), [=](sycl::id<1> j) {
        float sg = 0.0f, sb = 0.0f;
        for (int i = 0; i < rows; ++i) {
            sg += dout[i * cols + j] * input_norm[i * cols + j];
            sb += dout[i * cols + j];
        }
        dgamma[j] += sg;
        dbeta[j] += sb;
    });
}

// ============================================================================
// Add Row Bias (broadcast [1, cols] bias to [rows, cols])
// ============================================================================

void matrix_add_row_bias_gpu(const float* mat, const float* bias, float* out, int rows, int cols) {
    auto& q = GPUManager::get_queue();
    q.parallel_for(sycl::range<2>(rows, cols), [=](sycl::id<2> idx) {
        out[idx[0] * cols + idx[1]] = mat[idx[0] * cols + idx[1]] + bias[idx[1]];
    });
}

// ============================================================================
// Sum Rows: [rows, cols] → [1, cols]  (out[j] = sum_i mat[i,j])
// ============================================================================

void matrix_sum_rows_gpu(const float* mat, float* out, int rows, int cols) {
    auto& q = GPUManager::get_queue();
    q.parallel_for(sycl::range<1>(cols), [=](sycl::id<1> j) {
        float s = 0.0f;
        for (int i = 0; i < rows; ++i)
            s += mat[i * cols + j];
        out[j] = s;
    });
}

// ============================================================================
// Masked Fill: fill data[i] with fill_val where mask[i] == 0
// ============================================================================

void matrix_masked_fill_gpu(float* data, const float* mask, float fill_val, int size) {
    auto& q = GPUManager::get_queue();
    q.parallel_for(sycl::range<1>(size), [=](sycl::id<1> idx) {
        if (mask[idx] == 0.0f)
            data[idx] = fill_val;
    });
}

// ============================================================================
// Cross-Entropy Loss (returns scalar; logits are [seq_len, vocab_size])
// Numerically stable: log-sum-exp per row
// ============================================================================

float matrix_cross_entropy_loss_gpu(const float* logits, const int* targets, int seq_len,
                                    int vocab_size) {
    auto& q = GPUManager::get_queue();
    GPUMemory<float> d_losses(seq_len);
    constexpr int WG = 256;

    q.submit([&](sycl::handler& cgh) {
        sycl::local_accessor<float, 1> scratch(sycl::range<1>(WG), cgh);
        float* losses = d_losses.get();
        cgh.parallel_for(sycl::nd_range<1>(static_cast<size_t>(seq_len) * WG, WG),
                         [=](sycl::nd_item<1> item) {
                             const int row = static_cast<int>(item.get_group(0));
                             const int lid = static_cast<int>(item.get_local_id(0));
                             const float* row_ptr = logits + row * vocab_size;

                             // Max for numerical stability
                             float my_max = -1e30f;
                             for (int j = lid; j < vocab_size; j += WG)
                                 my_max = sycl::fmax(my_max, row_ptr[j]);
                             scratch[lid] = my_max;
                             item.barrier(sycl::access::fence_space::local_space);
                             for (int s = WG / 2; s > 0; s >>= 1) {
                                 if (lid < s)
                                     scratch[lid] = sycl::fmax(scratch[lid], scratch[lid + s]);
                                 item.barrier(sycl::access::fence_space::local_space);
                             }
                             const float row_max = scratch[0];

                             // Sum exp
                             float my_sum = 0.0f;
                             for (int j = lid; j < vocab_size; j += WG)
                                 my_sum += sycl::exp(row_ptr[j] - row_max);
                             scratch[lid] = my_sum;
                             item.barrier(sycl::access::fence_space::local_space);
                             for (int s = WG / 2; s > 0; s >>= 1) {
                                 if (lid < s)
                                     scratch[lid] += scratch[lid + s];
                                 item.barrier(sycl::access::fence_space::local_space);
                             }

                             if (lid == 0) {
                                 const int tgt = targets[row];
                                 const float log_sum_exp = sycl::log(scratch[0]) + row_max;
                                 losses[row] = log_sum_exp - row_ptr[tgt];
                             }
                         });
    });

    // Sum losses over seq_len
    return matrix_sum_gpu(d_losses.get(), seq_len) / static_cast<float>(seq_len);
}

// ============================================================================
// Cross-Entropy Gradient: grad[t][v] = (softmax(logits[t])[v] - 1_hot) / seq_len
// ============================================================================

void matrix_cross_entropy_grad_gpu(const float* logits, const int* targets, float* grad,
                                   int seq_len, int vocab_size) {
    auto& q = GPUManager::get_queue();
    constexpr int WG = 256;
    const float inv_seq = 1.0f / static_cast<float>(seq_len);

    q.submit([&](sycl::handler& cgh) {
        sycl::local_accessor<float, 1> scratch(sycl::range<1>(WG), cgh);
        cgh.parallel_for(sycl::nd_range<1>(static_cast<size_t>(seq_len) * WG, WG),
                         [=](sycl::nd_item<1> item) {
                             const int row = static_cast<int>(item.get_group(0));
                             const int lid = static_cast<int>(item.get_local_id(0));
                             const float* row_logits = logits + row * vocab_size;
                             float* row_grad = grad + row * vocab_size;

                             // Max for numerical stability
                             float my_max = -1e30f;
                             for (int j = lid; j < vocab_size; j += WG)
                                 my_max = sycl::fmax(my_max, row_logits[j]);
                             scratch[lid] = my_max;
                             item.barrier(sycl::access::fence_space::local_space);
                             for (int s = WG / 2; s > 0; s >>= 1) {
                                 if (lid < s)
                                     scratch[lid] = sycl::fmax(scratch[lid], scratch[lid + s]);
                                 item.barrier(sycl::access::fence_space::local_space);
                             }
                             const float row_max = scratch[0];

                             // Sum exp
                             float my_sum = 0.0f;
                             for (int j = lid; j < vocab_size; j += WG)
                                 my_sum += sycl::exp(row_logits[j] - row_max);
                             scratch[lid] = my_sum;
                             item.barrier(sycl::access::fence_space::local_space);
                             for (int s = WG / 2; s > 0; s >>= 1) {
                                 if (lid < s)
                                     scratch[lid] += scratch[lid + s];
                                 item.barrier(sycl::access::fence_space::local_space);
                             }
                             const float row_sum = scratch[0];

                             // grad = softmax - one_hot, scaled by 1/seq_len
                             const int tgt = targets[row];
                             for (int j = lid; j < vocab_size; j += WG) {
                                 float soft = sycl::exp(row_logits[j] - row_max) / row_sum;
                                 row_grad[j] = (soft - (j == tgt ? 1.0f : 0.0f)) * inv_seq;
                             }
                         });
    });
}

// ============================================================================
// Training-diagnostics reductions (activation saturation / attention entropy)
// ============================================================================

// Count of elements with |x| < threshold. Clone of matrix_sum_gpu's sub-group
// reduction shape (see above, and MatrixGPU.cu's count_below_threshold_kernel)
// with the per-thread load replaced by a thresholded predicate; recurses into
// matrix_sum_gpu itself for the final multi-group reduce.
float matrix_count_below_threshold_gpu(const float* data, int size, float threshold) {
    auto& q = GPUManager::get_queue();
    constexpr int WG_SIZE = 256;
    int num_groups = (size + WG_SIZE - 1) / WG_SIZE;

    GPUMemory<float> group_sums(num_groups);

    q.submit([&](sycl::handler& cgh) {
         sycl::local_accessor<float, 1> subgroup_sums(sycl::range<1>(WG_SIZE), cgh);
         float* out = group_sums.get();

         cgh.parallel_for(
             sycl::nd_range<1>(num_groups * WG_SIZE, WG_SIZE), [=](sycl::nd_item<1> item) {
                 const int global_id = static_cast<int>(item.get_global_id(0));
                 float val =
                     (global_id < size && sycl::fabs(data[global_id]) < threshold) ? 1.0f : 0.0f;

                 val = workgroup_reduce_sum(item, subgroup_sums, val);

                 if (item.get_local_id(0) == 0) {
                     out[item.get_group(0)] = val;
                 }
             });
     }).wait();

    if (num_groups == 1) {
        float result;
        group_sums.copy_to_host(&result, 1);
        return result;
    }

    return matrix_sum_gpu(group_sums.get(), num_groups);
}

// Per-row Shannon entropy -sum(p*log(p+eps)) of an already-normalized (e.g.
// post-softmax) matrix. One work-group per row, mirroring matrix_softmax_rows_gpu's
// per-row shape; out_row_entropy is a [rows]-sized buffer the caller reduces
// separately (via matrix_sum_gpu) to get a single scalar average.
void matrix_row_entropy_gpu(const float* data, float* out_row_entropy, int rows, int cols) {
    auto& q = GPUManager::get_queue();
    constexpr int WG = 256;
    q.submit([&](sycl::handler& cgh) {
        sycl::local_accessor<float, 1> scratch(sycl::range<1>(WG), cgh);
        cgh.parallel_for(sycl::nd_range<1>(static_cast<size_t>(rows) * WG, WG),
                         [=](sycl::nd_item<1> item) {
                             const int row = static_cast<int>(item.get_group(0));
                             const int lid = static_cast<int>(item.get_local_id(0));
                             const float* row_ptr = data + row * cols;

                             float my_entropy = 0.0f;
                             for (int j = lid; j < cols; j += WG) {
                                 const float p = row_ptr[j];
                                 if (p > 0.0f)
                                     my_entropy -= p * sycl::log(p + 1e-10f);
                             }
                             scratch[lid] = my_entropy;
                             item.barrier(sycl::access::fence_space::local_space);
                             for (int s = WG / 2; s > 0; s >>= 1) {
                                 if (lid < s)
                                     scratch[lid] += scratch[lid + s];
                                 item.barrier(sycl::access::fence_space::local_space);
                             }

                             if (lid == 0)
                                 out_row_entropy[row] = scratch[0];
                         });
    });
}

// ============================================================================
// Backend-agnostic transfer helpers (Part D — replace direct GPUManager::get_queue()
// calls at FeedForward/MultiHeadAttention/CrossAttention/EncoderDecoderModel/
// LanguageModelHead call sites so they compile under both backends).
// ============================================================================

void matrix_copy_device_to_device_gpu(const float* src, float* dst, int count) {
    auto& q = GPUManager::get_queue();
    // In-order queue: no .wait() needed, ordering with subsequent kernels on
    // the same queue is guaranteed.
    q.memcpy(dst, src, static_cast<size_t>(count) * sizeof(float));
}

void matrix_download_gpu(const float* device_ptr, float* host_ptr, int count) {
    auto& q = GPUManager::get_queue();
    q.memcpy(host_ptr, device_ptr, static_cast<size_t>(count) * sizeof(float)).wait();
}

}  // namespace gpu
}  // namespace adai

#endif  // ADAI_ENABLE_GPU
