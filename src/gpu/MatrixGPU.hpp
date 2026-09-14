#ifndef MATRIX_GPU_HPP
#define MATRIX_GPU_HPP

// @adai-status: beta        (TD-061 resolved — see MatrixGPU.cu's tag; this header just declares/wraps its kernels; TD-050 GPUKVCache added)
// @adai-version: 0.10.0
// @adai-reviewed: 2026-09-14


#include <stdexcept>
#include <vector>

#ifdef ADAI_ENABLE_GPU

#if defined(ADAI_GPU_BACKEND_SYCL)
#include "sycl/MatrixGPU_SYCL.hpp"
#else  // CUDA backend (default)

#include "GPUUtils.hpp"  // GPUMemory, GPUManager, CUDA_CHECK

namespace adai {
namespace gpu {

/**
 * @brief Activation function types for GPU operations
 */
enum class ActivationType { RELU = 0, SIGMOID = 1, TANH = 2, GELU = 3 };

/**
 * @brief GPU-accelerated matrix operations
 *
 * All functions assume data is already on the GPU device.
 * Use GPUMemory class from GPUUtils.hpp for memory management.
 */

/**
 * @brief Element-wise matrix addition on GPU
 * @param a First matrix (device pointer)
 * @param b Second matrix (device pointer)
 * @param c Result matrix (device pointer)
 * @param size Total number of elements
 */
void matrix_add_gpu(const float* a, const float* b, float* c, int size);

/**
 * @brief Add scalar to all matrix elements on GPU
 * @param a Input matrix (device pointer)
 * @param scalar Scalar value to add
 * @param c Result matrix (device pointer)
 * @param size Total number of elements
 */
void matrix_add_scalar_gpu(const float* a, float scalar, float* c, int size);

/**
 * @brief Element-wise matrix multiplication on GPU
 * @param a First matrix (device pointer)
 * @param b Second matrix (device pointer)
 * @param c Result matrix (device pointer)
 * @param size Total number of elements
 */
void matrix_multiply_elementwise_gpu(const float* a, const float* b, float* c, int size);

/**
 * @brief Multiply all matrix elements by scalar on GPU
 * @param a Input matrix (device pointer)
 * @param scalar Scalar value to multiply
 * @param c Result matrix (device pointer)
 * @param size Total number of elements
 */
void matrix_multiply_scalar_gpu(const float* a, float scalar, float* c, int size);

/**
 * @brief Matrix transpose on GPU
 * @param input Input matrix (device pointer)
 * @param output Output matrix (device pointer)
 * @param rows Number of rows in input matrix
 * @param cols Number of columns in input matrix
 */
void matrix_transpose_gpu(const float* input, float* output, int rows, int cols);

/**
 * @brief Matrix multiplication using cuBLAS (C = A * B)
 * @param a First matrix (m x k) (device pointer)
 * @param b Second matrix (k x n) (device pointer)
 * @param c Result matrix (m x n) (device pointer)
 * @param m Number of rows in A and C
 * @param k Number of columns in A and rows in B
 * @param n Number of columns in B and C
 */
void matrix_multiply_gpu(const float* a, const float* b, float* c, int m, int k, int n);

/**
 * @brief Apply activation function in-place on GPU
 * @param data Matrix data (device pointer, modified in-place)
 * @param size Total number of elements
 * @param type Activation function type
 */
void matrix_apply_activation_gpu(float* data, int size, ActivationType type);

/**
 * @brief Sum all elements in a matrix on GPU
 * @param data Matrix data (device pointer)
 * @param size Total number of elements
 * @return Sum of all elements
 */
float matrix_sum_gpu(const float* data, int size);

/**
 * @brief Batch element-wise addition on GPU
 * @param a_batch Array of input matrix pointers (device pointers)
 * @param b_batch Array of input matrix pointers (device pointers)
 * @param c_batch Array of output matrix pointers (device pointers)
 * @param batch_size Number of matrices in batch
 * @param size Elements per matrix
 */
void matrix_batch_add_gpu(const float** a_batch, const float** b_batch, float** c_batch,
                          int batch_size, int size);

/**
 * @brief Batch matrix multiplication on GPU
 * @param a_batch Array of input matrix pointers (device pointers)
 * @param b_batch Array of input matrix pointers (device pointers)
 * @param c_batch Array of output matrix pointers (device pointers)
 * @param batch_size Number of matrices in batch
 * @param m Rows in each A matrix
 * @param k Columns in each A matrix (rows in B)
 * @param n Columns in each B matrix
 */
void matrix_batch_multiply_gpu(const float** a_batch, const float** b_batch, float** c_batch,
                               int batch_size, int m, int k, int n);

// TD-003 training kernels (CUDA implementations in MatrixGPU.cu)
void matrix_add_inplace_gpu(const float* src, float* dst, int size);
void matrix_softmax_rows_gpu(float* data, int rows, int cols);
void matrix_softmax_backward_gpu(const float* s, const float* dout, float* din, int rows, int cols);
void matrix_gelu_backward_gpu(const float* pre_act, const float* dout, float* din, int size);
void matrix_layer_norm_fwd_gpu(const float* input, float* output, float* out_normed,
                               const float* gamma, const float* beta, float* out_mean,
                               float* out_rstd, int rows, int cols, float eps);
void matrix_layer_norm_bwd_gpu(const float* dout, const float* input_norm, const float* gamma,
                               const float* mean, const float* rstd, float* dx, float* dgamma,
                               float* dbeta, int rows, int cols);
void matrix_add_row_bias_gpu(const float* mat, const float* bias, float* out, int rows, int cols);
void matrix_sum_rows_gpu(const float* mat, float* out, int rows, int cols);
void matrix_masked_fill_gpu(float* data, const float* mask, float fill_val, int size);
float matrix_cross_entropy_loss_gpu(const float* logits, const int* targets, int seq_len,
                                    int vocab_size);
void matrix_cross_entropy_grad_gpu(const float* logits, const int* targets, float* grad,
                                   int seq_len, int vocab_size);

// Training-diagnostics reductions (GPU-native activation saturation / attention
// entropy — see ChatbotTrainer's gpu_activation_stats_hook_/gpu_attention_stats_hook_).
float matrix_count_below_threshold_gpu(const float* data, int size, float threshold);
void matrix_row_entropy_gpu(const float* data, float* out_row_entropy, int rows, int cols);

// Small backend-agnostic transfer helpers (replace direct GPUManager::get_queue()
// calls that only compiled under SYCL).
void matrix_copy_device_to_device_gpu(const float* src, float* dst, int count);
void matrix_download_gpu(const float* device_ptr, float* host_ptr, int count);

// ============================================================================
// GPUMatrix — persistent GPU-resident matrix (TD-003)
// ============================================================================

/**
 * @brief Persistent GPU-resident matrix.
 *
 * Keeps matrix data on the device across multiple operations, eliminating the
 * per-operation host↔device transfers incurred by Matrix::multiply_gpu() etc.
 *
 * Usage pattern (one upload, one download, N on-device ops):
 * @code
 *   auto A_gpu = A.to_gpu();
 *   auto B_gpu = B.to_gpu();
 *   auto C_gpu = A_gpu * B_gpu;           // on-device matmul (no PCIe traffic)
 *   auto D_gpu = C_gpu + A_gpu;           // on-device add
 *   auto E_gpu = D_gpu.transpose();       // on-device transpose
 *   Matrix E   = Matrix::from_gpu(E_gpu); // single download
 * @endcode
 *
 * GPUMatrix is move-only.  All device memory is managed through GPUMemory<float>
 * which automatically tracks the ADAI memory budget.
 */
class GPUMatrix {
   public:
    int rows = 0;
    int cols = 0;

   private:
    GPUMemory<float> data_;  ///< row-major device buffer

   public:
    GPUMatrix(int r, int c) : rows(r), cols(c), data_(r * c) {}

    // Move-only — GPU allocations are not trivially copyable
    GPUMatrix(const GPUMatrix&) = delete;
    GPUMatrix& operator=(const GPUMatrix&) = delete;
    GPUMatrix(GPUMatrix&&) = default;
    GPUMatrix& operator=(GPUMatrix&&) = default;

    float* device_ptr() {
        return data_.get();
    }
    const float* device_ptr() const {
        return data_.get();
    }
    int size() const {
        return rows * cols;
    }

    // ---- Host ↔ device transfers ----------------------------------------

    /** @brief Upload @p count floats from @p host_ptr (blocking). */
    void upload(const float* host_ptr, int count) {
        data_.copy_from_host(host_ptr, static_cast<size_t>(count));
    }

    /** @brief Download @p count floats to @p host_ptr (blocking). */
    void download(float* host_ptr, int count) const {
        data_.copy_to_host(host_ptr, static_cast<size_t>(count));
    }

    // ---- Device-to-device copy ------------------------------------------

    /** @brief Return a fresh on-device copy of this matrix. */
    GPUMatrix copy() const {
        GPUMatrix result(rows, cols);
        CUDA_CHECK(cudaMemcpyAsync(result.data_.get(), data_.get(),
                                   static_cast<size_t>(rows * cols) * sizeof(float),
                                   cudaMemcpyDeviceToDevice, GPUManager::get_stream()));
        return result;
    }

    // ---- On-device arithmetic -------------------------------------------

    /** @brief Matrix multiplication (C = this × other).  Uses cuBLAS SGEMM. */
    GPUMatrix operator*(const GPUMatrix& other) const {
        if (cols != other.rows)
            throw std::invalid_argument("GPUMatrix dimensions incompatible for multiply");
        GPUMatrix result(rows, other.cols);
        matrix_multiply_gpu(data_.get(), other.data_.get(), result.data_.get(), rows, cols,
                            other.cols);
        return result;
    }

    /** @brief Element-wise addition. */
    GPUMatrix operator+(const GPUMatrix& other) const {
        if (rows != other.rows || cols != other.cols)
            throw std::invalid_argument("GPUMatrix dimensions must match for add");
        GPUMatrix result(rows, cols);
        matrix_add_gpu(data_.get(), other.data_.get(), result.data_.get(), size());
        return result;
    }

    /** @brief Element-wise subtraction (implemented as a + (−1)×b). */
    GPUMatrix operator-(const GPUMatrix& other) const {
        if (rows != other.rows || cols != other.cols)
            throw std::invalid_argument("GPUMatrix dimensions must match for subtract");
        GPUMatrix neg_b(rows, cols);
        matrix_multiply_scalar_gpu(other.data_.get(), -1.0f, neg_b.data_.get(), size());
        GPUMatrix result(rows, cols);
        matrix_add_gpu(data_.get(), neg_b.data_.get(), result.data_.get(), size());
        return result;
    }

    /** @brief Scalar multiplication. */
    GPUMatrix scale(float scalar) const {
        GPUMatrix result(rows, cols);
        matrix_multiply_scalar_gpu(data_.get(), scalar, result.data_.get(), size());
        return result;
    }

    /** @brief Element-wise (Hadamard) multiplication. */
    GPUMatrix hadamard(const GPUMatrix& other) const {
        if (rows != other.rows || cols != other.cols)
            throw std::invalid_argument("GPUMatrix dimensions must match for hadamard");
        GPUMatrix result(rows, cols);
        matrix_multiply_elementwise_gpu(data_.get(), other.data_.get(), result.data_.get(), size());
        return result;
    }

    /** @brief Matrix transpose. */
    GPUMatrix transpose() const {
        GPUMatrix result(cols, rows);
        matrix_transpose_gpu(data_.get(), result.data_.get(), rows, cols);
        return result;
    }

    /** @brief Apply activation function in-place on device. */
    void apply_activation_inplace(ActivationType type) {
        matrix_apply_activation_gpu(data_.get(), size(), type);
    }

    /** @brief Sum all elements on device and return scalar to host. */
    float sum() const {
        return matrix_sum_gpu(data_.get(), size());
    }

    // ---- TD-003 persistent-training operations ----------------------------

    void zero() {
        CUDA_CHECK(cudaMemsetAsync(data_.get(), 0, static_cast<size_t>(rows * cols) * sizeof(float),
                                   GPUManager::get_stream()));
    }

    void add_inplace(const GPUMatrix& other) {
        matrix_add_inplace_gpu(other.data_.get(), data_.get(), size());
    }

    void softmax_rows_inplace() {
        matrix_softmax_rows_gpu(data_.get(), rows, cols);
    }

    GPUMatrix softmax_backward(const GPUMatrix& dout) const {
        GPUMatrix result(rows, cols);
        matrix_softmax_backward_gpu(data_.get(), dout.data_.get(), result.data_.get(), rows, cols);
        return result;
    }

    GPUMatrix gelu_backward(const GPUMatrix& dout) const {
        GPUMatrix result(rows, cols);
        matrix_gelu_backward_gpu(data_.get(), dout.data_.get(), result.data_.get(), size());
        return result;
    }

    GPUMatrix add_row_bias(const GPUMatrix& bias) const {
        GPUMatrix result(rows, cols);
        matrix_add_row_bias_gpu(data_.get(), bias.data_.get(), result.data_.get(), rows, cols);
        return result;
    }

    GPUMatrix sum_rows() const {
        GPUMatrix result(1, cols);
        matrix_sum_rows_gpu(data_.get(), result.data_.get(), rows, cols);
        return result;
    }

    void masked_fill_inplace(const GPUMatrix& mask, float fill_val) {
        matrix_masked_fill_gpu(data_.get(), mask.data_.get(), fill_val, size());
    }

    GPUMatrix layer_norm(const GPUMatrix& gamma, const GPUMatrix& beta, float eps,
                         GPUMatrix& out_normed, GPUMatrix& out_mean, GPUMatrix& out_rstd) const {
        GPUMatrix result(rows, cols);
        matrix_layer_norm_fwd_gpu(data_.get(), result.data_.get(), out_normed.data_.get(),
                                  gamma.data_.get(), beta.data_.get(), out_mean.data_.get(),
                                  out_rstd.data_.get(), rows, cols, eps);
        return result;
    }

    GPUMatrix layer_norm_backward(const GPUMatrix& input_norm, const GPUMatrix& gamma,
                                  const GPUMatrix& mean, const GPUMatrix& rstd, GPUMatrix& dgamma,
                                  GPUMatrix& dbeta) const {
        GPUMatrix dx(rows, cols);
        matrix_layer_norm_bwd_gpu(data_.get(), input_norm.data_.get(), gamma.data_.get(),
                                  mean.data_.get(), rstd.data_.get(), dx.data_.get(),
                                  dgamma.data_.get(), dbeta.data_.get(), rows, cols);
        return dx;
    }

    // ---- Training-diagnostics reductions (activation saturation / attention entropy) --

    /** @brief Fraction of elements with |x| < threshold (e.g. post-GELU saturation). */
    float count_below_threshold(float threshold) const {
        return matrix_count_below_threshold_gpu(data_.get(), size(), threshold) /
               static_cast<float>(size());
    }

    /** @brief Average per-row Shannon entropy of an already-normalized (e.g. post-softmax) matrix.
     */
    float row_entropy_avg() const {
        GPUMemory<float> row_ent(rows);
        matrix_row_entropy_gpu(data_.get(), row_ent.get(), rows, cols);
        return matrix_sum_gpu(row_ent.get(), rows) / static_cast<float>(rows);
    }
};

}  // namespace gpu
}  // namespace adai

#endif  // CUDA backend

namespace adai {
namespace gpu {

// ============================================================================
// GPUKVCache / GPUDecoderKVCache — GPU-resident KV cache for incremental
// autoregressive decoding (TD-050)
// ============================================================================

/**
 * @brief GPU-resident key-value cache for one attention layer's incremental decode (TD-050).
 *
 * Backend-agnostic by construction: placed here (outside the CUDA-vs-SYCL `#if`/`#else` split
 * above) and built entirely from GPUMatrix's own public interface plus the existing
 * matrix_copy_device_to_device_gpu() primitive — the same one MultiHeadAttention's
 * gpu_slice_head_columns()/gpu_scatter_head_columns() helpers already use for TD-059's per-head
 * GPU attention — so this single definition compiles unchanged under either the CUDA or SYCL
 * backend with no new low-level kernels. That matters because this environment can compile but
 * not runtime-verify either GPU backend (no physical GPU device available) — see TD-059's own
 * writeup in TECHNICAL_DEBT_RESOLVED.md for that residual verification gap, which applies here
 * identically: correctness rests on primitives already exercised elsewhere, not on new kernel
 * code with zero ability to catch a bug in it.
 *
 * Unlike the CPU KVCache (KVCache.hpp), which reallocates a larger Matrix on every append() call,
 * this pre-allocates a [max_seq_length, d_model] device buffer once at construction and grows
 * current_length_ in place — no per-decode-step device malloc/free churn, per this item's own
 * action item. get_keys()/get_values() materialize a freshly-sized [current_length, d_model]
 * GPUMatrix via a single bulk device-to-device copy of the populated prefix (rows are contiguous
 * in this row-major layout, so this is one copy, not the per-row loop TD-059's column-slicing
 * helpers need for non-contiguous column ranges).
 */
class GPUKVCache {
   public:
    GPUKVCache(int max_seq_length, int d_model)
        : max_seq_length_(max_seq_length),
          d_model_(d_model),
          keys_(max_seq_length, d_model),
          values_(max_seq_length, d_model),
          current_length_(0) {
        if (max_seq_length <= 0 || d_model <= 0) {
            throw std::invalid_argument("GPUKVCache: max_seq_length and d_model must be positive");
        }
    }

    // Move-only — owns device memory via GPUMatrix, which is itself move-only.
    GPUKVCache(const GPUKVCache&) = delete;
    GPUKVCache& operator=(const GPUKVCache&) = delete;
    GPUKVCache(GPUKVCache&&) = default;
    GPUKVCache& operator=(GPUKVCache&&) = default;

    bool is_empty() const {
        return current_length_ == 0;
    }
    int size() const {
        return current_length_;
    }
    int capacity() const {
        return max_seq_length_;
    }

    /** @brief Rewind to empty without freeing/reallocating the underlying device buffer. */
    void clear() {
        current_length_ = 0;
    }

    /**
     * @brief Append new_keys/new_values (each [num_new, d_model], already device-resident) at
     * the current write position.
     * @throws std::invalid_argument on shape mismatch, std::out_of_range if this would exceed
     *   the max_seq_length capacity supplied at construction.
     */
    void append(const GPUMatrix& new_keys, const GPUMatrix& new_values) {
        if (new_keys.rows != new_values.rows || new_keys.cols != d_model_ ||
            new_values.cols != d_model_) {
            throw std::invalid_argument("GPUKVCache::append: shape mismatch");
        }
        if (current_length_ + new_keys.rows > max_seq_length_) {
            throw std::out_of_range("GPUKVCache::append: exceeds max_seq_length capacity");
        }
        if (new_keys.rows > 0) {
            matrix_copy_device_to_device_gpu(new_keys.device_ptr(),
                                             keys_.device_ptr() + current_length_ * d_model_,
                                             new_keys.rows * d_model_);
            matrix_copy_device_to_device_gpu(new_values.device_ptr(),
                                             values_.device_ptr() + current_length_ * d_model_,
                                             new_values.rows * d_model_);
        }
        current_length_ += new_keys.rows;
    }

    /** @brief Materialize the populated [current_length, d_model] prefix as a fresh GPUMatrix. */
    GPUMatrix get_keys() const {
        return materialize(keys_);
    }
    /** @brief Materialize the populated [current_length, d_model] prefix as a fresh GPUMatrix. */
    GPUMatrix get_values() const {
        return materialize(values_);
    }

   private:
    GPUMatrix materialize(const GPUMatrix& buffer) const {
        GPUMatrix result(current_length_, d_model_);
        if (current_length_ > 0) {
            matrix_copy_device_to_device_gpu(buffer.device_ptr(), result.device_ptr(),
                                             current_length_ * d_model_);
        }
        return result;
    }

    int max_seq_length_;
    int d_model_;
    GPUMatrix keys_;
    GPUMatrix values_;
    int current_length_;
};

/**
 * @brief GPU-resident multi-layer KV cache for the full decoder (TD-050) — mirrors
 * DecoderKVCache's (KVCache.hpp) shape: one self-attention cache and one cross-attention cache
 * per decoder layer. Cross-attention caches hold the encoder's K/V projections, computed once on
 * the first decode step and reused unchanged for the rest of generation (encoder output never
 * changes across decode steps) — callers populate a layer's cross-attention cache via a single
 * append() with the full encoder sequence, then never append to it again.
 */
class GPUDecoderKVCache {
   public:
    GPUDecoderKVCache(int num_layers, int max_seq_length, int d_model) {
        self_attention_caches_.reserve(num_layers);
        cross_attention_caches_.reserve(num_layers);
        for (int i = 0; i < num_layers; ++i) {
            self_attention_caches_.emplace_back(max_seq_length, d_model);
            // Cross-attention K/V come from the encoder output, whose length is bounded by the
            // same MAX_SEQ_LENGTH config value this codebase already applies to both encoder and
            // decoder sequences — reusing it here as a safe, simple capacity upper bound.
            cross_attention_caches_.emplace_back(max_seq_length, d_model);
        }
    }

    GPUDecoderKVCache(const GPUDecoderKVCache&) = delete;
    GPUDecoderKVCache& operator=(const GPUDecoderKVCache&) = delete;
    GPUDecoderKVCache(GPUDecoderKVCache&&) = default;
    GPUDecoderKVCache& operator=(GPUDecoderKVCache&&) = default;

    GPUKVCache& self_attention_cache(int layer_idx) {
        return self_attention_caches_[layer_idx];
    }
    GPUKVCache& cross_attention_cache(int layer_idx) {
        return cross_attention_caches_[layer_idx];
    }

    void clear() {
        for (auto& c : self_attention_caches_) {
            c.clear();
        }
        for (auto& c : cross_attention_caches_) {
            c.clear();
        }
    }
    /** @brief Clear self-attention caches only (keep the one-time cross-attention K/V). */
    void clear_self_attention() {
        for (auto& c : self_attention_caches_) {
            c.clear();
        }
    }

    bool is_empty() const {
        for (const auto& c : self_attention_caches_) {
            if (!c.is_empty()) {
                return false;
            }
        }
        return true;
    }

    /** @brief Current sequence length, read from the first layer's self-attention cache. */
    int current_length() const {
        return self_attention_caches_.empty() ? 0 : self_attention_caches_[0].size();
    }

   private:
    std::vector<GPUKVCache> self_attention_caches_;
    std::vector<GPUKVCache> cross_attention_caches_;
};

}  // namespace gpu
}  // namespace adai

#endif  // ADAI_ENABLE_GPU

#endif  // MATRIX_GPU_HPP
