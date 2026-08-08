#ifndef MATRIX_GPU_HPP
#define MATRIX_GPU_HPP

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

#endif  // ADAI_ENABLE_GPU

#endif  // MATRIX_GPU_HPP
