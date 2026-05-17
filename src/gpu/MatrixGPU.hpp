#ifndef MATRIX_GPU_HPP
#define MATRIX_GPU_HPP

#ifdef ADAI_ENABLE_GPU

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
};

}  // namespace gpu
}  // namespace adai

#endif  // ADAI_ENABLE_GPU

#endif  // MATRIX_GPU_HPP
