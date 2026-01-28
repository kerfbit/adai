#ifndef MATRIX_GPU_HPP
#define MATRIX_GPU_HPP

#ifdef ADAI_ENABLE_GPU

namespace adai {
namespace gpu {

/**
 * @brief Activation function types for GPU operations
 */
enum class ActivationType {
    RELU = 0,
    SIGMOID = 1,
    TANH = 2,
    GELU = 3
};

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
void matrix_multiply_gpu(const float* a, const float* b, float* c, 
                        int m, int k, int n);

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
void matrix_batch_add_gpu(const float** a_batch, const float** b_batch, 
                         float** c_batch, int batch_size, int size);

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
void matrix_batch_multiply_gpu(const float** a_batch, const float** b_batch,
                              float** c_batch, int batch_size,
                              int m, int k, int n);

} // namespace gpu
} // namespace adai

#endif // ADAI_ENABLE_GPU

#endif // MATRIX_GPU_HPP
