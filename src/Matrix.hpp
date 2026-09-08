#pragma once

// @adai-status: beta        (capped by TD-033 — GPU dispatch still round-trips per op, see TECHNICAL_DEBT.md)
// @adai-version: 0.9.0
// @adai-reviewed: 2026-09-07


#include <cmath>
#include <iostream>
#include <random>
#include <stdexcept>
#include <vector>

#include "MatrixSIMD.hpp"

#ifdef ADAI_ENABLE_GPU
#include "gpu/GPUUtils.hpp"
#include "gpu/MatrixGPU.hpp"
#endif

/**
 * Matrix class for tensor operations in neural networks
 *
 * Provides basic matrix operations needed for neural network computations
 * including multiplication, addition, subtraction, transpose, and gradient operations.
 *
 * When compiled with GPU support (ENABLE_GPU=ON), provides GPU-accelerated operations.
 */
class Matrix {
   public:
    std::vector<std::vector<float>> data;
    int rows, cols;

    Matrix(const Matrix&) = default;
    Matrix& operator=(const Matrix&) = default;
    Matrix(Matrix&& other) noexcept;
    Matrix& operator=(Matrix&& other) noexcept;

    /**
     * Constructor: Create a matrix with given dimensions, initialized to zero
     *
     * @param r Number of rows
     * @param c Number of columns
     */
    Matrix(int r, int c);

    /**
     * Constructor: Create a matrix from existing 2D vector data
     *
     * @param d 2D vector containing matrix data
     */
    Matrix(const std::vector<std::vector<float>>& d);

    /**
     * Default constructor: Create an empty matrix
     */
    Matrix();

    /**
     * Element access operator (non-const)
     *
     * @param i Row index
     * @param j Column index
     * @return Reference to element at (i, j)
     */
    float& operator()(int i, int j);

    /**
     * Element access operator (const)
     *
     * @param i Row index
     * @param j Column index
     * @return Const reference to element at (i, j)
     */
    const float& operator()(int i, int j) const;

    /**
     * Matrix multiplication
     *
     * @param other Matrix to multiply with (this * other)
     * @return Result matrix of dimension [this.rows, other.cols]
     */
    Matrix operator*(const Matrix& other) const;

    /**
     * Matrix addition
     *
     * @param other Matrix to add
     * @return Result matrix (element-wise addition)
     */
    Matrix operator+(const Matrix& other) const;

    /**
     * Matrix subtraction
     *
     * @param other Matrix to subtract
     * @return Result matrix (element-wise subtraction)
     */
    Matrix operator-(const Matrix& other) const;

    /**
     * Matrix transpose
     *
     * @return Transposed matrix [cols, rows]
     */
    Matrix transpose() const;

    /**
     * Initialize matrix with random values using Xavier/He initialization
     *
     * @param scale Scaling factor for random values (default 0.1)
     */
    void randomize(float scale = 0.1f);

    /**
     * Scalar multiplication
     *
     * @param scalar Value to multiply each element by
     * @return Result matrix with all elements scaled
     */
    Matrix scale(float scalar) const;

    /**
     * Hadamard product (element-wise multiplication)
     *
     * @param other Matrix to multiply element-wise
     * @return Result matrix with element-wise products
     */
    Matrix hadamard(const Matrix& other) const;

    /**
     * Apply gradient update to matrix (used in backpropagation)
     * W = W - learning_rate * gradients
     *
     * @param gradients Gradient matrix
     * @param learning_rate Learning rate for update
     */
    void apply_gradients(const Matrix& gradients, float learning_rate);

    /**
     * Fill matrix with a constant value
     *
     * @param value Value to fill matrix with
     */
    void fill(float value);

    /**
     * Get sum of all elements in matrix
     *
     * @return Sum of all elements
     */
    float sum() const;

    /**
     * Get mean of all elements in matrix
     *
     * @return Mean of all elements
     */
    float mean() const;

    /**
     * Print matrix to console (for debugging)
     *
     * @param name Optional name to display
     * @param max_rows Maximum rows to print (default 10)
     * @param max_cols Maximum cols to print (default 10)
     */
    void print(const std::string& name = "", int max_rows = 10, int max_cols = 10) const;

    /**
     * Check if matrix dimensions are valid
     *
     * @return True if matrix has valid dimensions
     */
    bool is_valid() const;

    /**
     * Reshape matrix (if total elements match)
     *
     * @param new_rows New number of rows
     * @param new_cols New number of columns
     * @return Reshaped matrix
     */
    Matrix reshape(int new_rows, int new_cols) const;

    /**
     * Get a row as a vector
     *
     * @param row_idx Row index
     * @return Vector containing row data
     */
    std::vector<float> get_row(int row_idx) const;

    /**
     * Get a column as a vector
     *
     * @param col_idx Column index
     * @return Vector containing column data
     */
    std::vector<float> get_col(int col_idx) const;

    /**
     * Set a row from a vector
     *
     * @param row_idx Row index
     * @param values Vector of values to set
     */
    void set_row(int row_idx, const std::vector<float>& values);

    /**
     * Set a column from a vector
     *
     * @param col_idx Column index
     * @param values Vector of values to set
     */
    void set_col(int col_idx, const std::vector<float>& values);

    // -----------------------------------------------------------------------
    // GPU management — always declared so callers compile on CPU-only hosts.
    // When GPU support is not compiled in, these are inline no-ops / stubs.
    // -----------------------------------------------------------------------

    /**
     * Check if GPU acceleration is available and initialised.
     * Always returns false when compiled without ENABLE_GPU=ON.
     */
    static bool gpu_available();

    /**
     * Initialize GPU subsystem (must be called before using GPU operations).
     * @param device_id       CUDA device index to use (default: 0).
     * @param memory_fraction Fraction of total device memory ADAI may allocate
     *                        (0.0–1.0, default: 0.5).  Lower values leave more
     *                        headroom for other GPU tenants.
     * @return true  if the GPU was successfully initialised.
     * @return false if no CUDA device is present (operations will use CPU).
     * @throws std::runtime_error for unexpected CUDA initialisation errors.
     */
    static bool gpu_initialize(int device_id = 0, float memory_fraction = 0.5f,
                               bool use_low_priority = true);

    /**
     * Attempt GPU initialisation and silently fall back to CPU on any failure.
     *
     * Never throws.  Returns false immediately on CPU-only builds.
     * @param device_id        CUDA device index (default: 0).
     * @param memory_fraction  Memory budget fraction (default: 0.5).
     * @param use_low_priority true = low-priority stream (background mode);
     *                         false = high-priority stream (full mode).
     * @return true  if GPU is ready, false if CPU-only mode is in effect.
     */
    static bool gpu_try_initialize(int device_id = 0, float memory_fraction = 0.5f,
                                   bool use_low_priority = true);

    /**
     * Release all GPU resources owned by ADAI.  No-op on CPU-only builds.
     */
    static void gpu_cleanup();

    /**
     * Human-readable description of the selected GPU device and memory budget.
     * Returns a placeholder string on CPU-only builds.
     * @param device Device ID to describe (-1 = current device).
     */
    static std::string gpu_info(int device = -1);

#ifdef ADAI_ENABLE_GPU
    /**
     * GPU-accelerated matrix operations.
     * All _gpu methods fall back to their CPU equivalents when
     * gpu_available() returns false (i.e. no GPU was found at init time).
     */

    /**
     * Matrix multiplication using GPU (C = this * other).
     * Falls back to CPU multiply() if GPU is not available.
     * @param other Matrix to multiply with
     * @return Result matrix
     */
    Matrix multiply_gpu(const Matrix& other) const;

    /**
     * Matrix addition using GPU (C = this + other).
     * Falls back to CPU operator+() if GPU is not available.
     * @param other Matrix to add
     * @return Result matrix
     */
    Matrix add_gpu(const Matrix& other) const;

    /**
     * Matrix transpose using GPU.
     * Falls back to CPU transpose() if GPU is not available.
     * @return Transposed matrix
     */
    Matrix transpose_gpu() const;

    /**
     * Scalar multiplication using GPU.
     * Falls back to CPU scale() if GPU is not available.
     * @param scalar Value to multiply each element by
     * @return Result matrix
     */
    Matrix scale_gpu(float scalar) const;

    /**
     * Element-wise multiplication using GPU.
     * Falls back to CPU hadamard() if GPU is not available.
     * @param other Matrix to multiply element-wise
     * @return Result matrix
     */
    Matrix hadamard_gpu(const Matrix& other) const;

    /**
     * Upload this matrix to GPU-resident persistent storage.
     *
     * The returned GPUMatrix holds a device-side copy.  Chain arithmetic on
     * it without any further host↔device transfers, then call Matrix::from_gpu()
     * to retrieve the final result.
     *
     * Example:
     * @code
     *   auto A_gpu = A.to_gpu();
     *   auto B_gpu = B.to_gpu();
     *   auto C_gpu = A_gpu * B_gpu;           // on-device, no PCIe traffic
     *   auto D_gpu = C_gpu + A_gpu;
     *   Matrix D   = Matrix::from_gpu(D_gpu); // single download
     * @endcode
     *
     * @throws std::runtime_error if GPU has not been initialised.
     */
    adai::gpu::GPUMatrix to_gpu() const;

    /**
     * Download a GPU-resident matrix back to CPU host memory.
     * @param gm Source GPUMatrix on the device.
     * @return CPU Matrix with the same data and dimensions.
     */
    static Matrix from_gpu(const adai::gpu::GPUMatrix& gm);
#endif
};
