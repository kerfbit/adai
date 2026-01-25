#pragma once

#include <cmath>
#include <iostream>
#include <random>
#include <stdexcept>
#include <vector>

/**
 * Matrix class for tensor operations in neural networks
 *
 * Provides basic matrix operations needed for neural network computations
 * including multiplication, addition, subtraction, transpose, and gradient operations.
 */
class Matrix {
   public:
    std::vector<std::vector<float>> data;
    int rows, cols;

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
};
