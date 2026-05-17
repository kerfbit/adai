#pragma once

#include <gtest/gtest.h>
#include <cmath>
#include <vector>
#include "../src/Activation.hpp"
#include "../src/Matrix.hpp"

/**
 * @brief Base test fixture for transformer-related tests
 *
 * Provides common setup, teardown, and utility methods for testing
 * transformer components, reducing code duplication across test files.
 */
class TransformerTestBase : public ::testing::Test {
   protected:
    void SetUp() override {
        // Common test parameters
        test_d_model = 64;
        test_num_heads = 4;
        test_seq_len = 10;
        test_vocab_size = 100;
        test_batch_size = 1;
        test_d_ff = 256;
        test_epsilon = 1e-5f;
    }

    void TearDown() override {
        // Common cleanup if needed
    }

    // ========================================================================
    // Matrix Creation Helpers
    // ========================================================================

    /**
     * @brief Create a matrix filled with a constant value
     */
    Matrix create_test_matrix(int rows, int cols, float fill_value = 0.0f) {
        Matrix m(rows, cols);
        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < cols; ++j) {
                m.data[i][j] = fill_value;
            }
        }
        return m;
    }

    /**
     * @brief Create a matrix with random values in range [min, max]
     */
    Matrix create_random_matrix(int rows, int cols, float min = -1.0f, float max = 1.0f) {
        Matrix m(rows, cols);
        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < cols; ++j) {
                float random_val =
                    min + static_cast<float>(rand()) / (static_cast<float>(RAND_MAX / (max - min)));
                m.data[i][j] = random_val;
            }
        }
        return m;
    }

    /**
     * @brief Create an identity matrix
     */
    Matrix create_identity_matrix(int size) {
        Matrix m(size, size);
        for (int i = 0; i < size; ++i) {
            for (int j = 0; j < size; ++j) {
                m.data[i][j] = (i == j) ? 1.0f : 0.0f;
            }
        }
        return m;
    }

    /**
     * @brief Create a matrix with sequential values (for testing)
     */
    Matrix create_sequential_matrix(int rows, int cols, float start = 0.0f) {
        Matrix m(rows, cols);
        float val = start;
        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < cols; ++j) {
                m.data[i][j] = val++;
            }
        }
        return m;
    }

    // ========================================================================
    // Matrix Comparison Helpers
    // ========================================================================

    /**
     * @brief Check if two matrices are approximately equal within epsilon
     */
    bool matrices_near(const Matrix& a, const Matrix& b, float epsilon = 1e-5f) {
        if (a.rows != b.rows || a.cols != b.cols) {
            return false;
        }

        for (int i = 0; i < a.rows; ++i) {
            for (int j = 0; j < a.cols; ++j) {
                if (std::abs(a.data[i][j] - b.data[i][j]) > epsilon) {
                    return false;
                }
            }
        }
        return true;
    }

    /**
     * @brief Check if matrix contains only finite values (no NaN or Inf)
     */
    bool is_matrix_finite(const Matrix& m) {
        for (int i = 0; i < m.rows; ++i) {
            for (int j = 0; j < m.cols; ++j) {
                if (!std::isfinite(m.data[i][j])) {
                    return false;
                }
            }
        }
        return true;
    }

    /**
     * @brief Check if matrix has expected shape
     */
    bool has_shape(const Matrix& m, int expected_rows, int expected_cols) {
        return m.rows == expected_rows && m.cols == expected_cols;
    }

    /**
     * @brief Get maximum absolute difference between two matrices
     */
    float max_abs_diff(const Matrix& a, const Matrix& b) {
        if (a.rows != b.rows || a.cols != b.cols) {
            return std::numeric_limits<float>::infinity();
        }

        float max_diff = 0.0f;
        for (int i = 0; i < a.rows; ++i) {
            for (int j = 0; j < a.cols; ++j) {
                float diff = std::abs(a.data[i][j] - b.data[i][j]);
                if (diff > max_diff) {
                    max_diff = diff;
                }
            }
        }
        return max_diff;
    }

    // ========================================================================
    // Statistical Helpers
    // ========================================================================

    /**
     * @brief Calculate mean of all matrix values
     */
    float matrix_mean(const Matrix& m) {
        float sum = 0.0f;
        for (int i = 0; i < m.rows; ++i) {
            for (int j = 0; j < m.cols; ++j) {
                sum += m.data[i][j];
            }
        }
        return sum / (m.rows * m.cols);
    }

    /**
     * @brief Calculate variance of all matrix values
     */
    float matrix_variance(const Matrix& m) {
        float mean = matrix_mean(m);
        float sum_sq_diff = 0.0f;

        for (int i = 0; i < m.rows; ++i) {
            for (int j = 0; j < m.cols; ++j) {
                float diff = m.data[i][j] - mean;
                sum_sq_diff += diff * diff;
            }
        }
        return sum_sq_diff / (m.rows * m.cols);
    }

    /**
     * @brief Calculate L2 norm (Frobenius norm) of matrix
     */
    float matrix_norm(const Matrix& m) {
        float sum_sq = 0.0f;
        for (int i = 0; i < m.rows; ++i) {
            for (int j = 0; j < m.cols; ++j) {
                sum_sq += m.data[i][j] * m.data[i][j];
            }
        }
        return std::sqrt(sum_sq);
    }

    // ========================================================================
    // Gradient Checking Helpers
    // ========================================================================

    /**
     * @brief Numerical gradient check helper
     * Computes numerical gradient using finite differences
     */
    float compute_numerical_gradient(std::function<float(float)> loss_fn, float param_value,
                                     float epsilon = 1e-4f) {
        float loss_plus = loss_fn(param_value + epsilon);
        float loss_minus = loss_fn(param_value - epsilon);
        return (loss_plus - loss_minus) / (2.0f * epsilon);
    }

    // ========================================================================
    // Common Test Parameters
    // ========================================================================

    int test_d_model;     // Model dimension
    int test_num_heads;   // Number of attention heads
    int test_seq_len;     // Sequence length
    int test_vocab_size;  // Vocabulary size
    int test_batch_size;  // Batch size
    int test_d_ff;        // Feed-forward dimension
    float test_epsilon;   // Comparison epsilon
};

/**
 * @brief Base test fixture for optimizer-related tests
 */
class OptimizerTestBase : public ::testing::Test {
   protected:
    void SetUp() override {
        test_lr = 0.01f;
        test_momentum = 0.9f;
        test_beta1 = 0.9f;
        test_beta2 = 0.999f;
        test_epsilon = 1e-8f;
        test_weight_decay = 0.01f;
    }

    Matrix create_test_weights(int rows, int cols) {
        Matrix m(rows, cols);
        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < cols; ++j) {
                m.data[i][j] = static_cast<float>(rand()) / RAND_MAX * 0.1f;
            }
        }
        return m;
    }

    Matrix create_test_gradients(int rows, int cols) {
        Matrix m(rows, cols);
        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < cols; ++j) {
                m.data[i][j] = static_cast<float>(rand()) / RAND_MAX * 0.01f;
            }
        }
        return m;
    }

    float test_lr;
    float test_momentum;
    float test_beta1;
    float test_beta2;
    float test_epsilon;
    float test_weight_decay;
};

/**
 * @brief Base test fixture for NLP-related tests
 */
class NLPTestBase : public ::testing::Test {
   protected:
    void SetUp() override {
        test_vocab_size = 1000;
        test_max_seq_len = 512;
        test_pad_token = 0;
        test_unk_token = 1;
        test_bos_token = 2;
        test_eos_token = 3;
    }

    std::vector<int> create_test_sequence(int length) {
        std::vector<int> seq;
        for (int i = 0; i < length; ++i) {
            seq.push_back(rand() % test_vocab_size);
        }
        return seq;
    }

    int test_vocab_size;
    int test_max_seq_len;
    int test_pad_token;
    int test_unk_token;
    int test_bos_token;
    int test_eos_token;
};
