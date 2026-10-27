// @adai-status: stable
// @adai-version: 1.1.0
// @adai-reviewed: 2026-10-26

#include "Activation.hpp"
#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>

void Activation::softmax_row(const float* in, float* out, int n) {
    if (n <= 0) {
        return;
    }
    // Max subtraction for numerical stability
    float max_val = in[0];
    for (int j = 1; j < n; ++j) {
        max_val = std::max(max_val, in[j]);
    }
    float sum = 0.0f;
    for (int j = 0; j < n; ++j) {
        out[j] = std::exp(in[j] - max_val);
        sum += out[j];
    }
    if (sum > 0.0f) {
        for (int j = 0; j < n; ++j) {
            out[j] /= sum;
        }
    }
}

void Activation::softmax_row_backward(const float* y, const float* dy, float* dx, int n) {
    float sum = 0.0f;
    for (int k = 0; k < n; ++k) {
        sum += y[k] * dy[k];
    }
    for (int j = 0; j < n; ++j) {
        dx[j] = y[j] * (dy[j] - sum);
    }
}

float Activation::log_sum_exp_row(const float* row, int n) {
    if (n <= 0) {
        return -std::numeric_limits<float>::infinity();
    }
    float max_val = row[0];
    for (int j = 1; j < n; ++j) {
        max_val = std::max(max_val, row[j]);
    }
    float sum = 0.0f;
    for (int j = 0; j < n; ++j) {
        sum += std::exp(row[j] - max_val);
    }
    return max_val + std::log(sum);
}

// Softmax activation
Matrix Activation::softmax(const Matrix& input) {
    Matrix result(input.rows, input.cols);
    // TD-215: a matrix with no columns used to read input(i, 0) out of bounds.
    for (int i = 0; i < input.rows && input.cols > 0; i++) {
        softmax_row(input.data[i].data(), result.data[i].data(), input.cols);
    }
    return result;
}

// GELU activation (using tanh approximation)
Matrix Activation::gelu(const Matrix& input) {
    Matrix result(input.rows, input.cols);

    for (int i = 0; i < input.rows; i++) {
        for (int j = 0; j < input.cols; j++) {
            float x = input(i, j);

            // GELU approximation: 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³)))
            float x_cubed = x * x * x;
            float inner = SQRT_2_OVER_PI * (x + GELU_COEF * x_cubed);
            float tanh_inner = std::tanh(inner);

            result(i, j) = 0.5f * x * (1.0f + tanh_inner);
        }
    }

    return result;
}

// GELU derivative
Matrix Activation::gelu_derivative(const Matrix& input) {
    Matrix result(input.rows, input.cols);

    for (int i = 0; i < input.rows; i++) {
        for (int j = 0; j < input.cols; j++) {
            float x = input(i, j);

            // Derivative of GELU approximation
            float x_squared = x * x;
            float x_cubed = x_squared * x;

            float inner = SQRT_2_OVER_PI * (x + GELU_COEF * x_cubed);
            float tanh_inner = std::tanh(inner);
            float sech_squared = 1.0f - tanh_inner * tanh_inner;

            float tanh_derivative = SQRT_2_OVER_PI * (1.0f + 3.0f * GELU_COEF * x_squared);

            // d/dx[0.5 * x * (1 + tanh(inner))]
            float derivative =
                0.5f * (1.0f + tanh_inner) + 0.5f * x * sech_squared * tanh_derivative;

            result(i, j) = derivative;
        }
    }

    return result;
}

// Softmax derivative: the general softmax vector-Jacobian product (TD-215: not a cross-entropy shortcut)
Matrix Activation::softmax_derivative(const Matrix& output, const Matrix& grad_output) {
    // TD-215: a smaller grad_output used to be read out of bounds.
    if (output.rows != grad_output.rows || output.cols != grad_output.cols) {
        throw std::invalid_argument("Activation::softmax_derivative: output is " +
                                    std::to_string(output.rows) + "x" +
                                    std::to_string(output.cols) + ", grad_output is " +
                                    std::to_string(grad_output.rows) + "x" +
                                    std::to_string(grad_output.cols));
    }
    Matrix result(output.rows, output.cols);
    for (int i = 0; i < output.rows; i++) {
        softmax_row_backward(output.data[i].data(), grad_output.data[i].data(),
                             result.data[i].data(), output.cols);
    }
    return result;
}

// ReLU activation
Matrix Activation::relu(const Matrix& input) {
    Matrix result(input.rows, input.cols);

    for (int i = 0; i < input.rows; i++) {
        for (int j = 0; j < input.cols; j++) {
            result(i, j) = std::max(0.0f, input(i, j));
        }
    }

    return result;
}

// ReLU derivative
Matrix Activation::relu_derivative(const Matrix& input) {
    Matrix result(input.rows, input.cols);

    for (int i = 0; i < input.rows; i++) {
        for (int j = 0; j < input.cols; j++) {
            result(i, j) = (input(i, j) > 0.0f) ? 1.0f : 0.0f;
        }
    }

    return result;
}

// Sigmoid activation
Matrix Activation::sigmoid(const Matrix& input) {
    Matrix result(input.rows, input.cols);

    for (int i = 0; i < input.rows; i++) {
        for (int j = 0; j < input.cols; j++) {
            // Use numerically stable sigmoid
            float x = input(i, j);
            if (x >= 0) {
                float exp_neg_x = std::exp(-x);
                result(i, j) = 1.0f / (1.0f + exp_neg_x);
            } else {
                float exp_x = std::exp(x);
                result(i, j) = exp_x / (1.0f + exp_x);
            }
        }
    }

    return result;
}

// Sigmoid derivative
Matrix Activation::sigmoid_derivative(const Matrix& output) {
    Matrix result(output.rows, output.cols);

    for (int i = 0; i < output.rows; i++) {
        for (int j = 0; j < output.cols; j++) {
            float sig = output(i, j);
            result(i, j) = sig * (1.0f - sig);
        }
    }

    return result;
}

// Tanh activation
Matrix Activation::tanh(const Matrix& input) {
    Matrix result(input.rows, input.cols);

    for (int i = 0; i < input.rows; i++) {
        for (int j = 0; j < input.cols; j++) {
            result(i, j) = std::tanh(input(i, j));
        }
    }

    return result;
}

// Tanh derivative
Matrix Activation::tanh_derivative(const Matrix& output) {
    Matrix result(output.rows, output.cols);

    for (int i = 0; i < output.rows; i++) {
        for (int j = 0; j < output.cols; j++) {
            float tanh_val = output(i, j);
            result(i, j) = 1.0f - tanh_val * tanh_val;
        }
    }

    return result;
}

// Leaky ReLU activation
Matrix Activation::leaky_relu(const Matrix& input, float alpha) {
    Matrix result(input.rows, input.cols);

    for (int i = 0; i < input.rows; i++) {
        for (int j = 0; j < input.cols; j++) {
            float x = input(i, j);
            result(i, j) = (x > 0.0f) ? x : alpha * x;
        }
    }

    return result;
}

// Leaky ReLU derivative
Matrix Activation::leaky_relu_derivative(const Matrix& input, float alpha) {
    Matrix result(input.rows, input.cols);

    for (int i = 0; i < input.rows; i++) {
        for (int j = 0; j < input.cols; j++) {
            result(i, j) = (input(i, j) > 0.0f) ? 1.0f : alpha;
        }
    }

    return result;
}

// Swish/SiLU activation
Matrix Activation::swish(const Matrix& input) {
    Matrix sig = sigmoid(input);
    Matrix result(input.rows, input.cols);

    for (int i = 0; i < input.rows; i++) {
        for (int j = 0; j < input.cols; j++) {
            result(i, j) = input(i, j) * sig(i, j);
        }
    }

    return result;
}

// Swish derivative
Matrix Activation::swish_derivative(const Matrix& input) {
    Matrix sig = sigmoid(input);
    Matrix result(input.rows, input.cols);

    for (int i = 0; i < input.rows; i++) {
        for (int j = 0; j < input.cols; j++) {
            float s = sig(i, j);
            float x = input(i, j);
            // Swish'(x) = sigmoid(x) + x * sigmoid(x) * (1 - sigmoid(x))
            result(i, j) = s + x * s * (1.0f - s);
        }
    }

    return result;
}
