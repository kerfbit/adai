#include "Activation.hpp"
#include <algorithm>
#include <cmath>
#include <limits>

// Softmax activation
Matrix Activation::softmax(const Matrix& input) {
    Matrix result(input.rows, input.cols);

    for (int i = 0; i < input.rows; i++) {
        // Find max value in row for numerical stability
        float max_val = input(i, 0);
        for (int j = 1; j < input.cols; j++) {
            max_val = std::max(max_val, input(i, j));
        }

        // Compute exp(x - max) and sum
        float sum = 0.0f;
        for (int j = 0; j < input.cols; j++) {
            result(i, j) = std::exp(input(i, j) - max_val);
            sum += result(i, j);
        }

        // Normalize to get probabilities
        for (int j = 0; j < input.cols; j++) {
            result(i, j) /= sum;
        }
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

// Softmax derivative (efficient version for cross-entropy)
Matrix Activation::softmax_derivative(const Matrix& output, const Matrix& grad_output) {
    Matrix result(output.rows, output.cols);

    for (int i = 0; i < output.rows; i++) {
        // Compute sum of (output * grad_output) for this row
        float sum = 0.0f;
        for (int j = 0; j < output.cols; j++) {
            sum += output(i, j) * grad_output(i, j);
        }

        // Compute gradient: output * (grad_output - sum)
        for (int j = 0; j < output.cols; j++) {
            result(i, j) = output(i, j) * (grad_output(i, j) - sum);
        }
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
