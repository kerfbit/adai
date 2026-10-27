#pragma once

// @adai-status: stable
// @adai-version: 1.1.0
// @adai-reviewed: 2026-10-26


#include <algorithm>
#include <cmath>
#include "ActivationConstants.hpp"
#include "Matrix.hpp"

/**
 * Activation functions for neural network layers
 *
 * Provides static methods for common activation functions and their derivatives
 * used in neural network forward and backward passes.
 */
class Activation {
   public:
    /**
     * Softmax activation function
     *
     * Converts logits to probability distribution along each row.
     * Numerically stable implementation using max subtraction.
     *
     * @param input Matrix of logits [batch_size, num_classes]
     * @return Matrix of probabilities [batch_size, num_classes] where each row sums to 1
     *
     * Formula: softmax(x_i) = exp(x_i - max(x)) / Σ exp(x_j - max(x))
     * A matrix with no columns gives an empty result (TD-215).
     */
    static Matrix softmax(const Matrix& input);

    // TD-216: allocation-free row primitives, the one implementation of softmax and its
    // backward for every CPU hot path (attention, the losses, sampling, RLHF), which used to
    // re-type them inline.

    /** Max-subtracted softmax of `in` (n floats) into `out` (may be `in`). n <= 0 does nothing; a
     *  row whose exponentials sum to zero or NaN is left unnormalized. */
    static void softmax_row(const float* in, float* out, int n);
    static void softmax_row_inplace(float* row, int n) {
        softmax_row(row, row, n);
    }
    /** The softmax vector-Jacobian product of one row: dx = y * (dy - Σ y·dy), for softmax output
     *  y and upstream gradient dy (dx may be dy). */
    static void softmax_row_backward(const float* y, const float* dy, float* dx, int n);
    /** log Σ exp(row), max-subtracted (log-softmax of x_i is x_i - log_sum_exp_row()). */
    static float log_sum_exp_row(const float* row, int n);

    /**
     * GELU (Gaussian Error Linear Unit) activation function
     *
     * Smooth approximation to ReLU with better gradient properties.
     * Uses tanh approximation for computational efficiency.
     *
     * @param input Matrix of values
     * @return Matrix with GELU applied element-wise
     *
     * Formula: GELU(x) ≈ 0.5 * x * (1 + tanh(√(2/π) * (x + 0.044715 * x³)))
     */
    static Matrix gelu(const Matrix& input);

    /**
     * GELU derivative for backpropagation
     *
     * Computes the derivative of GELU with respect to its input.
     * Required for gradient computation in backward pass.
     *
     * @param input Original input to GELU (before activation)
     * @return Matrix of derivatives element-wise
     *
     * Formula: GELU'(x) = derivative of GELU approximation
     */
    static Matrix gelu_derivative(const Matrix& input);

    /**
     * Softmax derivative for backpropagation
     *
     * Computes gradient through softmax layer efficiently.
     * Uses the property that softmax Jacobian simplifies for common loss functions.
     *
     * @param output Softmax output (already computed)
     * @param grad_output Gradient from upstream layer
     * @return Gradient with respect to softmax input
     *
     * Formula: grad_input = output * (grad_output - sum(output * grad_output)), row by row: the
     * general softmax vector-Jacobian product, correct for any upstream gradient. Not for
     * softmax + cross-entropy, whose gradient with respect to the logits is y - one_hot(target)
     * directly (chaining this after it would differentiate through softmax twice). TD-215.
     *
     * @throws std::invalid_argument if output and grad_output have different shapes
     */
    static Matrix softmax_derivative(const Matrix& output, const Matrix& grad_output);

    /**
     * ReLU (Rectified Linear Unit) activation function
     *
     * Simple thresholding activation: max(0, x)
     *
     * @param input Matrix of values
     * @return Matrix with ReLU applied element-wise
     *
     * Formula: ReLU(x) = max(0, x)
     */
    static Matrix relu(const Matrix& input);

    /**
     * ReLU derivative for backpropagation
     *
     * Derivative is 1 for positive inputs, 0 for non-positive.
     *
     * @param input Original input to ReLU (before activation)
     * @return Matrix of derivatives (0 or 1)
     *
     * Formula: ReLU'(x) = 1 if x > 0, else 0
     */
    static Matrix relu_derivative(const Matrix& input);

    /**
     * Sigmoid activation function
     *
     * Squashes values to range (0, 1).
     *
     * @param input Matrix of values
     * @return Matrix with sigmoid applied element-wise
     *
     * Formula: sigmoid(x) = 1 / (1 + exp(-x))
     */
    static Matrix sigmoid(const Matrix& input);

    /**
     * Sigmoid derivative for backpropagation
     *
     * Can be computed efficiently from sigmoid output.
     *
     * @param output Sigmoid output (already computed)
     * @return Matrix of derivatives
     *
     * Formula: sigmoid'(x) = sigmoid(x) * (1 - sigmoid(x))
     */
    static Matrix sigmoid_derivative(const Matrix& output);

    /**
     * Tanh (Hyperbolic Tangent) activation function
     *
     * Squashes values to range (-1, 1).
     *
     * @param input Matrix of values
     * @return Matrix with tanh applied element-wise
     *
     * Formula: tanh(x) = (exp(x) - exp(-x)) / (exp(x) + exp(-x))
     */
    static Matrix tanh(const Matrix& input);

    /**
     * Tanh derivative for backpropagation
     *
     * Can be computed efficiently from tanh output.
     *
     * @param output Tanh output (already computed)
     * @return Matrix of derivatives
     *
     * Formula: tanh'(x) = 1 - tanh²(x)
     */
    static Matrix tanh_derivative(const Matrix& output);

    /**
     * Leaky ReLU activation function
     *
     * Allows small negative slope for negative inputs.
     *
     * @param input Matrix of values
     * @param alpha Slope for negative values (default 0.01)
     * @return Matrix with Leaky ReLU applied element-wise
     *
     * Formula: LeakyReLU(x) = max(alpha * x, x)
     */
    static Matrix leaky_relu(const Matrix& input, float alpha = 0.01f);

    /**
     * Leaky ReLU derivative for backpropagation
     *
     * @param input Original input to Leaky ReLU
     * @param alpha Slope for negative values (default 0.01)
     * @return Matrix of derivatives
     *
     * Formula: LeakyReLU'(x) = 1 if x > 0, else alpha (alpha at x == 0, TD-215)
     */
    static Matrix leaky_relu_derivative(const Matrix& input, float alpha = 0.01f);

    /**
     * Swish/SiLU activation function
     *
     * Self-gated activation: x * sigmoid(x)
     *
     * @param input Matrix of values
     * @return Matrix with Swish applied element-wise
     *
     * Formula: Swish(x) = x * sigmoid(x)
     */
    static Matrix swish(const Matrix& input);

    /**
     * Swish derivative for backpropagation
     *
     * @param input Original input to Swish
     * @return Matrix of derivatives
     *
     * Formula: Swish'(x) = sigmoid(x) + x * sigmoid(x) * (1 - sigmoid(x))
     */
    static Matrix swish_derivative(const Matrix& input);

   private:
    // GELU approximation constants, shared with the CUDA and SYCL kernels (TD-216)
    static constexpr float GELU_COEF = adai::activation::kGeluCoef;
    static constexpr float SQRT_2_OVER_PI = adai::activation::kSqrt2OverPi;
};
