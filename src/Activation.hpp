#pragma once

#include <algorithm>
#include <cmath>
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
     */
    static Matrix softmax(const Matrix& input);

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
     * Formula: For cross-entropy loss: grad_input = output * (grad_output - sum(output *
     * grad_output))
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
     * Formula: LeakyReLU'(x) = alpha if x < 0, else 1
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
    // Helper constant for GELU approximation
    static constexpr float GELU_COEF = 0.044715f;
    static constexpr float SQRT_2_OVER_PI = 0.7978845608f;  // sqrt(2/π)
};
