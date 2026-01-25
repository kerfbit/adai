#include "../src/Activation.hpp"
#include <../gtest/gtest.h>
#include <cmath>
#include <stdexcept>
#include "../src/Matrix.hpp"

// Helper function for numerical gradient checking
float numerical_derivative(Matrix& input, int i, int j,
                           std::function<Matrix(const Matrix&)> activation) {
    float epsilon = 1e-5f;

    float orig = input(i, j);

    input(i, j) = orig + epsilon;
    Matrix out_plus = activation(input);

    input(i, j) = orig - epsilon;
    Matrix out_minus = activation(input);

    input(i, j) = orig;

    return (out_plus(i, j) - out_minus(i, j)) / (2.0f * epsilon);
}

// Helper to check if value is close to expected
bool is_close(float actual, float expected, float tolerance = 1e-4f) {
    return std::abs(actual - expected) < tolerance;
}

// ============================================================================
// Softmax Tests
// ============================================================================

TEST(ActivationSoftmaxTest, BasicSoftmax) {
    Matrix input(1, 3);
    input(0, 0) = 1.0f;
    input(0, 1) = 2.0f;
    input(0, 2) = 3.0f;

    Matrix output = Activation::softmax(input);

    // Check probabilities sum to 1
    float sum = output(0, 0) + output(0, 1) + output(0, 2);
    EXPECT_NEAR(sum, 1.0f, 1e-6f);

    // Check all values in (0, 1)
    for (int j = 0; j < 3; j++) {
        EXPECT_GT(output(0, j), 0.0f);
        EXPECT_LT(output(0, j), 1.0f);
    }

    // Check ordering preserved (larger input -> larger probability)
    EXPECT_GT(output(0, 2), output(0, 1));
    EXPECT_GT(output(0, 1), output(0, 0));
}

TEST(ActivationSoftmaxTest, MultipleBatches) {
    Matrix input(2, 3);
    input(0, 0) = 1.0f;
    input(0, 1) = 2.0f;
    input(0, 2) = 3.0f;
    input(1, 0) = 0.5f;
    input(1, 1) = 1.5f;
    input(1, 2) = 2.5f;

    Matrix output = Activation::softmax(input);

    // Each row should sum to 1
    for (int i = 0; i < 2; i++) {
        float sum = 0.0f;
        for (int j = 0; j < 3; j++) {
            sum += output(i, j);
        }
        EXPECT_NEAR(sum, 1.0f, 1e-6f);
    }
}

TEST(ActivationSoftmaxTest, NumericalStability) {
    // Test with large values that would overflow without max subtraction
    Matrix input(1, 3);
    input(0, 0) = 1000.0f;
    input(0, 1) = 1001.0f;
    input(0, 2) = 1002.0f;

    Matrix output = Activation::softmax(input);

    // Should not contain NaN or Inf
    for (int j = 0; j < 3; j++) {
        EXPECT_FALSE(std::isnan(output(0, j)));
        EXPECT_FALSE(std::isinf(output(0, j)));
    }

    // Should still sum to 1
    float sum = output(0, 0) + output(0, 1) + output(0, 2);
    EXPECT_NEAR(sum, 1.0f, 1e-5f);
}

TEST(ActivationSoftmaxTest, UniformInput) {
    // All same values should give uniform distribution
    Matrix input(1, 4);
    input(0, 0) = 2.0f;
    input(0, 1) = 2.0f;
    input(0, 2) = 2.0f;
    input(0, 3) = 2.0f;

    Matrix output = Activation::softmax(input);

    // Each should be approximately 0.25
    for (int j = 0; j < 4; j++) {
        EXPECT_NEAR(output(0, j), 0.25f, 1e-6f);
    }
}

TEST(ActivationSoftmaxTest, SoftmaxDerivative) {
    Matrix output(1, 3);
    output(0, 0) = 0.1f;
    output(0, 1) = 0.3f;
    output(0, 2) = 0.6f;

    Matrix grad_output(1, 3);
    grad_output(0, 0) = 1.0f;
    grad_output(0, 1) = 0.5f;
    grad_output(0, 2) = 0.2f;

    Matrix grad_input = Activation::softmax_derivative(output, grad_output);

    // Check dimensions
    EXPECT_EQ(grad_input.rows, 1);
    EXPECT_EQ(grad_input.cols, 3);

    // Values should be finite
    for (int j = 0; j < 3; j++) {
        EXPECT_FALSE(std::isnan(grad_input(0, j)));
        EXPECT_FALSE(std::isinf(grad_input(0, j)));
    }
}

// ============================================================================
// GELU Tests
// ============================================================================

TEST(ActivationGELUTest, BasicGELU) {
    Matrix input(2, 3);
    input(0, 0) = -1.0f;
    input(0, 1) = 0.0f;
    input(0, 2) = 1.0f;
    input(1, 0) = -0.5f;
    input(1, 1) = 0.5f;
    input(1, 2) = 2.0f;

    Matrix output = Activation::gelu(input);

    // GELU(0) should be approximately 0
    EXPECT_NEAR(output(0, 1), 0.0f, 1e-5f);

    // GELU is approximately identity for large positive x
    EXPECT_GT(output(1, 2), 1.9f);

    // GELU allows small negative values (unlike ReLU)
    EXPECT_LT(output(0, 0), 0.0f);
    EXPECT_GT(output(0, 0), -0.2f);  // Small negative
}

TEST(ActivationGELUTest, PositiveValues) {
    Matrix input(1, 3);
    input(0, 0) = 1.0f;
    input(0, 1) = 2.0f;
    input(0, 2) = 3.0f;

    Matrix output = Activation::gelu(input);

    // For positive x, GELU(x) ≈ x
    EXPECT_NEAR(output(0, 0), 0.841f, 0.01f);  // GELU(1) ≈ 0.841
    EXPECT_NEAR(output(0, 1), 1.954f, 0.01f);  // GELU(2) ≈ 1.954
    EXPECT_GT(output(0, 2), 2.99f);            // GELU(3) ≈ 3
}

TEST(ActivationGELUTest, Smooth) {
    // GELU should be smooth (no sharp transitions like ReLU)
    Matrix input(1, 5);
    input(0, 0) = -0.2f;
    input(0, 1) = -0.1f;
    input(0, 2) = 0.0f;
    input(0, 3) = 0.1f;
    input(0, 4) = 0.2f;

    Matrix output = Activation::gelu(input);

    // Check smoothness - no large jumps
    for (int j = 0; j < 4; j++) {
        float diff = std::abs(output(0, j + 1) - output(0, j));
        EXPECT_LT(diff, 0.15f);  // Gradual change
    }
}

TEST(ActivationGELUTest, GELUDerivative) {
    Matrix input(3, 3);
    input.randomize(0.5f);

    Matrix analytical_grad = Activation::gelu_derivative(input);

    // Check numerical gradient for a few points
    // Note: Higher tolerance needed for GELU due to tanh approximation
    // and numerical precision issues at larger input values
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            float numerical_grad = numerical_derivative(input, i, j, Activation::gelu);
            EXPECT_NEAR(analytical_grad(i, j), numerical_grad, 1e-2f);
        }
    }
}

// ============================================================================
// ReLU Tests
// ============================================================================

TEST(ActivationReLUTest, BasicReLU) {
    Matrix input(2, 3);
    input(0, 0) = -1.0f;
    input(0, 1) = 0.0f;
    input(0, 2) = 1.0f;
    input(1, 0) = -5.0f;
    input(1, 1) = 3.0f;
    input(1, 2) = 0.5f;

    Matrix output = Activation::relu(input);

    // Negative values should be 0
    EXPECT_FLOAT_EQ(output(0, 0), 0.0f);
    EXPECT_FLOAT_EQ(output(1, 0), 0.0f);

    // Zero should remain 0
    EXPECT_FLOAT_EQ(output(0, 1), 0.0f);

    // Positive values unchanged
    EXPECT_FLOAT_EQ(output(0, 2), 1.0f);
    EXPECT_FLOAT_EQ(output(1, 1), 3.0f);
    EXPECT_FLOAT_EQ(output(1, 2), 0.5f);
}

TEST(ActivationReLUTest, AllNegative) {
    Matrix input(2, 2);
    input(0, 0) = -1.0f;
    input(0, 1) = -2.0f;
    input(1, 0) = -3.0f;
    input(1, 1) = -4.0f;

    Matrix output = Activation::relu(input);

    // All should be zero
    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 2; j++) {
            EXPECT_FLOAT_EQ(output(i, j), 0.0f);
        }
    }
}

TEST(ActivationReLUTest, AllPositive) {
    Matrix input(2, 2);
    input(0, 0) = 1.0f;
    input(0, 1) = 2.0f;
    input(1, 0) = 3.0f;
    input(1, 1) = 4.0f;

    Matrix output = Activation::relu(input);

    // All should be unchanged
    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 2; j++) {
            EXPECT_FLOAT_EQ(output(i, j), input(i, j));
        }
    }
}

TEST(ActivationReLUTest, ReLUDerivative) {
    Matrix input(2, 3);
    input(0, 0) = -1.0f;
    input(0, 1) = 0.0f;
    input(0, 2) = 1.0f;
    input(1, 0) = -5.0f;
    input(1, 1) = 3.0f;
    input(1, 2) = 0.5f;

    Matrix derivative = Activation::relu_derivative(input);

    // Negative: gradient = 0
    EXPECT_FLOAT_EQ(derivative(0, 0), 0.0f);
    EXPECT_FLOAT_EQ(derivative(1, 0), 0.0f);

    // Zero: gradient = 0 (convention)
    EXPECT_FLOAT_EQ(derivative(0, 1), 0.0f);

    // Positive: gradient = 1
    EXPECT_FLOAT_EQ(derivative(0, 2), 1.0f);
    EXPECT_FLOAT_EQ(derivative(1, 1), 1.0f);
    EXPECT_FLOAT_EQ(derivative(1, 2), 1.0f);
}

// ============================================================================
// Leaky ReLU Tests
// ============================================================================

TEST(ActivationLeakyReLUTest, BasicLeakyReLU) {
    Matrix input(2, 3);
    input(0, 0) = -1.0f;
    input(0, 1) = 0.0f;
    input(0, 2) = 1.0f;
    input(1, 0) = -2.0f;
    input(1, 1) = 2.0f;
    input(1, 2) = -0.5f;

    float alpha = 0.01f;
    Matrix output = Activation::leaky_relu(input, alpha);

    // Negative values scaled by alpha
    EXPECT_FLOAT_EQ(output(0, 0), -0.01f);
    EXPECT_FLOAT_EQ(output(1, 0), -0.02f);
    EXPECT_FLOAT_EQ(output(1, 2), -0.005f);

    // Zero should remain 0
    EXPECT_FLOAT_EQ(output(0, 1), 0.0f);

    // Positive values unchanged
    EXPECT_FLOAT_EQ(output(0, 2), 1.0f);
    EXPECT_FLOAT_EQ(output(1, 1), 2.0f);
}

TEST(ActivationLeakyReLUTest, DifferentAlpha) {
    Matrix input(1, 3);
    input(0, 0) = -1.0f;
    input(0, 1) = 0.0f;
    input(0, 2) = 1.0f;

    // Test with alpha = 0.1
    Matrix output = Activation::leaky_relu(input, 0.1f);

    EXPECT_FLOAT_EQ(output(0, 0), -0.1f);
    EXPECT_FLOAT_EQ(output(0, 1), 0.0f);
    EXPECT_FLOAT_EQ(output(0, 2), 1.0f);
}

TEST(ActivationLeakyReLUTest, LeakyReLUDerivative) {
    Matrix input(2, 2);
    input(0, 0) = -1.0f;
    input(0, 1) = 1.0f;
    input(1, 0) = -2.0f;
    input(1, 1) = 2.0f;

    float alpha = 0.01f;
    Matrix derivative = Activation::leaky_relu_derivative(input, alpha);

    // Negative: gradient = alpha
    EXPECT_FLOAT_EQ(derivative(0, 0), alpha);
    EXPECT_FLOAT_EQ(derivative(1, 0), alpha);

    // Positive: gradient = 1
    EXPECT_FLOAT_EQ(derivative(0, 1), 1.0f);
    EXPECT_FLOAT_EQ(derivative(1, 1), 1.0f);
}

TEST(ActivationLeakyReLUTest, NoDeadNeurons) {
    Matrix input(2, 2);
    input(0, 0) = -1.0f;
    input(0, 1) = -2.0f;
    input(1, 0) = -3.0f;
    input(1, 1) = -4.0f;

    Matrix derivative = Activation::leaky_relu_derivative(input, 0.01f);

    // All should have non-zero gradient
    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 2; j++) {
            EXPECT_GT(derivative(i, j), 0.0f);
        }
    }
}

// ============================================================================
// Sigmoid Tests
// ============================================================================

TEST(ActivationSigmoidTest, BasicSigmoid) {
    Matrix input(1, 5);
    input(0, 0) = -2.0f;
    input(0, 1) = -1.0f;
    input(0, 2) = 0.0f;
    input(0, 3) = 1.0f;
    input(0, 4) = 2.0f;

    Matrix output = Activation::sigmoid(input);

    // All values should be in (0, 1)
    for (int j = 0; j < 5; j++) {
        EXPECT_GT(output(0, j), 0.0f);
        EXPECT_LT(output(0, j), 1.0f);
    }

    // sigmoid(0) = 0.5
    EXPECT_NEAR(output(0, 2), 0.5f, 1e-6f);

    // sigmoid(-x) = 1 - sigmoid(x)
    EXPECT_NEAR(output(0, 0), 1.0f - output(0, 4), 1e-6f);
    EXPECT_NEAR(output(0, 1), 1.0f - output(0, 3), 1e-6f);
}

TEST(ActivationSigmoidTest, ExtremeValues) {
    Matrix input(1, 4);
    input(0, 0) = -100.0f;
    input(0, 1) = -10.0f;
    input(0, 2) = 10.0f;
    input(0, 3) = 100.0f;

    Matrix output = Activation::sigmoid(input);

    // Should not be NaN or Inf
    for (int j = 0; j < 4; j++) {
        EXPECT_FALSE(std::isnan(output(0, j)));
        EXPECT_FALSE(std::isinf(output(0, j)));
    }

    // Large negative -> near 0
    EXPECT_LT(output(0, 0), 1e-6f);
    EXPECT_LT(output(0, 1), 0.001f);

    // Large positive -> near 1
    EXPECT_GT(output(0, 2), 0.999f);
    EXPECT_GT(output(0, 3), 1.0f - 1e-6f);
}

TEST(ActivationSigmoidTest, SigmoidDerivative) {
    Matrix output(1, 3);
    output(0, 0) = 0.1f;
    output(0, 1) = 0.5f;
    output(0, 2) = 0.9f;

    Matrix derivative = Activation::sigmoid_derivative(output);

    // sigmoid'(x) = sigmoid(x) * (1 - sigmoid(x))
    EXPECT_NEAR(derivative(0, 0), 0.1f * 0.9f, 1e-6f);
    EXPECT_NEAR(derivative(0, 1), 0.5f * 0.5f, 1e-6f);
    EXPECT_NEAR(derivative(0, 2), 0.9f * 0.1f, 1e-6f);

    // Maximum gradient at 0.5 (input = 0)
    EXPECT_FLOAT_EQ(derivative(0, 1), 0.25f);
}

TEST(ActivationSigmoidTest, NumericalGradient) {
    Matrix input(2, 2);
    input.randomize(1.0f);

    Matrix output = Activation::sigmoid(input);
    Matrix analytical_grad = Activation::sigmoid_derivative(output);

    // Check numerical gradient
    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 2; j++) {
            float numerical_grad = numerical_derivative(input, i, j, Activation::sigmoid);
            EXPECT_NEAR(analytical_grad(i, j), numerical_grad, 5e-3f);
        }
    }
}

// ============================================================================
// Tanh Tests
// ============================================================================

TEST(ActivationTanhTest, BasicTanh) {
    Matrix input(1, 5);
    input(0, 0) = -2.0f;
    input(0, 1) = -1.0f;
    input(0, 2) = 0.0f;
    input(0, 3) = 1.0f;
    input(0, 4) = 2.0f;

    Matrix output = Activation::tanh(input);

    // All values should be in (-1, 1)
    for (int j = 0; j < 5; j++) {
        EXPECT_GT(output(0, j), -1.0f);
        EXPECT_LT(output(0, j), 1.0f);
    }

    // tanh(0) = 0
    EXPECT_NEAR(output(0, 2), 0.0f, 1e-6f);

    // tanh(-x) = -tanh(x)
    EXPECT_NEAR(output(0, 0), -output(0, 4), 1e-6f);
    EXPECT_NEAR(output(0, 1), -output(0, 3), 1e-6f);
}

TEST(ActivationTanhTest, ExtremeValues) {
    Matrix input(1, 4);
    input(0, 0) = -100.0f;
    input(0, 1) = -10.0f;
    input(0, 2) = 10.0f;
    input(0, 3) = 100.0f;

    Matrix output = Activation::tanh(input);

    // Should not be NaN or Inf
    for (int j = 0; j < 4; j++) {
        EXPECT_FALSE(std::isnan(output(0, j)));
        EXPECT_FALSE(std::isinf(output(0, j)));
    }

    // Large negative -> near -1
    EXPECT_LT(output(0, 0), -0.999f);

    // Large positive -> near 1
    EXPECT_GT(output(0, 3), 0.999f);
}

TEST(ActivationTanhTest, TanhDerivative) {
    Matrix output(1, 3);
    output(0, 0) = -0.5f;
    output(0, 1) = 0.0f;
    output(0, 2) = 0.5f;

    Matrix derivative = Activation::tanh_derivative(output);

    // tanh'(x) = 1 - tanh²(x)
    EXPECT_NEAR(derivative(0, 0), 1.0f - 0.25f, 1e-6f);
    EXPECT_NEAR(derivative(0, 1), 1.0f, 1e-6f);
    EXPECT_NEAR(derivative(0, 2), 1.0f - 0.25f, 1e-6f);
}

TEST(ActivationTanhTest, NumericalGradient) {
    Matrix input(2, 2);
    input.randomize(1.0f);

    Matrix output = Activation::tanh(input);
    Matrix analytical_grad = Activation::tanh_derivative(output);

    // Check numerical gradient
    // Note: Relaxed tolerance due to numerical precision at larger values
    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 2; j++) {
            float numerical_grad = numerical_derivative(input, i, j, Activation::tanh);
            EXPECT_NEAR(analytical_grad(i, j), numerical_grad, 5e-3f);
        }
    }
}

// ============================================================================
// Swish Tests
// ============================================================================

TEST(ActivationSwishTest, BasicSwish) {
    Matrix input(1, 5);
    input(0, 0) = -2.0f;
    input(0, 1) = -1.0f;
    input(0, 2) = 0.0f;
    input(0, 3) = 1.0f;
    input(0, 4) = 2.0f;

    Matrix output = Activation::swish(input);

    // Swish(0) = 0 * sigmoid(0) = 0
    EXPECT_NEAR(output(0, 2), 0.0f, 1e-6f);

    // For large positive x, Swish(x) approaches x (but not exactly)
    EXPECT_GT(output(0, 4), 1.7f);  // Swish(2) ≈ 1.76

    // Swish allows negative values
    EXPECT_LT(output(0, 0), 0.0f);
    EXPECT_LT(output(0, 1), 0.0f);
}

TEST(ActivationSwishTest, SelfGating) {
    Matrix input(1, 3);
    input(0, 0) = 1.0f;
    input(0, 1) = 2.0f;
    input(0, 2) = 3.0f;

    Matrix output = Activation::swish(input);
    Matrix sigmoid_out = Activation::sigmoid(input);

    // Swish(x) = x * sigmoid(x)
    for (int j = 0; j < 3; j++) {
        EXPECT_NEAR(output(0, j), input(0, j) * sigmoid_out(0, j), 1e-5f);
    }
}

TEST(ActivationSwishTest, SwishDerivative) {
    Matrix input(3, 3);
    input.randomize(1.0f);

    Matrix analytical_grad = Activation::swish_derivative(input);

    // Check numerical gradient for a few points
    // Note: Higher tolerance needed for Swish due to sigmoid's
    // numerical precision issues at larger input values
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            float numerical_grad = numerical_derivative(input, i, j, Activation::swish);
            EXPECT_NEAR(analytical_grad(i, j), numerical_grad, 1e-2f);
        }
    }
}

TEST(ActivationSwishTest, Smoothness) {
    // Swish should be smooth everywhere
    Matrix input(1, 5);
    input(0, 0) = -1.0f;
    input(0, 1) = -0.5f;
    input(0, 2) = 0.0f;
    input(0, 3) = 0.5f;
    input(0, 4) = 1.0f;

    Matrix derivative = Activation::swish_derivative(input);

    // All derivatives should be finite
    for (int j = 0; j < 5; j++) {
        EXPECT_FALSE(std::isnan(derivative(0, j)));
        EXPECT_FALSE(std::isinf(derivative(0, j)));
    }
}

// ============================================================================
// Integration Tests
// ============================================================================

TEST(ActivationIntegrationTest, ForwardBackwardGELU) {
    // Simulate forward and backward pass with GELU
    Matrix input(5, 10);
    input.randomize(1.0f);

    // Forward pass
    Matrix activated = Activation::gelu(input);

    // Simulate gradient from next layer
    Matrix grad_output(5, 10);
    grad_output.fill(1.0f);

    // Backward pass
    Matrix grad_input = Activation::gelu_derivative(input).hadamard(grad_output);

    EXPECT_EQ(grad_input.rows, 5);
    EXPECT_EQ(grad_input.cols, 10);

    // Gradients should be finite
    for (int i = 0; i < 5; i++) {
        for (int j = 0; j < 10; j++) {
            EXPECT_FALSE(std::isnan(grad_input(i, j)));
            EXPECT_FALSE(std::isinf(grad_input(i, j)));
        }
    }
}

TEST(ActivationIntegrationTest, ClassificationWithSoftmax) {
    // Simulate classification layer
    Matrix logits(32, 10);  // 32 samples, 10 classes
    logits.randomize(1.0f);

    // Forward pass
    Matrix probs = Activation::softmax(logits);

    // Check each sample has valid probability distribution
    for (int i = 0; i < 32; i++) {
        float sum = 0.0f;
        for (int j = 0; j < 10; j++) {
            EXPECT_GT(probs(i, j), 0.0f);
            EXPECT_LT(probs(i, j), 1.0f);
            sum += probs(i, j);
        }
        EXPECT_NEAR(sum, 1.0f, 1e-5f);
    }
}

TEST(ActivationIntegrationTest, BinaryClassificationWithSigmoid) {
    // Binary classification scenario
    Matrix logits(100, 1);
    logits.randomize(2.0f);

    Matrix probs = Activation::sigmoid(logits);

    // All probabilities in (0, 1)
    for (int i = 0; i < 100; i++) {
        EXPECT_GT(probs(i, 0), 0.0f);
        EXPECT_LT(probs(i, 0), 1.0f);
    }
}

TEST(ActivationIntegrationTest, ReLUSparsity) {
    // ReLU should create sparse activations
    Matrix input(100, 100);
    input.randomize(1.0f);  // Mean 0, some negative

    Matrix output = Activation::relu(input);

    // Count zeros
    int zero_count = 0;
    for (int i = 0; i < 100; i++) {
        for (int j = 0; j < 100; j++) {
            if (output(i, j) == 0.0f) {
                zero_count++;
            }
        }
    }

    // Should have some zeros (sparsity)
    EXPECT_GT(zero_count, 0);
}

TEST(ActivationIntegrationTest, AttentionScoresWithSoftmax) {
    // Simulate attention score normalization
    int seq_len = 20;
    Matrix scores(seq_len, seq_len);
    scores.randomize(1.0f);

    // Apply softmax to get attention weights
    Matrix attn_weights = Activation::softmax(scores);

    // Each query (row) should have valid attention distribution
    for (int i = 0; i < seq_len; i++) {
        float sum = 0.0f;
        for (int j = 0; j < seq_len; j++) {
            sum += attn_weights(i, j);
        }
        EXPECT_NEAR(sum, 1.0f, 1e-5f);
    }
}

// ============================================================================
// Comparison Tests
// ============================================================================

TEST(ActivationComparisonTest, GELUvsReLU) {
    Matrix input(1, 5);
    input(0, 0) = -2.0f;
    input(0, 1) = -1.0f;
    input(0, 2) = 0.0f;
    input(0, 3) = 1.0f;
    input(0, 4) = 2.0f;

    Matrix gelu_out = Activation::gelu(input);
    Matrix relu_out = Activation::relu(input);

    // GELU allows negative values, ReLU doesn't
    EXPECT_LT(gelu_out(0, 0), 0.0f);
    EXPECT_FLOAT_EQ(relu_out(0, 0), 0.0f);

    // For large positive values, both should be similar
    EXPECT_NEAR(gelu_out(0, 4), relu_out(0, 4), 0.1f);
}

TEST(ActivationComparisonTest, SigmoidVsTanh) {
    Matrix input(1, 3);
    input(0, 0) = -1.0f;
    input(0, 1) = 0.0f;
    input(0, 2) = 1.0f;

    Matrix sigmoid_out = Activation::sigmoid(input);
    Matrix tanh_out = Activation::tanh(input);

    // Sigmoid output in (0, 1)
    for (int j = 0; j < 3; j++) {
        EXPECT_GT(sigmoid_out(0, j), 0.0f);
        EXPECT_LT(sigmoid_out(0, j), 1.0f);
    }

    // Tanh output in (-1, 1)
    for (int j = 0; j < 3; j++) {
        EXPECT_GT(tanh_out(0, j), -1.0f);
        EXPECT_LT(tanh_out(0, j), 1.0f);
    }

    // Tanh is zero-centered
    EXPECT_NEAR(tanh_out(0, 1), 0.0f, 1e-6f);
    EXPECT_NEAR(sigmoid_out(0, 1), 0.5f, 1e-6f);
}

TEST(ActivationComparisonTest, LeakyReLUvsReLU) {
    Matrix input(1, 3);
    input(0, 0) = -1.0f;
    input(0, 1) = 0.0f;
    input(0, 2) = 1.0f;

    Matrix relu_out = Activation::relu(input);
    Matrix leaky_out = Activation::leaky_relu(input, 0.1f);
    Matrix relu_grad = Activation::relu_derivative(input);
    Matrix leaky_grad = Activation::leaky_relu_derivative(input, 0.1f);

    // Both preserve positive values
    EXPECT_FLOAT_EQ(relu_out(0, 2), leaky_out(0, 2));

    // Leaky ReLU has non-zero output/gradient for negative
    EXPECT_FLOAT_EQ(relu_out(0, 0), 0.0f);
    EXPECT_FLOAT_EQ(leaky_out(0, 0), -0.1f);
    EXPECT_FLOAT_EQ(relu_grad(0, 0), 0.0f);
    EXPECT_FLOAT_EQ(leaky_grad(0, 0), 0.1f);
}

// ============================================================================
// Edge Cases
// ============================================================================

TEST(ActivationEdgeCaseTest, SingleElement) {
    Matrix input(1, 1);
    input(0, 0) = 2.5f;

    // Test all activations
    Matrix relu_out = Activation::relu(input);
    Matrix gelu_out = Activation::gelu(input);
    Matrix sigmoid_out = Activation::sigmoid(input);
    Matrix tanh_out = Activation::tanh(input);

    EXPECT_FLOAT_EQ(relu_out(0, 0), 2.5f);
    EXPECT_GT(gelu_out(0, 0), 2.4f);
    EXPECT_GT(sigmoid_out(0, 0), 0.9f);
    EXPECT_GT(tanh_out(0, 0), 0.98f);
}

TEST(ActivationEdgeCaseTest, ZeroInput) {
    Matrix input(5, 5);
    input.fill(0.0f);

    Matrix relu_out = Activation::relu(input);
    Matrix gelu_out = Activation::gelu(input);
    Matrix sigmoid_out = Activation::sigmoid(input);
    Matrix tanh_out = Activation::tanh(input);

    // ReLU(0) = 0
    EXPECT_FLOAT_EQ(relu_out(0, 0), 0.0f);

    // GELU(0) ≈ 0
    EXPECT_NEAR(gelu_out(0, 0), 0.0f, 1e-5f);

    // sigmoid(0) = 0.5
    EXPECT_NEAR(sigmoid_out(0, 0), 0.5f, 1e-6f);

    // tanh(0) = 0
    EXPECT_NEAR(tanh_out(0, 0), 0.0f, 1e-6f);
}

TEST(ActivationEdgeCaseTest, LargeMatrix) {
    Matrix input(100, 100);
    input.randomize(1.0f);

    // Should handle large matrices efficiently
    Matrix relu_out = Activation::relu(input);
    Matrix gelu_out = Activation::gelu(input);

    EXPECT_EQ(relu_out.rows, 100);
    EXPECT_EQ(relu_out.cols, 100);
    EXPECT_EQ(gelu_out.rows, 100);
    EXPECT_EQ(gelu_out.cols, 100);
}

// ============================================================================
// Main
// ============================================================================

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
