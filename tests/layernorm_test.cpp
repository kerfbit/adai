#include "LayerNorm.hpp"
#include <gtest/gtest.h>
#include <cmath>
#include <vector>
#include "Matrix.hpp"
#include "Optimizer.hpp"

// ============================================================================
// Test Suite 1: Initialization Tests
// ============================================================================

TEST(LayerNormInitializationTest, ConstructorDefaultEpsilon) {
    LayerNorm ln(128);

    EXPECT_EQ(ln.get_dim(), 128);
    EXPECT_FLOAT_EQ(ln.get_epsilon(), 1e-5f);
    EXPECT_FLOAT_EQ(ln.learning_rate, 0.001f);
}

TEST(LayerNormInitializationTest, ConstructorCustomEpsilon) {
    LayerNorm ln(64, 1e-3f);

    EXPECT_EQ(ln.get_dim(), 64);
    EXPECT_FLOAT_EQ(ln.get_epsilon(), 1e-3f);
}

TEST(LayerNormInitializationTest, GammaInitializedToOne) {
    LayerNorm ln(256);
    const Matrix& gamma = ln.get_gamma();

    EXPECT_EQ(gamma.rows, 1);
    EXPECT_EQ(gamma.cols, 256);

    for (int j = 0; j < gamma.cols; ++j) {
        EXPECT_FLOAT_EQ(gamma(0, j), 1.0f);
    }
}

TEST(LayerNormInitializationTest, BetaInitializedToZero) {
    LayerNorm ln(256);
    const Matrix& beta = ln.get_beta();

    EXPECT_EQ(beta.rows, 1);
    EXPECT_EQ(beta.cols, 256);

    for (int j = 0; j < beta.cols; ++j) {
        EXPECT_FLOAT_EQ(beta(0, j), 0.0f);
    }
}

TEST(LayerNormInitializationTest, SmallDimension) {
    LayerNorm ln(1);

    EXPECT_EQ(ln.get_dim(), 1);
    EXPECT_FLOAT_EQ(ln.get_gamma()(0, 0), 1.0f);
    EXPECT_FLOAT_EQ(ln.get_beta()(0, 0), 0.0f);
}

TEST(LayerNormInitializationTest, LargeDimension) {
    LayerNorm ln(2048);

    EXPECT_EQ(ln.get_dim(), 2048);
    const Matrix& gamma = ln.get_gamma();
    const Matrix& beta = ln.get_beta();

    EXPECT_EQ(gamma.cols, 2048);
    EXPECT_EQ(beta.cols, 2048);
}

// ============================================================================
// Test Suite 2: Forward Pass - Normalization Correctness
// ============================================================================

TEST(LayerNormForwardTest, SingleSampleNormalization) {
    LayerNorm ln(4);

    Matrix input(1, 4);
    input(0, 0) = 1.0f;
    input(0, 1) = 2.0f;
    input(0, 2) = 3.0f;
    input(0, 3) = 4.0f;

    Matrix output = ln.forward(input);

    // With gamma=1, beta=0, output should have zero mean and unit variance
    float mean = 0.0f;
    for (int j = 0; j < 4; ++j) {
        mean += output(0, j);
    }
    mean /= 4.0f;
    EXPECT_NEAR(mean, 0.0f, 1e-5f);

    float var = 0.0f;
    for (int j = 0; j < 4; ++j) {
        var += (output(0, j) - mean) * (output(0, j) - mean);
    }
    var /= 4.0f;
    EXPECT_NEAR(var, 1.0f, 1e-5f);
}

TEST(LayerNormForwardTest, MultipleSamplesIndependentNormalization) {
    LayerNorm ln(3);

    Matrix input(2, 3);
    // Sample 1: [1, 2, 3]
    input(0, 0) = 1.0f;
    input(0, 1) = 2.0f;
    input(0, 2) = 3.0f;

    // Sample 2: [10, 20, 30]
    input(1, 0) = 10.0f;
    input(1, 1) = 20.0f;
    input(1, 2) = 30.0f;

    Matrix output = ln.forward(input);

    // Each sample should be normalized independently
    // Sample 1 output
    float mean1 = (output(0, 0) + output(0, 1) + output(0, 2)) / 3.0f;
    EXPECT_NEAR(mean1, 0.0f, 1e-5f);

    // Sample 2 output
    float mean2 = (output(1, 0) + output(1, 1) + output(1, 2)) / 3.0f;
    EXPECT_NEAR(mean2, 0.0f, 1e-5f);

    // Both should have the same normalized pattern
    // Since both inputs have same relative pattern [1,2,3] vs [10,20,30],
    // their normalized outputs should be identical
    EXPECT_NEAR(output(0, 0), output(1, 0), 1e-4f);
    EXPECT_NEAR(output(0, 1), output(1, 1), 1e-4f);
    EXPECT_NEAR(output(0, 2), output(1, 2), 1e-4f);
}

TEST(LayerNormForwardTest, ZeroMeanInput) {
    LayerNorm ln(4);

    Matrix input(1, 4);
    input(0, 0) = -1.0f;
    input(0, 1) = -0.5f;
    input(0, 2) = 0.5f;
    input(0, 3) = 1.0f;

    Matrix output = ln.forward(input);

    // Should still normalize to zero mean, unit variance
    float mean = (output(0, 0) + output(0, 1) + output(0, 2) + output(0, 3)) / 4.0f;
    EXPECT_NEAR(mean, 0.0f, 1e-5f);
}

TEST(LayerNormForwardTest, ConstantInput) {
    LayerNorm ln(5);

    Matrix input(1, 5);
    for (int j = 0; j < 5; ++j) {
        input(0, j) = 3.14f;
    }

    Matrix output = ln.forward(input);

    // With zero variance, all outputs should be equal (beta value = 0)
    for (int j = 0; j < 5; ++j) {
        EXPECT_NEAR(output(0, j), 0.0f, 1e-3f);
    }
}

TEST(LayerNormForwardTest, LargeBatchSize) {
    LayerNorm ln(64);

    Matrix input(128, 64);
    for (int i = 0; i < 128; ++i) {
        for (int j = 0; j < 64; ++j) {
            input(i, j) = static_cast<float>(i + j) * 0.1f;
        }
    }

    Matrix output = ln.forward(input);

    // Each sample should be normalized
    for (int i = 0; i < 128; ++i) {
        float mean = 0.0f;
        for (int j = 0; j < 64; ++j) {
            mean += output(i, j);
        }
        mean /= 64.0f;
        EXPECT_NEAR(mean, 0.0f, 1e-4f);
    }
}

// ============================================================================
// Test Suite 3: Forward Pass - Affine Transformation
// ============================================================================

TEST(LayerNormAffineTest, CustomGammaScale) {
    LayerNorm ln(3);

    // Set custom gamma (scale)
    Matrix gamma(1, 3);
    gamma(0, 0) = 2.0f;
    gamma(0, 1) = 2.0f;
    gamma(0, 2) = 2.0f;
    ln.set_gamma(gamma);

    Matrix input(1, 3);
    input(0, 0) = 1.0f;
    input(0, 1) = 2.0f;
    input(0, 2) = 3.0f;

    Matrix output = ln.forward(input);

    // Output variance should be scaled by gamma^2 = 4
    float mean = (output(0, 0) + output(0, 1) + output(0, 2)) / 3.0f;
    float var = 0.0f;
    for (int j = 0; j < 3; ++j) {
        var += (output(0, j) - mean) * (output(0, j) - mean);
    }
    var /= 3.0f;
    EXPECT_NEAR(var, 4.0f, 1e-4f);
}

TEST(LayerNormAffineTest, CustomBetaShift) {
    LayerNorm ln(3);

    // Set custom beta (shift)
    Matrix beta(1, 3);
    beta(0, 0) = 5.0f;
    beta(0, 1) = 5.0f;
    beta(0, 2) = 5.0f;
    ln.set_beta(beta);

    Matrix input(1, 3);
    input(0, 0) = 1.0f;
    input(0, 1) = 2.0f;
    input(0, 2) = 3.0f;

    Matrix output = ln.forward(input);

    // Mean should be shifted by beta = 5
    float mean = (output(0, 0) + output(0, 1) + output(0, 2)) / 3.0f;
    EXPECT_NEAR(mean, 5.0f, 1e-5f);
}

TEST(LayerNormAffineTest, GammaAndBetaTogether) {
    LayerNorm ln(4);

    Matrix gamma(1, 4);
    for (int j = 0; j < 4; ++j) {
        gamma(0, j) = 0.5f;
    }
    ln.set_gamma(gamma);

    Matrix beta(1, 4);
    for (int j = 0; j < 4; ++j) {
        beta(0, j) = 10.0f;
    }
    ln.set_beta(beta);

    Matrix input(1, 4);
    input(0, 0) = 1.0f;
    input(0, 1) = 2.0f;
    input(0, 2) = 3.0f;
    input(0, 3) = 4.0f;

    Matrix output = ln.forward(input);

    // Mean should be shifted to beta = 10
    float mean = 0.0f;
    for (int j = 0; j < 4; ++j) {
        mean += output(0, j);
    }
    mean /= 4.0f;
    EXPECT_NEAR(mean, 10.0f, 1e-4f);

    // Variance should be scaled by gamma^2 = 0.25
    float var = 0.0f;
    for (int j = 0; j < 4; ++j) {
        var += (output(0, j) - mean) * (output(0, j) - mean);
    }
    var /= 4.0f;
    EXPECT_NEAR(var, 0.25f, 1e-4f);
}

TEST(LayerNormAffineTest, PerFeatureGamma) {
    LayerNorm ln(3);

    // Different scale for each feature
    Matrix gamma(1, 3);
    gamma(0, 0) = 1.0f;
    gamma(0, 1) = 2.0f;
    gamma(0, 2) = 3.0f;
    ln.set_gamma(gamma);

    Matrix input(1, 3);
    input(0, 0) = 1.0f;
    input(0, 1) = 2.0f;
    input(0, 2) = 3.0f;

    Matrix output = ln.forward(input);

    // Features should be scaled differently
    // Cannot use simple variance test, but can verify computation
    EXPECT_TRUE(std::isfinite(output(0, 0)));
    EXPECT_TRUE(std::isfinite(output(0, 1)));
    EXPECT_TRUE(std::isfinite(output(0, 2)));
}

// ============================================================================
// Test Suite 4: Backward Pass - Gradient Computation
// ============================================================================

TEST(LayerNormBackwardTest, GammaGradientComputation) {
    LayerNorm ln(3);

    Matrix input(2, 3);
    input(0, 0) = 1.0f;
    input(0, 1) = 2.0f;
    input(0, 2) = 3.0f;
    input(1, 0) = 4.0f;
    input(1, 1) = 5.0f;
    input(1, 2) = 6.0f;

    Matrix output = ln.forward(input);

    // Uniform gradient
    Matrix grad_output(2, 3);
    for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < 3; ++j) {
            grad_output(i, j) = 1.0f;
        }
    }

    ln.zero_grad();
    Matrix grad_input = ln.backward(grad_output);
    ln.update_weights();

    // Gamma gradient should be non-zero
    const Matrix& gamma = ln.get_gamma();
    bool has_changed = false;
    for (int j = 0; j < 3; ++j) {
        if (std::abs(gamma(0, j) - 1.0f) > 1e-6f) {
            has_changed = true;
            break;
        }
    }
    EXPECT_TRUE(has_changed);
}

TEST(LayerNormBackwardTest, BetaGradientComputation) {
    LayerNorm ln(3);

    Matrix input(2, 3);
    input(0, 0) = 1.0f;
    input(0, 1) = 2.0f;
    input(0, 2) = 3.0f;
    input(1, 0) = 4.0f;
    input(1, 1) = 5.0f;
    input(1, 2) = 6.0f;

    Matrix output = ln.forward(input);

    Matrix grad_output(2, 3);
    for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < 3; ++j) {
            grad_output(i, j) = 0.5f;
        }
    }

    ln.zero_grad();
    Matrix grad_input = ln.backward(grad_output);
    ln.update_weights();

    // Beta should have been updated
    const Matrix& beta = ln.get_beta();
    bool has_changed = false;
    for (int j = 0; j < 3; ++j) {
        if (std::abs(beta(0, j) - 0.0f) > 1e-6f) {
            has_changed = true;
            break;
        }
    }
    EXPECT_TRUE(has_changed);
}

TEST(LayerNormBackwardTest, InputGradientShape) {
    LayerNorm ln(64);

    Matrix input(32, 64);
    for (int i = 0; i < 32; ++i) {
        for (int j = 0; j < 64; ++j) {
            input(i, j) = static_cast<float>(i + j) * 0.01f;
        }
    }

    Matrix output = ln.forward(input);

    Matrix grad_output(32, 64);
    for (int i = 0; i < 32; ++i) {
        for (int j = 0; j < 64; ++j) {
            grad_output(i, j) = 1.0f;
        }
    }

    ln.zero_grad();
    Matrix grad_input = ln.backward(grad_output);

    EXPECT_EQ(grad_input.rows, 32);
    EXPECT_EQ(grad_input.cols, 64);
}

TEST(LayerNormBackwardTest, ZeroGradReset) {
    LayerNorm ln(4);

    Matrix input(1, 4);
    input(0, 0) = 1.0f;
    input(0, 1) = 2.0f;
    input(0, 2) = 3.0f;
    input(0, 3) = 4.0f;

    ln.forward(input);

    Matrix grad_output(1, 4);
    for (int j = 0; j < 4; ++j) {
        grad_output(0, j) = 1.0f;
    }

    // First backward
    ln.zero_grad();
    ln.backward(grad_output);
    ln.update_weights();

    // Get gamma after first update
    Matrix gamma_after_first = ln.get_gamma();

    // Second backward with zero_grad
    ln.zero_grad();
    ln.backward(grad_output);
    ln.update_weights();

    // Parameters should continue to update (not accumulate from first)
    Matrix gamma_after_second = ln.get_gamma();

    // Should be different from first (continued updating)
    float diff = 0.0f;
    for (int j = 0; j < 4; ++j) {
        diff += std::abs(gamma_after_first(0, j) - gamma_after_second(0, j));
    }
    EXPECT_GT(diff, 1e-6f);
}

// ============================================================================
// Test Suite 5: Numerical Gradient Checking
// ============================================================================

TEST(LayerNormNumericalGradientTest, SimpleCase) {
    LayerNorm ln(3);
    ln.learning_rate = 0.0f;  // Disable parameter updates

    Matrix input(1, 3);
    input(0, 0) = 1.0f;
    input(0, 1) = 2.0f;
    input(0, 2) = 3.0f;

    Matrix output = ln.forward(input);

    Matrix grad_output(1, 3);
    for (int j = 0; j < 3; ++j) {
        grad_output(0, j) = 1.0f;
    }

    ln.zero_grad();
    Matrix grad_input = ln.backward(grad_output);

    // Numerical gradient check
    float epsilon = 1e-4f;
    for (int j = 0; j < 3; ++j) {
        float original = input(0, j);

        input(0, j) = original + epsilon;
        Matrix out_plus = ln.forward(input);
        float loss_plus = 0.0f;
        for (int k = 0; k < 3; ++k) {
            loss_plus += out_plus(0, k) * grad_output(0, k);  // Apply gradient weights
        }

        input(0, j) = original - epsilon;
        Matrix out_minus = ln.forward(input);
        float loss_minus = 0.0f;
        for (int k = 0; k < 3; ++k) {
            loss_minus += out_minus(0, k) * grad_output(0, k);  // Apply gradient weights
        }

        input(0, j) = original;

        float numerical_grad = (loss_plus - loss_minus) / (2.0f * epsilon);

        EXPECT_NEAR(grad_input(0, j), numerical_grad, 5e-3f);  // Relaxed tolerance
    }
}

TEST(LayerNormNumericalGradientTest, MultipleSamples) {
    LayerNorm ln(4);
    ln.learning_rate = 0.0f;

    Matrix input(2, 4);
    input(0, 0) = 0.5f;
    input(0, 1) = 1.0f;
    input(0, 2) = 1.5f;
    input(0, 3) = 2.0f;
    input(1, 0) = 2.5f;
    input(1, 1) = 3.0f;
    input(1, 2) = 3.5f;
    input(1, 3) = 4.0f;

    Matrix output = ln.forward(input);

    Matrix grad_output(2, 4);
    for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < 4; ++j) {
            grad_output(i, j) = 1.0f;
        }
    }

    ln.zero_grad();
    Matrix grad_input = ln.backward(grad_output);

    float epsilon = 1e-4f;

    // Check gradients for first sample only (to keep test fast)
    for (int j = 0; j < 4; ++j) {
        float original = input(0, j);

        input(0, j) = original + epsilon;
        Matrix out_plus = ln.forward(input);
        float loss_plus = 0.0f;
        for (int i = 0; i < 2; ++i) {
            for (int k = 0; k < 4; ++k) {
                loss_plus += out_plus(i, k);
            }
        }

        input(0, j) = original - epsilon;
        Matrix out_minus = ln.forward(input);
        float loss_minus = 0.0f;
        for (int i = 0; i < 2; ++i) {
            for (int k = 0; k < 4; ++k) {
                loss_minus += out_minus(i, k);
            }
        }

        input(0, j) = original;

        float numerical_grad = (loss_plus - loss_minus) / (2.0f * epsilon);

        EXPECT_NEAR(grad_input(0, j), numerical_grad, 5e-3f);
    }
}

TEST(LayerNormNumericalGradientTest, WithCustomAffineParameters) {
    LayerNorm ln(3);
    ln.learning_rate = 0.0f;

    // Custom gamma and beta
    Matrix gamma(1, 3);
    gamma(0, 0) = 1.5f;
    gamma(0, 1) = 2.0f;
    gamma(0, 2) = 0.5f;
    ln.set_gamma(gamma);

    Matrix beta(1, 3);
    beta(0, 0) = 0.1f;
    beta(0, 1) = -0.2f;
    beta(0, 2) = 0.3f;
    ln.set_beta(beta);

    Matrix input(1, 3);
    input(0, 0) = 1.0f;
    input(0, 1) = 2.0f;
    input(0, 2) = 3.0f;

    Matrix output = ln.forward(input);

    Matrix grad_output(1, 3);
    for (int j = 0; j < 3; ++j) {
        grad_output(0, j) = 1.0f;
    }

    ln.zero_grad();
    Matrix grad_input = ln.backward(grad_output);

    float epsilon = 1e-4f;

    for (int j = 0; j < 3; ++j) {
        float original = input(0, j);

        input(0, j) = original + epsilon;
        Matrix out_plus = ln.forward(input);
        float loss_plus = 0.0f;
        for (int k = 0; k < 3; ++k) {
            loss_plus += out_plus(0, k);
        }

        input(0, j) = original - epsilon;
        Matrix out_minus = ln.forward(input);
        float loss_minus = 0.0f;
        for (int k = 0; k < 3; ++k) {
            loss_minus += out_minus(0, k);
        }

        input(0, j) = original;

        float numerical_grad = (loss_plus - loss_minus) / (2.0f * epsilon);

        EXPECT_NEAR(grad_input(0, j), numerical_grad, 5e-3f);
    }
}

// ============================================================================
// Test Suite 6: Parameter Setting and Validation
// ============================================================================

TEST(LayerNormParameterTest, SetGammaValidDimension) {
    LayerNorm ln(5);

    Matrix new_gamma(1, 5);
    for (int j = 0; j < 5; ++j) {
        new_gamma(0, j) = 2.5f;
    }

    ln.set_gamma(new_gamma);

    const Matrix& gamma = ln.get_gamma();
    for (int j = 0; j < 5; ++j) {
        EXPECT_FLOAT_EQ(gamma(0, j), 2.5f);
    }
}

TEST(LayerNormParameterTest, SetBetaValidDimension) {
    LayerNorm ln(5);

    Matrix new_beta(1, 5);
    for (int j = 0; j < 5; ++j) {
        new_beta(0, j) = -1.5f;
    }

    ln.set_beta(new_beta);

    const Matrix& beta = ln.get_beta();
    for (int j = 0; j < 5; ++j) {
        EXPECT_FLOAT_EQ(beta(0, j), -1.5f);
    }
}

TEST(LayerNormParameterTest, CustomLearningRate) {
    LayerNorm ln(4);

    ln.learning_rate = 0.01f;
    EXPECT_FLOAT_EQ(ln.learning_rate, 0.01f);

    ln.learning_rate = 0.0001f;
    EXPECT_FLOAT_EQ(ln.learning_rate, 0.0001f);
}

// ============================================================================
// Test Suite 7: Edge Cases
// ============================================================================

TEST(LayerNormEdgeCaseTest, SingleFeatureDimension) {
    LayerNorm ln(1);

    Matrix input(3, 1);
    input(0, 0) = 5.0f;
    input(1, 0) = 10.0f;
    input(2, 0) = 15.0f;

    Matrix output = ln.forward(input);

    // With single feature, variance is 0, so output should be beta (0.0)
    EXPECT_NEAR(output(0, 0), 0.0f, 1e-3f);
    EXPECT_NEAR(output(1, 0), 0.0f, 1e-3f);
    EXPECT_NEAR(output(2, 0), 0.0f, 1e-3f);
}

TEST(LayerNormEdgeCaseTest, SingleSampleSingleFeature) {
    LayerNorm ln(1);

    Matrix input(1, 1);
    input(0, 0) = 42.0f;

    Matrix output = ln.forward(input);

    // Should handle gracefully
    EXPECT_TRUE(std::isfinite(output(0, 0)));
}

TEST(LayerNormEdgeCaseTest, VeryLargeValues) {
    LayerNorm ln(4);

    Matrix input(1, 4);
    input(0, 0) = 1000.0f;
    input(0, 1) = 2000.0f;
    input(0, 2) = 3000.0f;
    input(0, 3) = 4000.0f;

    Matrix output = ln.forward(input);

    // Should normalize properly
    float mean = 0.0f;
    for (int j = 0; j < 4; ++j) {
        mean += output(0, j);
    }
    mean /= 4.0f;
    EXPECT_NEAR(mean, 0.0f, 1e-3f);

    // All values should be finite
    for (int j = 0; j < 4; ++j) {
        EXPECT_TRUE(std::isfinite(output(0, j)));
    }
}

TEST(LayerNormEdgeCaseTest, VerySmallValues) {
    LayerNorm ln(4);

    Matrix input(1, 4);
    input(0, 0) = 1e-6f;
    input(0, 1) = 2e-6f;
    input(0, 2) = 3e-6f;
    input(0, 3) = 4e-6f;

    Matrix output = ln.forward(input);

    // Should still normalize
    for (int j = 0; j < 4; ++j) {
        EXPECT_TRUE(std::isfinite(output(0, j)));
    }
}

TEST(LayerNormEdgeCaseTest, NegativeValues) {
    LayerNorm ln(5);

    Matrix input(1, 5);
    input(0, 0) = -10.0f;
    input(0, 1) = -5.0f;
    input(0, 2) = 0.0f;
    input(0, 3) = 5.0f;
    input(0, 4) = 10.0f;

    Matrix output = ln.forward(input);

    // Should normalize to zero mean
    float mean = 0.0f;
    for (int j = 0; j < 5; ++j) {
        mean += output(0, j);
    }
    mean /= 5.0f;
    EXPECT_NEAR(mean, 0.0f, 1e-5f);
}

TEST(LayerNormEdgeCaseTest, ZeroVarianceWithEpsilon) {
    LayerNorm ln(3, 1e-3f);  // Larger epsilon

    Matrix input(1, 3);
    input(0, 0) = 5.0f;
    input(0, 1) = 5.0f;
    input(0, 2) = 5.0f;

    Matrix output = ln.forward(input);

    // Epsilon should prevent division by zero
    for (int j = 0; j < 3; ++j) {
        EXPECT_TRUE(std::isfinite(output(0, j)));
    }
}

// ============================================================================
// Test Suite 8: Training Simulation
// ============================================================================

TEST(LayerNormTrainingTest, ParameterUpdatesOverIterations) {
    LayerNorm ln(4);
    ln.learning_rate = 0.01f;

    Matrix gamma_initial = ln.get_gamma();
    Matrix beta_initial = ln.get_beta();

    // Simulate 10 training iterations
    for (int iter = 0; iter < 10; ++iter) {
        Matrix input(2, 4);
        for (int i = 0; i < 2; ++i) {
            for (int j = 0; j < 4; ++j) {
                input(i, j) = static_cast<float>(iter + i + j) * 0.1f;
            }
        }

        Matrix output = ln.forward(input);

        Matrix grad_output(2, 4);
        for (int i = 0; i < 2; ++i) {
            for (int j = 0; j < 4; ++j) {
                grad_output(i, j) = 0.1f;
            }
        }

        ln.zero_grad();
        ln.backward(grad_output);
        ln.update_weights();
    }

    // Parameters should have changed
    const Matrix& gamma_final = ln.get_gamma();
    const Matrix& beta_final = ln.get_beta();

    float gamma_diff = 0.0f;
    float beta_diff = 0.0f;
    for (int j = 0; j < 4; ++j) {
        gamma_diff += std::abs(gamma_final(0, j) - gamma_initial(0, j));
        beta_diff += std::abs(beta_final(0, j) - beta_initial(0, j));
    }

    EXPECT_GT(gamma_diff, 1e-4f);
    EXPECT_GT(beta_diff, 1e-4f);
}

TEST(LayerNormTrainingTest, ConvergenceWithSimpleLoss) {
    LayerNorm ln(3);
    ln.learning_rate = 0.1f;

    Matrix target(1, 3);
    target(0, 0) = 1.0f;
    target(0, 1) = 2.0f;
    target(0, 2) = 3.0f;

    float initial_loss = 0.0f;

    // Training iterations
    for (int iter = 0; iter < 50; ++iter) {
        Matrix input(1, 3);
        input(0, 0) = 0.5f;
        input(0, 1) = 1.5f;
        input(0, 2) = 2.5f;

        Matrix output = ln.forward(input);

        // Compute simple MSE loss
        float loss = 0.0f;
        Matrix grad(1, 3);
        for (int j = 0; j < 3; ++j) {
            float diff = output(0, j) - target(0, j);
            loss += diff * diff;
            grad(0, j) = 2.0f * diff;
        }
        loss /= 3.0f;

        if (iter == 0) {
            initial_loss = loss;
        }

        ln.zero_grad();
        ln.backward(grad);
        ln.update_weights();
    }

    // Final forward to check loss
    Matrix input(1, 3);
    input(0, 0) = 0.5f;
    input(0, 1) = 1.5f;
    input(0, 2) = 2.5f;

    Matrix output = ln.forward(input);
    float final_loss = 0.0f;
    for (int j = 0; j < 3; ++j) {
        float diff = output(0, j) - target(0, j);
        final_loss += diff * diff;
    }
    final_loss /= 3.0f;

    // Loss should decrease (learning happening)
    EXPECT_LT(final_loss, initial_loss);
}

// ============================================================================
// Test Suite 9: Integration Tests
// ============================================================================

TEST(LayerNormIntegrationTest, ChainedLayerNorms) {
    LayerNorm ln1(4);
    LayerNorm ln2(4);

    Matrix input(2, 4);
    for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < 4; ++j) {
            input(i, j) = static_cast<float>(i + j);
        }
    }

    // Forward through two layers
    Matrix out1 = ln1.forward(input);
    Matrix out2 = ln2.forward(out1);

    // Backward through two layers
    Matrix grad(2, 4);
    for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < 4; ++j) {
            grad(i, j) = 1.0f;
        }
    }

    ln2.zero_grad();
    Matrix grad2 = ln2.backward(grad);

    ln1.zero_grad();
    Matrix grad1 = ln1.backward(grad2);

    // Gradient should propagate through both layers
    EXPECT_EQ(grad1.rows, 2);
    EXPECT_EQ(grad1.cols, 4);

    // Check that gradients are reasonable
    for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < 4; ++j) {
            EXPECT_TRUE(std::isfinite(grad1(i, j)));
        }
    }
}

TEST(LayerNormIntegrationTest, WithResidualConnection) {
    LayerNorm ln(5);

    Matrix input(1, 5);
    for (int j = 0; j < 5; ++j) {
        input(0, j) = static_cast<float>(j) * 0.5f;
    }

    // Simulate sublayer output
    Matrix sublayer_output(1, 5);
    for (int j = 0; j < 5; ++j) {
        sublayer_output(0, j) = static_cast<float>(j) * 0.3f;
    }

    // Residual connection: input + sublayer_output
    Matrix residual(1, 5);
    for (int j = 0; j < 5; ++j) {
        residual(0, j) = input(0, j) + sublayer_output(0, j);
    }

    // LayerNorm after residual
    Matrix normalized = ln.forward(residual);

    // Should be normalized
    float mean = 0.0f;
    for (int j = 0; j < 5; ++j) {
        mean += normalized(0, j);
    }
    mean /= 5.0f;
    EXPECT_NEAR(mean, 0.0f, 1e-5f);
}

TEST(LayerNormIntegrationTest, BatchProcessing) {
    LayerNorm ln(8);

    std::vector<Matrix> batches;
    for (int b = 0; b < 5; ++b) {
        Matrix batch(4, 8);
        for (int i = 0; i < 4; ++i) {
            for (int j = 0; j < 8; ++j) {
                batch(i, j) = static_cast<float>(b * 10 + i + j) * 0.01f;
            }
        }
        batches.push_back(batch);
    }

    // Process all batches
    for (size_t b = 0; b < batches.size(); ++b) {
        Matrix output = ln.forward(batches[b]);

        Matrix grad(4, 8);
        for (int i = 0; i < 4; ++i) {
            for (int j = 0; j < 8; ++j) {
                grad(i, j) = 0.1f;
            }
        }

        ln.zero_grad();
        ln.backward(grad);
        ln.update_weights();
    }

    // Parameters should have been updated across batches
    const Matrix& gamma = ln.get_gamma();
    bool parameters_changed = false;
    for (int j = 0; j < 8; ++j) {
        if (std::abs(gamma(0, j) - 1.0f) > 1e-4f) {
            parameters_changed = true;
            break;
        }
    }
    EXPECT_TRUE(parameters_changed);
}

// ============================================================================
// Test Suite 10: Optimizer Integration Tests
// ============================================================================

TEST(LayerNormOptimizerTest, SetOptimizerBasic) {
    LayerNorm ln(64);
    Optimizer optimizer(OptimizerType::ADAM, 0.001f);

    // Should not throw
    EXPECT_NO_THROW(ln.set_optimizer(&optimizer));
}

TEST(LayerNormOptimizerTest, SetOptimizerNullptr) {
    LayerNorm ln(64);

    // Setting nullptr should work (revert to simple gradient descent)
    EXPECT_NO_THROW(ln.set_optimizer(nullptr));
}

TEST(LayerNormOptimizerTest, UpdateWithOptimizer) {
    LayerNorm ln(32);
    Optimizer optimizer(OptimizerType::ADAM, 0.01f);
    optimizer.set_betas(0.9f, 0.999f);

    ln.set_optimizer(&optimizer);

    // Create input
    Matrix input(4, 32);
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 32; ++j) {
            input(i, j) = 0.1f * (i + j);
        }
    }

    Matrix output = ln.forward(input);

    Matrix grad_output(4, 32);
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 32; ++j) {
            grad_output(i, j) = 0.01f;
        }
    }

    ln.backward(grad_output);

    // Update should use optimizer->step()
    EXPECT_NO_THROW(ln.update_weights());
}

TEST(LayerNormOptimizerTest, UpdateWithoutOptimizer) {
    LayerNorm ln(32);
    ln.learning_rate = 0.01f;

    // No optimizer set - should use simple gradient descent
    Matrix input(4, 32);
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 32; ++j) {
            input(i, j) = 0.1f * (i + j);
        }
    }

    Matrix output = ln.forward(input);

    Matrix grad_output(4, 32);
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 32; ++j) {
            grad_output(i, j) = 0.01f;
        }
    }

    ln.backward(grad_output);

    // Update should use apply_gradients fallback
    EXPECT_NO_THROW(ln.update_weights());
}

TEST(LayerNormOptimizerTest, OptimizerVsSimpleGradientDescent) {
    // Create two identical LayerNorm instances
    LayerNorm ln_with_opt(32);
    LayerNorm ln_without_opt(32);

    // Set up optimizer for first instance
    Optimizer optimizer(OptimizerType::ADAM, 0.01f);
    ln_with_opt.set_optimizer(&optimizer);

    // Set same learning rate for second instance
    ln_without_opt.learning_rate = 0.01f;

    // Create identical input
    Matrix input(4, 32);
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 32; ++j) {
            input(i, j) = 0.1f * (i + j);
        }
    }

    // Forward pass for both
    Matrix output1 = ln_with_opt.forward(input);
    Matrix output2 = ln_without_opt.forward(input);

    // Create gradient
    Matrix grad_output(4, 32);
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 32; ++j) {
            grad_output(i, j) = 0.01f;
        }
    }

    // Backward pass for both
    ln_with_opt.backward(grad_output);
    ln_without_opt.backward(grad_output);

    // Update weights
    ln_with_opt.update_weights();
    ln_without_opt.update_weights();

    // Both paths should work without errors
    EXPECT_TRUE(true);
}

TEST(LayerNormOptimizerTest, MultipleUpdatesWithOptimizer) {
    LayerNorm ln(32);
    Optimizer optimizer(OptimizerType::ADAM, 0.01f);
    optimizer.set_betas(0.9f, 0.999f);

    ln.set_optimizer(&optimizer);

    Matrix input(4, 32);
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 32; ++j) {
            input(i, j) = 0.1f * (i + j);
        }
    }

    Matrix grad_output(4, 32);
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 32; ++j) {
            grad_output(i, j) = 0.01f;
        }
    }

    // Perform multiple training steps
    for (int step = 0; step < 10; ++step) {
        Matrix output = ln.forward(input);
        ln.backward(grad_output);
        ln.update_weights();
    }

    // Should complete without errors
    EXPECT_TRUE(true);
}

TEST(LayerNormOptimizerTest, SwitchOptimizer) {
    LayerNorm ln(32);

    Optimizer optimizer1(OptimizerType::ADAM, 0.01f);
    ln.set_optimizer(&optimizer1);

    // Perform some training
    Matrix input(4, 32);
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 32; ++j) {
            input(i, j) = 0.1f;
        }
    }

    Matrix output = ln.forward(input);
    Matrix grad_output(4, 32);
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 32; ++j) {
            grad_output(i, j) = 0.01f;
        }
    }

    ln.backward(grad_output);
    ln.update_weights();

    // Switch to different optimizer
    Optimizer optimizer2(OptimizerType::ADAM, 0.001f);
    optimizer2.set_betas(0.95f, 0.999f);
    ln.set_optimizer(&optimizer2);

    // Continue training with new optimizer
    output = ln.forward(input);
    ln.backward(grad_output);

    EXPECT_NO_THROW(ln.update_weights());
}

TEST(LayerNormOptimizerTest, OptimizerWithDifferentLearningRates) {
    LayerNorm ln(32);
    Optimizer optimizer(OptimizerType::ADAM, 0.1f);  // High learning rate

    ln.set_optimizer(&optimizer);

    Matrix input(4, 32);
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 32; ++j) {
            input(i, j) = 0.1f;
        }
    }

    Matrix output1 = ln.forward(input);

    Matrix grad_output(4, 32);
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 32; ++j) {
            grad_output(i, j) = 0.1f;
        }
    }

    ln.backward(grad_output);
    ln.update_weights();

    // After update, output should be different
    Matrix output2 = ln.forward(input);

    bool outputs_different = false;
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 32; ++j) {
            if (std::abs(output1(i, j) - output2(i, j)) > 1e-6f) {
                outputs_different = true;
                break;
            }
        }
        if (outputs_different)
            break;
    }

    EXPECT_TRUE(outputs_different);
}

TEST(LayerNormOptimizerTest, RegisterParametersExplicit) {
    LayerNorm ln(32);
    Optimizer optimizer(OptimizerType::ADAM, 0.01f);

    ln.set_optimizer(&optimizer);

    // register_parameters() should be called automatically by set_optimizer()
    // but we can call it again without issues
    EXPECT_NO_THROW(ln.register_parameters());
}

TEST(LayerNormOptimizerTest, ParametersChangeWithOptimizer) {
    LayerNorm ln(32);
    Optimizer optimizer(OptimizerType::ADAM, 0.01f);

    ln.set_optimizer(&optimizer);

    // Store initial parameters
    Matrix gamma_initial(1, 32);
    Matrix beta_initial(1, 32);
    const Matrix& gamma = ln.get_gamma();
    const Matrix& beta = ln.get_beta();
    for (int j = 0; j < 32; ++j) {
        gamma_initial(0, j) = gamma(0, j);
        beta_initial(0, j) = beta(0, j);
    }

    Matrix input(4, 32);
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 32; ++j) {
            input(i, j) = 0.1f * (i + j);
        }
    }

    Matrix output = ln.forward(input);

    Matrix grad_output(4, 32);
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 32; ++j) {
            grad_output(i, j) = 0.1f;
        }
    }

    ln.backward(grad_output);
    ln.update_weights();

    // Parameters should have changed
    const Matrix& gamma_final = ln.get_gamma();
    const Matrix& beta_final = ln.get_beta();

    bool gamma_changed = false;
    bool beta_changed = false;

    for (int j = 0; j < 32; ++j) {
        if (std::abs(gamma_final(0, j) - gamma_initial(0, j)) > 1e-6f) {
            gamma_changed = true;
        }
        if (std::abs(beta_final(0, j) - beta_initial(0, j)) > 1e-6f) {
            beta_changed = true;
        }
    }

    EXPECT_TRUE(gamma_changed);
    EXPECT_TRUE(beta_changed);
}

TEST(LayerNormOptimizerTest, LearningRateScheduling) {
    LayerNorm ln(32);
    Optimizer optimizer(OptimizerType::ADAM, 0.01f);

    ln.set_optimizer(&optimizer);

    Matrix input(4, 32);
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 32; ++j) {
            input(i, j) = 0.1f * (i + j);
        }
    }

    Matrix grad_output(4, 32);
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 32; ++j) {
            grad_output(i, j) = 0.01f;
        }
    }

    // Train with different learning rates
    for (int step = 0; step < 5; ++step) {
        float lr = 0.01f * (1.0f / (step + 1));  // Decreasing schedule
        optimizer.set_learning_rate(lr);

        Matrix output = ln.forward(input);
        ln.backward(grad_output);
        ln.update_weights();
    }

    // Should complete without errors
    EXPECT_TRUE(true);
}

TEST(LayerNormOptimizerTest, BackwardCompatibilityNoOptimizer) {
    // Test that old code without optimizer still works
    LayerNorm ln(32);
    ln.learning_rate = 0.01f;

    Matrix input(4, 32);
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 32; ++j) {
            input(i, j) = 0.1f * (i + j);
        }
    }

    Matrix gamma_before = ln.get_gamma();

    Matrix output = ln.forward(input);

    Matrix grad_output(4, 32);
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 32; ++j) {
            grad_output(i, j) = 0.1f;
        }
    }

    ln.backward(grad_output);
    ln.update_weights();  // Should use simple gradient descent

    Matrix gamma_after = ln.get_gamma();

    // Parameters should have changed
    bool changed = false;
    for (int j = 0; j < 32; ++j) {
        if (std::abs(gamma_after(0, j) - gamma_before(0, j)) > 1e-6f) {
            changed = true;
            break;
        }
    }

    EXPECT_TRUE(changed);
}

// ============================================================================
// Main Function
// ============================================================================

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
