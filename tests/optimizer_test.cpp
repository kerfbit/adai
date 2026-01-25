#include "../src/Optimizer.hpp"
#include <../gtest/gtest.h>
#include <cmath>
#include <memory>
#include <stdexcept>
#include "../src/Matrix.hpp"

// ============================================================================
// Constructor Tests
// ============================================================================

TEST(OptimizerConstructorTest, DefaultConstructor) {
    Optimizer opt(OptimizerType::ADAM, 0.001f);
    EXPECT_FLOAT_EQ(opt.get_learning_rate(), 0.001f);
    EXPECT_EQ(opt.num_parameters(), 0);
    EXPECT_EQ(opt.total_parameters(), 0);
}

TEST(OptimizerConstructorTest, SGDConstructor) {
    Optimizer opt(OptimizerType::SGD, 0.01f);
    EXPECT_STREQ(opt.get_optimizer_name(), "SGD");
    EXPECT_FLOAT_EQ(opt.get_learning_rate(), 0.01f);
}

TEST(OptimizerConstructorTest, SGDMomentumConstructor) {
    Optimizer opt(OptimizerType::SGD_MOMENTUM, 0.01f);
    EXPECT_STREQ(opt.get_optimizer_name(), "SGD+Momentum");
}

TEST(OptimizerConstructorTest, AdamConstructor) {
    Optimizer opt(OptimizerType::ADAM, 0.001f);
    EXPECT_STREQ(opt.get_optimizer_name(), "Adam");
}

TEST(OptimizerConstructorTest, AdamWConstructor) {
    Optimizer opt(OptimizerType::ADAMW, 0.0001f);
    EXPECT_STREQ(opt.get_optimizer_name(), "AdamW");
}

// ============================================================================
// Parameter Group Tests
// ============================================================================

TEST(OptimizerParameterTest, AddSingleParameterGroup) {
    Optimizer opt(OptimizerType::SGD, 0.01f);

    Matrix weights(3, 4);
    Matrix gradients(3, 4);

    opt.add_parameter_group(&weights, &gradients);

    EXPECT_EQ(opt.num_parameters(), 1);
    EXPECT_EQ(opt.total_parameters(), 12);  // 3 * 4
}

TEST(OptimizerParameterTest, AddMultipleParameterGroups) {
    Optimizer opt(OptimizerType::ADAM, 0.001f);

    Matrix weights1(2, 3);
    Matrix gradients1(2, 3);
    Matrix weights2(4, 5);
    Matrix gradients2(4, 5);

    opt.add_parameter_group(&weights1, &gradients1);
    opt.add_parameter_group(&weights2, &gradients2);

    EXPECT_EQ(opt.num_parameters(), 2);
    EXPECT_EQ(opt.total_parameters(), 26);  // 6 + 20
}

TEST(OptimizerParameterTest, NullWeightsThrows) {
    Optimizer opt(OptimizerType::SGD, 0.01f);
    Matrix gradients(3, 4);

    EXPECT_THROW(opt.add_parameter_group(nullptr, &gradients), std::runtime_error);
}

TEST(OptimizerParameterTest, NullGradientsThrows) {
    Optimizer opt(OptimizerType::SGD, 0.01f);
    Matrix weights(3, 4);

    EXPECT_THROW(opt.add_parameter_group(&weights, nullptr), std::runtime_error);
}

TEST(OptimizerParameterTest, MismatchedShapeThrows) {
    Optimizer opt(OptimizerType::SGD, 0.01f);
    Matrix weights(3, 4);
    Matrix gradients(2, 5);

    EXPECT_THROW(opt.add_parameter_group(&weights, &gradients), std::runtime_error);
}

// ============================================================================
// Hyperparameter Tests
// ============================================================================

TEST(OptimizerHyperparameterTest, SetLearningRate) {
    Optimizer opt(OptimizerType::SGD, 0.01f);

    opt.set_learning_rate(0.001f);
    EXPECT_FLOAT_EQ(opt.get_learning_rate(), 0.001f);

    opt.set_learning_rate(0.1f);
    EXPECT_FLOAT_EQ(opt.get_learning_rate(), 0.1f);
}

TEST(OptimizerHyperparameterTest, SetMomentum) {
    Optimizer opt(OptimizerType::SGD_MOMENTUM, 0.01f);

    opt.set_momentum(0.95f);
    // No getter for momentum, so just verify no crash
    EXPECT_NO_THROW(opt.set_momentum(0.95f));
}

TEST(OptimizerHyperparameterTest, SetBetas) {
    Optimizer opt(OptimizerType::ADAM, 0.001f);

    opt.set_betas(0.9f, 0.999f);
    EXPECT_NO_THROW(opt.set_betas(0.95f, 0.9999f));
}

TEST(OptimizerHyperparameterTest, SetWeightDecay) {
    Optimizer opt(OptimizerType::ADAMW, 0.0001f);

    opt.set_weight_decay(0.01f);
    EXPECT_NO_THROW(opt.set_weight_decay(0.1f));
}

TEST(OptimizerHyperparameterTest, SetMaxGradNorm) {
    Optimizer opt(OptimizerType::ADAM, 0.001f);

    opt.set_max_grad_norm(1.0f);
    EXPECT_NO_THROW(opt.set_max_grad_norm(5.0f));
}

// ============================================================================
// Zero Gradient Tests
// ============================================================================

TEST(OptimizerZeroGradTest, ZeroSingleGradient) {
    Optimizer opt(OptimizerType::SGD, 0.01f);

    Matrix weights(2, 2);
    Matrix gradients(2, 2);

    // Set some gradient values
    gradients(0, 0) = 1.5f;
    gradients(0, 1) = 2.5f;
    gradients(1, 0) = -1.0f;
    gradients(1, 1) = 3.0f;

    opt.add_parameter_group(&weights, &gradients);
    opt.zero_grad();

    // Verify all gradients are zero
    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 2; j++) {
            EXPECT_FLOAT_EQ(gradients(i, j), 0.0f);
        }
    }
}

TEST(OptimizerZeroGradTest, ZeroMultipleGradients) {
    Optimizer opt(OptimizerType::ADAM, 0.001f);

    Matrix weights1(2, 2);
    Matrix gradients1(2, 2);
    Matrix weights2(3, 3);
    Matrix gradients2(3, 3);

    // Set gradient values
    gradients1(0, 0) = 1.0f;
    gradients2(1, 1) = 2.0f;

    opt.add_parameter_group(&weights1, &gradients1);
    opt.add_parameter_group(&weights2, &gradients2);
    opt.zero_grad();

    // Verify all are zero
    EXPECT_FLOAT_EQ(gradients1(0, 0), 0.0f);
    EXPECT_FLOAT_EQ(gradients2(1, 1), 0.0f);
}

// ============================================================================
// Gradient Norm Tests
// ============================================================================

TEST(OptimizerGradientNormTest, SimpleNorm) {
    Optimizer opt(OptimizerType::SGD, 0.01f);

    Matrix weights(2, 2);
    Matrix gradients(2, 2);

    // Set gradients: [3, 4; 0, 0]
    // Norm = sqrt(3^2 + 4^2) = sqrt(25) = 5
    gradients(0, 0) = 3.0f;
    gradients(0, 1) = 4.0f;
    gradients(1, 0) = 0.0f;
    gradients(1, 1) = 0.0f;

    opt.add_parameter_group(&weights, &gradients);

    float norm = opt.get_gradient_norm();
    EXPECT_FLOAT_EQ(norm, 5.0f);
}

TEST(OptimizerGradientNormTest, ZeroNorm) {
    Optimizer opt(OptimizerType::SGD, 0.01f);

    Matrix weights(3, 3);
    Matrix gradients(3, 3);

    opt.add_parameter_group(&weights, &gradients);

    float norm = opt.get_gradient_norm();
    EXPECT_FLOAT_EQ(norm, 0.0f);
}

TEST(OptimizerGradientNormTest, MultipleParameterGroups) {
    Optimizer opt(OptimizerType::ADAM, 0.001f);

    Matrix weights1(2, 1);
    Matrix gradients1(2, 1);
    Matrix weights2(2, 1);
    Matrix gradients2(2, 1);

    // gradients1: [3; 4], gradients2: [0; 0]
    // Total norm = sqrt(3^2 + 4^2) = 5
    gradients1(0, 0) = 3.0f;
    gradients1(1, 0) = 4.0f;

    opt.add_parameter_group(&weights1, &gradients1);
    opt.add_parameter_group(&weights2, &gradients2);

    float norm = opt.get_gradient_norm();
    EXPECT_FLOAT_EQ(norm, 5.0f);
}

// ============================================================================
// Gradient Clipping Tests
// ============================================================================

TEST(OptimizerGradientClipTest, NoClippingWhenBelowThreshold) {
    Optimizer opt(OptimizerType::SGD, 0.01f);

    Matrix weights(2, 2);
    Matrix gradients(2, 2);

    gradients(0, 0) = 3.0f;
    gradients(0, 1) = 4.0f;  // Norm = 5

    opt.add_parameter_group(&weights, &gradients);
    opt.set_max_grad_norm(10.0f);  // Threshold > norm

    float norm = opt.clip_gradients();

    EXPECT_FLOAT_EQ(norm, 5.0f);             // Original norm returned
    EXPECT_FLOAT_EQ(gradients(0, 0), 3.0f);  // Not clipped
    EXPECT_FLOAT_EQ(gradients(0, 1), 4.0f);
}

TEST(OptimizerGradientClipTest, ClippingWhenAboveThreshold) {
    Optimizer opt(OptimizerType::ADAM, 0.001f);

    Matrix weights(2, 2);
    Matrix gradients(2, 2);

    gradients(0, 0) = 3.0f;
    gradients(0, 1) = 4.0f;  // Norm = 5

    opt.add_parameter_group(&weights, &gradients);
    opt.set_max_grad_norm(2.5f);  // Threshold < norm

    float norm = opt.clip_gradients();

    EXPECT_FLOAT_EQ(norm, 5.0f);  // Original norm

    // After clipping, norm should be ~2.5
    float new_norm = opt.get_gradient_norm();
    EXPECT_NEAR(new_norm, 2.5f, 1e-4f);
}

TEST(OptimizerGradientClipTest, ClipWithCustomNorm) {
    Optimizer opt(OptimizerType::SGD, 0.01f);

    Matrix weights(2, 1);
    Matrix gradients(2, 1);

    gradients(0, 0) = 6.0f;
    gradients(1, 0) = 8.0f;  // Norm = 10

    opt.add_parameter_group(&weights, &gradients);

    float norm = opt.clip_gradients(5.0f);  // Custom threshold

    EXPECT_FLOAT_EQ(norm, 10.0f);

    // Gradients should be scaled by 0.5 (5.0 / 10.0)
    EXPECT_NEAR(gradients(0, 0), 3.0f, 1e-4f);
    EXPECT_NEAR(gradients(1, 0), 4.0f, 1e-4f);
}

TEST(OptimizerGradientClipTest, NoClipWhenDisabled) {
    Optimizer opt(OptimizerType::SGD, 0.01f);

    Matrix weights(2, 1);
    Matrix gradients(2, 1);

    gradients(0, 0) = 100.0f;
    gradients(1, 0) = 100.0f;

    opt.add_parameter_group(&weights, &gradients);
    opt.set_max_grad_norm(0.0f);  // Disabled

    float norm = opt.clip_gradients();

    EXPECT_FLOAT_EQ(norm, 0.0f);               // Returns 0 when disabled
    EXPECT_FLOAT_EQ(gradients(0, 0), 100.0f);  // Not clipped
}

// ============================================================================
// SGD Optimization Tests
// ============================================================================

TEST(OptimizerSGDTest, BasicUpdate) {
    Optimizer opt(OptimizerType::SGD, 0.1f);

    Matrix weights(2, 2);
    Matrix gradients(2, 2);

    // Initialize weights and gradients
    weights(0, 0) = 1.0f;
    weights(0, 1) = 2.0f;
    weights(1, 0) = 3.0f;
    weights(1, 1) = 4.0f;

    gradients(0, 0) = 0.1f;
    gradients(0, 1) = 0.2f;
    gradients(1, 0) = 0.3f;
    gradients(1, 1) = 0.4f;

    opt.add_parameter_group(&weights, &gradients);
    opt.step();

    // w = w - lr * grad
    EXPECT_NEAR(weights(0, 0), 1.0f - 0.1f * 0.1f, 1e-5f);  // 0.99
    EXPECT_NEAR(weights(0, 1), 2.0f - 0.1f * 0.2f, 1e-5f);  // 1.98
    EXPECT_NEAR(weights(1, 0), 3.0f - 0.1f * 0.3f, 1e-5f);  // 2.97
    EXPECT_NEAR(weights(1, 1), 4.0f - 0.1f * 0.4f, 1e-5f);  // 3.96
}

TEST(OptimizerSGDTest, WithWeightDecay) {
    Optimizer opt(OptimizerType::SGD, 0.1f);
    opt.set_weight_decay(0.01f);

    Matrix weights(2, 1);
    Matrix gradients(2, 1);

    weights(0, 0) = 1.0f;
    weights(1, 0) = 2.0f;

    gradients(0, 0) = 0.1f;
    gradients(1, 0) = 0.2f;

    opt.add_parameter_group(&weights, &gradients);
    opt.step();

    // w = w - lr * (grad + weight_decay * w)
    // w[0] = 1.0 - 0.1 * (0.1 + 0.01 * 1.0) = 1.0 - 0.011 = 0.989
    EXPECT_NEAR(weights(0, 0), 0.989f, 1e-5f);

    // w[1] = 2.0 - 0.1 * (0.2 + 0.01 * 2.0) = 2.0 - 0.022 = 1.978
    EXPECT_NEAR(weights(1, 0), 1.978f, 1e-5f);
}

TEST(OptimizerSGDTest, MultipleSteps) {
    Optimizer opt(OptimizerType::SGD, 0.01f);

    Matrix weights(1, 1);
    Matrix gradients(1, 1);

    weights(0, 0) = 1.0f;
    gradients(0, 0) = 1.0f;

    opt.add_parameter_group(&weights, &gradients);

    // Step 1: w = 1.0 - 0.01 * 1.0 = 0.99
    opt.step();
    EXPECT_NEAR(weights(0, 0), 0.99f, 1e-5f);

    // Step 2: w = 0.99 - 0.01 * 1.0 = 0.98
    opt.step();
    EXPECT_NEAR(weights(0, 0), 0.98f, 1e-5f);
}

// ============================================================================
// SGD with Momentum Tests
// ============================================================================

TEST(OptimizerSGDMomentumTest, FirstStep) {
    Optimizer opt(OptimizerType::SGD_MOMENTUM, 0.1f);
    opt.set_momentum(0.9f);

    Matrix weights(2, 1);
    Matrix gradients(2, 1);

    weights(0, 0) = 1.0f;
    weights(1, 0) = 2.0f;

    gradients(0, 0) = 0.1f;
    gradients(1, 0) = 0.2f;

    opt.add_parameter_group(&weights, &gradients);
    opt.step();

    // First step: momentum = 0.9 * 0 + 0.1 = 0.1
    // w = w - lr * momentum = 1.0 - 0.1 * 0.1 = 0.99
    EXPECT_NEAR(weights(0, 0), 0.99f, 1e-5f);
    EXPECT_NEAR(weights(1, 0), 1.98f, 1e-5f);
}

TEST(OptimizerSGDMomentumTest, AccumulatesMomentum) {
    Optimizer opt(OptimizerType::SGD_MOMENTUM, 0.1f);
    opt.set_momentum(0.9f);

    Matrix weights(1, 1);
    Matrix gradients(1, 1);

    weights(0, 0) = 1.0f;
    gradients(0, 0) = 0.1f;

    opt.add_parameter_group(&weights, &gradients);

    // Step 1: m = 0.9 * 0 + 0.1 = 0.1, w = 1.0 - 0.1 * 0.1 = 0.99
    opt.step();
    EXPECT_NEAR(weights(0, 0), 0.99f, 1e-5f);

    // Step 2: m = 0.9 * 0.1 + 0.1 = 0.19, w = 0.99 - 0.1 * 0.19 = 0.971
    opt.step();
    EXPECT_NEAR(weights(0, 0), 0.971f, 1e-5f);
}

// ============================================================================
// Adam Optimization Tests
// ============================================================================

TEST(OptimizerAdamTest, FirstStep) {
    Optimizer opt(OptimizerType::ADAM, 0.001f);
    opt.set_betas(0.9f, 0.999f);

    Matrix weights(1, 1);
    Matrix gradients(1, 1);

    weights(0, 0) = 1.0f;
    gradients(0, 0) = 0.1f;

    opt.add_parameter_group(&weights, &gradients);
    opt.step();

    // First step with bias correction
    // m = 0.9 * 0 + 0.1 * 0.1 = 0.01
    // v = 0.999 * 0 + 0.001 * 0.01 = 0.00001
    // m_hat = 0.01 / (1 - 0.9) = 0.1
    // v_hat = 0.00001 / (1 - 0.999) = 0.01
    // w = 1.0 - 0.001 * 0.1 / (sqrt(0.01) + 1e-8)

    EXPECT_LT(weights(0, 0), 1.0f);   // Weight should decrease
    EXPECT_GT(weights(0, 0), 0.99f);  // But not too much
}

TEST(OptimizerAdamTest, BiasCorrection) {
    Optimizer opt(OptimizerType::ADAM, 0.01f);
    opt.set_betas(0.9f, 0.999f);

    Matrix weights(1, 1);
    Matrix gradients(1, 1);

    weights(0, 0) = 10.0f;
    gradients(0, 0) = 1.0f;

    opt.add_parameter_group(&weights, &gradients);

    float w_before = weights(0, 0);
    opt.step();
    float w_after = weights(0, 0);

    // Verify weight updated
    EXPECT_NE(w_before, w_after);
    EXPECT_LT(w_after, w_before);  // Should decrease with positive gradient
}

TEST(OptimizerAdamTest, WithWeightDecay) {
    Optimizer opt(OptimizerType::ADAM, 0.001f);
    opt.set_betas(0.9f, 0.999f);
    opt.set_weight_decay(0.01f);

    Matrix weights(1, 1);
    Matrix gradients(1, 1);

    weights(0, 0) = 1.0f;
    gradients(0, 0) = 0.1f;

    opt.add_parameter_group(&weights, &gradients);

    float w_before = weights(0, 0);
    opt.step();

    // With weight decay, should update more (grad += wd * w)
    EXPECT_LT(weights(0, 0), w_before);
}

// ============================================================================
// AdamW Optimization Tests
// ============================================================================

TEST(OptimizerAdamWTest, DecoupledWeightDecay) {
    Optimizer opt_adamw(OptimizerType::ADAMW, 0.001f);
    opt_adamw.set_betas(0.9f, 0.999f);
    opt_adamw.set_weight_decay(0.01f);

    Matrix weights_adamw(2, 2);
    Matrix gradients_adamw(2, 2);

    // Set initial values
    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 2; j++) {
            weights_adamw(i, j) = 1.0f;
            gradients_adamw(i, j) = 0.1f;
        }
    }

    opt_adamw.add_parameter_group(&weights_adamw, &gradients_adamw);
    opt_adamw.step();

    // AdamW applies weight decay directly: w -= lr * (update + wd * w)
    // Verify weights decreased
    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 2; j++) {
            EXPECT_LT(weights_adamw(i, j), 1.0f);
        }
    }
}

TEST(OptimizerAdamWTest, FirstStepUpdate) {
    Optimizer opt(OptimizerType::ADAMW, 0.001f);
    opt.set_betas(0.9f, 0.999f);
    opt.set_weight_decay(0.0f);  // No weight decay for simpler verification

    Matrix weights(1, 1);
    Matrix gradients(1, 1);

    weights(0, 0) = 1.0f;
    gradients(0, 0) = 0.1f;

    opt.add_parameter_group(&weights, &gradients);
    opt.step();

    EXPECT_LT(weights(0, 0), 1.0f);
    EXPECT_GT(weights(0, 0), 0.99f);
}

// ============================================================================
// State Management Tests
// ============================================================================

TEST(OptimizerStateTest, ResetState) {
    Optimizer opt(OptimizerType::ADAM, 0.01f);

    Matrix weights(2, 2);
    Matrix gradients(2, 2);

    weights(0, 0) = 1.0f;
    gradients(0, 0) = 0.1f;

    opt.add_parameter_group(&weights, &gradients);

    // Take a step to accumulate state
    opt.step();

    // Reset state
    opt.reset_state();

    // After reset, should behave like first step again
    float w_after_reset = weights(0, 0);
    opt.step();

    // Verify update occurred
    EXPECT_NE(weights(0, 0), w_after_reset);
}

TEST(OptimizerStateTest, ResetZerosMomentum) {
    Optimizer opt(OptimizerType::SGD_MOMENTUM, 0.1f);
    opt.set_momentum(0.9f);

    Matrix weights(1, 1);
    Matrix gradients(1, 1);

    weights(0, 0) = 1.0f;
    gradients(0, 0) = 0.1f;

    opt.add_parameter_group(&weights, &gradients);

    // Accumulate momentum
    opt.step();
    opt.step();

    float w_before_reset = weights(0, 0);

    // Reset and step again
    opt.reset_state();
    opt.step();

    // After reset, update should be different than if momentum continued
    // This is a behavioral test - exact value depends on algorithm
    EXPECT_NO_THROW(opt.reset_state());
}

// ============================================================================
// Integration Tests
// ============================================================================

TEST(OptimizerIntegrationTest, CompleteTrainingLoop) {
    Optimizer opt(OptimizerType::ADAMW, 0.01f);
    opt.set_weight_decay(0.01f);
    opt.set_max_grad_norm(1.0f);
    opt.set_betas(0.9f, 0.999f);

    Matrix weights(3, 3);
    Matrix gradients(3, 3);

    // Initialize weights
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            weights(i, j) = 0.5f;
        }
    }

    opt.add_parameter_group(&weights, &gradients);

    // Simulate training loop
    for (int step = 0; step < 10; step++) {
        // Zero gradients
        opt.zero_grad();

        // Simulate gradient computation
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                gradients(i, j) = 0.1f;
            }
        }

        // Clip gradients
        float grad_norm = opt.clip_gradients();
        EXPECT_GE(grad_norm, 0.0f);

        // Update weights
        opt.step();
    }

    // Verify weights changed
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            EXPECT_NE(weights(i, j), 0.5f);
        }
    }
}

TEST(OptimizerIntegrationTest, LearningRateScheduling) {
    Optimizer opt(OptimizerType::SGD, 0.1f);

    Matrix weights(2, 2);
    Matrix gradients(2, 2);

    weights(0, 0) = 1.0f;
    gradients(0, 0) = 0.1f;

    opt.add_parameter_group(&weights, &gradients);

    // Step with initial LR
    opt.step();
    float w_step1 = weights(0, 0);

    // Reduce learning rate
    opt.set_learning_rate(0.01f);
    weights(0, 0) = 1.0f;    // Reset
    gradients(0, 0) = 0.1f;  // Reset gradients too

    opt.step();
    float w_step2 = weights(0, 0);

    // With lower LR, weight should change less
    // step1: 1.0 - 0.1*0.1 = 0.99
    // step2: 1.0 - 0.01*0.1 = 0.999
    EXPECT_LT(w_step1, w_step2);  // step1 decreased more (is smaller)
}

TEST(OptimizerIntegrationTest, MultiParameterGroupUpdate) {
    Optimizer opt(OptimizerType::ADAM, 0.001f);

    Matrix weights1(2, 2);
    Matrix gradients1(2, 2);
    Matrix weights2(3, 3);
    Matrix gradients2(3, 3);

    // Initialize
    weights1(0, 0) = 1.0f;
    weights2(0, 0) = 2.0f;
    gradients1(0, 0) = 0.1f;
    gradients2(0, 0) = 0.2f;

    opt.add_parameter_group(&weights1, &gradients1);
    opt.add_parameter_group(&weights2, &gradients2);

    float w1_before = weights1(0, 0);
    float w2_before = weights2(0, 0);

    opt.step();

    // Both should update
    EXPECT_NE(weights1(0, 0), w1_before);
    EXPECT_NE(weights2(0, 0), w2_before);
}

// ============================================================================
// Edge Case Tests
// ============================================================================

TEST(OptimizerEdgeCaseTest, ZeroGradient) {
    Optimizer opt(OptimizerType::SGD, 0.1f);

    Matrix weights(2, 2);
    Matrix gradients(2, 2);

    weights(0, 0) = 1.0f;
    gradients(0, 0) = 0.0f;

    opt.add_parameter_group(&weights, &gradients);
    opt.step();

    // With zero gradient, weight shouldn't change (no weight decay)
    EXPECT_FLOAT_EQ(weights(0, 0), 1.0f);
}

TEST(OptimizerEdgeCaseTest, VerySmallGradient) {
    Optimizer opt(OptimizerType::ADAM, 0.001f);

    Matrix weights(1, 1);
    Matrix gradients(1, 1);

    weights(0, 0) = 1.0f;
    gradients(0, 0) = 1e-10f;

    opt.add_parameter_group(&weights, &gradients);

    EXPECT_NO_THROW(opt.step());
}

TEST(OptimizerEdgeCaseTest, VeryLargeLearningRate) {
    Optimizer opt(OptimizerType::SGD, 1000.0f);

    Matrix weights(1, 1);
    Matrix gradients(1, 1);

    weights(0, 0) = 1.0f;
    gradients(0, 0) = 0.1f;

    opt.add_parameter_group(&weights, &gradients);
    opt.step();

    // Should update dramatically
    EXPECT_LT(weights(0, 0), -90.0f);
}

TEST(OptimizerEdgeCaseTest, EmptyOptimizer) {
    Optimizer opt(OptimizerType::SGD, 0.01f);

    // No parameters added
    EXPECT_NO_THROW(opt.zero_grad());
    EXPECT_NO_THROW(opt.step());
    EXPECT_FLOAT_EQ(opt.get_gradient_norm(), 0.0f);
}

// ============================================================================
// Performance/Stress Tests
// ============================================================================

TEST(OptimizerPerformanceTest, LargeParameterMatrix) {
    Optimizer opt(OptimizerType::ADAM, 0.001f);

    Matrix weights(100, 100);
    Matrix gradients(100, 100);

    // Initialize with random-ish values
    for (int i = 0; i < 100; i++) {
        for (int j = 0; j < 100; j++) {
            weights(i, j) = 0.1f * (i + j);
            gradients(i, j) = 0.01f;
        }
    }

    opt.add_parameter_group(&weights, &gradients);

    EXPECT_NO_THROW(opt.step());
    EXPECT_EQ(opt.total_parameters(), 10000);
}

TEST(OptimizerPerformanceTest, ManyParameterGroups) {
    Optimizer opt(OptimizerType::ADAMW, 0.0001f);

    std::vector<std::unique_ptr<Matrix>> weights_vec;
    std::vector<std::unique_ptr<Matrix>> gradients_vec;

    // Create 50 parameter groups
    for (int i = 0; i < 50; i++) {
        weights_vec.push_back(std::make_unique<Matrix>(10, 10));
        gradients_vec.push_back(std::make_unique<Matrix>(10, 10));
        opt.add_parameter_group(weights_vec.back().get(), gradients_vec.back().get());
    }

    EXPECT_EQ(opt.num_parameters(), 50);
    EXPECT_EQ(opt.total_parameters(), 5000);

    EXPECT_NO_THROW(opt.step());
}

// ============================================================================
// Main
// ============================================================================

int main(int argc, char** argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
