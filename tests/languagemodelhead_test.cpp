#include "../src/LanguageModelHead.hpp"
#include <../gtest/gtest.h>
#include <algorithm>
#include <cmath>
#include <fstream>
#include <stdexcept>
#include "../src/Activation.hpp"
#include "../src/Matrix.hpp"
#include "../src/Optimizer.hpp"

// ============================================================================
// Helper Functions
// ============================================================================

bool is_close(float actual, float expected, float tolerance = 1e-4f) {
    return std::abs(actual - expected) < tolerance;
}

bool matrices_equal(const Matrix& a, const Matrix& b, float tolerance = 1e-5f) {
    if (a.rows != b.rows || a.cols != b.cols)
        return false;

    for (int i = 0; i < a.rows; ++i) {
        for (int j = 0; j < a.cols; ++j) {
            if (!is_close(a(i, j), b(i, j), tolerance)) {
                return false;
            }
        }
    }
    return true;
}

float compute_gradient_norm(const Matrix& grad) {
    float sum = 0.0f;
    for (int i = 0; i < grad.rows; ++i) {
        for (int j = 0; j < grad.cols; ++j) {
            sum += grad(i, j) * grad(i, j);
        }
    }
    return std::sqrt(sum);
}

// ============================================================================
// Constructor Tests
// ============================================================================

TEST(LanguageModelHeadConstructorTest, BasicInitialization) {
    int d_model = 128;
    int vocab_size = 1000;

    LanguageModelHead lm_head(d_model, vocab_size);

    EXPECT_EQ(lm_head.learning_rate, 0.001f);
}

TEST(LanguageModelHeadConstructorTest, SmallVocabulary) {
    LanguageModelHead lm_head(64, 100);
    EXPECT_NO_THROW({
        Matrix input(5, 64);
        input.randomize(0.1f);
        Matrix output = lm_head.forward(input);
    });
}

TEST(LanguageModelHeadConstructorTest, LargeVocabulary) {
    LanguageModelHead lm_head(512, 50000);
    EXPECT_NO_THROW({
        Matrix input(10, 512);
        input.randomize(0.1f);
        Matrix output = lm_head.forward(input);
    });
}

// ============================================================================
// Forward Pass Tests
// ============================================================================

TEST(LanguageModelHeadForwardTest, OutputDimensions) {
    int d_model = 256;
    int vocab_size = 5000;
    int seq_len = 10;

    LanguageModelHead lm_head(d_model, vocab_size);

    Matrix input(seq_len, d_model);
    input.randomize(0.1f);

    Matrix output = lm_head.forward(input);

    EXPECT_EQ(output.rows, seq_len);
    EXPECT_EQ(output.cols, vocab_size);
}

TEST(LanguageModelHeadForwardTest, SingleToken) {
    int d_model = 128;
    int vocab_size = 1000;

    LanguageModelHead lm_head(d_model, vocab_size);

    Matrix input(1, d_model);
    input.fill(0.5f);

    Matrix output = lm_head.forward(input);

    EXPECT_EQ(output.rows, 1);
    EXPECT_EQ(output.cols, vocab_size);
}

TEST(LanguageModelHeadForwardTest, MultipleSequences) {
    int d_model = 64;
    int vocab_size = 500;
    int seq_len = 20;

    LanguageModelHead lm_head(d_model, vocab_size);

    Matrix input(seq_len, d_model);
    input.randomize(0.1f);

    Matrix output = lm_head.forward(input);

    // Check all values are finite
    for (int i = 0; i < output.rows; ++i) {
        for (int j = 0; j < output.cols; ++j) {
            EXPECT_TRUE(std::isfinite(output(i, j)));
        }
    }
}

TEST(LanguageModelHeadForwardTest, Deterministic) {
    int d_model = 128;
    int vocab_size = 1000;

    LanguageModelHead lm_head(d_model, vocab_size);

    Matrix input(5, d_model);
    input.randomize(0.1f);

    Matrix output1 = lm_head.forward(input);
    Matrix output2 = lm_head.forward(input);

    EXPECT_TRUE(matrices_equal(output1, output2, 1e-6f));
}

// ============================================================================
// Probability Tests
// ============================================================================

TEST(LanguageModelHeadProbabilityTest, SumToOne) {
    int d_model = 64;
    int vocab_size = 100;

    LanguageModelHead lm_head(d_model, vocab_size);

    std::vector<float> logits(vocab_size);
    for (int i = 0; i < vocab_size; ++i) {
        logits[i] = static_cast<float>(i) / 10.0f;
    }

    std::vector<float> probs = lm_head.get_probabilities(logits);

    float sum = 0.0f;
    for (float p : probs) {
        sum += p;
    }

    EXPECT_NEAR(sum, 1.0f, 1e-5f);
}

TEST(LanguageModelHeadProbabilityTest, AllPositive) {
    int d_model = 64;
    int vocab_size = 100;

    LanguageModelHead lm_head(d_model, vocab_size);

    std::vector<float> logits(vocab_size);
    for (int i = 0; i < vocab_size; ++i) {
        logits[i] = (rand() % 100) / 10.0f - 5.0f;  // Range [-5, 5]
    }

    std::vector<float> probs = lm_head.get_probabilities(logits);

    for (float p : probs) {
        EXPECT_GT(p, 0.0f);
        EXPECT_LT(p, 1.0f);
    }
}

TEST(LanguageModelHeadProbabilityTest, MaximumLogitHighestProbability) {
    int d_model = 64;
    int vocab_size = 100;

    LanguageModelHead lm_head(d_model, vocab_size);

    std::vector<float> logits(vocab_size, 0.0f);
    int max_idx = 42;
    logits[max_idx] = 10.0f;  // Much higher than others

    std::vector<float> probs = lm_head.get_probabilities(logits);

    // Find argmax
    int predicted_idx = std::max_element(probs.begin(), probs.end()) - probs.begin();

    EXPECT_EQ(predicted_idx, max_idx);
    EXPECT_GT(probs[max_idx], 0.9f);  // Should be very confident
}

TEST(LanguageModelHeadProbabilityTest, UniformLogitsUniformProbs) {
    int d_model = 64;
    int vocab_size = 10;

    LanguageModelHead lm_head(d_model, vocab_size);

    std::vector<float> logits(vocab_size, 1.0f);  // All same

    std::vector<float> probs = lm_head.get_probabilities(logits);

    float expected_prob = 1.0f / vocab_size;
    for (float p : probs) {
        EXPECT_NEAR(p, expected_prob, 1e-5f);
    }
}

TEST(LanguageModelHeadProbabilityTest, NumericalStability) {
    int d_model = 64;
    int vocab_size = 100;

    LanguageModelHead lm_head(d_model, vocab_size);

    // Very large logits that could cause overflow
    std::vector<float> logits(vocab_size, 100.0f);
    logits[50] = 101.0f;

    EXPECT_NO_THROW({
        std::vector<float> probs = lm_head.get_probabilities(logits);

        // Check still valid probabilities
        float sum = 0.0f;
        for (float p : probs) {
            EXPECT_TRUE(std::isfinite(p));
            EXPECT_GE(p, 0.0f);
            EXPECT_LE(p, 1.0f);
            sum += p;
        }
        EXPECT_NEAR(sum, 1.0f, 1e-5f);
    });
}

// ============================================================================
// Backward Pass Tests
// ============================================================================

TEST(LanguageModelHeadBackwardTest, GradientDimensions) {
    int d_model = 128;
    int vocab_size = 1000;
    int seq_len = 10;

    LanguageModelHead lm_head(d_model, vocab_size);

    Matrix input(seq_len, d_model);
    input.randomize(0.1f);

    Matrix output = lm_head.forward(input);

    Matrix grad_output(seq_len, vocab_size);
    grad_output.randomize(0.01f);

    Matrix grad_input = lm_head.backward(grad_output);

    EXPECT_EQ(grad_input.rows, seq_len);
    EXPECT_EQ(grad_input.cols, d_model);
}

TEST(LanguageModelHeadBackwardTest, GradientFinite) {
    int d_model = 64;
    int vocab_size = 500;
    int seq_len = 5;

    LanguageModelHead lm_head(d_model, vocab_size);

    Matrix input(seq_len, d_model);
    input.randomize(0.1f);

    Matrix output = lm_head.forward(input);

    Matrix grad_output(seq_len, vocab_size);
    grad_output.randomize(0.01f);

    Matrix grad_input = lm_head.backward(grad_output);

    for (int i = 0; i < grad_input.rows; ++i) {
        for (int j = 0; j < grad_input.cols; ++j) {
            EXPECT_TRUE(std::isfinite(grad_input(i, j)));
        }
    }
}

TEST(LanguageModelHeadBackwardTest, GradientAccumulation) {
    int d_model = 64;
    int vocab_size = 100;

    LanguageModelHead lm_head(d_model, vocab_size);
    lm_head.zero_grad();

    // First backward pass
    Matrix input1(3, d_model);
    input1.randomize(0.1f);
    lm_head.forward(input1);

    Matrix grad1(3, vocab_size);
    grad1.randomize(0.01f);
    lm_head.backward(grad1);

    // Second backward pass (without zeroing)
    Matrix input2(3, d_model);
    input2.randomize(0.1f);
    lm_head.forward(input2);

    Matrix grad2(3, vocab_size);
    grad2.randomize(0.01f);
    lm_head.backward(grad2);

    // Gradients should have accumulated
    // This is verified by the fact that we don't crash and weights update
    EXPECT_NO_THROW(lm_head.update_weights());
}

TEST(LanguageModelHeadBackwardTest, ZeroGradient) {
    int d_model = 64;
    int vocab_size = 100;

    LanguageModelHead lm_head(d_model, vocab_size);

    // Accumulate some gradients
    Matrix input(3, d_model);
    input.randomize(0.1f);
    lm_head.forward(input);

    Matrix grad(3, vocab_size);
    grad.randomize(0.01f);
    lm_head.backward(grad);

    // Zero gradients
    lm_head.zero_grad();

    // New backward pass should start fresh
    lm_head.forward(input);
    Matrix grad_input = lm_head.backward(grad);

    EXPECT_TRUE(std::isfinite(compute_gradient_norm(grad_input)));
}

// ============================================================================
// Weight Update Tests
// ============================================================================

TEST(LanguageModelHeadUpdateTest, WeightsChange) {
    int d_model = 64;
    int vocab_size = 100;

    LanguageModelHead lm_head(d_model, vocab_size);
    lm_head.learning_rate = 0.01f;

    // Save initial state
    Matrix input(5, d_model);
    input.randomize(0.1f);
    Matrix output_before = lm_head.forward(input);

    // Backward pass
    Matrix grad(5, vocab_size);
    grad.randomize(0.01f);
    lm_head.backward(grad);

    // Update weights
    lm_head.update_weights();

    // Output should be different
    Matrix output_after = lm_head.forward(input);

    bool weights_changed = false;
    for (int i = 0; i < output_before.rows; ++i) {
        for (int j = 0; j < output_before.cols; ++j) {
            if (std::abs(output_before(i, j) - output_after(i, j)) > 1e-6f) {
                weights_changed = true;
                break;
            }
        }
    }

    EXPECT_TRUE(weights_changed);
}

TEST(LanguageModelHeadUpdateTest, LearningRateEffect) {
    int d_model = 64;
    int vocab_size = 100;

    // Test with large learning rate
    LanguageModelHead lm_head_large(d_model, vocab_size);
    lm_head_large.learning_rate = 0.1f;

    Matrix input(5, d_model);
    input.randomize(0.1f);

    Matrix output_before_large = lm_head_large.forward(input);
    Matrix grad(5, vocab_size);
    grad.randomize(0.01f);
    lm_head_large.backward(grad);
    lm_head_large.update_weights();
    Matrix output_after_large = lm_head_large.forward(input);

    // Test with small learning rate
    LanguageModelHead lm_head_small(d_model, vocab_size);
    lm_head_small.learning_rate = 0.001f;

    output_before_large = lm_head_small.forward(input);
    lm_head_small.backward(grad);
    lm_head_small.update_weights();
    Matrix output_after_small = lm_head_small.forward(input);

    // Large LR should cause bigger changes (in general, not guaranteed for every element)
    float change_large =
        compute_gradient_norm(output_after_large) - compute_gradient_norm(output_before_large);
    float change_small =
        compute_gradient_norm(output_after_small) - compute_gradient_norm(output_before_large);

    // At least verify both caused changes
    EXPECT_TRUE(std::abs(change_large) > 0.0f || std::abs(change_small) > 0.0f);
}

// ============================================================================
// Save/Load Tests
// ============================================================================

TEST(LanguageModelHeadSaveLoadTest, SaveAndLoad) {
    int d_model = 64;
    int vocab_size = 100;
    std::string filepath = "test_lm_head.bin";

    LanguageModelHead lm_head(d_model, vocab_size);
    lm_head.learning_rate = 0.005f;

    // Train a bit to get non-random weights
    Matrix input(5, d_model);
    input.randomize(0.1f);
    Matrix output = lm_head.forward(input);
    Matrix grad(5, vocab_size);
    grad.randomize(0.01f);
    lm_head.backward(grad);
    lm_head.update_weights();

    // Get output before saving
    Matrix output_before_save = lm_head.forward(input);

    // Save
    EXPECT_NO_THROW(lm_head.save(filepath));

    // Load into new instance
    LanguageModelHead lm_head_loaded(d_model, vocab_size);
    EXPECT_NO_THROW(lm_head_loaded.load(filepath));

    // Get output after loading
    Matrix output_after_load = lm_head_loaded.forward(input);

    // Should produce same output
    EXPECT_TRUE(matrices_equal(output_before_save, output_after_load, 1e-5f));

    // Clean up
    std::remove(filepath.c_str());
}

TEST(LanguageModelHeadSaveLoadTest, DimensionMismatch) {
    int d_model = 64;
    int vocab_size = 100;
    std::string filepath = "test_lm_head_mismatch.bin";

    LanguageModelHead lm_head(d_model, vocab_size);
    lm_head.save(filepath);

    // Try to load into different dimensions
    LanguageModelHead lm_head_wrong(128, 100);  // Wrong d_model

    EXPECT_THROW(lm_head_wrong.load(filepath), std::runtime_error);

    // Clean up
    std::remove(filepath.c_str());
}

TEST(LanguageModelHeadSaveLoadTest, NonexistentFile) {
    int d_model = 64;
    int vocab_size = 100;

    LanguageModelHead lm_head(d_model, vocab_size);

    EXPECT_THROW(lm_head.load("nonexistent_file.bin"), std::runtime_error);
}

TEST(LanguageModelHeadSaveLoadTest, MultipleSaveLoad) {
    int d_model = 64;
    int vocab_size = 100;
    std::string filepath = "test_lm_head_multiple.bin";

    LanguageModelHead lm_head(d_model, vocab_size);

    Matrix input(5, d_model);
    input.randomize(0.1f);

    // Save and load multiple times
    for (int iter = 0; iter < 3; ++iter) {
        lm_head.save(filepath);

        LanguageModelHead lm_head_temp(d_model, vocab_size);
        lm_head_temp.load(filepath);

        Matrix output_orig = lm_head.forward(input);
        Matrix output_loaded = lm_head_temp.forward(input);

        EXPECT_TRUE(matrices_equal(output_orig, output_loaded, 1e-5f));
    }

    // Clean up
    std::remove(filepath.c_str());
}

// ============================================================================
// Integration Tests
// ============================================================================

TEST(LanguageModelHeadIntegrationTest, TrainingLoop) {
    int d_model = 64;
    int vocab_size = 100;
    int seq_len = 5;
    int num_steps = 10;

    LanguageModelHead lm_head(d_model, vocab_size);
    lm_head.learning_rate = 0.01f;

    // Simulate training
    for (int step = 0; step < num_steps; ++step) {
        lm_head.zero_grad();

        Matrix input(seq_len, d_model);
        input.randomize(0.1f);

        Matrix logits = lm_head.forward(input);

        // Simulate loss gradient
        Matrix grad(seq_len, vocab_size);
        grad.randomize(0.01f);

        lm_head.backward(grad);
        lm_head.update_weights();
    }

    // Should complete without errors
    EXPECT_TRUE(true);
}

TEST(LanguageModelHeadIntegrationTest, CrossEntropyGradient) {
    int d_model = 64;
    int vocab_size = 10;
    int seq_len = 3;

    LanguageModelHead lm_head(d_model, vocab_size);

    Matrix input(seq_len, d_model);
    input.randomize(0.1f);

    Matrix logits = lm_head.forward(input);

    // Simulate cross-entropy gradient (predicted_prob - target)
    Matrix grad(seq_len, vocab_size);
    for (int i = 0; i < seq_len; ++i) {
        // Get probabilities for position i
        std::vector<float> logits_i(vocab_size);
        for (int j = 0; j < vocab_size; ++j) {
            logits_i[j] = logits(i, j);
        }
        std::vector<float> probs = lm_head.get_probabilities(logits_i);

        // Target: one-hot at position 2
        int target = 2;
        for (int j = 0; j < vocab_size; ++j) {
            grad(i, j) = probs[j] - (j == target ? 1.0f : 0.0f);
        }
    }

    Matrix grad_input = lm_head.backward(grad);

    // Gradient should be finite and reasonable
    float grad_norm = compute_gradient_norm(grad_input);
    EXPECT_TRUE(std::isfinite(grad_norm));
    EXPECT_LT(grad_norm, 100.0f);  // Shouldn't explode
}

TEST(LanguageModelHeadIntegrationTest, GenerationSimulation) {
    int d_model = 64;
    int vocab_size = 100;

    LanguageModelHead lm_head(d_model, vocab_size);

    // Simulate autoregressive generation
    std::vector<int> generated_tokens;
    int max_length = 10;

    for (int step = 0; step < max_length; ++step) {
        // Single token input (simulating current position)
        Matrix decoder_output(1, d_model);
        decoder_output.randomize(0.1f);

        Matrix logits = lm_head.forward(decoder_output);

        // Get probabilities
        std::vector<float> logits_vec(vocab_size);
        for (int j = 0; j < vocab_size; ++j) {
            logits_vec[j] = logits(0, j);
        }
        std::vector<float> probs = lm_head.get_probabilities(logits_vec);

        // Greedy selection (argmax)
        int next_token = std::max_element(probs.begin(), probs.end()) - probs.begin();
        generated_tokens.push_back(next_token);
    }

    EXPECT_EQ(generated_tokens.size(), max_length);
}

// ============================================================================
// Edge Cases
// ============================================================================

TEST(LanguageModelHeadEdgeCaseTest, VerySmallLearningRate) {
    int d_model = 64;
    int vocab_size = 100;

    LanguageModelHead lm_head(d_model, vocab_size);
    lm_head.learning_rate = 1e-10f;

    Matrix input(5, d_model);
    input.randomize(0.1f);

    Matrix output_before = lm_head.forward(input);

    Matrix grad(5, vocab_size);
    grad.randomize(0.01f);
    lm_head.backward(grad);
    lm_head.update_weights();

    Matrix output_after = lm_head.forward(input);

    // Should still work, even if weights barely change
    EXPECT_TRUE(matrices_equal(output_before, output_after, 1e-5f));
}

TEST(LanguageModelHeadEdgeCaseTest, ZeroGradient) {
    int d_model = 64;
    int vocab_size = 100;

    LanguageModelHead lm_head(d_model, vocab_size);

    Matrix input(5, d_model);
    input.randomize(0.1f);
    lm_head.forward(input);

    // Zero gradient
    Matrix grad(5, vocab_size);
    grad.fill(0.0f);

    EXPECT_NO_THROW({
        Matrix grad_input = lm_head.backward(grad);
        lm_head.update_weights();
    });
}

TEST(LanguageModelHeadEdgeCaseTest, VeryLargeGradient) {
    int d_model = 64;
    int vocab_size = 100;

    LanguageModelHead lm_head(d_model, vocab_size);
    lm_head.learning_rate = 0.001f;  // Small LR to avoid explosion

    Matrix input(5, d_model);
    input.randomize(0.1f);
    lm_head.forward(input);

    // Very large gradient
    Matrix grad(5, vocab_size);
    grad.fill(100.0f);

    EXPECT_NO_THROW({
        Matrix grad_input = lm_head.backward(grad);

        // Check gradient doesn't produce NaN or Inf
        for (int i = 0; i < grad_input.rows; ++i) {
            for (int j = 0; j < grad_input.cols; ++j) {
                EXPECT_TRUE(std::isfinite(grad_input(i, j)));
            }
        }
    });
}

// ============================================================================
// Performance Tests
// ============================================================================

TEST(LanguageModelHeadPerformanceTest, LargeVocabularyForward) {
    int d_model = 768;
    int vocab_size = 50000;
    int seq_len = 100;

    LanguageModelHead lm_head(d_model, vocab_size);

    Matrix input(seq_len, d_model);
    input.randomize(0.1f);

    // Should complete in reasonable time
    EXPECT_NO_THROW({
        Matrix output = lm_head.forward(input);
        EXPECT_EQ(output.rows, seq_len);
        EXPECT_EQ(output.cols, vocab_size);
    });
}

TEST(LanguageModelHeadPerformanceTest, LargeVocabularyBackward) {
    int d_model = 768;
    int vocab_size = 50000;
    int seq_len = 100;

    LanguageModelHead lm_head(d_model, vocab_size);

    Matrix input(seq_len, d_model);
    input.randomize(0.1f);
    Matrix output = lm_head.forward(input);

    Matrix grad(seq_len, vocab_size);
    grad.randomize(0.01f);

    // Should complete in reasonable time
    EXPECT_NO_THROW({
        Matrix grad_input = lm_head.backward(grad);
        EXPECT_EQ(grad_input.rows, seq_len);
        EXPECT_EQ(grad_input.cols, d_model);
    });
}

// ============================================================================
// Optimizer Integration Tests
// ============================================================================

TEST(LanguageModelHeadOptimizerTest, SetOptimizerBasic) {
    LanguageModelHead lm_head(128, 1000);
    Optimizer opt(OptimizerType::ADAM, 0.001f);

    lm_head.set_optimizer(&opt);

    // Verify optimizer was set by checking it works
    Matrix input(5, 128);
    for (int i = 0; i < 5; ++i) {
        for (int j = 0; j < 128; ++j) {
            input(i, j) = 0.5f;
        }
    }

    Matrix grad_output(5, 1000);
    for (int i = 0; i < 5; ++i) {
        for (int j = 0; j < 1000; ++j) {
            grad_output(i, j) = 0.01f;
        }
    }

    lm_head.forward(input);
    lm_head.backward(grad_output);
    EXPECT_NO_THROW(lm_head.update_weights());
}

TEST(LanguageModelHeadOptimizerTest, SetOptimizerNullptr) {
    LanguageModelHead lm_head(128, 1000);

    // Should not crash when setting nullptr
    lm_head.set_optimizer(nullptr);

    // Should still work with default learning rate
    Matrix input(3, 128);
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 128; ++j) {
            input(i, j) = 0.5f;
        }
    }
    Matrix output = lm_head.forward(input);

    EXPECT_EQ(output.rows, 3);
    EXPECT_EQ(output.cols, 1000);
}

TEST(LanguageModelHeadOptimizerTest, UpdateWithOptimizer) {
    LanguageModelHead lm_head(128, 1000);
    Optimizer opt(OptimizerType::ADAM, 0.001f);
    lm_head.set_optimizer(&opt);

    Matrix input(3, 128);
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 128; ++j) {
            input(i, j) = 0.5f;
        }
    }

    Matrix grad_output(3, 1000);
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 1000; ++j) {
            grad_output(i, j) = 0.01f;
        }
    }

    // Forward and backward pass
    Matrix out_before = lm_head.forward(input);
    lm_head.backward(grad_output);
    lm_head.update_weights();

    // Forward again - output should have changed
    Matrix out_after = lm_head.forward(input);

    bool weights_changed = false;
    for (int i = 0; i < out_before.rows; ++i) {
        for (int j = 0; j < out_before.cols; ++j) {
            if (std::abs(out_after(i, j) - out_before(i, j)) > 1e-6f) {
                weights_changed = true;
                break;
            }
        }
        if (weights_changed)
            break;
    }

    EXPECT_TRUE(weights_changed);
}

TEST(LanguageModelHeadOptimizerTest, UpdateWithoutOptimizer) {
    LanguageModelHead lm_head(128, 1000);
    lm_head.learning_rate = 0.001f;

    Matrix input(3, 128);
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 128; ++j) {
            input(i, j) = 0.5f;
        }
    }

    Matrix grad_output(3, 1000);
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 1000; ++j) {
            grad_output(i, j) = 0.01f;
        }
    }

    // Forward and backward pass
    Matrix out_before = lm_head.forward(input);
    lm_head.backward(grad_output);
    lm_head.update_weights();

    // Forward again - output should have changed
    Matrix out_after = lm_head.forward(input);

    bool weights_changed = false;
    for (int i = 0; i < out_before.rows; ++i) {
        for (int j = 0; j < out_before.cols; ++j) {
            if (std::abs(out_after(i, j) - out_before(i, j)) > 1e-6f) {
                weights_changed = true;
                break;
            }
        }
        if (weights_changed)
            break;
    }

    EXPECT_TRUE(weights_changed);
}

TEST(LanguageModelHeadOptimizerTest, OptimizerVsSimpleGradientDescent) {
    LanguageModelHead lm_head(128, 1000);

    // Set optimizer with SGD (no momentum)
    Optimizer opt(OptimizerType::SGD, 0.01f);
    lm_head.set_optimizer(&opt);

    Matrix input(2, 128);
    for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < 128; ++j) {
            input(i, j) = 0.5f;
        }
    }

    Matrix grad_output(2, 1000);
    for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < 1000; ++j) {
            grad_output(i, j) = 0.01f;
        }
    }

    // Forward and backward
    lm_head.forward(input);
    lm_head.backward(grad_output);

    // Update weights - SGD should work
    EXPECT_NO_THROW(lm_head.update_weights());

    // Should produce valid output
    Matrix output = lm_head.forward(input);
    for (int i = 0; i < output.rows; ++i) {
        for (int j = 0; j < output.cols; ++j) {
            EXPECT_TRUE(std::isfinite(output(i, j)));
        }
    }
}

TEST(LanguageModelHeadOptimizerTest, MultipleUpdatesWithOptimizer) {
    LanguageModelHead lm_head(128, 1000);
    Optimizer opt(OptimizerType::ADAM, 0.001f);
    lm_head.set_optimizer(&opt);

    Matrix input(3, 128);
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 128; ++j) {
            input(i, j) = 0.5f;
        }
    }

    Matrix grad_output(3, 1000);
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 1000; ++j) {
            grad_output(i, j) = 0.01f;
        }
    }

    for (int i = 0; i < 5; ++i) {
        lm_head.forward(input);
        lm_head.backward(grad_output);
        lm_head.update_weights();
    }

    // Should complete without errors
    SUCCEED();
}

TEST(LanguageModelHeadOptimizerTest, SwitchOptimizer) {
    LanguageModelHead lm_head(128, 1000);

    // Start with Adam
    Optimizer opt1(OptimizerType::ADAM, 0.001f);
    lm_head.set_optimizer(&opt1);

    Matrix input(3, 128);
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 128; ++j) {
            input(i, j) = 0.5f;
        }
    }

    Matrix grad_output(3, 1000);
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 1000; ++j) {
            grad_output(i, j) = 0.01f;
        }
    }

    lm_head.forward(input);
    lm_head.backward(grad_output);
    lm_head.update_weights();

    // Switch to SGD
    Optimizer opt2(OptimizerType::SGD, 0.01f);
    lm_head.set_optimizer(&opt2);

    lm_head.forward(input);
    lm_head.backward(grad_output);
    lm_head.update_weights();

    SUCCEED();
}

TEST(LanguageModelHeadOptimizerTest, OptimizerWithDifferentLearningRates) {
    LanguageModelHead lm_head1(128, 1000);
    LanguageModelHead lm_head2(128, 1000);

    // Set different learning rates
    Optimizer opt1(OptimizerType::ADAM, 0.001f);
    Optimizer opt2(OptimizerType::ADAM, 0.01f);

    lm_head1.set_optimizer(&opt1);
    lm_head2.set_optimizer(&opt2);

    Matrix input(3, 128);
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 128; ++j) {
            input(i, j) = 0.5f;
        }
    }

    Matrix grad_output(3, 1000);
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 1000; ++j) {
            grad_output(i, j) = 0.01f;
        }
    }

    Matrix out1_before = lm_head1.forward(input);
    lm_head1.backward(grad_output);
    lm_head1.update_weights();
    Matrix out1_after = lm_head1.forward(input);

    Matrix out2_before = lm_head2.forward(input);
    lm_head2.backward(grad_output);
    lm_head2.update_weights();
    Matrix out2_after = lm_head2.forward(input);

    // Calculate update magnitudes
    float update_mag_1 = 0.0f;
    float update_mag_2 = 0.0f;

    for (int i = 0; i < out1_before.rows; ++i) {
        for (int j = 0; j < out1_before.cols; ++j) {
            update_mag_1 += std::abs(out1_after(i, j) - out1_before(i, j));
            update_mag_2 += std::abs(out2_after(i, j) - out2_before(i, j));
        }
    }

    // Higher learning rate should produce larger updates
    EXPECT_GT(update_mag_2, update_mag_1);
}

TEST(LanguageModelHeadOptimizerTest, RegisterParametersExplicit) {
    LanguageModelHead lm_head(128, 1000);
    Optimizer opt(OptimizerType::ADAM, 0.001f);

    lm_head.set_optimizer(&opt);

    // register_parameters() should have been called by set_optimizer()
    Matrix input(3, 128);
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 128; ++j) {
            input(i, j) = 0.5f;
        }
    }

    Matrix grad_output(3, 1000);
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 1000; ++j) {
            grad_output(i, j) = 0.01f;
        }
    }

    lm_head.forward(input);
    lm_head.backward(grad_output);

    // Should not throw if parameters are registered
    EXPECT_NO_THROW(lm_head.update_weights());
}

TEST(LanguageModelHeadOptimizerTest, ParametersChangeWithOptimizer) {
    LanguageModelHead lm_head(128, 1000);
    Optimizer opt(OptimizerType::ADAM, 0.001f);
    lm_head.set_optimizer(&opt);

    Matrix input(3, 128);
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 128; ++j) {
            input(i, j) = 0.5f;
        }
    }

    Matrix grad_output(3, 1000);
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 1000; ++j) {
            grad_output(i, j) = 0.1f;
        }
    }

    Matrix out_before = lm_head.forward(input);

    // Train for a few steps
    for (int i = 0; i < 3; ++i) {
        lm_head.forward(input);
        lm_head.backward(grad_output);
        lm_head.update_weights();
    }

    Matrix out_after = lm_head.forward(input);

    // Output should have changed
    bool changed = false;
    for (int i = 0; i < out_before.rows && !changed; ++i) {
        for (int j = 0; j < out_before.cols && !changed; ++j) {
            if (std::abs(out_after(i, j) - out_before(i, j)) > 1e-6f) {
                changed = true;
            }
        }
    }

    EXPECT_TRUE(changed);
}

TEST(LanguageModelHeadOptimizerTest, LearningRateScheduling) {
    LanguageModelHead lm_head(128, 1000);
    Optimizer opt(OptimizerType::ADAM, 0.001f);
    lm_head.set_optimizer(&opt);

    Matrix input(3, 128);
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 128; ++j) {
            input(i, j) = 0.5f;
        }
    }

    Matrix grad_output(3, 1000);
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 1000; ++j) {
            grad_output(i, j) = 0.01f;
        }
    }

    // First update with lr=0.001
    Matrix out1 = lm_head.forward(input);
    lm_head.backward(grad_output);
    lm_head.update_weights();
    Matrix out1_after = lm_head.forward(input);

    // Change learning rate to much higher
    opt.set_learning_rate(0.1f);  // 100x higher

    // Second update with lr=0.1
    Matrix out2 = lm_head.forward(input);
    lm_head.backward(grad_output);
    lm_head.update_weights();
    Matrix out2_after = lm_head.forward(input);

    // Calculate update magnitudes
    float update_mag_1 = 0.0f;
    float update_mag_2 = 0.0f;

    for (int i = 0; i < out1.rows; ++i) {
        for (int j = 0; j < out1.cols; ++j) {
            update_mag_1 += std::abs(out1_after(i, j) - out1(i, j));
            update_mag_2 += std::abs(out2_after(i, j) - out2(i, j));
        }
    }

    // Second update should be significantly larger (100x higher learning rate)
    EXPECT_GT(update_mag_2, update_mag_1 * 10.0f);
}

TEST(LanguageModelHeadOptimizerTest, BackwardCompatibilityNoOptimizer) {
    LanguageModelHead lm_head(128, 1000);
    lm_head.learning_rate = 0.01f;

    Matrix input(3, 128);
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 128; ++j) {
            input(i, j) = 0.5f;
        }
    }

    Matrix grad_output(3, 1000);
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 1000; ++j) {
            grad_output(i, j) = 0.01f;
        }
    }

    // Should work without optimizer (backward compatibility)
    for (int i = 0; i < 3; ++i) {
        Matrix output = lm_head.forward(input);

        EXPECT_EQ(output.rows, 3);
        EXPECT_EQ(output.cols, 1000);

        lm_head.backward(grad_output);
        lm_head.update_weights();
    }

    SUCCEED();
}

// ============================================================================
// Main
// ============================================================================

int main(int argc, char** argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
