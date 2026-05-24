#include "../src/EncoderBlock.hpp"
#include <../gtest/gtest.h>
#include <cmath>
#include <cstdio>
#include <fstream>
#include "../src/Matrix.hpp"

// Test fixture for EncoderBlock tests
class EncoderBlockTest : public ::testing::Test {
   protected:
    void SetUp() override {
        test_weights_file = "test_encoderblock_weights.bin";
        d_model = 64;
        num_heads = 4;
        d_ff = 256;
        dropout = 0.1f;
    }

    void TearDown() override {
        // Clean up test files — save_weights() strips the extension before
        // appending _attention.bin / _feedforward.bin, so we must do the same.
        std::string base = test_weights_file.substr(0, test_weights_file.find_last_of('.'));
        std::remove(test_weights_file.c_str());
        std::remove((base + "_attention.bin").c_str());
        std::remove((base + "_feedforward.bin").c_str());
        std::remove("test_block_2.bin");
        std::remove("test_block_2_attention.bin");
        std::remove("test_block_2_feedforward.bin");
    }

    std::string test_weights_file;
    int d_model;
    int num_heads;
    int d_ff;
    float dropout;

    // Helper function to check if values are close
    bool is_close(float a, float b, float epsilon = 1e-5f) {
        return std::abs(a - b) < epsilon;
    }

    // Helper to check matrix equality
    bool matrices_close(const Matrix& a, const Matrix& b, float epsilon = 1e-4f) {
        if (a.rows != b.rows || a.cols != b.cols)
            return false;
        for (int i = 0; i < a.rows; ++i) {
            for (int j = 0; j < a.cols; ++j) {
                if (!is_close(a.data[i][j], b.data[i][j], epsilon)) {
                    return false;
                }
            }
        }
        return true;
    }

    // Helper to create a simple causal mask
    Matrix create_causal_mask(int seq_len) {
        Matrix mask(seq_len, seq_len);
        for (int i = 0; i < seq_len; ++i) {
            for (int j = 0; j < seq_len; ++j) {
                mask.data[i][j] = (j <= i) ? 0.0f : -1e9f;
            }
        }
        return mask;
    }

    // Helper to check if matrix has reasonable values (not NaN/Inf)
    bool has_valid_values(const Matrix& m) {
        for (int i = 0; i < m.rows; ++i) {
            for (int j = 0; j < m.cols; ++j) {
                if (std::isnan(m.data[i][j]) || std::isinf(m.data[i][j])) {
                    return false;
                }
            }
        }
        return true;
    }
};

// ============================================================================
// Construction Tests
// ============================================================================

TEST_F(EncoderBlockTest, BasicConstruction) {
    EncoderBlock block(d_model, num_heads, d_ff, dropout);

    EXPECT_EQ(block.learning_rate, 0.001f);
    EXPECT_NO_THROW(block.print_config("TestBlock"));
}

TEST_F(EncoderBlockTest, ConstructionWithDifferentDimensions) {
    EncoderBlock small_block(32, 2, 128);
    EncoderBlock medium_block(128, 8, 512);
    EncoderBlock large_block(512, 16, 2048);

    EXPECT_NO_THROW(small_block.print_config("Small"));
    EXPECT_NO_THROW(medium_block.print_config("Medium"));
    EXPECT_NO_THROW(large_block.print_config("Large"));
}

TEST_F(EncoderBlockTest, ConstructionWithZeroDropout) {
    EncoderBlock block(d_model, num_heads, d_ff, 0.0f);

    Matrix input(5, d_model);
    for (int i = 0; i < input.rows; ++i) {
        for (int j = 0; j < input.cols; ++j) {
            input.data[i][j] = 0.1f * (i + j);
        }
    }

    Matrix output = block.forward(input);
    EXPECT_EQ(output.rows, 5);
    EXPECT_EQ(output.cols, d_model);
    EXPECT_TRUE(has_valid_values(output));
}

// ============================================================================
// Forward Pass Tests
// ============================================================================

TEST_F(EncoderBlockTest, ForwardPassBasic) {
    EncoderBlock block(d_model, num_heads, d_ff);

    Matrix input(10, d_model);
    for (int i = 0; i < input.rows; ++i) {
        for (int j = 0; j < input.cols; ++j) {
            input.data[i][j] = 0.01f * (i * d_model + j);
        }
    }

    Matrix output = block.forward(input);

    EXPECT_EQ(output.rows, 10);
    EXPECT_EQ(output.cols, d_model);
    EXPECT_TRUE(has_valid_values(output));
}

TEST_F(EncoderBlockTest, ForwardPassWithMask) {
    EncoderBlock block(d_model, num_heads, d_ff);

    int seq_len = 8;
    Matrix input(seq_len, d_model);
    for (int i = 0; i < input.rows; ++i) {
        for (int j = 0; j < input.cols; ++j) {
            input.data[i][j] = 0.05f * (i + j);
        }
    }

    Matrix mask = create_causal_mask(seq_len);
    Matrix output = block.forward(input, &mask);

    EXPECT_EQ(output.rows, seq_len);
    EXPECT_EQ(output.cols, d_model);
    EXPECT_TRUE(has_valid_values(output));
}

TEST_F(EncoderBlockTest, ForwardPassVariableSequenceLengths) {
    EncoderBlock block(d_model, num_heads, d_ff);

    std::vector<int> seq_lengths = {1, 5, 10, 20, 50};

    for (int seq_len : seq_lengths) {
        Matrix input(seq_len, d_model);
        for (int i = 0; i < input.rows; ++i) {
            for (int j = 0; j < input.cols; ++j) {
                input.data[i][j] = 0.01f * (i + j);
            }
        }

        Matrix output = block.forward(input);

        EXPECT_EQ(output.rows, seq_len);
        EXPECT_EQ(output.cols, d_model);
        EXPECT_TRUE(has_valid_values(output));
    }
}

TEST_F(EncoderBlockTest, ForwardPassDeterministic) {
    EncoderBlock block(d_model, num_heads, d_ff, 0.0f);  // No dropout for determinism

    Matrix input(7, d_model);
    for (int i = 0; i < input.rows; ++i) {
        for (int j = 0; j < input.cols; ++j) {
            input.data[i][j] = 0.02f * (i * d_model + j);
        }
    }

    Matrix output1 = block.forward(input);
    Matrix output2 = block.forward(input);

    EXPECT_TRUE(matrices_close(output1, output2, 1e-6f));
}

TEST_F(EncoderBlockTest, ForwardPassNormalizationEffect) {
    EncoderBlock block(d_model, num_heads, d_ff);

    Matrix input(10, d_model);
    for (int i = 0; i < input.rows; ++i) {
        for (int j = 0; j < input.cols; ++j) {
            input.data[i][j] = 10.0f * (i + j);  // Large values
        }
    }

    Matrix output = block.forward(input);

    // Check that output is normalized (roughly mean 0, std 1 per position)
    for (int i = 0; i < output.rows; ++i) {
        float mean = 0.0f;
        for (int j = 0; j < output.cols; ++j) {
            mean += output.data[i][j];
        }
        mean /= output.cols;

        // Mean should be close to 0 after layer norm
        EXPECT_LT(std::abs(mean), 1.0f);
    }

    EXPECT_TRUE(has_valid_values(output));
}

// ============================================================================
// Backward Pass Tests
// ============================================================================

TEST_F(EncoderBlockTest, BackwardPassBasic) {
    EncoderBlock block(d_model, num_heads, d_ff);

    Matrix input(5, d_model);
    for (int i = 0; i < input.rows; ++i) {
        for (int j = 0; j < input.cols; ++j) {
            input.data[i][j] = 0.01f * (i + j);
        }
    }

    Matrix output = block.forward(input);

    Matrix grad_output(5, d_model);
    for (int i = 0; i < grad_output.rows; ++i) {
        for (int j = 0; j < grad_output.cols; ++j) {
            grad_output.data[i][j] = 0.001f;
        }
    }

    Matrix grad_input = block.backward(grad_output);

    EXPECT_EQ(grad_input.rows, 5);
    EXPECT_EQ(grad_input.cols, d_model);
    EXPECT_TRUE(has_valid_values(grad_input));
}

TEST_F(EncoderBlockTest, BackwardPassGradientFlow) {
    EncoderBlock block(d_model, num_heads, d_ff);

    Matrix input(8, d_model);
    for (int i = 0; i < input.rows; ++i) {
        for (int j = 0; j < input.cols; ++j) {
            input.data[i][j] = 0.05f * (i * d_model + j);
        }
    }

    block.forward(input);

    // Use non-constant gradients: uniform 1.0 cancels exactly through LayerNorm
    // backward (sum of deviations from mean = 0), producing mathematically correct
    // but test-defeating zeros on compilers that don't introduce FP rounding noise.
    Matrix grad_output(8, d_model);
    for (int i = 0; i < grad_output.rows; ++i) {
        for (int j = 0; j < grad_output.cols; ++j) {
            grad_output.data[i][j] = 0.1f * (i + 1) + 0.01f * (j + 1);
        }
    }

    Matrix grad_input = block.backward(grad_output);

    // Check that gradients are non-zero (gradient flow is working)
    float grad_sum = 0.0f;
    for (int i = 0; i < grad_input.rows; ++i) {
        for (int j = 0; j < grad_input.cols; ++j) {
            grad_sum += std::abs(grad_input.data[i][j]);
        }
    }

    EXPECT_GT(grad_sum, 0.0f);
    EXPECT_TRUE(has_valid_values(grad_input));
}

TEST_F(EncoderBlockTest, BackwardPassWithMask) {
    EncoderBlock block(d_model, num_heads, d_ff);

    int seq_len = 6;
    Matrix input(seq_len, d_model);
    for (int i = 0; i < input.rows; ++i) {
        for (int j = 0; j < input.cols; ++j) {
            input.data[i][j] = 0.02f * (i + j);
        }
    }

    Matrix mask = create_causal_mask(seq_len);
    block.forward(input, &mask);

    Matrix grad_output(seq_len, d_model);
    for (int i = 0; i < grad_output.rows; ++i) {
        for (int j = 0; j < grad_output.cols; ++j) {
            grad_output.data[i][j] = 0.5f;
        }
    }

    Matrix grad_input = block.backward(grad_output);

    EXPECT_EQ(grad_input.rows, seq_len);
    EXPECT_EQ(grad_input.cols, d_model);
    EXPECT_TRUE(has_valid_values(grad_input));
}

// ============================================================================
// Gradient Management Tests
// ============================================================================

TEST_F(EncoderBlockTest, ZeroGradients) {
    EncoderBlock block(d_model, num_heads, d_ff);

    Matrix input(5, d_model);
    for (int i = 0; i < input.rows; ++i) {
        for (int j = 0; j < input.cols; ++j) {
            input.data[i][j] = 0.1f * (i + j);
        }
    }

    block.forward(input);

    Matrix grad_output(5, d_model);
    for (int i = 0; i < grad_output.rows; ++i) {
        for (int j = 0; j < grad_output.cols; ++j) {
            grad_output.data[i][j] = 1.0f;
        }
    }

    block.backward(grad_output);

    float grad_norm_before = block.get_gradient_norm();
    EXPECT_GT(grad_norm_before, 0.0f);

    block.zero_grad();

    float grad_norm_after = block.get_gradient_norm();
    EXPECT_FLOAT_EQ(grad_norm_after, 0.0f);
}

TEST_F(EncoderBlockTest, GradientNormComputation) {
    EncoderBlock block(d_model, num_heads, d_ff);

    Matrix input(10, d_model);
    for (int i = 0; i < input.rows; ++i) {
        for (int j = 0; j < input.cols; ++j) {
            input.data[i][j] = 0.01f * (i * d_model + j);
        }
    }

    block.forward(input);

    Matrix grad_output(10, d_model);
    for (int i = 0; i < grad_output.rows; ++i) {
        for (int j = 0; j < grad_output.cols; ++j) {
            grad_output.data[i][j] = 0.1f;
        }
    }

    block.backward(grad_output);

    float grad_norm = block.get_gradient_norm();
    EXPECT_GT(grad_norm, 0.0f);
    EXPECT_FALSE(std::isnan(grad_norm));
    EXPECT_FALSE(std::isinf(grad_norm));
}

TEST_F(EncoderBlockTest, GradientClipping) {
    EncoderBlock block(d_model, num_heads, d_ff);

    Matrix input(8, d_model);
    for (int i = 0; i < input.rows; ++i) {
        for (int j = 0; j < input.cols; ++j) {
            input.data[i][j] = 0.05f * (i + j);
        }
    }

    block.forward(input);

    Matrix grad_output(8, d_model);
    for (int i = 0; i < grad_output.rows; ++i) {
        for (int j = 0; j < grad_output.cols; ++j) {
            grad_output.data[i][j] = 100.0f;  // Large gradients
        }
    }

    block.backward(grad_output);

    float max_norm = 5.0f;
    block.clip_gradients(max_norm);

    float grad_norm_after = block.get_gradient_norm();
    EXPECT_LE(grad_norm_after, max_norm * 1.01f);  // Small tolerance for floating point
}

TEST_F(EncoderBlockTest, GradientAccumulation) {
    EncoderBlock block(d_model, num_heads, d_ff);

    Matrix input(5, d_model);
    for (int i = 0; i < input.rows; ++i) {
        for (int j = 0; j < input.cols; ++j) {
            input.data[i][j] = 0.02f * (i + j);
        }
    }

    Matrix grad_output(5, d_model);
    for (int i = 0; i < grad_output.rows; ++i) {
        for (int j = 0; j < grad_output.cols; ++j) {
            grad_output.data[i][j] = 0.5f;
        }
    }

    // Test that gradients exist after backward
    block.forward(input);
    block.backward(grad_output);
    float norm1 = block.get_gradient_norm();
    EXPECT_GT(norm1, 0.0f);

    // After zeroing, norm should be zero
    block.zero_grad();
    float norm_after_zero = block.get_gradient_norm();
    EXPECT_FLOAT_EQ(norm_after_zero, 0.0f);

    // After another backward pass, gradients should exist again
    block.forward(input);
    block.backward(grad_output);
    float norm2 = block.get_gradient_norm();
    EXPECT_GT(norm2, 0.0f);
}

// ============================================================================
// Weight Update Tests
// ============================================================================

TEST_F(EncoderBlockTest, WeightUpdateBasic) {
    EncoderBlock block(d_model, num_heads, d_ff);
    block.learning_rate = 0.01f;

    Matrix input(5, d_model);
    for (int i = 0; i < input.rows; ++i) {
        for (int j = 0; j < input.cols; ++j) {
            input.data[i][j] = 0.1f * (i + j);
        }
    }

    Matrix output_before = block.forward(input);

    Matrix grad_output(5, d_model);
    for (int i = 0; i < grad_output.rows; ++i) {
        for (int j = 0; j < grad_output.cols; ++j) {
            grad_output.data[i][j] = 0.1f;
        }
    }

    block.backward(grad_output);
    block.update_weights();
    block.zero_grad();

    Matrix output_after = block.forward(input);

    // Outputs should differ after weight update
    EXPECT_FALSE(matrices_close(output_before, output_after, 1e-6f));
}

TEST_F(EncoderBlockTest, MultipleWeightUpdates) {
    EncoderBlock block(d_model, num_heads, d_ff);
    block.learning_rate = 0.01f;

    Matrix input(8, d_model);
    for (int i = 0; i < input.rows; ++i) {
        for (int j = 0; j < input.cols; ++j) {
            input.data[i][j] = 0.05f * (i * d_model + j);
        }
    }

    std::vector<Matrix> outputs;
    outputs.push_back(block.forward(input));

    for (int iter = 0; iter < 5; ++iter) {
        Matrix grad_output(8, d_model);
        for (int i = 0; i < grad_output.rows; ++i) {
            for (int j = 0; j < grad_output.cols; ++j) {
                grad_output.data[i][j] = 0.1f;
            }
        }

        block.backward(grad_output);
        block.update_weights();
        block.zero_grad();

        outputs.push_back(block.forward(input));
    }

    // Each output should be different
    for (size_t i = 1; i < outputs.size(); ++i) {
        EXPECT_FALSE(matrices_close(outputs[0], outputs[i], 1e-5f));
    }
}

TEST_F(EncoderBlockTest, LearningRateEffect) {
    EncoderBlock block1(d_model, num_heads, d_ff);
    EncoderBlock block2(d_model, num_heads, d_ff);

    block1.learning_rate = 0.01f;
    block2.learning_rate = 0.001f;

    Matrix input(5, d_model);
    for (int i = 0; i < input.rows; ++i) {
        for (int j = 0; j < input.cols; ++j) {
            input.data[i][j] = 0.1f * (i + j);
        }
    }

    Matrix grad_output(5, d_model);
    for (int i = 0; i < grad_output.rows; ++i) {
        for (int j = 0; j < grad_output.cols; ++j) {
            grad_output.data[i][j] = 1.0f;
        }
    }

    // Save initial outputs
    Matrix out1_before = block1.forward(input);
    Matrix out2_before = block2.forward(input);

    // Update both blocks
    block1.backward(grad_output);
    block1.update_weights();

    block2.backward(grad_output);
    block2.update_weights();

    Matrix out1_after = block1.forward(input);
    Matrix out2_after = block2.forward(input);

    // Compute differences
    float diff1 = 0.0f, diff2 = 0.0f;
    for (int i = 0; i < out1_before.rows; ++i) {
        for (int j = 0; j < out1_before.cols; ++j) {
            diff1 += std::abs(out1_after.data[i][j] - out1_before.data[i][j]);
            diff2 += std::abs(out2_after.data[i][j] - out2_before.data[i][j]);
        }
    }

    // Both should cause changes (verify updates are happening)
    EXPECT_GT(diff1, 0.0f);
    EXPECT_GT(diff2, 0.0f);
    // Learning rates differ by 10x, so changes should differ
    EXPECT_NE(diff1, diff2);
}

// ============================================================================
// Persistence Tests (Save/Load)
// ============================================================================

TEST_F(EncoderBlockTest, SaveAndLoadWeights) {
    EncoderBlock block1(d_model, num_heads, d_ff);

    Matrix input(10, d_model);
    for (int i = 0; i < input.rows; ++i) {
        for (int j = 0; j < input.cols; ++j) {
            input.data[i][j] = 0.01f * (i * d_model + j);
        }
    }

    Matrix output1 = block1.forward(input);

    block1.save_weights(test_weights_file);

    EncoderBlock block2(d_model, num_heads, d_ff);
    block2.load_weights(test_weights_file);

    Matrix output2 = block2.forward(input);

    EXPECT_TRUE(matrices_close(output1, output2, 1e-5f));
}

TEST_F(EncoderBlockTest, SaveLoadAfterTraining) {
    EncoderBlock block1(d_model, num_heads, d_ff);
    block1.learning_rate = 0.01f;

    Matrix input(8, d_model);
    for (int i = 0; i < input.rows; ++i) {
        for (int j = 0; j < input.cols; ++j) {
            input.data[i][j] = 0.05f * (i + j);
        }
    }

    // Train for a few iterations
    for (int iter = 0; iter < 10; ++iter) {
        block1.forward(input);

        Matrix grad_output(8, d_model);
        for (int i = 0; i < grad_output.rows; ++i) {
            for (int j = 0; j < grad_output.cols; ++j) {
                grad_output.data[i][j] = 0.1f;
            }
        }

        block1.backward(grad_output);
        block1.update_weights();
        block1.zero_grad();
    }

    Matrix output1 = block1.forward(input);

    block1.save_weights("test_block_2.bin");

    EncoderBlock block2(d_model, num_heads, d_ff);
    block2.load_weights("test_block_2.bin");

    Matrix output2 = block2.forward(input);

    EXPECT_TRUE(matrices_close(output1, output2, 1e-5f));
}

TEST_F(EncoderBlockTest, LoadWeightsDimensionMismatch) {
    EncoderBlock block1(64, 4, 256);
    block1.save_weights(test_weights_file);

    EncoderBlock block2(128, 8, 512);  // Different dimensions

    EXPECT_THROW(block2.load_weights(test_weights_file), std::runtime_error);
}

TEST_F(EncoderBlockTest, LoadWeightsNonexistentFile) {
    EncoderBlock block(d_model, num_heads, d_ff);

    EXPECT_THROW(block.load_weights("nonexistent_file.bin"), std::runtime_error);
}

// ============================================================================
// Integration Tests
// ============================================================================

TEST_F(EncoderBlockTest, StackedBlocks) {
    EncoderBlock block1(d_model, num_heads, d_ff);
    EncoderBlock block2(d_model, num_heads, d_ff);
    EncoderBlock block3(d_model, num_heads, d_ff);

    Matrix input(10, d_model);
    for (int i = 0; i < input.rows; ++i) {
        for (int j = 0; j < input.cols; ++j) {
            input.data[i][j] = 0.01f * (i + j);
        }
    }

    Matrix out1 = block1.forward(input);
    Matrix out2 = block2.forward(out1);
    Matrix out3 = block3.forward(out2);

    EXPECT_EQ(out3.rows, 10);
    EXPECT_EQ(out3.cols, d_model);
    EXPECT_TRUE(has_valid_values(out3));
}

TEST_F(EncoderBlockTest, StackedBlocksBackward) {
    EncoderBlock block1(d_model, num_heads, d_ff);
    EncoderBlock block2(d_model, num_heads, d_ff);

    Matrix input(5, d_model);
    for (int i = 0; i < input.rows; ++i) {
        for (int j = 0; j < input.cols; ++j) {
            input.data[i][j] = 0.1f * (i + j);
        }
    }

    Matrix out1 = block1.forward(input);
    Matrix out2 = block2.forward(out1);

    Matrix grad_output(5, d_model);
    for (int i = 0; i < grad_output.rows; ++i) {
        for (int j = 0; j < grad_output.cols; ++j) {
            grad_output.data[i][j] = 0.5f;
        }
    }

    Matrix grad2 = block2.backward(grad_output);
    Matrix grad1 = block1.backward(grad2);

    EXPECT_EQ(grad1.rows, 5);
    EXPECT_EQ(grad1.cols, d_model);
    EXPECT_TRUE(has_valid_values(grad1));
}

TEST_F(EncoderBlockTest, TrainingLoop) {
    EncoderBlock block(d_model, num_heads, d_ff);
    block.learning_rate = 0.01f;

    Matrix input(10, d_model);
    for (int i = 0; i < input.rows; ++i) {
        for (int j = 0; j < input.cols; ++j) {
            input.data[i][j] = 0.01f * (i * d_model + j);
        }
    }

    std::vector<float> losses;

    for (int epoch = 0; epoch < 20; ++epoch) {
        Matrix output = block.forward(input);

        // Compute simple MSE loss (target = input)
        float loss = 0.0f;
        Matrix grad_output(output.rows, output.cols);
        for (int i = 0; i < output.rows; ++i) {
            for (int j = 0; j < output.cols; ++j) {
                float diff = output.data[i][j] - input.data[i][j];
                loss += diff * diff;
                grad_output.data[i][j] = 2.0f * diff / (output.rows * output.cols);
            }
        }

        losses.push_back(loss);

        block.backward(grad_output);

        float grad_norm = block.get_gradient_norm();
        if (grad_norm > 5.0f) {
            block.clip_gradients(5.0f);
        }

        block.update_weights();
        block.zero_grad();
    }

    // Loss should generally decrease (with some fluctuation)
    EXPECT_LT(losses.back(), losses.front() * 2.0f);
}

TEST_F(EncoderBlockTest, BatchProcessing) {
    EncoderBlock block(d_model, num_heads, d_ff);

    // Process different batch sizes
    std::vector<int> batch_sizes = {1, 4, 8, 16};

    for (int batch_size : batch_sizes) {
        Matrix input(batch_size, d_model);
        for (int i = 0; i < input.rows; ++i) {
            for (int j = 0; j < input.cols; ++j) {
                input.data[i][j] = 0.01f * (i + j);
            }
        }

        Matrix output = block.forward(input);

        EXPECT_EQ(output.rows, batch_size);
        EXPECT_EQ(output.cols, d_model);
        EXPECT_TRUE(has_valid_values(output));
    }
}

// ============================================================================
// Edge Cases and Robustness
// ============================================================================

TEST_F(EncoderBlockTest, SingleTokenSequence) {
    EncoderBlock block(d_model, num_heads, d_ff);

    Matrix input(1, d_model);
    for (int j = 0; j < input.cols; ++j) {
        input.data[0][j] = 0.1f * j;
    }

    Matrix output = block.forward(input);

    EXPECT_EQ(output.rows, 1);
    EXPECT_EQ(output.cols, d_model);
    EXPECT_TRUE(has_valid_values(output));
}

TEST_F(EncoderBlockTest, LongSequence) {
    EncoderBlock block(d_model, num_heads, d_ff);

    int long_seq_len = 100;
    Matrix input(long_seq_len, d_model);
    for (int i = 0; i < input.rows; ++i) {
        for (int j = 0; j < input.cols; ++j) {
            input.data[i][j] = 0.001f * (i * d_model + j);
        }
    }

    Matrix output = block.forward(input);

    EXPECT_EQ(output.rows, long_seq_len);
    EXPECT_EQ(output.cols, d_model);
    EXPECT_TRUE(has_valid_values(output));
}

TEST_F(EncoderBlockTest, ZeroInput) {
    EncoderBlock block(d_model, num_heads, d_ff);

    Matrix input(5, d_model);
    for (int i = 0; i < input.rows; ++i) {
        for (int j = 0; j < input.cols; ++j) {
            input.data[i][j] = 0.0f;
        }
    }

    Matrix output = block.forward(input);

    EXPECT_TRUE(has_valid_values(output));
}

TEST_F(EncoderBlockTest, LargeValueInput) {
    EncoderBlock block(d_model, num_heads, d_ff);

    Matrix input(5, d_model);
    for (int i = 0; i < input.rows; ++i) {
        for (int j = 0; j < input.cols; ++j) {
            input.data[i][j] = 100.0f * (i + j);  // Large values
        }
    }

    Matrix output = block.forward(input);

    EXPECT_TRUE(has_valid_values(output));
}

TEST_F(EncoderBlockTest, ResidualConnectionPreservation) {
    EncoderBlock block(d_model, num_heads, d_ff, 0.0f);  // No dropout

    Matrix input(5, d_model);
    for (int i = 0; i < input.rows; ++i) {
        for (int j = 0; j < input.cols; ++j) {
            input.data[i][j] = 0.1f * (i + j);
        }
    }

    Matrix output = block.forward(input);

    // Output should be significantly influenced by input (residual connections)
    // We can't directly test this without knowing internal weights,
    // but we can verify output is valid and differs from zero
    float output_magnitude = 0.0f;
    for (int i = 0; i < output.rows; ++i) {
        for (int j = 0; j < output.cols; ++j) {
            output_magnitude += std::abs(output.data[i][j]);
        }
    }

    EXPECT_GT(output_magnitude, 0.0f);
}

// ============================================================================
// Configuration Tests
// ============================================================================

TEST_F(EncoderBlockTest, PrintConfiguration) {
    EncoderBlock block(512, 8, 2048, 0.1f);

    // Just verify it doesn't crash
    EXPECT_NO_THROW(block.print_config("TestEncoder"));
}

TEST_F(EncoderBlockTest, DifferentConfigurations) {
    std::vector<std::tuple<int, int, int>> configs = {
        {64, 4, 256}, {128, 8, 512}, {256, 8, 1024}, {512, 16, 2048}};

    for (const auto& [dm, nh, dff] : configs) {
        EncoderBlock block(dm, nh, dff);

        Matrix input(10, dm);
        for (int i = 0; i < input.rows; ++i) {
            for (int j = 0; j < input.cols; ++j) {
                input.data[i][j] = 0.01f * (i + j);
            }
        }

        Matrix output = block.forward(input);

        EXPECT_EQ(output.rows, 10);
        EXPECT_EQ(output.cols, dm);
        EXPECT_TRUE(has_valid_values(output));
    }
}

// Run all tests
int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
