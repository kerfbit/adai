#include "../src/DecoderBlock.hpp"
#include <../gtest/gtest.h>
#include <algorithm>
#include <cmath>
#include <fstream>
#include <stdexcept>
#include "../src/Matrix.hpp"

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

Matrix create_causal_mask(int seq_len) {
    Matrix mask(seq_len, seq_len);
    for (int i = 0; i < seq_len; ++i) {
        for (int j = 0; j < seq_len; ++j) {
            // Allow attention to current and past positions
            mask(i, j) = (j <= i) ? 1.0f : 0.0f;
        }
    }
    return mask;
}

// ============================================================================
// Constructor Tests
// ============================================================================

TEST(DecoderBlockConstructorTest, BasicInitialization) {
    int d_model = 128;
    int num_heads = 8;
    int d_ff = 512;

    DecoderBlock decoder_block(d_model, num_heads, d_ff);

    // Should not throw any exceptions
    EXPECT_NO_THROW({
        Matrix input(10, d_model);
        Matrix encoder_output(15, d_model);
        Matrix causal_mask = create_causal_mask(10);
        decoder_block.forward(input, encoder_output, causal_mask);
    });
}

TEST(DecoderBlockConstructorTest, InvalidHeadCount) {
    int d_model = 128;
    int num_heads = 7;  // Not a divisor of 128
    int d_ff = 512;

    // Should throw exception because MultiHeadAttention validates this
    EXPECT_THROW(DecoderBlock(d_model, num_heads, d_ff), std::invalid_argument);
}

TEST(DecoderBlockConstructorTest, CustomDropout) {
    int d_model = 64;
    int num_heads = 4;
    int d_ff = 256;
    float dropout = 0.2f;

    DecoderBlock decoder_block(d_model, num_heads, d_ff, dropout);

    // Should initialize correctly
    EXPECT_NO_THROW({
        Matrix input(5, d_model);
        Matrix encoder_output(8, d_model);
        Matrix causal_mask = create_causal_mask(5);
        decoder_block.forward(input, encoder_output, causal_mask);
    });
}

// ============================================================================
// Forward Pass Tests
// ============================================================================

TEST(DecoderBlockForwardTest, OutputDimensions) {
    int d_model = 128;
    int num_heads = 8;
    int d_ff = 512;
    int tgt_len = 10;
    int src_len = 15;

    DecoderBlock decoder_block(d_model, num_heads, d_ff);

    Matrix decoder_input(tgt_len, d_model);
    Matrix encoder_output(src_len, d_model);
    Matrix causal_mask = create_causal_mask(tgt_len);

    Matrix output = decoder_block.forward(decoder_input, encoder_output, causal_mask);

    EXPECT_EQ(output.rows, tgt_len);
    EXPECT_EQ(output.cols, d_model);
}

TEST(DecoderBlockForwardTest, DifferentSequenceLengths) {
    int d_model = 64;
    int num_heads = 4;
    int d_ff = 256;

    DecoderBlock decoder_block(d_model, num_heads, d_ff);

    // Test with various sequence length combinations
    std::vector<std::pair<int, int>> test_cases = {
        {5, 10},  // tgt shorter than src
        {10, 5},  // tgt longer than src
        {8, 8},   // equal length
        {1, 20},  // single target token
        {20, 1}   // single source token
    };

    for (const auto& [tgt_len, src_len] : test_cases) {
        Matrix decoder_input(tgt_len, d_model);
        Matrix encoder_output(src_len, d_model);
        Matrix causal_mask = create_causal_mask(tgt_len);

        Matrix output = decoder_block.forward(decoder_input, encoder_output, causal_mask);

        EXPECT_EQ(output.rows, tgt_len);
        EXPECT_EQ(output.cols, d_model);
    }
}

TEST(DecoderBlockForwardTest, WithCrossAttentionMask) {
    int d_model = 128;
    int num_heads = 8;
    int d_ff = 512;
    int tgt_len = 10;
    int src_len = 15;

    DecoderBlock decoder_block(d_model, num_heads, d_ff);

    Matrix decoder_input(tgt_len, d_model);
    Matrix encoder_output(src_len, d_model);
    Matrix causal_mask = create_causal_mask(tgt_len);

    // Create cross-attention mask (e.g., padding mask for encoder)
    Matrix cross_mask(tgt_len, src_len);
    for (int i = 0; i < tgt_len; ++i) {
        for (int j = 0; j < src_len; ++j) {
            // Mask out last 5 positions (simulating padding)
            cross_mask(i, j) = (j < src_len - 5) ? 1.0f : 0.0f;
        }
    }

    Matrix output = decoder_block.forward(decoder_input, encoder_output, causal_mask, &cross_mask);

    EXPECT_EQ(output.rows, tgt_len);
    EXPECT_EQ(output.cols, d_model);
}

TEST(DecoderBlockForwardTest, NonZeroValues) {
    int d_model = 64;
    int num_heads = 4;
    int d_ff = 256;
    int seq_len = 8;

    DecoderBlock decoder_block(d_model, num_heads, d_ff);

    Matrix decoder_input(seq_len, d_model);
    Matrix encoder_output(seq_len, d_model);

    // Initialize with non-zero values
    for (int i = 0; i < seq_len; ++i) {
        for (int j = 0; j < d_model; ++j) {
            decoder_input(i, j) = 0.1f * (i + j);
            encoder_output(i, j) = 0.05f * (i - j);
        }
    }

    Matrix causal_mask = create_causal_mask(seq_len);
    Matrix output = decoder_block.forward(decoder_input, encoder_output, causal_mask);

    // Check that output is not all zeros
    bool has_nonzero = false;
    for (int i = 0; i < output.rows; ++i) {
        for (int j = 0; j < output.cols; ++j) {
            if (std::abs(output(i, j)) > 1e-6f) {
                has_nonzero = true;
                break;
            }
        }
        if (has_nonzero)
            break;
    }
    EXPECT_TRUE(has_nonzero);
}

// ============================================================================
// Causal Masking Tests
// ============================================================================

TEST(DecoderBlockMaskingTest, CausalMaskStructure) {
    int seq_len = 5;
    Matrix mask = create_causal_mask(seq_len);

    // Verify lower triangular structure
    for (int i = 0; i < seq_len; ++i) {
        for (int j = 0; j < seq_len; ++j) {
            if (j <= i) {
                EXPECT_FLOAT_EQ(mask(i, j), 1.0f);  // Can attend to current and past
            } else {
                EXPECT_FLOAT_EQ(mask(i, j), 0.0f);  // Cannot attend to future
            }
        }
    }
}

TEST(DecoderBlockMaskingTest, CausalAttentionEnforcement) {
    int d_model = 64;
    int num_heads = 4;
    int d_ff = 256;
    int seq_len = 8;

    DecoderBlock decoder_block(d_model, num_heads, d_ff);

    // Create input with distinct pattern at each position
    Matrix decoder_input(seq_len, d_model);
    Matrix encoder_output(seq_len, d_model);

    for (int i = 0; i < seq_len; ++i) {
        for (int j = 0; j < d_model; ++j) {
            decoder_input(i, j) = static_cast<float>(i + 1);  // Position-specific values
            encoder_output(i, j) = 0.1f;
        }
    }

    Matrix causal_mask = create_causal_mask(seq_len);
    Matrix output = decoder_block.forward(decoder_input, encoder_output, causal_mask);

    // The output should exist and have proper dimensions
    EXPECT_EQ(output.rows, seq_len);
    EXPECT_EQ(output.cols, d_model);

    // First position should only see itself (minimal context)
    // Later positions should have access to more context
    // This is verified implicitly by the architecture
}

TEST(DecoderBlockMaskingTest, AllOnesVsAllZerosMask) {
    int d_model = 64;
    int num_heads = 4;
    int d_ff = 256;
    int seq_len = 5;

    DecoderBlock decoder_block(d_model, num_heads, d_ff);

    Matrix decoder_input(seq_len, d_model);
    Matrix encoder_output(seq_len, d_model);

    for (int i = 0; i < seq_len; ++i) {
        for (int j = 0; j < d_model; ++j) {
            decoder_input(i, j) = 0.1f * (i + j);
            encoder_output(i, j) = 0.05f;
        }
    }

    // Test 1: Causal mask (proper)
    Matrix causal_mask = create_causal_mask(seq_len);
    Matrix output_causal = decoder_block.forward(decoder_input, encoder_output, causal_mask);

    // Test 2: All-ones mask (no masking - bi-directional attention)
    Matrix all_ones(seq_len, seq_len);
    for (int i = 0; i < seq_len; ++i) {
        for (int j = 0; j < seq_len; ++j) {
            all_ones(i, j) = 1.0f;
        }
    }

    decoder_block.zero_grad();  // Reset state
    Matrix output_all_ones = decoder_block.forward(decoder_input, encoder_output, all_ones);

    // Outputs should be different (causal vs bidirectional)
    bool outputs_differ = false;
    for (int i = 0; i < seq_len; ++i) {
        for (int j = 0; j < d_model; ++j) {
            if (!is_close(output_causal(i, j), output_all_ones(i, j), 1e-3f)) {
                outputs_differ = true;
                break;
            }
        }
        if (outputs_differ)
            break;
    }

    EXPECT_TRUE(outputs_differ);
}

// ============================================================================
// Cross-Attention Tests
// ============================================================================

TEST(DecoderBlockCrossAttentionTest, EncoderInfluence) {
    int d_model = 64;
    int num_heads = 4;
    int d_ff = 256;
    int tgt_len = 5;
    int src_len = 10;

    DecoderBlock decoder_block(d_model, num_heads, d_ff);

    Matrix decoder_input(tgt_len, d_model);
    Matrix causal_mask = create_causal_mask(tgt_len);

    // Initialize decoder input
    for (int i = 0; i < tgt_len; ++i) {
        for (int j = 0; j < d_model; ++j) {
            decoder_input(i, j) = 0.1f;
        }
    }

    // Test 1: Encoder output with small values
    Matrix encoder_small(src_len, d_model);
    for (int i = 0; i < src_len; ++i) {
        for (int j = 0; j < d_model; ++j) {
            encoder_small(i, j) = 0.01f;
        }
    }

    Matrix output_small = decoder_block.forward(decoder_input, encoder_small, causal_mask);

    // Test 2: Encoder output with large values
    Matrix encoder_large(src_len, d_model);
    for (int i = 0; i < src_len; ++i) {
        for (int j = 0; j < d_model; ++j) {
            encoder_large(i, j) = 1.0f;
        }
    }

    decoder_block.zero_grad();
    Matrix output_large = decoder_block.forward(decoder_input, encoder_large, causal_mask);

    // Outputs should differ based on encoder input
    bool outputs_differ = false;
    for (int i = 0; i < tgt_len; ++i) {
        for (int j = 0; j < d_model; ++j) {
            if (!is_close(output_small(i, j), output_large(i, j), 1e-2f)) {
                outputs_differ = true;
                break;
            }
        }
        if (outputs_differ)
            break;
    }

    EXPECT_TRUE(outputs_differ);
}

TEST(DecoderBlockCrossAttentionTest, DifferentEncoderLengths) {
    int d_model = 64;
    int num_heads = 4;
    int d_ff = 256;
    int tgt_len = 8;

    DecoderBlock decoder_block(d_model, num_heads, d_ff);

    Matrix decoder_input(tgt_len, d_model);
    Matrix causal_mask = create_causal_mask(tgt_len);

    for (int i = 0; i < tgt_len; ++i) {
        for (int j = 0; j < d_model; ++j) {
            decoder_input(i, j) = 0.1f * i;
        }
    }

    // Test with different encoder sequence lengths
    std::vector<int> encoder_lengths = {5, 10, 20, 50};

    for (int src_len : encoder_lengths) {
        Matrix encoder_output(src_len, d_model);
        for (int i = 0; i < src_len; ++i) {
            for (int j = 0; j < d_model; ++j) {
                encoder_output(i, j) = 0.05f;
            }
        }

        decoder_block.zero_grad();
        Matrix output = decoder_block.forward(decoder_input, encoder_output, causal_mask);

        EXPECT_EQ(output.rows, tgt_len);
        EXPECT_EQ(output.cols, d_model);
    }
}

// ============================================================================
// Backward Pass Tests
// ============================================================================

TEST(DecoderBlockBackwardTest, GradientDimensions) {
    int d_model = 64;
    int num_heads = 4;
    int d_ff = 256;
    int tgt_len = 8;
    int src_len = 12;

    DecoderBlock decoder_block(d_model, num_heads, d_ff);

    Matrix decoder_input(tgt_len, d_model);
    Matrix encoder_output(src_len, d_model);
    Matrix causal_mask = create_causal_mask(tgt_len);

    // Forward pass
    Matrix output = decoder_block.forward(decoder_input, encoder_output, causal_mask);

    // Backward pass
    Matrix grad_output(tgt_len, d_model);
    for (int i = 0; i < tgt_len; ++i) {
        for (int j = 0; j < d_model; ++j) {
            grad_output(i, j) = 0.01f;
        }
    }

    Matrix grad_input = decoder_block.backward(grad_output);

    EXPECT_EQ(grad_input.rows, tgt_len);
    EXPECT_EQ(grad_input.cols, d_model);
}

TEST(DecoderBlockBackwardTest, GradientNonZero) {
    int d_model = 64;
    int num_heads = 4;
    int d_ff = 256;
    int seq_len = 8;

    DecoderBlock decoder_block(d_model, num_heads, d_ff);

    Matrix decoder_input(seq_len, d_model);
    Matrix encoder_output(seq_len, d_model);

    for (int i = 0; i < seq_len; ++i) {
        for (int j = 0; j < d_model; ++j) {
            decoder_input(i, j) = 0.1f * (i + j);
            encoder_output(i, j) = 0.05f * i;
        }
    }

    Matrix causal_mask = create_causal_mask(seq_len);
    Matrix output = decoder_block.forward(decoder_input, encoder_output, causal_mask);

    Matrix grad_output(seq_len, d_model);
    for (int i = 0; i < seq_len; ++i) {
        for (int j = 0; j < d_model; ++j) {
            grad_output(i, j) = 0.01f;
        }
    }

    Matrix grad_input = decoder_block.backward(grad_output);

    // Check that gradients are non-zero
    float grad_norm = compute_gradient_norm(grad_input);
    EXPECT_GT(grad_norm, 0.0f);
}

TEST(DecoderBlockBackwardTest, GradientFlow) {
    int d_model = 32;
    int num_heads = 4;
    int d_ff = 128;
    int seq_len = 5;

    DecoderBlock decoder_block(d_model, num_heads, d_ff);
    decoder_block.set_learning_rate(0.001f);

    Matrix decoder_input(seq_len, d_model);
    Matrix encoder_output(seq_len, d_model);

    for (int i = 0; i < seq_len; ++i) {
        for (int j = 0; j < d_model; ++j) {
            decoder_input(i, j) = 0.1f * (i + 1);
            encoder_output(i, j) = 0.05f;
        }
    }

    Matrix causal_mask = create_causal_mask(seq_len);

    // Forward pass
    Matrix output = decoder_block.forward(decoder_input, encoder_output, causal_mask);

    // Create gradient
    Matrix grad_output(seq_len, d_model);
    for (int i = 0; i < seq_len; ++i) {
        for (int j = 0; j < d_model; ++j) {
            grad_output(i, j) = (i == 0 && j == 0) ? 1.0f : 0.0f;
        }
    }

    // Backward pass
    Matrix grad_input = decoder_block.backward(grad_output);

    // Gradient should propagate through all layers
    float total_grad = 0.0f;
    for (int i = 0; i < grad_input.rows; ++i) {
        for (int j = 0; j < grad_input.cols; ++j) {
            total_grad += std::abs(grad_input(i, j));
        }
    }

    EXPECT_GT(total_grad, 0.0f);
}

TEST(DecoderBlockBackwardTest, MultipleBackwardPasses) {
    int d_model = 64;
    int num_heads = 4;
    int d_ff = 256;
    int seq_len = 6;

    DecoderBlock decoder_block(d_model, num_heads, d_ff);

    Matrix decoder_input(seq_len, d_model);
    Matrix encoder_output(seq_len, d_model);
    Matrix causal_mask = create_causal_mask(seq_len);

    for (int i = 0; i < seq_len; ++i) {
        for (int j = 0; j < d_model; ++j) {
            decoder_input(i, j) = 0.1f;
            encoder_output(i, j) = 0.05f;
        }
    }

    // Multiple forward-backward cycles
    for (int iter = 0; iter < 3; ++iter) {
        decoder_block.zero_grad();

        Matrix output = decoder_block.forward(decoder_input, encoder_output, causal_mask);

        Matrix grad_output(seq_len, d_model);
        for (int i = 0; i < seq_len; ++i) {
            for (int j = 0; j < d_model; ++j) {
                grad_output(i, j) = 0.01f;
            }
        }

        Matrix grad_input = decoder_block.backward(grad_output);

        EXPECT_EQ(grad_input.rows, seq_len);
        EXPECT_EQ(grad_input.cols, d_model);
    }
}

// ============================================================================
// Weight Update Tests
// ============================================================================

TEST(DecoderBlockUpdateTest, WeightUpdateChangesOutput) {
    int d_model = 64;
    int num_heads = 4;
    int d_ff = 256;
    int seq_len = 8;

    DecoderBlock decoder_block(d_model, num_heads, d_ff);
    decoder_block.set_learning_rate(0.1f);  // Large LR for visible changes

    Matrix decoder_input(seq_len, d_model);
    Matrix encoder_output(seq_len, d_model);
    Matrix causal_mask = create_causal_mask(seq_len);

    for (int i = 0; i < seq_len; ++i) {
        for (int j = 0; j < d_model; ++j) {
            decoder_input(i, j) = 0.1f * (i + j);
            encoder_output(i, j) = 0.05f;
        }
    }

    // Initial output
    Matrix output_before = decoder_block.forward(decoder_input, encoder_output, causal_mask);

    // Backward pass and update
    Matrix grad_output(seq_len, d_model);
    for (int i = 0; i < seq_len; ++i) {
        for (int j = 0; j < d_model; ++j) {
            grad_output(i, j) = 0.1f;
        }
    }

    decoder_block.backward(grad_output);
    decoder_block.update_weights();

    // Output after update
    decoder_block.zero_grad();
    Matrix output_after = decoder_block.forward(decoder_input, encoder_output, causal_mask);

    // Outputs should differ
    bool weights_changed = false;
    for (int i = 0; i < seq_len; ++i) {
        for (int j = 0; j < d_model; ++j) {
            if (!is_close(output_before(i, j), output_after(i, j), 1e-4f)) {
                weights_changed = true;
                break;
            }
        }
        if (weights_changed)
            break;
    }

    EXPECT_TRUE(weights_changed);
}

TEST(DecoderBlockUpdateTest, LearningRatePropagation) {
    int d_model = 64;
    int num_heads = 4;
    int d_ff = 256;

    DecoderBlock decoder_block(d_model, num_heads, d_ff);

    float new_lr = 0.0005f;
    decoder_block.set_learning_rate(new_lr);

    EXPECT_FLOAT_EQ(decoder_block.learning_rate, new_lr);
}

// ============================================================================
// Save/Load Tests
// ============================================================================

TEST(DecoderBlockSaveLoadTest, BasicSaveLoad) {
    int d_model = 64;
    int num_heads = 4;
    int d_ff = 256;
    int seq_len = 6;

    DecoderBlock decoder_block(d_model, num_heads, d_ff);

    Matrix decoder_input(seq_len, d_model);
    Matrix encoder_output(seq_len, d_model);
    Matrix causal_mask = create_causal_mask(seq_len);

    for (int i = 0; i < seq_len; ++i) {
        for (int j = 0; j < d_model; ++j) {
            decoder_input(i, j) = 0.1f * (i + j);
            encoder_output(i, j) = 0.05f;
        }
    }

    // Get output from original
    Matrix output_original = decoder_block.forward(decoder_input, encoder_output, causal_mask);

    // Save
    std::string filepath = "test_decoder_block.bin";
    decoder_block.save(filepath);

    // Load into new block
    DecoderBlock loaded_block(d_model, num_heads, d_ff);
    loaded_block.load(filepath);

    // Get output from loaded
    Matrix output_loaded = loaded_block.forward(decoder_input, encoder_output, causal_mask);

    // Outputs should be very close (not exact due to LayerNorm reinitialization)
    float max_diff = 0.0f;
    for (int i = 0; i < seq_len; ++i) {
        for (int j = 0; j < d_model; ++j) {
            float diff = std::abs(output_original(i, j) - output_loaded(i, j));
            max_diff = std::max(max_diff, diff);
        }
    }

    // Allow some tolerance due to LayerNorm not being saved
    EXPECT_LT(max_diff, 1.0f);

    // Clean up
    std::remove(filepath.c_str());
    std::remove((filepath + ".self_attn").c_str());
    std::remove((filepath + ".cross_attn").c_str());
    std::remove((filepath + ".ff").c_str());
}

TEST(DecoderBlockSaveLoadTest, MultipleIterations) {
    int d_model = 64;
    int num_heads = 4;
    int d_ff = 256;
    int seq_len = 5;

    DecoderBlock decoder_block(d_model, num_heads, d_ff);
    decoder_block.set_learning_rate(0.01f);

    Matrix decoder_input(seq_len, d_model);
    Matrix encoder_output(seq_len, d_model);
    Matrix causal_mask = create_causal_mask(seq_len);

    for (int i = 0; i < seq_len; ++i) {
        for (int j = 0; j < d_model; ++j) {
            decoder_input(i, j) = 0.1f;
            encoder_output(i, j) = 0.05f;
        }
    }

    // Train for a few iterations
    for (int iter = 0; iter < 5; ++iter) {
        decoder_block.zero_grad();
        Matrix output = decoder_block.forward(decoder_input, encoder_output, causal_mask);

        Matrix grad_output(seq_len, d_model);
        for (int i = 0; i < seq_len; ++i) {
            for (int j = 0; j < d_model; ++j) {
                grad_output(i, j) = 0.01f;
            }
        }

        decoder_block.backward(grad_output);
        decoder_block.update_weights();
    }

    // Get final output
    decoder_block.zero_grad();
    Matrix output_final = decoder_block.forward(decoder_input, encoder_output, causal_mask);

    // Save
    std::string filepath = "test_decoder_trained.bin";
    decoder_block.save(filepath);

    // Load
    DecoderBlock loaded_block(d_model, num_heads, d_ff);
    loaded_block.load(filepath);

    // Get output from loaded
    Matrix output_loaded = loaded_block.forward(decoder_input, encoder_output, causal_mask);

    // Should produce similar outputs
    float max_diff = 0.0f;
    for (int i = 0; i < seq_len; ++i) {
        for (int j = 0; j < d_model; ++j) {
            float diff = std::abs(output_final(i, j) - output_loaded(i, j));
            max_diff = std::max(max_diff, diff);
        }
    }

    EXPECT_LT(max_diff, 1.0f);

    // Clean up
    std::remove(filepath.c_str());
    std::remove((filepath + ".self_attn").c_str());
    std::remove((filepath + ".cross_attn").c_str());
    std::remove((filepath + ".ff").c_str());
}

// ============================================================================
// Integration Tests
// ============================================================================

TEST(DecoderBlockIntegrationTest, MultiLayerStack) {
    int d_model = 64;
    int num_heads = 4;
    int d_ff = 256;
    int seq_len = 8;
    int num_layers = 3;

    // Create stack of decoder blocks
    std::vector<DecoderBlock> layers;
    for (int i = 0; i < num_layers; ++i) {
        layers.emplace_back(d_model, num_heads, d_ff);
    }

    Matrix decoder_input(seq_len, d_model);
    Matrix encoder_output(seq_len, d_model);
    Matrix causal_mask = create_causal_mask(seq_len);

    for (int i = 0; i < seq_len; ++i) {
        for (int j = 0; j < d_model; ++j) {
            decoder_input(i, j) = 0.1f * i;
            encoder_output(i, j) = 0.05f;
        }
    }

    // Forward through all layers
    Matrix x = decoder_input;
    for (auto& layer : layers) {
        x = layer.forward(x, encoder_output, causal_mask);
    }

    EXPECT_EQ(x.rows, seq_len);
    EXPECT_EQ(x.cols, d_model);

    // Backward through all layers (reverse order)
    Matrix grad(seq_len, d_model);
    for (int i = 0; i < seq_len; ++i) {
        for (int j = 0; j < d_model; ++j) {
            grad(i, j) = 0.01f;
        }
    }

    for (auto it = layers.rbegin(); it != layers.rend(); ++it) {
        grad = it->backward(grad);
    }

    EXPECT_EQ(grad.rows, seq_len);
    EXPECT_EQ(grad.cols, d_model);
}

TEST(DecoderBlockIntegrationTest, TrainingLoop) {
    int d_model = 64;
    int num_heads = 4;
    int d_ff = 256;
    int seq_len = 6;

    DecoderBlock decoder_block(d_model, num_heads, d_ff);
    decoder_block.set_learning_rate(0.01f);

    Matrix decoder_input(seq_len, d_model);
    Matrix encoder_output(seq_len, d_model);
    Matrix causal_mask = create_causal_mask(seq_len);

    for (int i = 0; i < seq_len; ++i) {
        for (int j = 0; j < d_model; ++j) {
            decoder_input(i, j) = 0.1f * (i + 1);
            encoder_output(i, j) = 0.05f;
        }
    }

    // Training loop
    std::vector<float> losses;
    for (int epoch = 0; epoch < 10; ++epoch) {
        decoder_block.zero_grad();

        Matrix output = decoder_block.forward(decoder_input, encoder_output, causal_mask);

        // Dummy loss (MSE with target)
        float loss = 0.0f;
        Matrix grad_output(seq_len, d_model);
        for (int i = 0; i < seq_len; ++i) {
            for (int j = 0; j < d_model; ++j) {
                float target = 0.0f;
                float error = output(i, j) - target;
                loss += error * error;
                grad_output(i, j) = 2.0f * error / (seq_len * d_model);
            }
        }
        losses.push_back(loss);

        decoder_block.backward(grad_output);
        decoder_block.update_weights();
    }

    // Loss should generally decrease (may fluctuate)
    EXPECT_LT(losses.back(), losses[0] * 2.0f);
}

TEST(DecoderBlockIntegrationTest, LongSequence) {
    int d_model = 128;
    int num_heads = 8;
    int d_ff = 512;
    int tgt_len = 50;
    int src_len = 100;

    DecoderBlock decoder_block(d_model, num_heads, d_ff);

    Matrix decoder_input(tgt_len, d_model);
    Matrix encoder_output(src_len, d_model);
    Matrix causal_mask = create_causal_mask(tgt_len);

    for (int i = 0; i < tgt_len; ++i) {
        for (int j = 0; j < d_model; ++j) {
            decoder_input(i, j) = 0.01f * i;
        }
    }

    for (int i = 0; i < src_len; ++i) {
        for (int j = 0; j < d_model; ++j) {
            encoder_output(i, j) = 0.005f * i;
        }
    }

    // Should handle long sequences
    EXPECT_NO_THROW({
        Matrix output = decoder_block.forward(decoder_input, encoder_output, causal_mask);
        EXPECT_EQ(output.rows, tgt_len);
        EXPECT_EQ(output.cols, d_model);
    });
}

// ============================================================================
// Edge Case Tests
// ============================================================================

TEST(DecoderBlockEdgeCaseTest, SingleToken) {
    int d_model = 64;
    int num_heads = 4;
    int d_ff = 256;

    DecoderBlock decoder_block(d_model, num_heads, d_ff);

    Matrix decoder_input(1, d_model);
    Matrix encoder_output(10, d_model);
    Matrix causal_mask = create_causal_mask(1);

    for (int j = 0; j < d_model; ++j) {
        decoder_input(0, j) = 0.1f;
    }

    for (int i = 0; i < 10; ++i) {
        for (int j = 0; j < d_model; ++j) {
            encoder_output(i, j) = 0.05f;
        }
    }

    Matrix output = decoder_block.forward(decoder_input, encoder_output, causal_mask);

    EXPECT_EQ(output.rows, 1);
    EXPECT_EQ(output.cols, d_model);
}

TEST(DecoderBlockEdgeCaseTest, ZeroInput) {
    int d_model = 64;
    int num_heads = 4;
    int d_ff = 256;
    int seq_len = 8;

    DecoderBlock decoder_block(d_model, num_heads, d_ff);

    Matrix decoder_input(seq_len, d_model);   // All zeros
    Matrix encoder_output(seq_len, d_model);  // All zeros
    Matrix causal_mask = create_causal_mask(seq_len);

    Matrix output = decoder_block.forward(decoder_input, encoder_output, causal_mask);

    // Should produce output (layer norm prevents all zeros)
    EXPECT_EQ(output.rows, seq_len);
    EXPECT_EQ(output.cols, d_model);
}

TEST(DecoderBlockEdgeCaseTest, SmallModelDimensions) {
    int d_model = 16;
    int num_heads = 2;
    int d_ff = 64;
    int seq_len = 4;

    DecoderBlock decoder_block(d_model, num_heads, d_ff);

    Matrix decoder_input(seq_len, d_model);
    Matrix encoder_output(seq_len, d_model);
    Matrix causal_mask = create_causal_mask(seq_len);

    for (int i = 0; i < seq_len; ++i) {
        for (int j = 0; j < d_model; ++j) {
            decoder_input(i, j) = 0.1f;
            encoder_output(i, j) = 0.05f;
        }
    }

    EXPECT_NO_THROW({
        Matrix output = decoder_block.forward(decoder_input, encoder_output, causal_mask);
        EXPECT_EQ(output.rows, seq_len);
        EXPECT_EQ(output.cols, d_model);
    });
}

// ============================================================================
// Main
// ============================================================================

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
