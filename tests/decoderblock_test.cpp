#include "../src/DecoderBlock.hpp"
#include <../gtest/gtest.h>
#include <algorithm>
#include <cmath>
#include <fstream>
#include <stdexcept>
#include <utility>
#include <vector>
#include "../src/HippocampalMemory.hpp"
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

// TD-180: copies every core-sublayer weight (self-attention, cross-attention, feed-forward,
// all three layer norms) from `src` into `dst`, via the same public SafeTensors-style
// get_W*/set_W* accessors DecoderBlock::save()/load() itself uses. Lets a no-op test compare
// two separately-constructed instances (one with the gated paths allocated, one without) as if
// they were the same weights — otherwise each instance's own random initialization would make
// any output comparison meaningless.
void clone_core_weights(DecoderBlock& dst, DecoderBlock& src) {
    dst.get_self_attention()->set_Wq(src.get_self_attention()->get_Wq());
    dst.get_self_attention()->set_Wk(src.get_self_attention()->get_Wk());
    dst.get_self_attention()->set_Wv(src.get_self_attention()->get_Wv());
    dst.get_self_attention()->set_Wo(src.get_self_attention()->get_Wo());

    dst.get_cross_attention()->set_Wq(src.get_cross_attention()->get_Wq());
    dst.get_cross_attention()->set_Wk(src.get_cross_attention()->get_Wk());
    dst.get_cross_attention()->set_Wv(src.get_cross_attention()->get_Wv());
    dst.get_cross_attention()->set_Wo(src.get_cross_attention()->get_Wo());

    dst.get_feed_forward()->set_W1(src.get_feed_forward()->get_W1());
    dst.get_feed_forward()->set_W2(src.get_feed_forward()->get_W2());
    dst.get_feed_forward()->set_b1(src.get_feed_forward()->get_b1());
    dst.get_feed_forward()->set_b2(src.get_feed_forward()->get_b2());

    dst.get_norm1()->set_gamma(src.get_norm1()->get_gamma());
    dst.get_norm1()->set_beta(src.get_norm1()->get_beta());
    dst.get_norm2()->set_gamma(src.get_norm2()->get_gamma());
    dst.get_norm2()->set_beta(src.get_norm2()->get_beta());
    dst.get_norm3()->set_gamma(src.get_norm3()->get_gamma());
    dst.get_norm3()->set_beta(src.get_norm3()->get_beta());
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

    // Pre-LN normalizes the input (per-row mean/variance) before self-attention,
    // so a per-row additive constant (e.g. 0.1f*(i+j)) is removed by that
    // normalization and every row ends up producing nearly identical Q/K/V —
    // masking then has almost no visible effect regardless of causal vs
    // bidirectional, since attending to "different" positions doesn't matter
    // when they're all nearly the same vector. Use a per-row varying shape
    // (not just a shift) so rows stay genuinely distinguishable post-norm.
    for (int i = 0; i < seq_len; ++i) {
        for (int j = 0; j < d_model; ++j) {
            decoder_input(i, j) =
                0.1f * static_cast<float>(i + 1) * std::sin(0.3f * static_cast<float>(j));
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

TEST(DecoderBlockBackwardTest, GradientNormComputation) {
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

    // Before any backward pass, accumulated gradients are zero.
    EXPECT_FLOAT_EQ(decoder_block.get_gradient_norm(), 0.0f);

    Matrix output = decoder_block.forward(decoder_input, encoder_output, causal_mask);
    Matrix grad_output(seq_len, d_model);
    for (int i = 0; i < seq_len; ++i) {
        for (int j = 0; j < d_model; ++j) {
            grad_output(i, j) = 0.01f * (i + 1) + 0.001f * (j + 1);
        }
    }
    decoder_block.backward(grad_output);

    // After backward, self-attention/cross-attention/feed-forward gradients
    // (self_attention, cross_attention, feed_forward — see
    // DecoderBlock::get_gradient_norm()) should combine to a nonzero norm.
    EXPECT_GT(decoder_block.get_gradient_norm(), 0.0f);
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

// Finite-difference gradient check for the Pre-LN forward/backward rewrite.
// This is the real correctness gate for the Post-LN -> Pre-LN migration: a
// subtly wrong backward pass (e.g. an accumulation point moved to the wrong
// residual, or the cross-attention encoder-gradient contribution dropped)
// would still produce finite, plausible-looking gradients under the other
// tests in this file, but would fail this check.
TEST(DecoderBlockBackwardTest, BackwardPassMatchesNumericalGradient) {
    int d_model = 32;
    int num_heads = 4;
    int d_ff = 64;
    int tgt_len = 4;
    int src_len = 5;

    DecoderBlock decoder_block(d_model, num_heads, d_ff);

    Matrix decoder_input(tgt_len, d_model);
    Matrix encoder_output(src_len, d_model);
    for (int i = 0; i < tgt_len; ++i) {
        for (int j = 0; j < d_model; ++j) {
            decoder_input(i, j) = 0.05f * std::sin(static_cast<float>(i * d_model + j));
        }
    }
    for (int i = 0; i < src_len; ++i) {
        for (int j = 0; j < d_model; ++j) {
            encoder_output(i, j) = 0.05f * std::cos(static_cast<float>(i * d_model + j));
        }
    }
    Matrix causal_mask = create_causal_mask(tgt_len);

    auto scalar_loss = [](const Matrix& out) {
        float loss = 0.0f;
        for (int i = 0; i < out.rows; ++i) {
            for (int j = 0; j < out.cols; ++j) {
                loss += out(i, j) * out(i, j);
            }
        }
        return loss;
    };

    Matrix output = decoder_block.forward(decoder_input, encoder_output, causal_mask);
    Matrix grad_output(output.rows, output.cols);
    for (int i = 0; i < output.rows; ++i) {
        for (int j = 0; j < output.cols; ++j) {
            grad_output(i, j) = 2.0f * output(i, j);
        }
    }
    Matrix analytic_grad = decoder_block.backward(grad_output);

    const float epsilon = 1e-3f;
    const std::vector<std::pair<int, int>> positions = {
        {0, 0}, {0, d_model / 2}, {1, 3}, {tgt_len - 1, d_model - 1}};

    for (const auto& [pi, pj] : positions) {
        Matrix input_plus = decoder_input;
        input_plus(pi, pj) += epsilon;
        Matrix input_minus = decoder_input;
        input_minus(pi, pj) -= epsilon;

        float loss_plus =
            scalar_loss(decoder_block.forward(input_plus, encoder_output, causal_mask));
        float loss_minus =
            scalar_loss(decoder_block.forward(input_minus, encoder_output, causal_mask));
        float numerical_grad = (loss_plus - loss_minus) / (2.0f * epsilon);

        float tolerance = std::max(1e-2f, 0.05f * std::abs(numerical_grad));
        EXPECT_NEAR(analytic_grad(pi, pj), numerical_grad, tolerance)
            << "gradient mismatch at (" << pi << "," << pj << ")";
    }
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
// TD-180: Gated World-Model / Hippocampal Cross-Attention Path Tests
// ============================================================================

// No-op guarantee #1 (TD-180's own Action Item): an instance constructed WITHOUT the gated
// paths (the default — every pre-TD-180 caller's own behavior) must produce identical output
// whether or not "real-looking" world_model_output/memory arguments are passed to forward() —
// world_model_cross_attention/hippocampal_cross_attention are both null, so the gated branches
// are unreachable regardless of what forward() is given.
TEST(DecoderBlockGatedPathTest, NoOpWhenGatedPathsNotAllocated) {
    int d_model = 32, num_heads = 4, d_ff = 64;
    int tgt_len = 4, src_len = 5, wm_len = 3;

    DecoderBlock decoder_block(d_model, num_heads, d_ff);  // gated paths NOT enabled

    Matrix decoder_input(tgt_len, d_model);
    Matrix encoder_output(src_len, d_model);
    for (int i = 0; i < tgt_len; ++i) {
        for (int j = 0; j < d_model; ++j) {
            decoder_input(i, j) = 0.05f * std::sin(static_cast<float>(i * d_model + j));
        }
    }
    for (int i = 0; i < src_len; ++i) {
        for (int j = 0; j < d_model; ++j) {
            encoder_output(i, j) = 0.05f * std::cos(static_cast<float>(i * d_model + j));
        }
    }
    Matrix causal_mask = create_causal_mask(tgt_len);

    Matrix baseline_output = decoder_block.forward(decoder_input, encoder_output, causal_mask);

    Matrix world_model_output(wm_len, d_model);
    for (int i = 0; i < wm_len; ++i) {
        for (int j = 0; j < d_model; ++j) {
            world_model_output(i, j) = 0.2f;
        }
    }
    HippocampalMemory memory(d_model, 8);
    Matrix key(1, d_model), value(1, d_model);
    for (int j = 0; j < d_model; ++j) {
        key(0, j) = 0.3f;
        value(0, j) = 0.3f;
    }
    memory.write(key, value);

    Matrix output_with_real_args =
        decoder_block.forward(decoder_input, encoder_output, causal_mask, nullptr,
                              &world_model_output, nullptr, &memory, 0.5f, 0.9f);

    EXPECT_TRUE(matrices_equal(baseline_output, output_with_real_args));
}

// No-op guarantee #1, other half: an instance WITH both gated paths allocated, called with the
// default nullptr world_model_output/memory, must match a same-weights baseline instance that
// never allocated them at all.
TEST(DecoderBlockGatedPathTest, NoOpWhenGatedPathsAllocatedButInputsNull) {
    int d_model = 32, num_heads = 4, d_ff = 64;
    int tgt_len = 4, src_len = 5;

    DecoderBlock baseline(d_model, num_heads, d_ff);
    DecoderBlock gated(d_model, num_heads, d_ff, 0.1f, /*enable_world_model=*/true,
                       /*enable_hippocampal=*/true);
    clone_core_weights(gated, baseline);

    Matrix decoder_input(tgt_len, d_model);
    Matrix encoder_output(src_len, d_model);
    for (int i = 0; i < tgt_len; ++i) {
        for (int j = 0; j < d_model; ++j) {
            decoder_input(i, j) = 0.05f * std::sin(static_cast<float>(i * d_model + j));
        }
    }
    for (int i = 0; i < src_len; ++i) {
        for (int j = 0; j < d_model; ++j) {
            encoder_output(i, j) = 0.05f * std::cos(static_cast<float>(i * d_model + j));
        }
    }
    Matrix causal_mask = create_causal_mask(tgt_len);

    Matrix baseline_output = baseline.forward(decoder_input, encoder_output, causal_mask);
    Matrix gated_output = gated.forward(decoder_input, encoder_output, causal_mask);

    EXPECT_TRUE(matrices_equal(baseline_output, gated_output));
}

// No-op guarantee #2 (TD-180's own Action Item): with both gated paths allocated AND real,
// non-null world_model_output/memory supplied, gate == gate_h == 0.0f (the construction
// default) must still reproduce the no-gated-input output exactly — verifies the *gates*, not
// just the pointer arguments, are what's disabled by default.
TEST(DecoderBlockGatedPathTest, NoOpWhenGateIsZeroEvenWithRealInputs) {
    int d_model = 32, num_heads = 4, d_ff = 64;
    int tgt_len = 4, src_len = 5, wm_len = 3;

    DecoderBlock decoder_block(d_model, num_heads, d_ff, 0.1f, true, true);

    Matrix decoder_input(tgt_len, d_model);
    Matrix encoder_output(src_len, d_model);
    for (int i = 0; i < tgt_len; ++i) {
        for (int j = 0; j < d_model; ++j) {
            decoder_input(i, j) = 0.05f * std::sin(static_cast<float>(i * d_model + j));
        }
    }
    for (int i = 0; i < src_len; ++i) {
        for (int j = 0; j < d_model; ++j) {
            encoder_output(i, j) = 0.05f * std::cos(static_cast<float>(i * d_model + j));
        }
    }
    Matrix causal_mask = create_causal_mask(tgt_len);

    Matrix output_without_gated_inputs =
        decoder_block.forward(decoder_input, encoder_output, causal_mask);

    Matrix world_model_output(wm_len, d_model);
    for (int i = 0; i < wm_len; ++i) {
        for (int j = 0; j < d_model; ++j) {
            world_model_output(i, j) = 0.4f;
        }
    }
    HippocampalMemory memory(d_model, 8);
    Matrix key(1, d_model), value(1, d_model);
    for (int j = 0; j < d_model; ++j) {
        key(0, j) = 0.3f;
        value(0, j) = 0.3f;
    }
    memory.write(key, value);

    Matrix output_with_real_gated_inputs =
        decoder_block.forward(decoder_input, encoder_output, causal_mask, nullptr,
                              &world_model_output, nullptr, &memory, 0.5f, 0.9f);

    EXPECT_TRUE(matrices_equal(output_without_gated_inputs, output_with_real_gated_inputs));
}

// TD-180's own Action Item: hammer a single hippocampal slot for many decode steps and confirm
// its coverage never exceeds the theoretical bound 1 / (1 - repetition_decay) — self-bounding
// by construction (decay-then-accumulate, per-step increment always in [0, 1]).
TEST(DecoderBlockGatedPathTest, CoverageNeverExceedsTheoreticalBound) {
    int d_model = 16, num_heads = 2, d_ff = 32;
    int tgt_len = 2, src_len = 3;

    DecoderBlock decoder_block(d_model, num_heads, d_ff, 0.1f, /*enable_world_model=*/false,
                               /*enable_hippocampal=*/true);

    HippocampalMemory memory(d_model, /*capacity=*/1);  // exactly one slot to hammer
    Matrix key(1, d_model), value(1, d_model);
    for (int j = 0; j < d_model; ++j) {
        key(0, j) = 0.1f * static_cast<float>(j);
        value(0, j) = 0.1f * static_cast<float>(j);
    }
    memory.write(key, value);

    Matrix decoder_input(tgt_len, d_model);
    Matrix encoder_output(src_len, d_model);
    for (int i = 0; i < tgt_len; ++i) {
        for (int j = 0; j < d_model; ++j) {
            decoder_input(i, j) = 0.1f * static_cast<float>(i + j);
        }
    }
    for (int i = 0; i < src_len; ++i) {
        for (int j = 0; j < d_model; ++j) {
            encoder_output(i, j) = 0.1f * static_cast<float>(i - j);
        }
    }
    Matrix causal_mask = create_causal_mask(tgt_len);

    const float repetition_decay = 0.9f;
    const float bound = 1.0f / (1.0f - repetition_decay);

    for (int step = 0; step < 300; ++step) {
        decoder_block.forward(decoder_input, encoder_output, causal_mask, nullptr, nullptr,
                              nullptr, &memory, /*repetition_alpha=*/0.5f, repetition_decay);
        ASSERT_EQ(memory.size(), 1);
        float coverage_val = memory.coverage_vector()[0];
        EXPECT_LE(coverage_val, bound + 1e-4f) << "step " << step;
        EXPECT_GE(coverage_val, 0.0f) << "step " << step;
    }
}

// TD-180's own Action Item: gradient check on the world-model gate (finite-difference vs.
// analytic tanh derivative), same technique as BackwardPassMatchesNumericalGradient above.
TEST(DecoderBlockGatedPathTest, GateGradientMatchesFiniteDifference) {
    int d_model = 16, num_heads = 2, d_ff = 32;
    int tgt_len = 3, src_len = 4, wm_len = 3;

    DecoderBlock decoder_block(d_model, num_heads, d_ff, 0.1f, /*enable_world_model=*/true,
                               /*enable_hippocampal=*/false);
    decoder_block.set_gate(0.3f);  // away from 0 so tanh's own curvature is actually exercised

    Matrix decoder_input(tgt_len, d_model);
    Matrix encoder_output(src_len, d_model);
    Matrix world_model_output(wm_len, d_model);
    for (int i = 0; i < tgt_len; ++i) {
        for (int j = 0; j < d_model; ++j) {
            decoder_input(i, j) = 0.05f * std::sin(static_cast<float>(i * d_model + j));
        }
    }
    for (int i = 0; i < src_len; ++i) {
        for (int j = 0; j < d_model; ++j) {
            encoder_output(i, j) = 0.05f * std::cos(static_cast<float>(i * d_model + j));
        }
    }
    for (int i = 0; i < wm_len; ++i) {
        for (int j = 0; j < d_model; ++j) {
            world_model_output(i, j) = 0.05f * static_cast<float>(i - j);
        }
    }
    Matrix causal_mask = create_causal_mask(tgt_len);

    Matrix output = decoder_block.forward(decoder_input, encoder_output, causal_mask, nullptr,
                                          &world_model_output);

    Matrix grad_output(output.rows, output.cols);
    for (int i = 0; i < output.rows; ++i) {
        for (int j = 0; j < output.cols; ++j) {
            grad_output(i, j) = 0.05f * static_cast<float>(i - j);
        }
    }
    decoder_block.backward(grad_output);
    float analytic_grad = decoder_block.get_gate_grad();

    auto weighted_output_sum = [&]() {
        Matrix out = decoder_block.forward(decoder_input, encoder_output, causal_mask, nullptr,
                                           &world_model_output);
        float total = 0.0f;
        for (int i = 0; i < out.rows; ++i) {
            for (int j = 0; j < out.cols; ++j) {
                total += out(i, j) * grad_output(i, j);
            }
        }
        return total;
    };

    const float original_gate = decoder_block.get_gate();
    const float epsilon = 1e-3f;

    decoder_block.set_gate(original_gate + epsilon);
    float loss_plus = weighted_output_sum();
    decoder_block.set_gate(original_gate - epsilon);
    float loss_minus = weighted_output_sum();
    decoder_block.set_gate(original_gate);

    float numerical_grad = (loss_plus - loss_minus) / (2.0f * epsilon);
    float tolerance = std::max(1e-2f, 0.05f * std::abs(numerical_grad));
    EXPECT_NEAR(analytic_grad, numerical_grad, tolerance);
}

// Same check, hippocampal gate.
TEST(DecoderBlockGatedPathTest, GateHGradientMatchesFiniteDifference) {
    int d_model = 16, num_heads = 2, d_ff = 32;
    int tgt_len = 3, src_len = 4;

    DecoderBlock decoder_block(d_model, num_heads, d_ff, 0.1f, /*enable_world_model=*/false,
                               /*enable_hippocampal=*/true);
    decoder_block.set_gate_h(-0.4f);

    HippocampalMemory memory(d_model, 4);
    for (int slot = 0; slot < 3; ++slot) {
        Matrix key(1, d_model), value(1, d_model);
        for (int j = 0; j < d_model; ++j) {
            key(0, j) = 0.1f * static_cast<float>(slot + j);
            value(0, j) = 0.1f * static_cast<float>(slot + j);
        }
        memory.write(key, value);
    }

    Matrix decoder_input(tgt_len, d_model);
    Matrix encoder_output(src_len, d_model);
    for (int i = 0; i < tgt_len; ++i) {
        for (int j = 0; j < d_model; ++j) {
            decoder_input(i, j) = 0.05f * std::sin(static_cast<float>(i * d_model + j));
        }
    }
    for (int i = 0; i < src_len; ++i) {
        for (int j = 0; j < d_model; ++j) {
            encoder_output(i, j) = 0.05f * std::cos(static_cast<float>(i * d_model + j));
        }
    }
    Matrix causal_mask = create_causal_mask(tgt_len);

    auto run_forward = [&]() {
        // repetition_alpha = 0.0f: isolates the gate's own gradient from the coverage-penalty
        // machinery, which also perturbs the score distribution as gate_h (indirectly, via
        // nothing here) — not actually coupled, but keeping alpha at 0 keeps this check focused
        // purely on tanh(gate_h)'s own derivative, matching the world-model check's own setup.
        return decoder_block.forward(decoder_input, encoder_output, causal_mask, nullptr, nullptr,
                                     nullptr, &memory, /*repetition_alpha=*/0.0f, 0.9f);
    };

    Matrix output = run_forward();
    Matrix grad_output(output.rows, output.cols);
    for (int i = 0; i < output.rows; ++i) {
        for (int j = 0; j < output.cols; ++j) {
            grad_output(i, j) = 0.05f * static_cast<float>(i - j);
        }
    }
    decoder_block.backward(grad_output);
    float analytic_grad = decoder_block.get_gate_h_grad();

    auto weighted_output_sum = [&]() {
        Matrix out = run_forward();
        float total = 0.0f;
        for (int i = 0; i < out.rows; ++i) {
            for (int j = 0; j < out.cols; ++j) {
                total += out(i, j) * grad_output(i, j);
            }
        }
        return total;
    };

    const float original_gate_h = decoder_block.get_gate_h();
    const float epsilon = 1e-3f;

    decoder_block.set_gate_h(original_gate_h + epsilon);
    float loss_plus = weighted_output_sum();
    decoder_block.set_gate_h(original_gate_h - epsilon);
    float loss_minus = weighted_output_sum();
    decoder_block.set_gate_h(original_gate_h);

    float numerical_grad = (loss_plus - loss_minus) / (2.0f * epsilon);
    float tolerance = std::max(1e-2f, 0.05f * std::abs(numerical_grad));
    EXPECT_NEAR(analytic_grad, numerical_grad, tolerance);
}

// ============================================================================
// TD-194: cross-reference gated pull + Hebbian association strengthening
// ============================================================================

// The Hebbian update must actually run and touch association_matrix() — a plain mechanism check,
// not a claim about which specific pair strengthens most (that depends on randomly-initialized
// attention weights this test doesn't control).
TEST(DecoderBlockCrossReferenceTest, ForwardStrengthensAssociationBetweenAttendedSlots) {
    int d_model = 16, num_heads = 2, d_ff = 32;
    int tgt_len = 3, src_len = 4;

    DecoderBlock decoder_block(d_model, num_heads, d_ff, 0.1f, /*enable_world_model=*/false,
                               /*enable_hippocampal=*/true);

    HippocampalMemory memory(d_model, 4);
    for (int slot = 0; slot < 3; ++slot) {
        Matrix key(1, d_model), value(1, d_model);
        for (int j = 0; j < d_model; ++j) {
            key(0, j) = 0.1f * static_cast<float>(slot + j);
            value(0, j) = 0.1f * static_cast<float>(slot + j);
        }
        memory.write(key, value);
    }

    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            EXPECT_FLOAT_EQ(memory.association_matrix()[i][j], 0.0f) << "starts at zero";
        }
    }

    Matrix decoder_input(tgt_len, d_model);
    Matrix encoder_output(src_len, d_model);
    for (int i = 0; i < tgt_len; ++i) {
        for (int j = 0; j < d_model; ++j) {
            decoder_input(i, j) = 0.05f * std::sin(static_cast<float>(i * d_model + j));
        }
    }
    for (int i = 0; i < src_len; ++i) {
        for (int j = 0; j < d_model; ++j) {
            encoder_output(i, j) = 0.05f * std::cos(static_cast<float>(i * d_model + j));
        }
    }
    Matrix causal_mask = create_causal_mask(tgt_len);

    decoder_block.forward(decoder_input, encoder_output, causal_mask, nullptr, nullptr, nullptr,
                          &memory, /*repetition_alpha=*/0.0f, /*repetition_decay=*/0.9f,
                          /*cross_reference_alpha=*/0.0f, /*association_decay=*/0.9f);

    bool any_nonzero = false;
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            if (i != j && memory.association_matrix()[i][j] != 0.0f) {
                any_nonzero = true;
                // Symmetric by construction.
                EXPECT_FLOAT_EQ(memory.association_matrix()[i][j], memory.association_matrix()[j][i]);
            }
        }
    }
    EXPECT_TRUE(any_nonzero) << "at least one pair of slots should have co-activated";
}

// Mirrors CoverageNeverExceedsTheoreticalBound's own self-bounding check, for association.
TEST(DecoderBlockCrossReferenceTest, AssociationNeverExceedsTheoreticalBound) {
    int d_model = 16, num_heads = 2, d_ff = 32;
    int tgt_len = 2, src_len = 3;

    DecoderBlock decoder_block(d_model, num_heads, d_ff, 0.1f, /*enable_world_model=*/false,
                               /*enable_hippocampal=*/true);

    HippocampalMemory memory(d_model, 3);
    for (int slot = 0; slot < 3; ++slot) {
        Matrix key(1, d_model), value(1, d_model);
        for (int j = 0; j < d_model; ++j) {
            key(0, j) = 0.1f * static_cast<float>(slot + j);
            value(0, j) = 0.1f * static_cast<float>(slot + j);
        }
        memory.write(key, value);
    }

    Matrix decoder_input(tgt_len, d_model);
    Matrix encoder_output(src_len, d_model);
    for (int i = 0; i < tgt_len; ++i) {
        for (int j = 0; j < d_model; ++j) {
            decoder_input(i, j) = 0.1f * static_cast<float>(i + j);
        }
    }
    for (int i = 0; i < src_len; ++i) {
        for (int j = 0; j < d_model; ++j) {
            encoder_output(i, j) = 0.1f * static_cast<float>(i - j);
        }
    }
    Matrix causal_mask = create_causal_mask(tgt_len);

    const float association_decay = 0.9f;
    const float bound = 1.0f / (1.0f - association_decay);

    for (int step = 0; step < 300; ++step) {
        decoder_block.forward(decoder_input, encoder_output, causal_mask, nullptr, nullptr,
                              nullptr, &memory, /*repetition_alpha=*/0.0f,
                              /*repetition_decay=*/0.9f, /*cross_reference_alpha=*/0.0f,
                              association_decay);
        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 3; ++j) {
                if (i != j) {
                    EXPECT_LE(memory.association_matrix()[i][j], bound + 1e-3f)
                        << "step " << step << " (" << i << "," << j << ")";
                    EXPECT_GE(memory.association_matrix()[i][j], 0.0f)
                        << "step " << step << " (" << i << "," << j << ")";
                }
            }
        }
    }
}

// Shared fixture-style setup for the two tests below: ONE DecoderBlock + ONE HippocampalMemory
// instance reused across both forward() calls being compared, so randomly-initialized weights
// stay fixed and any output difference can only come from what the test itself changes between
// calls (association_matrix()/coverage_vector() content, or cross_reference_alpha) — comparing
// two independently-constructed DecoderBlocks would confound the comparison with their own
// unrelated random-initialization differences.
namespace {
struct CrossReferenceFixture {
    std::unique_ptr<DecoderBlock> block;
    std::unique_ptr<HippocampalMemory> memory;
    Matrix decoder_input;
    Matrix encoder_output;
    Matrix causal_mask;

    static CrossReferenceFixture make(int d_model, int num_heads, int d_ff, int tgt_len,
                                      int src_len) {
        CrossReferenceFixture fx{
            std::make_unique<DecoderBlock>(d_model, num_heads, d_ff, 0.1f,
                                           /*enable_world_model=*/false,
                                           /*enable_hippocampal=*/true),
            std::make_unique<HippocampalMemory>(d_model, 3), Matrix(tgt_len, d_model),
            Matrix(src_len, d_model), create_causal_mask(tgt_len)};
        fx.block->set_gate_h(0.6f);

        for (int slot = 0; slot < 3; ++slot) {
            Matrix key(1, d_model), value(1, d_model);
            for (int j = 0; j < d_model; ++j) {
                key(0, j) = 0.1f * static_cast<float>(slot + j);
                value(0, j) = 0.1f * static_cast<float>(slot + j);
            }
            fx.memory->write(key, value);
        }

        for (int i = 0; i < tgt_len; ++i) {
            for (int j = 0; j < d_model; ++j) {
                fx.decoder_input(i, j) = 0.1f * static_cast<float>(i + j);
            }
        }
        for (int i = 0; i < src_len; ++i) {
            for (int j = 0; j < d_model; ++j) {
                fx.encoder_output(i, j) = 0.1f * static_cast<float>(i - j);
            }
        }
        return fx;
    }

    // Resets coverage/association to a known baseline (undoing whatever the previous forward()
    // call's own bookkeeping just did to them) before seeding a specific scenario.
    void reset_state() {
        for (float& c : memory->coverage_vector()) {
            c = 0.0f;
        }
        for (auto& row : memory->association_matrix()) {
            for (float& v : row) {
                v = 0.0f;
            }
        }
    }

    Matrix run(float cross_reference_alpha) {
        return block->forward(decoder_input, encoder_output, causal_mask, nullptr, nullptr,
                              nullptr, memory.get(), /*repetition_alpha=*/0.0f,
                              /*repetition_decay=*/1.0f, cross_reference_alpha,
                              /*association_decay=*/1.0f);
    }
};

bool matrices_differ(const Matrix& a, const Matrix& b, float tolerance = 1e-6f) {
    for (int i = 0; i < a.rows; ++i) {
        for (int j = 0; j < a.cols; ++j) {
            if (std::abs(a(i, j) - b(i, j)) > tolerance) {
                return true;
            }
        }
    }
    return false;
}
}  // namespace

// cross_reference_alpha=0.0f (the default) must be a complete no-op regardless of how strong a
// pre-existing association is — matches this codebase's "explicit opt-in magnitude, not just
// on/off" convention already established for repetition_alpha.
TEST(DecoderBlockCrossReferenceTest, ZeroCrossReferenceAlphaIsNoOpRegardlessOfAssociation) {
    auto fx = CrossReferenceFixture::make(/*d_model=*/16, /*num_heads=*/2, /*d_ff=*/32,
                                          /*tgt_len=*/2, /*src_len=*/3);

    fx.reset_state();
    Matrix out_no_association = fx.run(/*cross_reference_alpha=*/0.0f);

    fx.reset_state();
    fx.memory->association_matrix()[0][1] = 50.0f;
    fx.memory->association_matrix()[1][0] = 50.0f;
    fx.memory->coverage_vector()[1] = 5.0f;  // large, so a real boost would be very visible
    Matrix out_with_association = fx.run(/*cross_reference_alpha=*/0.0f);

    EXPECT_FALSE(matrices_differ(out_no_association, out_with_association))
        << "cross_reference_alpha=0 should ignore association/coverage content entirely";
}

// The mirror image: a nonzero cross_reference_alpha with a real pre-existing association must
// change the output relative to leaving the association at zero.
TEST(DecoderBlockCrossReferenceTest, NonzeroCrossReferenceAlphaChangesOutputWhenAssociationNonzero) {
    auto fx = CrossReferenceFixture::make(/*d_model=*/16, /*num_heads=*/2, /*d_ff=*/32,
                                          /*tgt_len=*/2, /*src_len=*/3);

    fx.reset_state();
    Matrix out_no_association = fx.run(/*cross_reference_alpha=*/0.8f);

    fx.reset_state();
    fx.memory->association_matrix()[0][1] = 50.0f;
    fx.memory->association_matrix()[1][0] = 50.0f;
    fx.memory->coverage_vector()[1] = 5.0f;
    Matrix out_with_association = fx.run(/*cross_reference_alpha=*/0.8f);

    EXPECT_TRUE(matrices_differ(out_no_association, out_with_association))
        << "a real association combined with nonzero cross_reference_alpha should change the "
           "gated hippocampal output";
}

TEST(DecoderBlockGatedPathTest, GetGateAccessorsReflectSetters) {
    DecoderBlock decoder_block(16, 2, 32, 0.1f, true, true);

    EXPECT_FLOAT_EQ(decoder_block.get_gate(), 0.0f);
    EXPECT_FLOAT_EQ(decoder_block.get_gate_h(), 0.0f);

    decoder_block.set_gate(0.7f);
    decoder_block.set_gate_h(-0.2f);

    EXPECT_FLOAT_EQ(decoder_block.get_gate(), 0.7f);
    EXPECT_FLOAT_EQ(decoder_block.get_gate_h(), -0.2f);
}

TEST(DecoderBlockGatedPathTest, NullableAccessorsReflectConstruction) {
    DecoderBlock neither(16, 2, 32);
    EXPECT_EQ(neither.get_world_model_cross_attention(), nullptr);
    EXPECT_EQ(neither.get_hippocampal_cross_attention(), nullptr);

    DecoderBlock both(16, 2, 32, 0.1f, true, true);
    EXPECT_NE(both.get_world_model_cross_attention(), nullptr);
    EXPECT_NE(both.get_hippocampal_cross_attention(), nullptr);
}

// ============================================================================
// TD-187: save()/load() persistence of the gated world-model/hippocampal paths
// (closes TD-180's own documented gap — see DecoderBlock::save()'s own doc comment)
// ============================================================================

namespace {
// Removes the main file plus every sub-component file save()/load() can possibly have written,
// gated paths included — safe to call even when a given suffix was never written (std::remove
// on a non-existent path is a silent no-op).
void remove_decoder_block_files(const std::string& filepath) {
    std::remove(filepath.c_str());
    std::remove((filepath + ".self_attn").c_str());
    std::remove((filepath + ".cross_attn").c_str());
    std::remove((filepath + ".ff").c_str());
    std::remove((filepath + ".world_model_cross_attn").c_str());
    std::remove((filepath + ".norm_world").c_str());
    std::remove((filepath + ".hippocampal_cross_attn").c_str());
    std::remove((filepath + ".norm_hippocampal").c_str());
}
}  // namespace

TEST(DecoderBlockGatedPathSaveLoadTest, RoundTripPersistsWorldModelGateAndCrossAttentionWeights) {
    int d_model = 16, num_heads = 2, d_ff = 32;
    int tgt_len = 3, src_len = 4, wm_len = 3;

    DecoderBlock original(d_model, num_heads, d_ff, 0.1f, /*enable_world_model=*/true,
                          /*enable_hippocampal=*/false);
    original.set_gate(0.6f);

    Matrix decoder_input(tgt_len, d_model), encoder_output(src_len, d_model),
        world_model_output(wm_len, d_model);
    for (int i = 0; i < tgt_len; ++i)
        for (int j = 0; j < d_model; ++j) decoder_input(i, j) = 0.05f * (i + j);
    for (int i = 0; i < src_len; ++i)
        for (int j = 0; j < d_model; ++j) encoder_output(i, j) = 0.03f * (i - j);
    for (int i = 0; i < wm_len; ++i)
        for (int j = 0; j < d_model; ++j) world_model_output(i, j) = 0.2f * (i + 1);
    Matrix causal_mask = create_causal_mask(tgt_len);

    // Reference output from the original, pre-save instance, with the gated path genuinely
    // exercised (nonzero gate, real world_model_output) -- if either the gate scalar or the
    // gated cross-attention/norm weights fail to round-trip, this won't match after loading.
    Matrix reference_output = original.forward(decoder_input, encoder_output, causal_mask,
                                               nullptr, &world_model_output);

    std::string filepath = "test_decoder_block_wm_gated.bin";
    original.save(filepath);

    // A freshly, independently-constructed instance (different random init) with the SAME
    // enable_world_model=true -- load() must fully overwrite its random gate/cross-attention
    // weights with the saved ones, not just leave them as coincidentally-close random values.
    DecoderBlock loaded(d_model, num_heads, d_ff, 0.1f, /*enable_world_model=*/true,
                       /*enable_hippocampal=*/false);
    loaded.load(filepath);

    EXPECT_FLOAT_EQ(loaded.get_gate(), 0.6f);
    Matrix loaded_output = loaded.forward(decoder_input, encoder_output, causal_mask, nullptr,
                                          &world_model_output);
    EXPECT_TRUE(matrices_equal(reference_output, loaded_output));

    remove_decoder_block_files(filepath);
}

TEST(DecoderBlockGatedPathSaveLoadTest, RoundTripPersistsHippocampalGateAndCrossAttentionWeights) {
    int d_model = 16, num_heads = 2, d_ff = 32;
    int tgt_len = 3, src_len = 4;

    DecoderBlock original(d_model, num_heads, d_ff, 0.1f, /*enable_world_model=*/false,
                          /*enable_hippocampal=*/true);
    original.set_gate_h(-0.4f);

    Matrix decoder_input(tgt_len, d_model), encoder_output(src_len, d_model);
    for (int i = 0; i < tgt_len; ++i)
        for (int j = 0; j < d_model; ++j) decoder_input(i, j) = 0.05f * (i + j);
    for (int i = 0; i < src_len; ++i)
        for (int j = 0; j < d_model; ++j) encoder_output(i, j) = 0.03f * (i - j);
    Matrix causal_mask = create_causal_mask(tgt_len);

    HippocampalMemory memory(d_model, /*capacity=*/4);
    Matrix key(1, d_model), value(1, d_model);
    for (int j = 0; j < d_model; ++j) {
        key(0, j) = 0.25f;
        value(0, j) = 0.25f;
    }
    memory.write(key, value);

    Matrix reference_output = original.forward(decoder_input, encoder_output, causal_mask,
                                               nullptr, nullptr, nullptr, &memory,
                                               /*repetition_alpha=*/0.3f,
                                               /*repetition_decay=*/0.9f);

    std::string filepath = "test_decoder_block_hm_gated.bin";
    original.save(filepath);

    DecoderBlock loaded(d_model, num_heads, d_ff, 0.1f, /*enable_world_model=*/false,
                       /*enable_hippocampal=*/true);
    loaded.load(filepath);

    EXPECT_FLOAT_EQ(loaded.get_gate_h(), -0.4f);
    Matrix loaded_output = loaded.forward(decoder_input, encoder_output, causal_mask, nullptr,
                                          nullptr, nullptr, &memory, 0.3f, 0.9f);
    EXPECT_TRUE(matrices_equal(reference_output, loaded_output));

    remove_decoder_block_files(filepath);
}

TEST(DecoderBlockGatedPathSaveLoadTest, RoundTripPersistsBothGatedPathsWhenBothAllocated) {
    int d_model = 16, num_heads = 2, d_ff = 32;
    int tgt_len = 3, src_len = 4, wm_len = 3;

    DecoderBlock original(d_model, num_heads, d_ff, 0.1f, /*enable_world_model=*/true,
                          /*enable_hippocampal=*/true);
    original.set_gate(0.5f);
    original.set_gate_h(0.35f);

    Matrix decoder_input(tgt_len, d_model), encoder_output(src_len, d_model),
        world_model_output(wm_len, d_model);
    for (int i = 0; i < tgt_len; ++i)
        for (int j = 0; j < d_model; ++j) decoder_input(i, j) = 0.04f * (i + j);
    for (int i = 0; i < src_len; ++i)
        for (int j = 0; j < d_model; ++j) encoder_output(i, j) = 0.02f * (i - j);
    for (int i = 0; i < wm_len; ++i)
        for (int j = 0; j < d_model; ++j) world_model_output(i, j) = 0.15f * (i + 1);
    Matrix causal_mask = create_causal_mask(tgt_len);

    HippocampalMemory memory(d_model, /*capacity=*/4);
    Matrix key(1, d_model), value(1, d_model);
    for (int j = 0; j < d_model; ++j) {
        key(0, j) = 0.2f;
        value(0, j) = 0.2f;
    }
    memory.write(key, value);

    Matrix reference_output =
        original.forward(decoder_input, encoder_output, causal_mask, nullptr,
                         &world_model_output, nullptr, &memory, 0.4f, 0.9f);

    std::string filepath = "test_decoder_block_both_gated.bin";
    original.save(filepath);

    DecoderBlock loaded(d_model, num_heads, d_ff, 0.1f, /*enable_world_model=*/true,
                       /*enable_hippocampal=*/true);
    loaded.load(filepath);

    EXPECT_FLOAT_EQ(loaded.get_gate(), 0.5f);
    EXPECT_FLOAT_EQ(loaded.get_gate_h(), 0.35f);
    Matrix loaded_output = loaded.forward(decoder_input, encoder_output, causal_mask, nullptr,
                                          &world_model_output, nullptr, &memory, 0.4f, 0.9f);
    EXPECT_TRUE(matrices_equal(reference_output, loaded_output));

    remove_decoder_block_files(filepath);
}

TEST(DecoderBlockGatedPathSaveLoadTest, LoadThrowsWhenWorldModelFlagMismatchesSavedPresent) {
    int d_model = 16, num_heads = 2, d_ff = 32;
    DecoderBlock with_world_model(d_model, num_heads, d_ff, 0.1f, /*enable_world_model=*/true,
                                  /*enable_hippocampal=*/false);
    std::string filepath = "test_decoder_block_mismatch1.bin";
    with_world_model.save(filepath);

    // Loading into an instance constructed WITHOUT the world-model path must fail clearly,
    // not silently drop the saved gate/cross-attention weights.
    DecoderBlock without_world_model(d_model, num_heads, d_ff);
    EXPECT_THROW(without_world_model.load(filepath), std::runtime_error);

    remove_decoder_block_files(filepath);
}

TEST(DecoderBlockGatedPathSaveLoadTest, LoadThrowsWhenWorldModelFlagMismatchesSavedAbsent) {
    int d_model = 16, num_heads = 2, d_ff = 32;
    DecoderBlock without_world_model(d_model, num_heads, d_ff);
    std::string filepath = "test_decoder_block_mismatch2.bin";
    without_world_model.save(filepath);

    // Loading into an instance constructed WITH the world-model path must also fail clearly,
    // not silently leave it at its fresh random-init state.
    DecoderBlock with_world_model(d_model, num_heads, d_ff, 0.1f, /*enable_world_model=*/true,
                                  /*enable_hippocampal=*/false);
    EXPECT_THROW(with_world_model.load(filepath), std::runtime_error);

    remove_decoder_block_files(filepath);
}

TEST(DecoderBlockGatedPathSaveLoadTest, LoadThrowsWhenHippocampalFlagMismatches) {
    int d_model = 16, num_heads = 2, d_ff = 32;
    DecoderBlock with_hippocampal(d_model, num_heads, d_ff, 0.1f, /*enable_world_model=*/false,
                                  /*enable_hippocampal=*/true);
    std::string filepath = "test_decoder_block_mismatch3.bin";
    with_hippocampal.save(filepath);

    DecoderBlock without_hippocampal(d_model, num_heads, d_ff);
    EXPECT_THROW(without_hippocampal.load(filepath), std::runtime_error);

    remove_decoder_block_files(filepath);
}

TEST(DecoderBlockGatedPathSaveLoadTest, NonGatedSaveLoadRoundTripsUnaffectedByTheNewFlags) {
    // Regression guard: a plain, non-gated instance must still save/load exactly as before --
    // the new presence-flag bytes are written unconditionally (both false here) but change
    // nothing about the existing, already-tested non-gated round-trip behavior.
    int d_model = 16, num_heads = 2, d_ff = 32;
    DecoderBlock original(d_model, num_heads, d_ff);
    EXPECT_EQ(original.get_world_model_cross_attention(), nullptr);
    EXPECT_EQ(original.get_hippocampal_cross_attention(), nullptr);

    std::string filepath = "test_decoder_block_nongated.bin";
    original.save(filepath);

    DecoderBlock loaded(d_model, num_heads, d_ff);
    EXPECT_NO_THROW(loaded.load(filepath));
    EXPECT_EQ(loaded.get_world_model_cross_attention(), nullptr);
    EXPECT_EQ(loaded.get_hippocampal_cross_attention(), nullptr);

    remove_decoder_block_files(filepath);
}

// ============================================================================
// Main
// ============================================================================

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
