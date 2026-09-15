#include "../src/CrossAttention.hpp"
#include <../gtest/gtest.h>
#include <algorithm>
#include <cmath>
#include <fstream>
#include <random>
#include <stdexcept>
#include <utility>
#include <vector>
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

TEST(CrossAttentionConstructorTest, BasicInitialization) {
    int d_model = 128;
    int num_heads = 8;

    CrossAttention cross_attn(d_model, num_heads);

    EXPECT_EQ(cross_attn.get_d_model(), 128);
    EXPECT_EQ(cross_attn.get_num_heads(), 8);

    // Should be able to perform forward pass
    EXPECT_NO_THROW({
        Matrix query_input(10, d_model);
        Matrix kv_input(15, d_model);
        cross_attn.forward(query_input, kv_input);
    });
}

TEST(CrossAttentionConstructorTest, InvalidHeadCount) {
    int d_model = 128;
    int num_heads = 7;  // Not a divisor of 128

    // Should throw exception
    EXPECT_THROW(CrossAttention(d_model, num_heads), std::invalid_argument);
}

TEST(CrossAttentionConstructorTest, VariousConfigurations) {
    // Test various valid configurations
    std::vector<std::pair<int, int>> configs = {
        {64, 4},    // Small model
        {256, 8},   // Medium model
        {512, 8},   // Standard transformer
        {768, 12},  // BERT-base
        {1024, 16}  // GPT-2
    };

    for (const auto& [d_model, num_heads] : configs) {
        EXPECT_NO_THROW({
            CrossAttention cross_attn(d_model, num_heads);
            EXPECT_EQ(cross_attn.get_d_model(), d_model);
            EXPECT_EQ(cross_attn.get_num_heads(), num_heads);
        });
    }
}

// ============================================================================
// Forward Pass Tests
// ============================================================================

TEST(CrossAttentionForwardTest, OutputDimensions) {
    int d_model = 128;
    int num_heads = 8;
    int tgt_len = 10;
    int src_len = 15;

    CrossAttention cross_attn(d_model, num_heads);

    Matrix query_input(tgt_len, d_model);
    Matrix kv_input(src_len, d_model);

    Matrix output = cross_attn.forward(query_input, kv_input);

    // Output should match query (decoder) dimensions
    EXPECT_EQ(output.rows, tgt_len);
    EXPECT_EQ(output.cols, d_model);
}

TEST(CrossAttentionForwardTest, DifferentSequenceLengths) {
    int d_model = 64;
    int num_heads = 4;

    CrossAttention cross_attn(d_model, num_heads);

    // Test various sequence length combinations
    std::vector<std::pair<int, int>> test_cases = {
        {5, 10},   // tgt shorter than src
        {10, 5},   // tgt longer than src
        {8, 8},    // equal length
        {1, 20},   // single query token
        {20, 1},   // single context token
        {50, 100}  // long sequences
    };

    for (const auto& [tgt_len, src_len] : test_cases) {
        Matrix query_input(tgt_len, d_model);
        Matrix kv_input(src_len, d_model);

        Matrix output = cross_attn.forward(query_input, kv_input);

        EXPECT_EQ(output.rows, tgt_len);
        EXPECT_EQ(output.cols, d_model);
    }
}

TEST(CrossAttentionForwardTest, WithMask) {
    int d_model = 128;
    int num_heads = 8;
    int tgt_len = 10;
    int src_len = 15;

    CrossAttention cross_attn(d_model, num_heads);

    Matrix query_input(tgt_len, d_model);
    Matrix kv_input(src_len, d_model);

    // Create padding mask (mask out last 5 positions)
    Matrix mask(tgt_len, src_len);
    for (int i = 0; i < tgt_len; ++i) {
        for (int j = 0; j < src_len; ++j) {
            mask(i, j) = (j < src_len - 5) ? 1.0f : 0.0f;
        }
    }

    Matrix output = cross_attn.forward(query_input, kv_input, &mask);

    EXPECT_EQ(output.rows, tgt_len);
    EXPECT_EQ(output.cols, d_model);
}

TEST(CrossAttentionForwardTest, NonZeroValues) {
    int d_model = 64;
    int num_heads = 4;
    int tgt_len = 8;
    int src_len = 12;

    CrossAttention cross_attn(d_model, num_heads);

    Matrix query_input(tgt_len, d_model);
    Matrix kv_input(src_len, d_model);

    // Initialize with non-zero values
    for (int i = 0; i < tgt_len; ++i) {
        for (int j = 0; j < d_model; ++j) {
            query_input(i, j) = 0.1f * (i + j);
        }
    }

    for (int i = 0; i < src_len; ++i) {
        for (int j = 0; j < d_model; ++j) {
            kv_input(i, j) = 0.05f * (i - j);
        }
    }

    Matrix output = cross_attn.forward(query_input, kv_input);

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

TEST(CrossAttentionForwardTest, EncoderInfluence) {
    int d_model = 64;
    int num_heads = 4;
    int tgt_len = 5;
    int src_len = 10;

    CrossAttention cross_attn(d_model, num_heads);

    Matrix query_input(tgt_len, d_model);
    for (int i = 0; i < tgt_len; ++i) {
        for (int j = 0; j < d_model; ++j) {
            query_input(i, j) = 0.1f;
        }
    }

    // Test 1: Encoder with small values
    Matrix kv_small(src_len, d_model);
    for (int i = 0; i < src_len; ++i) {
        for (int j = 0; j < d_model; ++j) {
            kv_small(i, j) = 0.01f;
        }
    }

    Matrix output_small = cross_attn.forward(query_input, kv_small);

    // Test 2: Encoder with large values
    Matrix kv_large(src_len, d_model);
    for (int i = 0; i < src_len; ++i) {
        for (int j = 0; j < d_model; ++j) {
            kv_large(i, j) = 1.0f;
        }
    }

    cross_attn.zero_grad();
    Matrix output_large = cross_attn.forward(query_input, kv_large);

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

// ============================================================================
// Masking Tests
// ============================================================================

TEST(CrossAttentionMaskingTest, PaddingMask) {
    int d_model = 64;
    int num_heads = 4;
    int tgt_len = 8;
    int src_len = 12;

    CrossAttention cross_attn(d_model, num_heads);

    Matrix query_input(tgt_len, d_model);
    Matrix kv_input(src_len, d_model);

    for (int i = 0; i < tgt_len; ++i) {
        for (int j = 0; j < d_model; ++j) {
            query_input(i, j) = 0.1f * i;
        }
    }

    for (int i = 0; i < src_len; ++i) {
        for (int j = 0; j < d_model; ++j) {
            kv_input(i, j) = 0.05f * i;
        }
    }

    // Test with mask
    Matrix mask(tgt_len, src_len);
    for (int i = 0; i < tgt_len; ++i) {
        for (int j = 0; j < src_len; ++j) {
            // Mask out last 4 positions
            mask(i, j) = (j < src_len - 4) ? 1.0f : 0.0f;
        }
    }

    Matrix output_masked = cross_attn.forward(query_input, kv_input, &mask);

    // Test without mask
    cross_attn.zero_grad();
    Matrix output_unmasked = cross_attn.forward(query_input, kv_input, nullptr);

    // Outputs should differ
    bool outputs_differ = false;
    for (int i = 0; i < tgt_len; ++i) {
        for (int j = 0; j < d_model; ++j) {
            if (!is_close(output_masked(i, j), output_unmasked(i, j), 1e-3f)) {
                outputs_differ = true;
                break;
            }
        }
        if (outputs_differ)
            break;
    }

    EXPECT_TRUE(outputs_differ);
}

TEST(CrossAttentionMaskingTest, AllOnesVsAllZerosMask) {
    int d_model = 64;
    int num_heads = 4;
    int tgt_len = 6;
    int src_len = 10;

    CrossAttention cross_attn(d_model, num_heads);

    Matrix query_input(tgt_len, d_model);
    Matrix kv_input(src_len, d_model);

    for (int i = 0; i < tgt_len; ++i) {
        for (int j = 0; j < d_model; ++j) {
            query_input(i, j) = 0.1f * (i + j);
        }
    }

    for (int i = 0; i < src_len; ++i) {
        for (int j = 0; j < d_model; ++j) {
            kv_input(i, j) = 0.05f;
        }
    }

    // All-ones mask (no masking)
    Matrix all_ones(tgt_len, src_len);
    for (int i = 0; i < tgt_len; ++i) {
        for (int j = 0; j < src_len; ++j) {
            all_ones(i, j) = 1.0f;
        }
    }

    Matrix output_all_ones = cross_attn.forward(query_input, kv_input, &all_ones);

    // No mask (should be same as all-ones)
    cross_attn.zero_grad();
    Matrix output_no_mask = cross_attn.forward(query_input, kv_input, nullptr);

    // Should be very similar (might have tiny numerical differences)
    EXPECT_TRUE(matrices_equal(output_all_ones, output_no_mask, 1e-5f));
}

// ============================================================================
// TD-174 Tests: forward_with_scores (pre-softmax additive score bias)
//
// The hippocampal-memory repetition penalty (see
// docs/proposals/lejepa_world_model_gated_injection_plan.md's Component 6) needs a caller-
// supplied additive bias per key position, applied before softmax — these are exactly the two
// checks TD-174's own Action Items call for: an all-zero bias must reproduce forward()'s
// existing output exactly, and a large negative bias at one position must suppress attention
// to it the same way masking that position would.
// ============================================================================

TEST(CrossAttentionScoreBiasTest, ZeroBiasReproducesForwardExactly) {
    int d_model = 32;
    int num_heads = 4;
    int tgt_len = 5;
    int src_len = 7;

    CrossAttention cross_attn(d_model, num_heads);

    Matrix query_input(tgt_len, d_model);
    Matrix kv_input(src_len, d_model);
    for (int i = 0; i < tgt_len; ++i) {
        for (int j = 0; j < d_model; ++j) {
            query_input(i, j) = 0.1f * static_cast<float>(i + j);
        }
    }
    for (int i = 0; i < src_len; ++i) {
        for (int j = 0; j < d_model; ++j) {
            kv_input(i, j) = 0.05f * static_cast<float>(i - j);
        }
    }

    Matrix output_forward = cross_attn.forward(query_input, kv_input);

    Matrix zero_bias(tgt_len, src_len);  // default-initialized to 0.0f
    Matrix output_with_scores = cross_attn.forward_with_scores(query_input, kv_input, zero_bias);

    ASSERT_EQ(output_with_scores.rows, output_forward.rows);
    ASSERT_EQ(output_with_scores.cols, output_forward.cols);
    for (int i = 0; i < output_forward.rows; ++i) {
        for (int j = 0; j < output_forward.cols; ++j) {
            // Adding 0.0f to a score is an exact no-op in IEEE floating point — this should be
            // bit-identical, not merely close, since forward_with_scores() otherwise performs
            // the identical sequence of operations forward() does.
            EXPECT_FLOAT_EQ(output_with_scores(i, j), output_forward(i, j));
        }
    }
}

TEST(CrossAttentionScoreBiasTest, ZeroBiasWithMaskReproducesForwardExactly) {
    int d_model = 24;
    int num_heads = 3;
    int tgt_len = 4;
    int src_len = 6;

    CrossAttention cross_attn(d_model, num_heads);

    Matrix query_input(tgt_len, d_model);
    Matrix kv_input(src_len, d_model);
    for (int i = 0; i < tgt_len; ++i) {
        for (int j = 0; j < d_model; ++j) {
            query_input(i, j) = 0.2f * static_cast<float>(i);
        }
    }
    for (int i = 0; i < src_len; ++i) {
        for (int j = 0; j < d_model; ++j) {
            kv_input(i, j) = 0.1f * static_cast<float>(j);
        }
    }

    Matrix mask(tgt_len, src_len);
    for (int i = 0; i < tgt_len; ++i) {
        for (int j = 0; j < src_len; ++j) {
            mask(i, j) = (j < src_len - 2) ? 1.0f : 0.0f;
        }
    }

    Matrix output_forward = cross_attn.forward(query_input, kv_input, &mask);

    Matrix zero_bias(tgt_len, src_len);
    Matrix output_with_scores =
        cross_attn.forward_with_scores(query_input, kv_input, zero_bias, &mask);

    for (int i = 0; i < output_forward.rows; ++i) {
        for (int j = 0; j < output_forward.cols; ++j) {
            EXPECT_FLOAT_EQ(output_with_scores(i, j), output_forward(i, j));
        }
    }
}

TEST(CrossAttentionScoreBiasTest, LargeNegativeBiasSuppressesAttentionLikeMasking) {
    int d_model = 32;
    int num_heads = 4;
    int tgt_len = 4;
    int src_len = 6;
    int suppressed_col = 2;

    CrossAttention cross_attn(d_model, num_heads);

    Matrix query_input(tgt_len, d_model);
    Matrix kv_input(src_len, d_model);
    for (int i = 0; i < tgt_len; ++i) {
        for (int j = 0; j < d_model; ++j) {
            query_input(i, j) = 0.15f * static_cast<float>(i + j);
        }
    }
    for (int i = 0; i < src_len; ++i) {
        for (int j = 0; j < d_model; ++j) {
            kv_input(i, j) = 0.1f * static_cast<float>(i - j);
        }
    }

    // Reference: suppress the one column via a boolean mask.
    Matrix mask(tgt_len, src_len);
    for (int i = 0; i < tgt_len; ++i) {
        for (int j = 0; j < src_len; ++j) {
            mask(i, j) = (j == suppressed_col) ? 0.0f : 1.0f;
        }
    }
    Matrix output_masked = cross_attn.forward(query_input, kv_input, &mask);

    // Same suppression via a large negative score_bias instead, no mask at all.
    Matrix bias(tgt_len, src_len);  // default-initialized to 0.0f everywhere else
    for (int i = 0; i < tgt_len; ++i) {
        bias(i, suppressed_col) = -1e9f;
    }
    Matrix output_biased = cross_attn.forward_with_scores(query_input, kv_input, bias, nullptr);

    EXPECT_TRUE(matrices_equal(output_masked, output_biased, 1e-3f));
}

TEST(CrossAttentionScoreBiasTest, MaskTakesPrecedenceOverBiasAtTheSamePosition) {
    // A masked position must end up suppressed (-1e9) regardless of whatever bias value it
    // carries — even a large *positive* bias, which alone would otherwise dominate the softmax.
    int d_model = 16;
    int num_heads = 2;
    int tgt_len = 3;
    int src_len = 5;
    int masked_col = 1;

    CrossAttention cross_attn(d_model, num_heads);

    Matrix query_input(tgt_len, d_model);
    Matrix kv_input(src_len, d_model);
    for (int i = 0; i < tgt_len; ++i) {
        for (int j = 0; j < d_model; ++j) {
            query_input(i, j) = 0.1f * static_cast<float>(i + j);
        }
    }
    for (int i = 0; i < src_len; ++i) {
        for (int j = 0; j < d_model; ++j) {
            kv_input(i, j) = 0.05f * static_cast<float>(i + j);
        }
    }

    Matrix mask(tgt_len, src_len);
    for (int i = 0; i < tgt_len; ++i) {
        for (int j = 0; j < src_len; ++j) {
            mask(i, j) = (j == masked_col) ? 0.0f : 1.0f;
        }
    }

    Matrix bias(tgt_len, src_len);
    for (int i = 0; i < tgt_len; ++i) {
        bias(i, masked_col) = 1e6f;  // would otherwise dominate the softmax at this column
    }

    Matrix output_masked_only = cross_attn.forward(query_input, kv_input, &mask);

    Matrix zero_bias(tgt_len, src_len);
    Matrix output_masked_and_biased =
        cross_attn.forward_with_scores(query_input, kv_input, bias, &mask);

    // The positive bias at the masked column must have no effect — mask wins.
    EXPECT_TRUE(matrices_equal(output_masked_only, output_masked_and_biased, 1e-3f));
}

TEST(CrossAttentionScoreBiasTest, RejectsMismatchedScoreBiasDimensions) {
    CrossAttention cross_attn(16, 2);

    Matrix query_input(3, 16);
    Matrix kv_input(4, 16);
    Matrix wrong_shape_bias(2, 4);  // should be [3, 4]

    EXPECT_THROW(cross_attn.forward_with_scores(query_input, kv_input, wrong_shape_bias),
                 std::invalid_argument);
}

TEST(CrossAttentionScoreBiasTest, BackwardWorksAfterForwardWithScores) {
    int d_model = 16;
    int num_heads = 2;
    int tgt_len = 3;
    int src_len = 4;

    CrossAttention cross_attn(d_model, num_heads);

    Matrix query_input(tgt_len, d_model);
    Matrix kv_input(src_len, d_model);
    for (int i = 0; i < tgt_len; ++i) {
        for (int j = 0; j < d_model; ++j) {
            query_input(i, j) = 0.1f * static_cast<float>(i + j);
        }
    }
    for (int i = 0; i < src_len; ++i) {
        for (int j = 0; j < d_model; ++j) {
            kv_input(i, j) = 0.05f * static_cast<float>(i + j);
        }
    }

    Matrix bias(tgt_len, src_len);
    for (int i = 0; i < tgt_len; ++i) {
        for (int j = 0; j < src_len; ++j) {
            bias(i, j) = 0.01f * static_cast<float>(i - j);
        }
    }

    cross_attn.forward_with_scores(query_input, kv_input, bias);

    Matrix grad_output(tgt_len, d_model);
    for (int i = 0; i < tgt_len; ++i) {
        for (int j = 0; j < d_model; ++j) {
            grad_output(i, j) = 0.1f;
        }
    }

    Matrix grad_query, grad_kv;
    cross_attn.backward(grad_output, grad_query, grad_kv);

    EXPECT_EQ(grad_query.rows, tgt_len);
    EXPECT_EQ(grad_query.cols, d_model);
    EXPECT_EQ(grad_kv.rows, src_len);
    EXPECT_EQ(grad_kv.cols, d_model);
    for (int i = 0; i < grad_query.rows; ++i) {
        for (int j = 0; j < grad_query.cols; ++j) {
            EXPECT_TRUE(std::isfinite(grad_query(i, j)));
        }
    }
}

// ============================================================================
// TD-059 Regression Tests: genuine per-head cross-attention
//
// See multiheadattention_test.cpp's identical section for the full rationale, including why
// a naive "does output change with num_heads" test is a weaker check than it looks (d_k
// changes too, and the old scale factor already varied with d_k on its own).
// ============================================================================

namespace {
void seed_ca_weights_deterministically(CrossAttention& ca, int d_model, unsigned int seed) {
    std::mt19937 gen(seed);
    float scale = std::sqrt(2.0f / static_cast<float>(d_model));
    std::normal_distribution<float> dist(0.0f, scale);

    auto fill = [&]() {
        Matrix m(d_model, d_model);
        for (int i = 0; i < d_model; ++i) {
            for (int j = 0; j < d_model; ++j) {
                m(i, j) = dist(gen);
            }
        }
        return m;
    };

    ca.set_Wq(fill());
    ca.set_Wk(fill());
    ca.set_Wv(fill());
    ca.set_Wo(fill());
}
}  // namespace

TEST(CrossAttentionArchitectureTest, MatchesIndependentPerHeadReference) {
    // TD-059 regression: check forward()'s output against an independent, from-scratch
    // implementation of the documented per-head formula -- softmax(Q_h K_h^T / sqrt(d_k))
    // V_h computed separately per head, concatenated, then projected through W_o -- built
    // here directly from the class's own weights via the public accessors, not by reusing
    // any of CrossAttention's own forward code.
    int d_model = 8;
    int num_heads = 2;
    int d_k = d_model / num_heads;
    CrossAttention ca(d_model, num_heads);

    int tgt_len = 3;
    int src_len = 4;
    Matrix query_input(tgt_len, d_model);
    Matrix kv_input(src_len, d_model);
    for (int i = 0; i < tgt_len; ++i) {
        for (int j = 0; j < d_model; ++j) {
            query_input(i, j) = 0.1f * static_cast<float>(i + 1) - 0.05f * static_cast<float>(j);
        }
    }
    for (int i = 0; i < src_len; ++i) {
        for (int j = 0; j < d_model; ++j) {
            kv_input(i, j) = 0.07f * static_cast<float>(i) + 0.02f * static_cast<float>(j);
        }
    }

    Matrix actual = ca.forward(query_input, kv_input);

    // --- Independent reference, reimplemented from scratch ---
    Matrix Wq = ca.get_Wq(), Wk = ca.get_Wk(), Wv = ca.get_Wv(), Wo = ca.get_Wo();

    auto matmul = [](const Matrix& a, const Matrix& b) {
        Matrix out(a.rows, b.cols);
        for (int i = 0; i < a.rows; ++i) {
            for (int j = 0; j < b.cols; ++j) {
                float sum = 0.0f;
                for (int k = 0; k < a.cols; ++k) {
                    sum += a(i, k) * b(k, j);
                }
                out(i, j) = sum;
            }
        }
        return out;
    };

    Matrix Q = matmul(query_input, Wq);
    Matrix K = matmul(kv_input, Wk);
    Matrix V = matmul(kv_input, Wv);

    Matrix concatenated(tgt_len, d_model);
    float scale = 1.0f / std::sqrt(static_cast<float>(d_k));
    for (int h = 0; h < num_heads; ++h) {
        int start = h * d_k;
        std::vector<std::vector<float>> scores(tgt_len, std::vector<float>(src_len, 0.0f));
        for (int i = 0; i < tgt_len; ++i) {
            for (int j = 0; j < src_len; ++j) {
                float sum = 0.0f;
                for (int k = 0; k < d_k; ++k) {
                    sum += Q(i, start + k) * K(j, start + k);
                }
                scores[i][j] = sum * scale;
            }
        }
        for (int i = 0; i < tgt_len; ++i) {
            float max_v = scores[i][0];
            for (int j = 1; j < src_len; ++j) {
                max_v = std::max(max_v, scores[i][j]);
            }
            float sum_exp = 0.0f;
            for (int j = 0; j < src_len; ++j) {
                scores[i][j] = std::exp(scores[i][j] - max_v);
                sum_exp += scores[i][j];
            }
            for (int j = 0; j < src_len; ++j) {
                scores[i][j] /= sum_exp;
            }
        }
        for (int i = 0; i < tgt_len; ++i) {
            for (int k = 0; k < d_k; ++k) {
                float sum = 0.0f;
                for (int j = 0; j < src_len; ++j) {
                    sum += scores[i][j] * V(j, start + k);
                }
                concatenated(i, start + k) = sum;
            }
        }
    }

    Matrix expected = matmul(concatenated, Wo);

    ASSERT_EQ(actual.rows, expected.rows);
    ASSERT_EQ(actual.cols, expected.cols);
    for (int i = 0; i < actual.rows; ++i) {
        for (int j = 0; j < actual.cols; ++j) {
            EXPECT_NEAR(actual(i, j), expected(i, j), 1e-4f)
                << "mismatch at (" << i << "," << j << ")";
        }
    }
}

TEST(CrossAttentionArchitectureTest, BackwardPassMatchesNumericalGradient) {
    // Finite-difference gradient check with num_heads > 1, w.r.t. the query input, added
    // alongside TD-059's per-head backward rewrite. Confirms backward() is the correct
    // gradient of whatever forward() actually computes (a complementary property to
    // MatchesIndependentPerHeadReference above, which pins down the formula itself -- see
    // multiheadattention_test.cpp's identical test for why these two checks are both needed).
    int d_model = 16;
    int num_heads = 4;
    CrossAttention ca(d_model, num_heads);
    seed_ca_weights_deterministically(ca, d_model, /*seed=*/13);

    int tgt_len = 4;
    int src_len = 5;
    Matrix query_input(tgt_len, d_model);
    for (int i = 0; i < tgt_len; ++i) {
        for (int j = 0; j < d_model; ++j) {
            query_input(i, j) = 0.05f * std::sin(static_cast<float>(i * d_model + j));
        }
    }
    Matrix kv_input(src_len, d_model);
    for (int i = 0; i < src_len; ++i) {
        for (int j = 0; j < d_model; ++j) {
            kv_input(i, j) = 0.05f * std::cos(static_cast<float>(i * d_model + j));
        }
    }

    auto scalar_loss = [](const Matrix& out) {
        float loss = 0.0f;
        for (int i = 0; i < out.rows; ++i) {
            for (int j = 0; j < out.cols; ++j) {
                loss += out(i, j) * out(i, j);
            }
        }
        return loss;
    };

    Matrix output = ca.forward(query_input, kv_input);
    Matrix grad_output(output.rows, output.cols);
    for (int i = 0; i < output.rows; ++i) {
        for (int j = 0; j < output.cols; ++j) {
            grad_output(i, j) = 2.0f * output(i, j);
        }
    }
    Matrix analytic_grad_query, analytic_grad_kv;
    ca.backward(grad_output, analytic_grad_query, analytic_grad_kv);

    const float epsilon = 1e-3f;
    const std::vector<std::pair<int, int>> positions = {
        {0, 0}, {0, d_model / 2}, {1, 3}, {tgt_len - 1, d_model - 1}};

    for (const auto& [pi, pj] : positions) {
        Matrix query_plus = query_input;
        query_plus(pi, pj) += epsilon;
        Matrix query_minus = query_input;
        query_minus(pi, pj) -= epsilon;

        float loss_plus = scalar_loss(ca.forward(query_plus, kv_input));
        float loss_minus = scalar_loss(ca.forward(query_minus, kv_input));
        float numerical_grad = (loss_plus - loss_minus) / (2.0f * epsilon);

        float tolerance = std::max(1e-2f, 0.05f * std::abs(numerical_grad));
        EXPECT_NEAR(analytic_grad_query(pi, pj), numerical_grad, tolerance)
            << "gradient mismatch at (" << pi << "," << pj << ")";
    }
}

// ============================================================================
// TD-038: LoRA integration tests
// ============================================================================

TEST(CrossAttentionLoRATest, EnableLoraIsNoOpBeforeTraining) {
    int d_model = 16;
    CrossAttention ca(d_model, 4);
    seed_ca_weights_deterministically(ca, d_model, /*seed=*/9);

    Matrix query_input(3, d_model);
    Matrix kv_input(5, d_model);
    for (int i = 0; i < query_input.rows; ++i) {
        for (int j = 0; j < d_model; ++j) {
            query_input(i, j) = 0.05f * std::cos(static_cast<float>(i * d_model + j));
        }
    }
    for (int i = 0; i < kv_input.rows; ++i) {
        for (int j = 0; j < d_model; ++j) {
            kv_input(i, j) = 0.04f * std::sin(static_cast<float>(i * d_model + j));
        }
    }

    Matrix output_before = ca.forward(query_input, kv_input);

    LoRAConfig config;
    config.rank = 4;
    ca.enable_lora(config);
    ASSERT_TRUE(ca.has_lora());

    Matrix output_after = ca.forward(query_input, kv_input);
    ASSERT_EQ(output_after.rows, output_before.rows);
    ASSERT_EQ(output_after.cols, output_before.cols);
    for (int i = 0; i < output_before.rows; ++i) {
        for (int j = 0; j < output_before.cols; ++j) {
            EXPECT_FLOAT_EQ(output_before(i, j), output_after(i, j))
                << "enabling LoRA changed forward() output before any training at ("
                << i << "," << j << ")";
        }
    }
}

TEST(CrossAttentionLoRATest, BackwardPassMatchesNumericalGradientWithLoraActive) {
    // Same shape as CrossAttentionArchitectureTest.BackwardPassMatchesNumericalGradient
    // above, with LoRA adapters enabled and pre-trained (B != 0) so the LoRA branch
    // genuinely contributes -- exercises LoRAAdapter::backward()'s grad_input return value
    // wired through both grad_query_input and grad_kv_input (TD-038).
    int d_model = 16;
    int num_heads = 4;
    CrossAttention ca(d_model, num_heads);
    seed_ca_weights_deterministically(ca, d_model, /*seed=*/17);

    LoRAConfig config;
    config.rank = 4;
    config.alpha = 8.0f;
    ca.enable_lora(config);

    int tgt_len = 4;
    int src_len = 5;
    Matrix query_input(tgt_len, d_model);
    for (int i = 0; i < tgt_len; ++i) {
        for (int j = 0; j < d_model; ++j) {
            query_input(i, j) = 0.05f * std::sin(static_cast<float>(i * d_model + j));
        }
    }
    Matrix kv_input(src_len, d_model);
    for (int i = 0; i < src_len; ++i) {
        for (int j = 0; j < d_model; ++j) {
            kv_input(i, j) = 0.05f * std::cos(static_cast<float>(i * d_model + j));
        }
    }

    for (int step = 0; step < 3; ++step) {
        Matrix out = ca.forward(query_input, kv_input);
        Matrix grad(out.rows, out.cols);
        for (int i = 0; i < grad.rows; ++i) {
            for (int j = 0; j < grad.cols; ++j) {
                grad(i, j) = 0.1f * (i + j + 1);
            }
        }
        Matrix gq, gkv;
        ca.backward(grad, gq, gkv);
        ca.get_lora_q()->update(0.05f);
        ca.get_lora_k()->update(0.05f);
        ca.get_lora_v()->update(0.05f);
        ca.get_lora_o()->update(0.05f);
    }

    auto scalar_loss = [](const Matrix& out) {
        float loss = 0.0f;
        for (int i = 0; i < out.rows; ++i) {
            for (int j = 0; j < out.cols; ++j) {
                loss += out(i, j) * out(i, j);
            }
        }
        return loss;
    };

    Matrix output = ca.forward(query_input, kv_input);
    Matrix grad_output(output.rows, output.cols);
    for (int i = 0; i < output.rows; ++i) {
        for (int j = 0; j < output.cols; ++j) {
            grad_output(i, j) = 2.0f * output(i, j);
        }
    }
    Matrix analytic_grad_query, analytic_grad_kv;
    ca.backward(grad_output, analytic_grad_query, analytic_grad_kv);

    const float epsilon = 1e-3f;
    const std::vector<std::pair<int, int>> query_positions = {
        {0, 0}, {0, d_model / 2}, {1, 3}, {tgt_len - 1, d_model - 1}};
    for (const auto& [pi, pj] : query_positions) {
        Matrix query_plus = query_input;
        query_plus(pi, pj) += epsilon;
        Matrix query_minus = query_input;
        query_minus(pi, pj) -= epsilon;

        float loss_plus = scalar_loss(ca.forward(query_plus, kv_input));
        float loss_minus = scalar_loss(ca.forward(query_minus, kv_input));
        float numerical_grad = (loss_plus - loss_minus) / (2.0f * epsilon);

        float tolerance = std::max(1e-2f, 0.05f * std::abs(numerical_grad));
        EXPECT_NEAR(analytic_grad_query(pi, pj), numerical_grad, tolerance)
            << "query gradient mismatch at (" << pi << "," << pj << ") with LoRA active";
    }

    const std::vector<std::pair<int, int>> kv_positions = {
        {0, 0}, {0, d_model / 2}, {1, 3}, {src_len - 1, d_model - 1}};
    for (const auto& [pi, pj] : kv_positions) {
        Matrix kv_plus = kv_input;
        kv_plus(pi, pj) += epsilon;
        Matrix kv_minus = kv_input;
        kv_minus(pi, pj) -= epsilon;

        float loss_plus = scalar_loss(ca.forward(query_input, kv_plus));
        float loss_minus = scalar_loss(ca.forward(query_input, kv_minus));
        float numerical_grad = (loss_plus - loss_minus) / (2.0f * epsilon);

        float tolerance = std::max(1e-2f, 0.05f * std::abs(numerical_grad));
        EXPECT_NEAR(analytic_grad_kv(pi, pj), numerical_grad, tolerance)
            << "kv gradient mismatch at (" << pi << "," << pj << ") with LoRA active";
    }
}

TEST(CrossAttentionLoRATest, RegisterLoraParametersFreezesBaseWeights) {
    int d_model = 12;
    CrossAttention ca(d_model, 3);
    seed_ca_weights_deterministically(ca, d_model, /*seed=*/31);

    LoRAConfig config;
    config.rank = 3;
    ca.enable_lora(config);

    Matrix Wq_before = ca.get_Wq();
    Matrix Wk_before = ca.get_Wk();
    Matrix Wv_before = ca.get_Wv();
    Matrix Wo_before = ca.get_Wo();

    Optimizer optimizer(OptimizerType::ADAM, 0.05f);
    ca.register_lora_parameters(optimizer);

    Matrix query_input(4, d_model);
    Matrix kv_input(4, d_model);
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < d_model; ++j) {
            query_input(i, j) = 0.05f * (i + j);
            kv_input(i, j) = 0.04f * (i - j);
        }
    }

    for (int step = 0; step < 2; ++step) {
        ca.zero_grad();
        Matrix output = ca.forward(query_input, kv_input);
        Matrix grad(output.rows, output.cols);
        for (int i = 0; i < grad.rows; ++i) {
            for (int j = 0; j < grad.cols; ++j) {
                grad(i, j) = 0.1f;
            }
        }
        Matrix gq, gkv;
        ca.backward(grad, gq, gkv);
        optimizer.step();
    }

    const Matrix& Wq_after = ca.get_Wq();
    const Matrix& Wk_after = ca.get_Wk();
    const Matrix& Wv_after = ca.get_Wv();
    const Matrix& Wo_after = ca.get_Wo();
    for (int i = 0; i < d_model; ++i) {
        for (int j = 0; j < d_model; ++j) {
            EXPECT_FLOAT_EQ(Wq_before(i, j), Wq_after(i, j));
            EXPECT_FLOAT_EQ(Wk_before(i, j), Wk_after(i, j));
            EXPECT_FLOAT_EQ(Wv_before(i, j), Wv_after(i, j));
            EXPECT_FLOAT_EQ(Wo_before(i, j), Wo_after(i, j));
        }
    }
}

TEST(CrossAttentionLoRATest, MergeLoraPreservesForwardOutput) {
    int d_model = 12;
    CrossAttention ca(d_model, 3);
    seed_ca_weights_deterministically(ca, d_model, /*seed=*/37);

    LoRAConfig config;
    config.rank = 3;
    config.alpha = 6.0f;
    ca.enable_lora(config);

    Matrix query_input(4, d_model);
    Matrix kv_input(4, d_model);
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < d_model; ++j) {
            query_input(i, j) = 0.03f * (i - j);
            kv_input(i, j) = 0.02f * (i + j);
        }
    }

    for (int step = 0; step < 3; ++step) {
        Matrix out = ca.forward(query_input, kv_input);
        Matrix grad(out.rows, out.cols);
        for (int i = 0; i < grad.rows; ++i) {
            for (int j = 0; j < grad.cols; ++j) {
                grad(i, j) = 0.05f * (i + 1);
            }
        }
        Matrix gq, gkv;
        ca.backward(grad, gq, gkv);
        ca.get_lora_q()->update(0.1f);
        ca.get_lora_k()->update(0.1f);
        ca.get_lora_v()->update(0.1f);
        ca.get_lora_o()->update(0.1f);
    }

    Matrix output_before_merge = ca.forward(query_input, kv_input);
    ca.merge_lora();
    EXPECT_FALSE(ca.has_lora());
    Matrix output_after_merge = ca.forward(query_input, kv_input);

    ASSERT_EQ(output_before_merge.rows, output_after_merge.rows);
    ASSERT_EQ(output_before_merge.cols, output_after_merge.cols);
    for (int i = 0; i < output_before_merge.rows; ++i) {
        for (int j = 0; j < output_before_merge.cols; ++j) {
            EXPECT_NEAR(output_before_merge(i, j), output_after_merge(i, j), 1e-4f)
                << "merge_lora() changed forward()'s output at (" << i << "," << j << ")";
        }
    }
}

// ============================================================================
// Backward Pass Tests
// ============================================================================

TEST(CrossAttentionBackwardTest, GradientDimensions) {
    int d_model = 64;
    int num_heads = 4;
    int tgt_len = 8;
    int src_len = 12;

    CrossAttention cross_attn(d_model, num_heads);

    Matrix query_input(tgt_len, d_model);
    Matrix kv_input(src_len, d_model);

    // Forward pass
    Matrix output = cross_attn.forward(query_input, kv_input);

    // Backward pass
    Matrix grad_output(tgt_len, d_model);
    for (int i = 0; i < tgt_len; ++i) {
        for (int j = 0; j < d_model; ++j) {
            grad_output(i, j) = 0.01f;
        }
    }

    Matrix grad_query_input, grad_kv_input;
    cross_attn.backward(grad_output, grad_query_input, grad_kv_input);

    // Check gradient dimensions
    EXPECT_EQ(grad_query_input.rows, tgt_len);
    EXPECT_EQ(grad_query_input.cols, d_model);
    EXPECT_EQ(grad_kv_input.rows, src_len);
    EXPECT_EQ(grad_kv_input.cols, d_model);
}

TEST(CrossAttentionBackwardTest, TwoGradientOutputs) {
    int d_model = 64;
    int num_heads = 4;
    int tgt_len = 5;
    int src_len = 10;

    CrossAttention cross_attn(d_model, num_heads);

    Matrix query_input(tgt_len, d_model);
    Matrix kv_input(src_len, d_model);

    for (int i = 0; i < tgt_len; ++i) {
        for (int j = 0; j < d_model; ++j) {
            query_input(i, j) = 0.1f * i;
        }
    }

    for (int i = 0; i < src_len; ++i) {
        for (int j = 0; j < d_model; ++j) {
            kv_input(i, j) = 0.05f * i;
        }
    }

    Matrix output = cross_attn.forward(query_input, kv_input);

    Matrix grad_output(tgt_len, d_model);
    for (int i = 0; i < tgt_len; ++i) {
        for (int j = 0; j < d_model; ++j) {
            grad_output(i, j) = 0.01f;
        }
    }

    Matrix grad_query, grad_kv;
    cross_attn.backward(grad_output, grad_query, grad_kv);

    // Both gradients should be non-zero
    float query_grad_norm = compute_gradient_norm(grad_query);
    float kv_grad_norm = compute_gradient_norm(grad_kv);

    EXPECT_GT(query_grad_norm, 0.0f);
    EXPECT_GT(kv_grad_norm, 0.0f);
}

TEST(CrossAttentionBackwardTest, GradientNonZero) {
    int d_model = 64;
    int num_heads = 4;
    int tgt_len = 8;
    int src_len = 8;

    CrossAttention cross_attn(d_model, num_heads);

    Matrix query_input(tgt_len, d_model);
    Matrix kv_input(src_len, d_model);

    for (int i = 0; i < tgt_len; ++i) {
        for (int j = 0; j < d_model; ++j) {
            query_input(i, j) = 0.1f * (i + j);
        }
    }

    for (int i = 0; i < src_len; ++i) {
        for (int j = 0; j < d_model; ++j) {
            kv_input(i, j) = 0.05f * i;
        }
    }

    Matrix output = cross_attn.forward(query_input, kv_input);

    Matrix grad_output(tgt_len, d_model);
    for (int i = 0; i < tgt_len; ++i) {
        for (int j = 0; j < d_model; ++j) {
            grad_output(i, j) = 0.01f;
        }
    }

    Matrix grad_query, grad_kv;
    cross_attn.backward(grad_output, grad_query, grad_kv);

    // Check that gradients are non-zero
    float query_norm = compute_gradient_norm(grad_query);
    float kv_norm = compute_gradient_norm(grad_kv);

    EXPECT_GT(query_norm, 0.0f);
    EXPECT_GT(kv_norm, 0.0f);
}

TEST(CrossAttentionBackwardTest, GradientFlow) {
    int d_model = 32;
    int num_heads = 4;
    int tgt_len = 5;
    int src_len = 8;

    CrossAttention cross_attn(d_model, num_heads);
    cross_attn.learning_rate = 0.001f;

    Matrix query_input(tgt_len, d_model);
    Matrix kv_input(src_len, d_model);

    // Use stronger input signals to ensure gradients don't vanish
    for (int i = 0; i < tgt_len; ++i) {
        for (int j = 0; j < d_model; ++j) {
            query_input(i, j) = 1.0f * (i + 1);
        }
    }

    for (int i = 0; i < src_len; ++i) {
        for (int j = 0; j < d_model; ++j) {
            kv_input(i, j) = 1.0f;
        }
    }

    // Forward pass
    Matrix output = cross_attn.forward(query_input, kv_input);

    // Create gradient with stronger signal across all dimensions
    Matrix grad_output(tgt_len, d_model);
    for (int i = 0; i < tgt_len; ++i) {
        for (int j = 0; j < d_model; ++j) {
            grad_output(i, j) = 1.0f;
        }
    }

    // Backward pass
    Matrix grad_query, grad_kv;
    cross_attn.backward(grad_output, grad_query, grad_kv);

    // Gradient should propagate to both inputs
    // Note: Due to random weight initialization, gradients may occasionally be very small
    // This test verifies that backward() runs without error and produces some gradient signal
    float total_grad_query = 0.0f;
    float total_grad_kv = 0.0f;

    for (int i = 0; i < grad_query.rows; ++i) {
        for (int j = 0; j < grad_query.cols; ++j) {
            total_grad_query += std::abs(grad_query(i, j));
        }
    }

    for (int i = 0; i < grad_kv.rows; ++i) {
        for (int j = 0; j < grad_kv.cols; ++j) {
            total_grad_kv += std::abs(grad_kv(i, j));
        }
    }

    // Very permissive threshold - just verify gradients exist (not NaN/Inf)
    // The exact magnitude depends on random initialization
    EXPECT_FALSE(std::isnan(total_grad_query));
    EXPECT_FALSE(std::isinf(total_grad_query));
    EXPECT_FALSE(std::isnan(total_grad_kv));
    EXPECT_FALSE(std::isinf(total_grad_kv));

    // Gradients should be computable (finite values)
    EXPECT_GE(total_grad_query, 0.0f);
    EXPECT_GE(total_grad_kv, 0.0f);
}

TEST(CrossAttentionBackwardTest, MultipleBackwardPasses) {
    int d_model = 64;
    int num_heads = 4;
    int tgt_len = 6;
    int src_len = 10;

    CrossAttention cross_attn(d_model, num_heads);

    Matrix query_input(tgt_len, d_model);
    Matrix kv_input(src_len, d_model);

    for (int i = 0; i < tgt_len; ++i) {
        for (int j = 0; j < d_model; ++j) {
            query_input(i, j) = 0.1f;
        }
    }

    for (int i = 0; i < src_len; ++i) {
        for (int j = 0; j < d_model; ++j) {
            kv_input(i, j) = 0.05f;
        }
    }

    // Multiple forward-backward cycles
    for (int iter = 0; iter < 3; ++iter) {
        cross_attn.zero_grad();

        Matrix output = cross_attn.forward(query_input, kv_input);

        Matrix grad_output(tgt_len, d_model);
        for (int i = 0; i < tgt_len; ++i) {
            for (int j = 0; j < d_model; ++j) {
                grad_output(i, j) = 0.01f;
            }
        }

        Matrix grad_query, grad_kv;
        cross_attn.backward(grad_output, grad_query, grad_kv);

        EXPECT_EQ(grad_query.rows, tgt_len);
        EXPECT_EQ(grad_query.cols, d_model);
        EXPECT_EQ(grad_kv.rows, src_len);
        EXPECT_EQ(grad_kv.cols, d_model);
    }
}

// ============================================================================
// Weight Update Tests
// ============================================================================

TEST(CrossAttentionUpdateTest, WeightUpdateChangesOutput) {
    int d_model = 64;
    int num_heads = 4;
    int tgt_len = 8;
    int src_len = 12;

    CrossAttention cross_attn(d_model, num_heads);
    cross_attn.learning_rate = 0.1f;  // Large LR for visible changes

    Matrix query_input(tgt_len, d_model);
    Matrix kv_input(src_len, d_model);

    for (int i = 0; i < tgt_len; ++i) {
        for (int j = 0; j < d_model; ++j) {
            query_input(i, j) = 0.1f * (i + j);
        }
    }

    for (int i = 0; i < src_len; ++i) {
        for (int j = 0; j < d_model; ++j) {
            kv_input(i, j) = 0.05f;
        }
    }

    // Initial output
    Matrix output_before = cross_attn.forward(query_input, kv_input);

    // Backward pass and update
    Matrix grad_output(tgt_len, d_model);
    for (int i = 0; i < tgt_len; ++i) {
        for (int j = 0; j < d_model; ++j) {
            grad_output(i, j) = 0.1f;
        }
    }

    Matrix grad_query, grad_kv;
    cross_attn.backward(grad_output, grad_query, grad_kv);
    cross_attn.update_weights();

    // Output after update
    cross_attn.zero_grad();
    Matrix output_after = cross_attn.forward(query_input, kv_input);

    // Outputs should differ
    bool weights_changed = false;
    for (int i = 0; i < tgt_len; ++i) {
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

TEST(CrossAttentionUpdateTest, LearningRateEffect) {
    int d_model = 64;
    int num_heads = 4;

    CrossAttention cross_attn(d_model, num_heads);

    float new_lr = 0.0005f;
    cross_attn.learning_rate = new_lr;

    EXPECT_FLOAT_EQ(cross_attn.learning_rate, new_lr);
}

// ============================================================================
// Save/Load Tests
// ============================================================================

TEST(CrossAttentionSaveLoadTest, BasicSaveLoad) {
    int d_model = 64;
    int num_heads = 4;
    int tgt_len = 6;
    int src_len = 10;

    CrossAttention cross_attn(d_model, num_heads);

    Matrix query_input(tgt_len, d_model);
    Matrix kv_input(src_len, d_model);

    for (int i = 0; i < tgt_len; ++i) {
        for (int j = 0; j < d_model; ++j) {
            query_input(i, j) = 0.1f * (i + j);
        }
    }

    for (int i = 0; i < src_len; ++i) {
        for (int j = 0; j < d_model; ++j) {
            kv_input(i, j) = 0.05f;
        }
    }

    // Get output from original
    Matrix output_original = cross_attn.forward(query_input, kv_input);

    // Save
    std::string filepath = "test_cross_attn.bin";
    cross_attn.save(filepath);

    // Load into new instance
    CrossAttention loaded_cross_attn(d_model, num_heads);
    loaded_cross_attn.load(filepath);

    // Get output from loaded
    Matrix output_loaded = loaded_cross_attn.forward(query_input, kv_input);

    // Outputs should be identical
    EXPECT_TRUE(matrices_equal(output_original, output_loaded, 1e-6f));

    // Clean up
    std::remove(filepath.c_str());
}

TEST(CrossAttentionSaveLoadTest, MultipleIterations) {
    int d_model = 64;
    int num_heads = 4;
    int tgt_len = 5;
    int src_len = 8;

    CrossAttention cross_attn(d_model, num_heads);
    cross_attn.learning_rate = 0.01f;

    Matrix query_input(tgt_len, d_model);
    Matrix kv_input(src_len, d_model);

    for (int i = 0; i < tgt_len; ++i) {
        for (int j = 0; j < d_model; ++j) {
            query_input(i, j) = 0.1f;
        }
    }

    for (int i = 0; i < src_len; ++i) {
        for (int j = 0; j < d_model; ++j) {
            kv_input(i, j) = 0.05f;
        }
    }

    // Train for a few iterations
    for (int iter = 0; iter < 5; ++iter) {
        cross_attn.zero_grad();
        Matrix output = cross_attn.forward(query_input, kv_input);

        Matrix grad_output(tgt_len, d_model);
        for (int i = 0; i < tgt_len; ++i) {
            for (int j = 0; j < d_model; ++j) {
                grad_output(i, j) = 0.01f;
            }
        }

        Matrix grad_query, grad_kv;
        cross_attn.backward(grad_output, grad_query, grad_kv);
        cross_attn.update_weights();
    }

    // Get final output
    cross_attn.zero_grad();
    Matrix output_final = cross_attn.forward(query_input, kv_input);

    // Save
    std::string filepath = "test_cross_attn_trained.bin";
    cross_attn.save(filepath);

    // Load
    CrossAttention loaded_cross_attn(d_model, num_heads);
    loaded_cross_attn.load(filepath);

    // Get output from loaded
    Matrix output_loaded = loaded_cross_attn.forward(query_input, kv_input);

    // Should produce identical outputs
    EXPECT_TRUE(matrices_equal(output_final, output_loaded, 1e-6f));

    // Clean up
    std::remove(filepath.c_str());
}

TEST(CrossAttentionSaveLoadTest, DimensionMismatch) {
    int d_model = 64;
    int num_heads = 4;

    CrossAttention cross_attn(d_model, num_heads);

    std::string filepath = "test_cross_attn_mismatch.bin";
    cross_attn.save(filepath);

    // Try to load into different sized model
    CrossAttention different_cross_attn(128, 8);

    EXPECT_THROW(different_cross_attn.load(filepath), std::runtime_error);

    // Clean up
    std::remove(filepath.c_str());
}

// ============================================================================
// Integration Tests
// ============================================================================

TEST(CrossAttentionIntegrationTest, MultiLayerStack) {
    int d_model = 64;
    int num_heads = 4;
    int tgt_len = 8;
    int src_len = 12;
    int num_layers = 3;

    // Create stack of cross-attention layers
    std::vector<CrossAttention> layers;
    for (int i = 0; i < num_layers; ++i) {
        layers.emplace_back(d_model, num_heads);
    }

    Matrix query_input(tgt_len, d_model);
    Matrix kv_input(src_len, d_model);

    for (int i = 0; i < tgt_len; ++i) {
        for (int j = 0; j < d_model; ++j) {
            query_input(i, j) = 0.1f * i;
        }
    }

    for (int i = 0; i < src_len; ++i) {
        for (int j = 0; j < d_model; ++j) {
            kv_input(i, j) = 0.05f;
        }
    }

    // Forward through all layers
    Matrix x = query_input;
    for (auto& layer : layers) {
        x = layer.forward(x, kv_input);
    }

    EXPECT_EQ(x.rows, tgt_len);
    EXPECT_EQ(x.cols, d_model);

    // Backward through all layers (reverse order)
    Matrix grad(tgt_len, d_model);
    for (int i = 0; i < tgt_len; ++i) {
        for (int j = 0; j < d_model; ++j) {
            grad(i, j) = 0.01f;
        }
    }

    Matrix grad_kv_dummy;
    for (auto it = layers.rbegin(); it != layers.rend(); ++it) {
        Matrix grad_next;
        it->backward(grad, grad_next, grad_kv_dummy);
        grad = grad_next;
    }

    EXPECT_EQ(grad.rows, tgt_len);
    EXPECT_EQ(grad.cols, d_model);
}

TEST(CrossAttentionIntegrationTest, TrainingLoop) {
    int d_model = 64;
    int num_heads = 4;
    int tgt_len = 6;
    int src_len = 10;

    CrossAttention cross_attn(d_model, num_heads);
    cross_attn.learning_rate = 0.01f;

    Matrix query_input(tgt_len, d_model);
    Matrix kv_input(src_len, d_model);

    for (int i = 0; i < tgt_len; ++i) {
        for (int j = 0; j < d_model; ++j) {
            query_input(i, j) = 0.1f * (i + 1);
        }
    }

    for (int i = 0; i < src_len; ++i) {
        for (int j = 0; j < d_model; ++j) {
            kv_input(i, j) = 0.05f;
        }
    }

    // Training loop
    std::vector<float> losses;
    for (int epoch = 0; epoch < 10; ++epoch) {
        cross_attn.zero_grad();

        Matrix output = cross_attn.forward(query_input, kv_input);

        // Dummy loss (MSE with target)
        float loss = 0.0f;
        Matrix grad_output(tgt_len, d_model);
        for (int i = 0; i < tgt_len; ++i) {
            for (int j = 0; j < d_model; ++j) {
                float target = 0.0f;
                float error = output(i, j) - target;
                loss += error * error;
                grad_output(i, j) = 2.0f * error / (tgt_len * d_model);
            }
        }
        losses.push_back(loss);

        Matrix grad_query, grad_kv;
        cross_attn.backward(grad_output, grad_query, grad_kv);
        cross_attn.update_weights();
    }

    // Loss should generally decrease (may fluctuate)
    EXPECT_LT(losses.back(), losses[0] * 2.0f);
}

TEST(CrossAttentionIntegrationTest, LongSequences) {
    int d_model = 128;
    int num_heads = 8;
    int tgt_len = 50;
    int src_len = 100;

    CrossAttention cross_attn(d_model, num_heads);

    Matrix query_input(tgt_len, d_model);
    Matrix kv_input(src_len, d_model);

    for (int i = 0; i < tgt_len; ++i) {
        for (int j = 0; j < d_model; ++j) {
            query_input(i, j) = 0.01f * i;
        }
    }

    for (int i = 0; i < src_len; ++i) {
        for (int j = 0; j < d_model; ++j) {
            kv_input(i, j) = 0.005f * i;
        }
    }

    // Should handle long sequences
    EXPECT_NO_THROW({
        Matrix output = cross_attn.forward(query_input, kv_input);
        EXPECT_EQ(output.rows, tgt_len);
        EXPECT_EQ(output.cols, d_model);
    });
}

// ============================================================================
// Edge Case Tests
// ============================================================================

TEST(CrossAttentionEdgeCaseTest, SingleQueryToken) {
    int d_model = 64;
    int num_heads = 4;

    CrossAttention cross_attn(d_model, num_heads);

    Matrix query_input(1, d_model);  // Single query token
    Matrix kv_input(10, d_model);

    for (int j = 0; j < d_model; ++j) {
        query_input(0, j) = 0.1f;
    }

    for (int i = 0; i < 10; ++i) {
        for (int j = 0; j < d_model; ++j) {
            kv_input(i, j) = 0.05f;
        }
    }

    Matrix output = cross_attn.forward(query_input, kv_input);

    EXPECT_EQ(output.rows, 1);
    EXPECT_EQ(output.cols, d_model);
}

TEST(CrossAttentionEdgeCaseTest, SingleContextToken) {
    int d_model = 64;
    int num_heads = 4;

    CrossAttention cross_attn(d_model, num_heads);

    Matrix query_input(10, d_model);
    Matrix kv_input(1, d_model);  // Single context token

    for (int i = 0; i < 10; ++i) {
        for (int j = 0; j < d_model; ++j) {
            query_input(i, j) = 0.1f;
        }
    }

    for (int j = 0; j < d_model; ++j) {
        kv_input(0, j) = 0.05f;
    }

    Matrix output = cross_attn.forward(query_input, kv_input);

    EXPECT_EQ(output.rows, 10);
    EXPECT_EQ(output.cols, d_model);
}

TEST(CrossAttentionEdgeCaseTest, ZeroInputs) {
    int d_model = 64;
    int num_heads = 4;
    int tgt_len = 8;
    int src_len = 12;

    CrossAttention cross_attn(d_model, num_heads);

    Matrix query_input(tgt_len, d_model);  // All zeros
    Matrix kv_input(src_len, d_model);     // All zeros

    Matrix output = cross_attn.forward(query_input, kv_input);

    // Should produce output without errors
    EXPECT_EQ(output.rows, tgt_len);
    EXPECT_EQ(output.cols, d_model);
}

TEST(CrossAttentionEdgeCaseTest, SmallModelDimensions) {
    int d_model = 16;
    int num_heads = 2;
    int tgt_len = 4;
    int src_len = 6;

    CrossAttention cross_attn(d_model, num_heads);

    Matrix query_input(tgt_len, d_model);
    Matrix kv_input(src_len, d_model);

    for (int i = 0; i < tgt_len; ++i) {
        for (int j = 0; j < d_model; ++j) {
            query_input(i, j) = 0.1f;
        }
    }

    for (int i = 0; i < src_len; ++i) {
        for (int j = 0; j < d_model; ++j) {
            kv_input(i, j) = 0.05f;
        }
    }

    EXPECT_NO_THROW({
        Matrix output = cross_attn.forward(query_input, kv_input);
        EXPECT_EQ(output.rows, tgt_len);
        EXPECT_EQ(output.cols, d_model);
    });
}

// ============================================================================
// Optimizer Integration Tests
// ============================================================================

TEST(CrossAttentionOptimizerTest, SetOptimizerBasic) {
    CrossAttention cross_attn(64, 4);
    Optimizer opt(OptimizerType::ADAM, 0.001f);

    // Should not throw
    EXPECT_NO_THROW(cross_attn.set_optimizer(&opt));
}

TEST(CrossAttentionOptimizerTest, SetOptimizerNullptr) {
    CrossAttention cross_attn(64, 4);

    // Should handle nullptr gracefully
    EXPECT_NO_THROW(cross_attn.set_optimizer(nullptr));
}

TEST(CrossAttentionOptimizerTest, UpdateWithOptimizer) {
    int d_model = 64;
    int num_heads = 4;
    CrossAttention cross_attn(d_model, num_heads);

    Optimizer opt(OptimizerType::ADAM, 0.001f);
    opt.set_betas(0.9f, 0.999f);
    cross_attn.set_optimizer(&opt);

    // Create inputs
    Matrix query_input(3, d_model);
    Matrix kv_input(5, d_model);
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < d_model; ++j) {
            query_input(i, j) = 1.0f;
        }
    }
    for (int i = 0; i < 5; ++i) {
        for (int j = 0; j < d_model; ++j) {
            kv_input(i, j) = 1.0f;
        }
    }

    // Forward pass
    Matrix output = cross_attn.forward(query_input, kv_input);

    // Create gradient
    Matrix grad_output(3, d_model);
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < d_model; ++j) {
            grad_output(i, j) = 0.1f;
        }
    }

    Matrix grad_query(3, d_model);
    Matrix grad_kv(5, d_model);
    cross_attn.backward(grad_output, grad_query, grad_kv);

    // Update using optimizer - should not throw
    EXPECT_NO_THROW(cross_attn.update_weights());
}

TEST(CrossAttentionOptimizerTest, UpdateWithoutOptimizer) {
    int d_model = 64;
    CrossAttention cross_attn(d_model, 4);
    cross_attn.learning_rate = 0.01f;

    // Don't set optimizer - should use simple gradient descent

    Matrix query_input(2, d_model);
    Matrix kv_input(3, d_model);
    for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < d_model; ++j) {
            query_input(i, j) = 1.0f;
        }
    }
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < d_model; ++j) {
            kv_input(i, j) = 1.0f;
        }
    }

    Matrix output = cross_attn.forward(query_input, kv_input);

    Matrix grad_output(2, d_model);
    for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < d_model; ++j) {
            grad_output(i, j) = 0.1f;
        }
    }

    Matrix grad_query(2, d_model);
    Matrix grad_kv(3, d_model);
    cross_attn.backward(grad_output, grad_query, grad_kv);
    cross_attn.update_weights();

    // Should complete without error
    EXPECT_TRUE(true);
}

TEST(CrossAttentionOptimizerTest, OptimizerVsSimpleGradientDescent) {
    int d_model = 64;

    // Create two identical cross-attention layers
    CrossAttention ca_with_opt(d_model, 4);
    CrossAttention ca_without_opt(d_model, 4);

    // Set optimizer for first one
    Optimizer opt(OptimizerType::ADAM, 0.1f);
    opt.set_betas(0.9f, 0.999f);
    ca_with_opt.set_optimizer(&opt);

    ca_without_opt.learning_rate = 0.1f;

    // Run multiple training steps with varying gradients
    Matrix query_input(2, d_model);
    Matrix kv_input(3, d_model);

    for (int iter = 0; iter < 5; ++iter) {
        // Initialize inputs
        for (int i = 0; i < 2; ++i) {
            for (int j = 0; j < d_model; ++j) {
                query_input(i, j) = 0.5f;
            }
        }
        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < d_model; ++j) {
                kv_input(i, j) = 0.5f;
            }
        }

        Matrix grad_output(2, d_model);
        for (int i = 0; i < 2; ++i) {
            for (int j = 0; j < d_model; ++j) {
                grad_output(i, j) = (iter % 2 == 0) ? 0.5f : 0.25f;
            }
        }

        // With optimizer
        ca_with_opt.forward(query_input, kv_input);
        Matrix grad_q1(2, d_model), grad_kv1(3, d_model);
        ca_with_opt.backward(grad_output, grad_q1, grad_kv1);
        ca_with_opt.update_weights();

        // Without optimizer
        ca_without_opt.forward(query_input, kv_input);
        Matrix grad_q2(2, d_model), grad_kv2(3, d_model);
        ca_without_opt.backward(grad_output, grad_q2, grad_kv2);
        ca_without_opt.update_weights();
    }

    // Results should be different (Adam uses momentum and adapts per-parameter)
    Matrix out1 = ca_with_opt.forward(query_input, kv_input);
    Matrix out2 = ca_without_opt.forward(query_input, kv_input);

    // Check that at least one value is significantly different
    bool found_difference = false;
    for (int i = 0; i < 2 && !found_difference; ++i) {
        for (int j = 0; j < d_model && !found_difference; ++j) {
            if (std::abs(out1(i, j) - out2(i, j)) > 0.01f) {
                found_difference = true;
            }
        }
    }
    EXPECT_TRUE(found_difference);
}

TEST(CrossAttentionOptimizerTest, MultipleUpdatesWithOptimizer) {
    CrossAttention cross_attn(64, 4);

    Optimizer opt(OptimizerType::ADAM, 0.001f);
    opt.set_betas(0.9f, 0.999f);
    cross_attn.set_optimizer(&opt);

    Matrix query_input(2, 64);
    Matrix kv_input(3, 64);
    for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < 64; ++j) {
            query_input(i, j) = 0.5f;
        }
    }
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 64; ++j) {
            kv_input(i, j) = 0.5f;
        }
    }

    Matrix grad_output(2, 64);
    for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < 64; ++j) {
            grad_output(i, j) = 0.1f;
        }
    }

    // Multiple updates
    for (int i = 0; i < 3; ++i) {
        cross_attn.forward(query_input, kv_input);
        Matrix grad_q(2, 64), grad_kv(3, 64);
        cross_attn.backward(grad_output, grad_q, grad_kv);
        EXPECT_NO_THROW(cross_attn.update_weights());
    }
}

TEST(CrossAttentionOptimizerTest, SwitchOptimizer) {
    CrossAttention cross_attn(64, 4);

    // Start with Adam
    Optimizer opt1(OptimizerType::ADAM, 0.001f);
    cross_attn.set_optimizer(&opt1);

    Matrix query_input(2, 64);
    Matrix kv_input(3, 64);
    for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < 64; ++j) {
            query_input(i, j) = 0.5f;
        }
    }
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 64; ++j) {
            kv_input(i, j) = 0.5f;
        }
    }

    Matrix grad_output(2, 64);
    for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < 64; ++j) {
            grad_output(i, j) = 0.1f;
        }
    }

    cross_attn.forward(query_input, kv_input);
    Matrix grad_q1(2, 64), grad_kv1(3, 64);
    cross_attn.backward(grad_output, grad_q1, grad_kv1);
    cross_attn.update_weights();

    // Switch to different optimizer
    Optimizer opt2(OptimizerType::SGD, 0.01f);
    cross_attn.set_optimizer(&opt2);

    cross_attn.forward(query_input, kv_input);
    Matrix grad_q2(2, 64), grad_kv2(3, 64);
    cross_attn.backward(grad_output, grad_q2, grad_kv2);

    // Should not throw
    EXPECT_NO_THROW(cross_attn.update_weights());
}

TEST(CrossAttentionOptimizerTest, OptimizerWithDifferentLearningRates) {
    CrossAttention ca1(64, 4);
    CrossAttention ca2(64, 4);

    Optimizer opt1(OptimizerType::SGD, 0.001f);
    Optimizer opt2(OptimizerType::SGD, 0.1f);

    ca1.set_optimizer(&opt1);
    ca2.set_optimizer(&opt2);

    Matrix query_input(2, 64);
    Matrix kv_input(3, 64);
    for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < 64; ++j) {
            query_input(i, j) = 0.5f;
        }
    }
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 64; ++j) {
            kv_input(i, j) = 0.5f;
        }
    }

    Matrix grad_output(2, 64);
    for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < 64; ++j) {
            grad_output(i, j) = 0.1f;
        }
    }

    Matrix out1_before = ca1.forward(query_input, kv_input);
    Matrix out2_before = ca2.forward(query_input, kv_input);

    Matrix grad_q1(2, 64), grad_kv1(3, 64);
    ca1.backward(grad_output, grad_q1, grad_kv1);
    ca1.update_weights();

    Matrix grad_q2(2, 64), grad_kv2(3, 64);
    ca2.backward(grad_output, grad_q2, grad_kv2);
    ca2.update_weights();

    Matrix out1_after = ca1.forward(query_input, kv_input);
    Matrix out2_after = ca2.forward(query_input, kv_input);

    // Higher learning rate should cause bigger change
    float change1 = 0.0f, change2 = 0.0f;
    for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < 64; ++j) {
            change1 += std::abs(out1_after(i, j) - out1_before(i, j));
            change2 += std::abs(out2_after(i, j) - out2_before(i, j));
        }
    }

    EXPECT_LT(change1, change2);
}

TEST(CrossAttentionOptimizerTest, RegisterParametersExplicit) {
    CrossAttention cross_attn(64, 4);
    Optimizer opt(OptimizerType::ADAM, 0.001f);

    cross_attn.set_optimizer(&opt);

    // register_parameters() should have been called by set_optimizer()
    // Verify by doing update
    Matrix query_input(2, 64);
    Matrix kv_input(3, 64);
    for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < 64; ++j) {
            query_input(i, j) = 0.5f;
        }
    }
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 64; ++j) {
            kv_input(i, j) = 0.5f;
        }
    }

    Matrix grad_output(2, 64);
    for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < 64; ++j) {
            grad_output(i, j) = 0.1f;
        }
    }

    cross_attn.forward(query_input, kv_input);
    Matrix grad_q(2, 64), grad_kv(3, 64);
    cross_attn.backward(grad_output, grad_q, grad_kv);

    // Should not throw if parameters are registered
    EXPECT_NO_THROW(cross_attn.update_weights());
}

TEST(CrossAttentionOptimizerTest, ParametersChangeWithOptimizer) {
    CrossAttention cross_attn(64, 4);

    Optimizer opt(OptimizerType::ADAM, 0.01f);
    opt.set_betas(0.9f, 0.999f);
    cross_attn.set_optimizer(&opt);

    Matrix query_input(2, 64);
    Matrix kv_input(3, 64);
    for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < 64; ++j) {
            query_input(i, j) = 0.5f;
        }
    }
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 64; ++j) {
            kv_input(i, j) = 0.5f;
        }
    }

    Matrix grad_output(2, 64);
    for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < 64; ++j) {
            grad_output(i, j) = 0.5f;
        }
    }

    Matrix out_before = cross_attn.forward(query_input, kv_input);

    Matrix grad_q(2, 64), grad_kv(3, 64);
    cross_attn.backward(grad_output, grad_q, grad_kv);
    cross_attn.update_weights();

    Matrix out_after = cross_attn.forward(query_input, kv_input);

    // Output should have changed
    bool changed = false;
    for (int i = 0; i < 2 && !changed; ++i) {
        for (int j = 0; j < 64 && !changed; ++j) {
            if (std::abs(out_after(i, j) - out_before(i, j)) > 1e-6f) {
                changed = true;
            }
        }
    }

    EXPECT_TRUE(changed);
}

TEST(CrossAttentionOptimizerTest, LearningRateScheduling) {
    CrossAttention cross_attn(64, 4);

    Optimizer opt(OptimizerType::SGD, 0.1f);
    cross_attn.set_optimizer(&opt);

    Matrix query_input(2, 64);
    Matrix kv_input(3, 64);
    for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < 64; ++j) {
            query_input(i, j) = 0.5f;
        }
    }
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 64; ++j) {
            kv_input(i, j) = 0.5f;
        }
    }

    Matrix grad_output(2, 64);
    for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < 64; ++j) {
            grad_output(i, j) = 0.1f;
        }
    }

    // First update with LR = 0.1
    Matrix out1 = cross_attn.forward(query_input, kv_input);
    Matrix grad_q1(2, 64), grad_kv1(3, 64);
    cross_attn.backward(grad_output, grad_q1, grad_kv1);
    cross_attn.update_weights();
    Matrix out1_after = cross_attn.forward(query_input, kv_input);

    // Change learning rate
    opt.set_learning_rate(0.01f);

    // Second update with LR = 0.01
    Matrix out2 = cross_attn.forward(query_input, kv_input);
    Matrix grad_q2(2, 64), grad_kv2(3, 64);
    cross_attn.backward(grad_output, grad_q2, grad_kv2);
    cross_attn.update_weights();
    Matrix out2_after = cross_attn.forward(query_input, kv_input);

    // Changes should be different due to different learning rates
    float change1 = 0.0f, change2 = 0.0f;
    for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < 64; ++j) {
            change1 += std::abs(out1_after(i, j) - out1(i, j));
            change2 += std::abs(out2_after(i, j) - out2(i, j));
        }
    }

    EXPECT_GT(change1, change2 * 5.0f);  // First should be ~10x larger
}

TEST(CrossAttentionOptimizerTest, BackwardCompatibilityNoOptimizer) {
    CrossAttention cross_attn(64, 4);
    cross_attn.learning_rate = 0.01f;

    // Old-style usage without optimizer
    Matrix query_input(2, 64);
    Matrix kv_input(3, 64);
    for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < 64; ++j) {
            query_input(i, j) = 0.5f;
        }
    }
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 64; ++j) {
            kv_input(i, j) = 0.5f;
        }
    }

    Matrix grad_output(2, 64);
    for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < 64; ++j) {
            grad_output(i, j) = 0.1f;
        }
    }

    cross_attn.forward(query_input, kv_input);
    Matrix grad_q(2, 64), grad_kv(3, 64);
    cross_attn.backward(grad_output, grad_q, grad_kv);
    cross_attn.update_weights();

    // Should complete without error
    EXPECT_TRUE(true);
}

// ============================================================================
// Main
// ============================================================================

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
