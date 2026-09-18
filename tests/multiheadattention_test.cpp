#include "../src/MultiHeadAttention.hpp"
#include <gtest/gtest.h>
#include <cmath>
#include <fstream>
#include <random>
#include <stdexcept>
#include <utility>
#include <vector>
#include "../src/Activation.hpp"
#include "../src/Matrix.hpp"
#include "../src/Optimizer.hpp"

// ============================================================================
// Constructor Tests
// ============================================================================

TEST(MultiHeadAttentionConstructorTest, BasicConstruction) {
    MultiHeadAttention mha(512, 8);

    EXPECT_EQ(mha.get_d_model(), 512);
    EXPECT_EQ(mha.get_num_heads(), 8);
    EXPECT_EQ(mha.get_d_k(), 64);  // 512 / 8
    EXPECT_FLOAT_EQ(mha.learning_rate, 0.001f);
}

TEST(MultiHeadAttentionConstructorTest, SmallModel) {
    MultiHeadAttention mha(256, 4);

    EXPECT_EQ(mha.get_d_model(), 256);
    EXPECT_EQ(mha.get_num_heads(), 4);
    EXPECT_EQ(mha.get_d_k(), 64);
}

TEST(MultiHeadAttentionConstructorTest, LargeModel) {
    MultiHeadAttention mha(1024, 16);

    EXPECT_EQ(mha.get_d_model(), 1024);
    EXPECT_EQ(mha.get_num_heads(), 16);
    EXPECT_EQ(mha.get_d_k(), 64);
}

TEST(MultiHeadAttentionConstructorTest, DimensionNotDivisible) {
    // d_model must be divisible by num_heads
    EXPECT_THROW(MultiHeadAttention(512, 7), std::invalid_argument);
    EXPECT_THROW(MultiHeadAttention(100, 7), std::invalid_argument);
}

TEST(MultiHeadAttentionConstructorTest, SingleHead) {
    MultiHeadAttention mha(512, 1);

    EXPECT_EQ(mha.get_d_model(), 512);
    EXPECT_EQ(mha.get_num_heads(), 1);
    EXPECT_EQ(mha.get_d_k(), 512);
}

TEST(MultiHeadAttentionConstructorTest, LearningRateModification) {
    MultiHeadAttention mha(512, 8);
    mha.learning_rate = 0.0001f;

    EXPECT_FLOAT_EQ(mha.learning_rate, 0.0001f);
}

// ============================================================================
// Forward Pass Tests
// ============================================================================

TEST(MultiHeadAttentionForwardTest, BasicForward) {
    MultiHeadAttention mha(64, 4);

    Matrix input(10, 64);  // seq_len=10, d_model=64
    for (int i = 0; i < 10; ++i) {
        for (int j = 0; j < 64; ++j) {
            input(i, j) = 0.01f * (i + j);
        }
    }

    Matrix output = mha.forward(input);

    EXPECT_EQ(output.rows, 10);
    EXPECT_EQ(output.cols, 64);
}

TEST(MultiHeadAttentionForwardTest, SingleToken) {
    MultiHeadAttention mha(128, 4);

    Matrix input(1, 128);
    for (int j = 0; j < 128; ++j) {
        input(0, j) = 0.1f;
    }

    Matrix output = mha.forward(input);

    EXPECT_EQ(output.rows, 1);
    EXPECT_EQ(output.cols, 128);
}

TEST(MultiHeadAttentionForwardTest, LongSequence) {
    MultiHeadAttention mha(256, 8);

    Matrix input(100, 256);
    for (int i = 0; i < 100; ++i) {
        for (int j = 0; j < 256; ++j) {
            input(i, j) = 0.001f * i;
        }
    }

    Matrix output = mha.forward(input);

    EXPECT_EQ(output.rows, 100);
    EXPECT_EQ(output.cols, 256);
}

TEST(MultiHeadAttentionForwardTest, InvalidInputDimension) {
    MultiHeadAttention mha(512, 8);

    Matrix input(10, 256);  // Wrong d_model (should be 512)

    EXPECT_THROW(mha.forward(input), std::invalid_argument);
}

TEST(MultiHeadAttentionForwardTest, OutputNotZero) {
    MultiHeadAttention mha(64, 4);

    Matrix input(5, 64);
    for (int i = 0; i < 5; ++i) {
        for (int j = 0; j < 64; ++j) {
            input(i, j) = 1.0f;
        }
    }

    Matrix output = mha.forward(input);

    // Check that output is not all zeros
    bool has_nonzero = false;
    for (int i = 0; i < output.rows; ++i) {
        for (int j = 0; j < output.cols; ++j) {
            if (std::abs(output(i, j)) > 1e-6f) {
                has_nonzero = true;
                break;
            }
        }
    }
    EXPECT_TRUE(has_nonzero);
}

// ============================================================================
// Attention Weights Tests
// ============================================================================

TEST(MultiHeadAttentionWeightsTest, AttentionWeightsShape) {
    MultiHeadAttention mha(128, 4);

    int seq_len = 10;
    Matrix input(seq_len, 128);
    for (int i = 0; i < seq_len; ++i) {
        for (int j = 0; j < 128; ++j) {
            input(i, j) = 0.1f * i;
        }
    }

    mha.forward(input);

    const Matrix& attn_weights = mha.get_attention_weights();
    EXPECT_EQ(attn_weights.rows, seq_len);
    EXPECT_EQ(attn_weights.cols, seq_len);
}

TEST(MultiHeadAttentionWeightsTest, AttentionWeightsSumToOne) {
    MultiHeadAttention mha(64, 4);

    Matrix input(5, 64);
    for (int i = 0; i < 5; ++i) {
        for (int j = 0; j < 64; ++j) {
            input(i, j) = 0.5f + 0.1f * i;
        }
    }

    mha.forward(input);

    const Matrix& attn_weights = mha.get_attention_weights();

    // Each row should sum to approximately 1.0
    for (int i = 0; i < attn_weights.rows; ++i) {
        float sum = 0.0f;
        for (int j = 0; j < attn_weights.cols; ++j) {
            sum += attn_weights(i, j);
        }
        EXPECT_NEAR(sum, 1.0f, 1e-5f);
    }
}

TEST(MultiHeadAttentionWeightsTest, AttentionWeightsNonNegative) {
    MultiHeadAttention mha(128, 8);

    Matrix input(8, 128);
    for (int i = 0; i < 8; ++i) {
        for (int j = 0; j < 128; ++j) {
            input(i, j) = static_cast<float>(i + j) / 10.0f;
        }
    }

    mha.forward(input);

    const Matrix& attn_weights = mha.get_attention_weights();

    // All weights should be non-negative
    for (int i = 0; i < attn_weights.rows; ++i) {
        for (int j = 0; j < attn_weights.cols; ++j) {
            EXPECT_GE(attn_weights(i, j), 0.0f);
        }
    }
}

// ============================================================================
// Masking Tests
// ============================================================================

TEST(MultiHeadAttentionMaskTest, CausalMask) {
    MultiHeadAttention mha(64, 4);

    int seq_len = 5;
    Matrix input(seq_len, 64);
    for (int i = 0; i < seq_len; ++i) {
        for (int j = 0; j < 64; ++j) {
            input(i, j) = 0.1f * i;
        }
    }

    // Create causal mask (lower triangular)
    Matrix mask(seq_len, seq_len);
    for (int i = 0; i < seq_len; ++i) {
        for (int j = 0; j < seq_len; ++j) {
            mask(i, j) = (j <= i) ? 1.0f : 0.0f;
        }
    }

    Matrix output = mha.forward(input, &mask);

    EXPECT_EQ(output.rows, seq_len);
    EXPECT_EQ(output.cols, 64);

    // Check that attention weights respect the mask
    const Matrix& attn_weights = mha.get_attention_weights();
    for (int i = 0; i < seq_len; ++i) {
        for (int j = 0; j < seq_len; ++j) {
            if (j > i) {
                // Masked positions should have near-zero attention
                EXPECT_NEAR(attn_weights(i, j), 0.0f, 1e-5f);
            }
        }
    }
}

TEST(MultiHeadAttentionMaskTest, PaddingMask) {
    MultiHeadAttention mha(128, 8);

    int seq_len = 6;
    int valid_len = 4;  // First 4 positions are valid

    Matrix input(seq_len, 128);
    for (int i = 0; i < seq_len; ++i) {
        for (int j = 0; j < 128; ++j) {
            input(i, j) = (i < valid_len) ? 1.0f : 0.0f;
        }
    }

    // Create padding mask
    Matrix mask(seq_len, seq_len);
    for (int i = 0; i < seq_len; ++i) {
        for (int j = 0; j < seq_len; ++j) {
            mask(i, j) = (j < valid_len) ? 1.0f : 0.0f;
        }
    }

    Matrix output = mha.forward(input, &mask);

    const Matrix& attn_weights = mha.get_attention_weights();

    // Check that padded positions have near-zero attention
    for (int i = 0; i < seq_len; ++i) {
        for (int j = valid_len; j < seq_len; ++j) {
            EXPECT_NEAR(attn_weights(i, j), 0.0f, 1e-5f);
        }
    }
}

TEST(MultiHeadAttentionMaskTest, InvalidMaskDimensions) {
    MultiHeadAttention mha(64, 4);

    Matrix input(5, 64);
    Matrix wrong_mask(3, 3);  // Wrong dimensions

    EXPECT_THROW(mha.forward(input, &wrong_mask), std::invalid_argument);
}

TEST(MultiHeadAttentionMaskTest, NoMask) {
    MultiHeadAttention mha(64, 4);

    Matrix input(5, 64);
    for (int i = 0; i < 5; ++i) {
        for (int j = 0; j < 64; ++j) {
            input(i, j) = 0.1f;
        }
    }

    // Without mask, all positions can attend to all others
    Matrix output = mha.forward(input, nullptr);

    EXPECT_EQ(output.rows, 5);
    EXPECT_EQ(output.cols, 64);
}

// ============================================================================
// TD-059 Regression Tests: genuine per-head attention
//
// Before the fix, forward()/forward_with_cache() computed one global attention pattern
// over the full d_model width regardless of num_heads -- every test above this section
// passes identically whether or not heads are actually split (shape, mask, sum-to-one,
// non-negativity all hold either way), which is exactly why the original bug went
// uncaught. These tests specifically exercise "does num_heads actually change the
// computation" and "is the analytic per-head backward pass correct", which the old
// implementation would have failed.
// ============================================================================

namespace {
// Deterministic weight draw shared by two MultiHeadAttention instances -- any output
// difference between them is then attributable only to num_heads (real per-head
// splitting), never to different random initialization.
void seed_weights_deterministically(MultiHeadAttention& mha, int d_model, unsigned int seed) {
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

    mha.set_Wq(fill());
    mha.set_Wk(fill());
    mha.set_Wv(fill());
    mha.set_Wo(fill());
}
}  // namespace

TEST(MultiHeadAttentionArchitectureTest, MatchesIndependentPerHeadReference) {
    // TD-059 regression. A first draft of this test compared two instances differing only in
    // num_heads (with shared weights) and asserted their outputs differ -- that turned out to be
    // a weak check: d_k = d_model/num_heads changes whenever num_heads does, and the pre-fix
    // code's scale factor (1/sqrt(d_k), applied to a d_model-wide contraction) already varied
    // with d_k on its own, so output changing between num_heads configurations doesn't, by
    // itself, prove real per-head splitting happened. Confirmed directly: that draft passed
    // against BOTH the pre-fix and post-fix implementation. The only way to actually pin down the
    // formula is to check forward()'s output against an independent, from-scratch implementation
    // of the documented per-head formula -- softmax(Q_h K_h^T / sqrt(d_k)) V_h computed
    // separately per head, concatenated, then projected through W_o -- built here directly from
    // the class's own weights via the public accessors, not by reusing any of
    // MultiHeadAttention's own forward/forward_parallel code.
    int d_model = 8;
    int num_heads = 2;
    int d_k = d_model / num_heads;
    MultiHeadAttention mha(d_model, num_heads);

    int seq_len = 3;
    Matrix input(seq_len, d_model);
    for (int i = 0; i < seq_len; ++i) {
        for (int j = 0; j < d_model; ++j) {
            input(i, j) = 0.1f * static_cast<float>(i + 1) - 0.05f * static_cast<float>(j);
        }
    }

    Matrix actual = mha.forward(input);

    // --- Independent reference, reimplemented from scratch ---
    Matrix Wq = mha.get_Wq(), Wk = mha.get_Wk(), Wv = mha.get_Wv(), Wo = mha.get_Wo();

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

    Matrix Q = matmul(input, Wq);
    Matrix K = matmul(input, Wk);
    Matrix V = matmul(input, Wv);

    Matrix concatenated(seq_len, d_model);
    float scale = 1.0f / std::sqrt(static_cast<float>(d_k));
    for (int h = 0; h < num_heads; ++h) {
        int start = h * d_k;
        std::vector<std::vector<float>> scores(seq_len, std::vector<float>(seq_len, 0.0f));
        for (int i = 0; i < seq_len; ++i) {
            for (int j = 0; j < seq_len; ++j) {
                float sum = 0.0f;
                for (int k = 0; k < d_k; ++k) {
                    sum += Q(i, start + k) * K(j, start + k);
                }
                scores[i][j] = sum * scale;
            }
        }
        for (int i = 0; i < seq_len; ++i) {
            float max_v = scores[i][0];
            for (int j = 1; j < seq_len; ++j) {
                max_v = std::max(max_v, scores[i][j]);
            }
            float sum_exp = 0.0f;
            for (int j = 0; j < seq_len; ++j) {
                scores[i][j] = std::exp(scores[i][j] - max_v);
                sum_exp += scores[i][j];
            }
            for (int j = 0; j < seq_len; ++j) {
                scores[i][j] /= sum_exp;
            }
        }
        for (int i = 0; i < seq_len; ++i) {
            for (int k = 0; k < d_k; ++k) {
                float sum = 0.0f;
                for (int j = 0; j < seq_len; ++j) {
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

TEST(MultiHeadAttentionArchitectureTest, SingleHeadMatchesGlobalAttention) {
    // Sanity check on the same fix from the other direction: num_heads=1 has exactly one
    // head spanning the full d_model width, so it degenerates to plain (single-head)
    // scaled dot-product attention over the whole vector -- forward_parallel()'s per-head
    // loop with num_heads=1 should reproduce that directly.
    int d_model = 16;
    MultiHeadAttention mha(d_model, 1);

    Matrix input(5, d_model);
    for (int i = 0; i < input.rows; ++i) {
        for (int j = 0; j < input.cols; ++j) {
            input(i, j) = 0.05f * static_cast<float>(i + j);
        }
    }

    Matrix output = mha.forward(input);
    const Matrix& weights = mha.get_attention_weights();

    // Single head means cached_attention_weights (the mean-across-heads summary) IS the
    // one head's real weights -- still a valid [seq_len, seq_len] probability distribution.
    for (int i = 0; i < weights.rows; ++i) {
        float sum = 0.0f;
        for (int j = 0; j < weights.cols; ++j) {
            sum += weights(i, j);
        }
        EXPECT_NEAR(sum, 1.0f, 1e-5f);
    }
    EXPECT_EQ(output.rows, 5);
    EXPECT_EQ(output.cols, d_model);
}

TEST(MultiHeadAttentionArchitectureTest, BackwardPassMatchesNumericalGradient) {
    // Finite-difference gradient check with num_heads > 1, added alongside TD-059's per-head
    // backward rewrite. Note this checks a different property than
    // MatchesIndependentPerHeadReference above: it confirms backward() is the correct gradient
    // of *whatever forward() actually computes*, which passes equally well for a self-consistent
    // wrong implementation (confirmed: this test also passed against the pre-fix code, since the
    // old forward()/backward() pair was internally consistent with each other, just consistently
    // wrong) -- MatchesIndependentPerHeadReference is what actually pins down the formula itself.
    // Still valuable in its own right: a subtly wrong per-head gradient (e.g. a head's
    // softmax-backward reading another head's weights, or a column slice written to the wrong
    // offset) would produce finite, plausible-shaped gradients under every other test in this
    // file, but would fail this check.
    int d_model = 16;
    int num_heads = 4;
    MultiHeadAttention mha(d_model, num_heads);
    seed_weights_deterministically(mha, d_model, /*seed=*/11);

    const int seq_len = 5;
    Matrix input(seq_len, d_model);
    for (int i = 0; i < input.rows; ++i) {
        for (int j = 0; j < input.cols; ++j) {
            input(i, j) = 0.05f * std::sin(static_cast<float>(i * d_model + j));
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

    Matrix output = mha.forward(input);
    Matrix grad_output(output.rows, output.cols);
    for (int i = 0; i < output.rows; ++i) {
        for (int j = 0; j < output.cols; ++j) {
            grad_output(i, j) = 2.0f * output(i, j);
        }
    }
    Matrix analytic_grad = mha.backward(grad_output);

    const float epsilon = 1e-3f;
    const std::vector<std::pair<int, int>> positions = {
        {0, 0}, {0, d_model / 2}, {1, 3}, {2, d_model - 1}, {seq_len - 1, 7}};

    for (const auto& [pi, pj] : positions) {
        Matrix input_plus = input;
        input_plus(pi, pj) += epsilon;
        Matrix input_minus = input;
        input_minus(pi, pj) -= epsilon;

        float loss_plus = scalar_loss(mha.forward(input_plus));
        float loss_minus = scalar_loss(mha.forward(input_minus));
        float numerical_grad = (loss_plus - loss_minus) / (2.0f * epsilon);

        float tolerance = std::max(1e-2f, 0.05f * std::abs(numerical_grad));
        EXPECT_NEAR(analytic_grad(pi, pj), numerical_grad, tolerance)
            << "gradient mismatch at (" << pi << "," << pj << ")";
    }
}

// ============================================================================
// TD-038: LoRA integration tests
// ============================================================================

TEST(MultiHeadAttentionLoRATest, EnableLoraIsNoOpBeforeTraining) {
    int d_model = 16;
    MultiHeadAttention mha(d_model, 4);
    seed_weights_deterministically(mha, d_model, /*seed=*/7);

    Matrix input(5, d_model);
    for (int i = 0; i < input.rows; ++i) {
        for (int j = 0; j < input.cols; ++j) {
            input(i, j) = 0.05f * std::cos(static_cast<float>(i * d_model + j));
        }
    }

    Matrix output_before = mha.forward(input);

    LoRAConfig config;
    config.rank = 4;
    mha.enable_lora(config);
    ASSERT_TRUE(mha.has_lora());

    // LoRAAdapter's B matrix starts at zero, so ΔW=0 -- forward() must be untouched.
    Matrix output_after = mha.forward(input);
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

TEST(MultiHeadAttentionLoRATest, BackwardPassMatchesNumericalGradientWithLoraActive) {
    // Same finite-difference shape as
    // MultiHeadAttentionArchitectureTest.BackwardPassMatchesNumericalGradient above, but with
    // LoRA adapters enabled AND pre-trained (so B != 0 and the LoRA branch actually
    // contributes to the output) -- this is what actually exercises the new
    // LoRAAdapter::backward() grad_input return value wired into
    // MultiHeadAttention::backward() (TD-038), not just the pre-existing frozen-weight path.
    int d_model = 16;
    int num_heads = 4;
    MultiHeadAttention mha(d_model, num_heads);
    seed_weights_deterministically(mha, d_model, /*seed=*/13);

    LoRAConfig config;
    config.rank = 4;
    config.alpha = 8.0f;
    mha.enable_lora(config);

    const int seq_len = 5;
    Matrix input(seq_len, d_model);
    for (int i = 0; i < input.rows; ++i) {
        for (int j = 0; j < input.cols; ++j) {
            input(i, j) = 0.05f * std::sin(static_cast<float>(i * d_model + j));
        }
    }

    // Pre-train the adapters a few steps with plain manual-learning-rate descent so every
    // adapter's B matrix is genuinely non-zero before the gradient check below.
    for (int step = 0; step < 3; ++step) {
        Matrix out = mha.forward(input);
        Matrix grad(out.rows, out.cols);
        for (int i = 0; i < grad.rows; ++i) {
            for (int j = 0; j < grad.cols; ++j) {
                grad(i, j) = 0.1f * (i + j + 1);
            }
        }
        mha.backward(grad);
        mha.get_lora_q()->update(0.05f);
        mha.get_lora_k()->update(0.05f);
        mha.get_lora_v()->update(0.05f);
        mha.get_lora_o()->update(0.05f);
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

    Matrix output = mha.forward(input);
    Matrix grad_output(output.rows, output.cols);
    for (int i = 0; i < output.rows; ++i) {
        for (int j = 0; j < output.cols; ++j) {
            grad_output(i, j) = 2.0f * output(i, j);
        }
    }
    Matrix analytic_grad = mha.backward(grad_output);

    const float epsilon = 1e-3f;
    const std::vector<std::pair<int, int>> positions = {
        {0, 0}, {0, d_model / 2}, {1, 3}, {2, d_model - 1}, {seq_len - 1, 7}};

    for (const auto& [pi, pj] : positions) {
        Matrix input_plus = input;
        input_plus(pi, pj) += epsilon;
        Matrix input_minus = input;
        input_minus(pi, pj) -= epsilon;

        float loss_plus = scalar_loss(mha.forward(input_plus));
        float loss_minus = scalar_loss(mha.forward(input_minus));
        float numerical_grad = (loss_plus - loss_minus) / (2.0f * epsilon);

        float tolerance = std::max(1e-2f, 0.05f * std::abs(numerical_grad));
        EXPECT_NEAR(analytic_grad(pi, pj), numerical_grad, tolerance)
            << "gradient mismatch at (" << pi << "," << pj << ") with LoRA active";
    }
}

TEST(MultiHeadAttentionLoRATest, RegisterLoraParametersFreezesBaseWeights) {
    int d_model = 12;
    MultiHeadAttention mha(d_model, 3);
    seed_weights_deterministically(mha, d_model, /*seed=*/23);

    LoRAConfig config;
    config.rank = 3;
    mha.enable_lora(config);

    Matrix Wq_before = mha.get_Wq();
    Matrix Wk_before = mha.get_Wk();
    Matrix Wv_before = mha.get_Wv();
    Matrix Wo_before = mha.get_Wo();
    Matrix A_before = mha.get_lora_q()->get_A();
    Matrix B_before = mha.get_lora_q()->get_B();

    Optimizer optimizer(OptimizerType::ADAM, 0.05f);
    // Deliberately register ONLY the LoRA parameters -- never register_parameters() on this
    // optimizer -- so the base weights are never in its parameter_groups at all.
    mha.register_lora_parameters(optimizer);

    Matrix input(4, d_model);
    for (int i = 0; i < input.rows; ++i) {
        for (int j = 0; j < input.cols; ++j) {
            input(i, j) = 0.05f * (i + j);
        }
    }

    // Two steps: on the first, grad_A is identically zero (LoRAAdapter's own grad_A
    // computation multiplies through B, which starts at zero -- see this codebase's own
    // LoRATest.UpdateWeights for the same documented property), so only B moves initially;
    // A only starts moving once B is non-zero, from the second step onward.
    for (int step = 0; step < 2; ++step) {
        mha.zero_grad();
        Matrix output = mha.forward(input);
        Matrix grad(output.rows, output.cols);
        for (int i = 0; i < grad.rows; ++i) {
            for (int j = 0; j < grad.cols; ++j) {
                grad(i, j) = 0.1f;
            }
        }
        mha.backward(grad);
        optimizer.step();
    }

    // Base weights must be bit-for-bit untouched -- they were never registered anywhere.
    const Matrix& Wq_after = mha.get_Wq();
    const Matrix& Wk_after = mha.get_Wk();
    const Matrix& Wv_after = mha.get_Wv();
    const Matrix& Wo_after = mha.get_Wo();
    for (int i = 0; i < d_model; ++i) {
        for (int j = 0; j < d_model; ++j) {
            EXPECT_FLOAT_EQ(Wq_before(i, j), Wq_after(i, j));
            EXPECT_FLOAT_EQ(Wk_before(i, j), Wk_after(i, j));
            EXPECT_FLOAT_EQ(Wv_before(i, j), Wv_after(i, j));
            EXPECT_FLOAT_EQ(Wo_before(i, j), Wo_after(i, j));
        }
    }

    // But the LoRA adapter's own A/B must have actually moved.
    const Matrix& A_after = mha.get_lora_q()->get_A();
    const Matrix& B_after = mha.get_lora_q()->get_B();
    bool a_changed = false, b_changed = false;
    for (int i = 0; i < A_before.rows && !a_changed; ++i) {
        for (int j = 0; j < A_before.cols; ++j) {
            if (std::abs(A_before(i, j) - A_after(i, j)) > 1e-6f) {
                a_changed = true;
                break;
            }
        }
    }
    for (int i = 0; i < B_before.rows && !b_changed; ++i) {
        for (int j = 0; j < B_before.cols; ++j) {
            if (std::abs(B_before(i, j) - B_after(i, j)) > 1e-6f) {
                b_changed = true;
                break;
            }
        }
    }
    EXPECT_TRUE(a_changed) << "LoRA_q's A matrix never moved -- register_lora_parameters() "
                              "isn't actually training the adapter";
    EXPECT_TRUE(b_changed) << "LoRA_q's B matrix never moved -- register_lora_parameters() "
                              "isn't actually training the adapter";
}

TEST(MultiHeadAttentionLoRATest, MergeLoraPreservesForwardOutput) {
    int d_model = 12;
    MultiHeadAttention mha(d_model, 3);
    seed_weights_deterministically(mha, d_model, /*seed=*/29);

    LoRAConfig config;
    config.rank = 3;
    config.alpha = 6.0f;
    mha.enable_lora(config);

    Matrix input(4, d_model);
    for (int i = 0; i < input.rows; ++i) {
        for (int j = 0; j < input.cols; ++j) {
            input(i, j) = 0.03f * (i - j);
        }
    }

    // Train the adapters a bit so ΔW != 0, otherwise this test would pass trivially.
    for (int step = 0; step < 3; ++step) {
        Matrix out = mha.forward(input);
        Matrix grad(out.rows, out.cols);
        for (int i = 0; i < grad.rows; ++i) {
            for (int j = 0; j < grad.cols; ++j) {
                grad(i, j) = 0.05f * (i + 1);
            }
        }
        mha.backward(grad);
        mha.get_lora_q()->update(0.1f);
        mha.get_lora_k()->update(0.1f);
        mha.get_lora_v()->update(0.1f);
        mha.get_lora_o()->update(0.1f);
    }

    Matrix output_before_merge = mha.forward(input);
    mha.merge_lora();
    EXPECT_FALSE(mha.has_lora());
    Matrix output_after_merge = mha.forward(input);

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

TEST(MultiHeadAttentionBackwardTest, BasicBackward) {
    MultiHeadAttention mha(64, 4);

    Matrix input(5, 64);
    for (int i = 0; i < 5; ++i) {
        for (int j = 0; j < 64; ++j) {
            input(i, j) = 0.1f * (i + j);
        }
    }

    Matrix output = mha.forward(input);

    Matrix grad_output(5, 64);
    for (int i = 0; i < 5; ++i) {
        for (int j = 0; j < 64; ++j) {
            grad_output(i, j) = 0.01f;
        }
    }

    Matrix grad_input = mha.backward(grad_output);

    EXPECT_EQ(grad_input.rows, 5);
    EXPECT_EQ(grad_input.cols, 64);
}

TEST(MultiHeadAttentionBackwardTest, GradientDimensionMismatch) {
    MultiHeadAttention mha(64, 4);

    Matrix input(5, 64);
    mha.forward(input);

    Matrix wrong_grad(3, 64);  // Wrong number of rows

    EXPECT_THROW(mha.backward(wrong_grad), std::invalid_argument);
}

TEST(MultiHeadAttentionBackwardTest, GradientAccumulation) {
    MultiHeadAttention mha(64, 4);

    Matrix input(3, 64);
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 64; ++j) {
            input(i, j) = 1.0f;
        }
    }

    mha.forward(input);

    Matrix grad_output(3, 64);
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 64; ++j) {
            grad_output(i, j) = 0.5f;
        }
    }

    mha.backward(grad_output);

    // Gradient should have been accumulated
    float grad_norm = mha.get_gradient_norm();
    EXPECT_GT(grad_norm, 0.0f);
}

TEST(MultiHeadAttentionBackwardTest, MultipleBackwardPasses) {
    MultiHeadAttention mha(64, 4);

    Matrix input(4, 64);
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 64; ++j) {
            input(i, j) = 0.1f;
        }
    }

    Matrix grad_output(4, 64);
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 64; ++j) {
            grad_output(i, j) = 0.1f;
        }
    }

    // First forward and backward
    mha.forward(input);
    mha.backward(grad_output);

    float norm1 = mha.get_gradient_norm();
    EXPECT_GT(norm1, 0.0f);

    // Second forward and backward (without zero_grad()/update_weights() in between)
    // TD-195: backward() accumulates (+=) into W_q_grad/W_k_grad/W_v_grad/W_o_grad rather than
    // overwriting them — a second call with the identical input/grad_output adds the identical
    // gradient a second time, so every element (and therefore the L2 norm) should double, not
    // stay the same. This is the actual, intended behavior gradient-accumulation training
    // (ChatbotTrainer.cpp's own GRADIENT_ACCUMULATION_STEPS) relies on: multiple backward() calls
    // between one zero_grad() and one update_weights() are meant to sum, not replace.
    mha.forward(input);
    mha.backward(grad_output);

    float norm2 = mha.get_gradient_norm();

    EXPECT_NEAR(norm2, 2.0f * norm1, 1e-2f);
}

// ============================================================================
// Weight Update Tests
// ============================================================================

TEST(MultiHeadAttentionUpdateTest, BasicUpdate) {
    MultiHeadAttention mha(64, 4);
    mha.learning_rate = 0.1f;

    Matrix input(3, 64);
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 64; ++j) {
            input(i, j) = 1.0f;
        }
    }

    Matrix grad_output(3, 64);
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 64; ++j) {
            grad_output(i, j) = 0.1f;
        }
    }

    mha.forward(input);
    mha.backward(grad_output);

    float grad_norm_before = mha.get_gradient_norm();
    EXPECT_GT(grad_norm_before, 0.0f);

    mha.update_weights();

    // After update, gradients should be zeroed
    float grad_norm_after = mha.get_gradient_norm();
    EXPECT_FLOAT_EQ(grad_norm_after, 0.0f);
}

TEST(MultiHeadAttentionUpdateTest, LearningRateEffect) {
    // Test that different learning rates produce different results
    MultiHeadAttention mha1(64, 4);
    MultiHeadAttention mha2(64, 4);

    mha1.learning_rate = 0.001f;
    mha2.learning_rate = 0.1f;

    Matrix input(3, 64);
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 64; ++j) {
            input(i, j) = 0.5f;
        }
    }

    Matrix grad_output(3, 64);
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 64; ++j) {
            grad_output(i, j) = 0.1f;
        }
    }

    // Same forward and backward
    mha1.forward(input);
    mha1.backward(grad_output);

    mha2.forward(input);
    mha2.backward(grad_output);

    // Different learning rates should result in different updates
    // (We can't directly compare weights, but we can verify the process works)
    mha1.update_weights();
    mha2.update_weights();

    EXPECT_FLOAT_EQ(mha1.get_gradient_norm(), 0.0f);
    EXPECT_FLOAT_EQ(mha2.get_gradient_norm(), 0.0f);
}

// ============================================================================
// Gradient Monitoring Tests
// ============================================================================

TEST(MultiHeadAttentionGradientTest, ZeroGrad) {
    MultiHeadAttention mha(64, 4);

    Matrix input(3, 64);
    Matrix grad_output(3, 64);

    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 64; ++j) {
            input(i, j) = 1.0f;
            grad_output(i, j) = 0.5f;
        }
    }

    mha.forward(input);
    mha.backward(grad_output);

    EXPECT_GT(mha.get_gradient_norm(), 0.0f);

    mha.zero_grad();

    EXPECT_FLOAT_EQ(mha.get_gradient_norm(), 0.0f);
}

TEST(MultiHeadAttentionGradientTest, GradientNormIncreases) {
    MultiHeadAttention mha(64, 4);

    Matrix input(3, 64);
    Matrix grad_output(3, 64);

    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 64; ++j) {
            input(i, j) = 0.1f;
            grad_output(i, j) = 1.0f;
        }
    }

    // First backward pass
    mha.forward(input);
    mha.backward(grad_output);

    float norm1 = mha.get_gradient_norm();

    // Second backward with larger gradient
    Matrix grad_output2(3, 64);
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 64; ++j) {
            grad_output2(i, j) = 10.0f;  // Larger gradient
        }
    }

    mha.forward(input);
    mha.backward(grad_output2);

    float norm2 = mha.get_gradient_norm();

    // Larger gradient should produce larger gradient norm
    EXPECT_GT(norm2, norm1);
}

TEST(MultiHeadAttentionGradientTest, ClipGradients) {
    MultiHeadAttention mha(64, 4);

    Matrix input(5, 64);
    Matrix grad_output(5, 64);

    for (int i = 0; i < 5; ++i) {
        for (int j = 0; j < 64; ++j) {
            input(i, j) = 1.0f;
            grad_output(i, j) = 10.0f;  // Large gradients
        }
    }

    mha.forward(input);
    mha.backward(grad_output);

    float norm_before = mha.get_gradient_norm();
    EXPECT_GT(norm_before, 5.0f);

    mha.clip_gradients(5.0f);

    float norm_after = mha.get_gradient_norm();
    EXPECT_NEAR(norm_after, 5.0f, 1e-3f);
}

TEST(MultiHeadAttentionGradientTest, ClipGradientsNoEffect) {
    MultiHeadAttention mha(64, 4);

    Matrix input(3, 64);
    Matrix grad_output(3, 64);

    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 64; ++j) {
            input(i, j) = 0.1f;
            grad_output(i, j) = 0.01f;  // Small gradients
        }
    }

    mha.forward(input);
    mha.backward(grad_output);

    float norm_before = mha.get_gradient_norm();
    mha.clip_gradients(100.0f);  // High threshold
    float norm_after = mha.get_gradient_norm();

    // Should be unchanged
    EXPECT_NEAR(norm_before, norm_after, 1e-6f);
}

// ============================================================================
// Persistence Tests
// ============================================================================

TEST(MultiHeadAttentionPersistenceTest, SaveAndLoad) {
    MultiHeadAttention mha1(128, 8);

    Matrix input(5, 128);
    for (int i = 0; i < 5; ++i) {
        for (int j = 0; j < 128; ++j) {
            input(i, j) = 0.1f * i;
        }
    }

    Matrix output1 = mha1.forward(input);

    std::string filename = "test_mha_weights.bin";
    mha1.save_weights(filename);

    MultiHeadAttention mha2(128, 8);
    mha2.load_weights(filename);

    Matrix output2 = mha2.forward(input);

    // Outputs should be identical
    for (int i = 0; i < output1.rows; ++i) {
        for (int j = 0; j < output1.cols; ++j) {
            EXPECT_FLOAT_EQ(output1(i, j), output2(i, j));
        }
    }

    // Clean up
    std::remove(filename.c_str());
}

TEST(MultiHeadAttentionPersistenceTest, LoadDimensionMismatch) {
    MultiHeadAttention mha1(128, 8);
    mha1.save_weights("test_mha_mismatch.bin");

    MultiHeadAttention mha2(256, 8);  // Different d_model
    EXPECT_THROW(mha2.load_weights("test_mha_mismatch.bin"), std::runtime_error);

    MultiHeadAttention mha3(128, 4);  // Different num_heads
    EXPECT_THROW(mha3.load_weights("test_mha_mismatch.bin"), std::runtime_error);

    // Clean up
    std::remove("test_mha_mismatch.bin");
}

TEST(MultiHeadAttentionPersistenceTest, LoadNonexistentFile) {
    MultiHeadAttention mha(128, 8);
    EXPECT_THROW(mha.load_weights("nonexistent_file.bin"), std::runtime_error);
}

// ============================================================================
// Integration Tests
// ============================================================================

TEST(MultiHeadAttentionIntegrationTest, TrainingLoop) {
    MultiHeadAttention mha(64, 4);
    mha.learning_rate = 0.01f;

    Matrix input(5, 64);
    for (int i = 0; i < 5; ++i) {
        for (int j = 0; j < 64; ++j) {
            input(i, j) = 0.1f * (i + j);
        }
    }

    Matrix grad_output(5, 64);
    for (int i = 0; i < 5; ++i) {
        for (int j = 0; j < 64; ++j) {
            grad_output(i, j) = 0.01f;
        }
    }

    // Simulate training for several steps
    for (int step = 0; step < 10; ++step) {
        mha.forward(input);
        mha.backward(grad_output);
        mha.update_weights();
    }

    // After training, gradients should be zero
    EXPECT_FLOAT_EQ(mha.get_gradient_norm(), 0.0f);
}

TEST(MultiHeadAttentionIntegrationTest, ForwardBackwardConsistency) {
    MultiHeadAttention mha(128, 8);

    Matrix input(10, 128);
    for (int i = 0; i < 10; ++i) {
        for (int j = 0; j < 128; ++j) {
            input(i, j) = static_cast<float>(i + j) / 100.0f;
        }
    }

    // Forward pass
    Matrix output1 = mha.forward(input);

    // Create gradient
    Matrix grad_output(10, 128);
    for (int i = 0; i < 10; ++i) {
        for (int j = 0; j < 128; ++j) {
            grad_output(i, j) = 0.01f;
        }
    }

    // Backward pass
    Matrix grad_input = mha.backward(grad_output);

    // Forward again with same input should give same output
    Matrix output2 = mha.forward(input);

    for (int i = 0; i < output1.rows; ++i) {
        for (int j = 0; j < output1.cols; ++j) {
            EXPECT_FLOAT_EQ(output1(i, j), output2(i, j));
        }
    }
}

TEST(MultiHeadAttentionIntegrationTest, WithCausalMaskTraining) {
    MultiHeadAttention mha(64, 4);
    mha.learning_rate = 0.01f;

    int seq_len = 8;
    Matrix input(seq_len, 64);
    for (int i = 0; i < seq_len; ++i) {
        for (int j = 0; j < 64; ++j) {
            input(i, j) = 0.1f * i;
        }
    }

    // Causal mask
    Matrix mask(seq_len, seq_len);
    for (int i = 0; i < seq_len; ++i) {
        for (int j = 0; j < seq_len; ++j) {
            mask(i, j) = (j <= i) ? 1.0f : 0.0f;
        }
    }

    Matrix grad_output(seq_len, 64);
    for (int i = 0; i < seq_len; ++i) {
        for (int j = 0; j < 64; ++j) {
            grad_output(i, j) = 0.01f;
        }
    }

    // Train with mask
    for (int step = 0; step < 5; ++step) {
        mha.forward(input, &mask);
        mha.backward(grad_output);
        mha.update_weights();
    }

    // Verify mask is still respected
    mha.forward(input, &mask);
    const Matrix& attn_weights = mha.get_attention_weights();

    for (int i = 0; i < seq_len; ++i) {
        for (int j = 0; j < seq_len; ++j) {
            if (j > i) {
                EXPECT_NEAR(attn_weights(i, j), 0.0f, 1e-5f);
            }
        }
    }
}

// ============================================================================
// Edge Cases
// ============================================================================

TEST(MultiHeadAttentionEdgeCaseTest, VerySmallModel) {
    MultiHeadAttention mha(16, 2);

    Matrix input(3, 16);
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 16; ++j) {
            input(i, j) = 1.0f;
        }
    }

    Matrix output = mha.forward(input);

    EXPECT_EQ(output.rows, 3);
    EXPECT_EQ(output.cols, 16);
}

TEST(MultiHeadAttentionEdgeCaseTest, SingleTokenSequence) {
    MultiHeadAttention mha(128, 8);

    Matrix input(1, 128);
    for (int j = 0; j < 128; ++j) {
        input(0, j) = 0.5f;
    }

    Matrix output = mha.forward(input);

    EXPECT_EQ(output.rows, 1);
    EXPECT_EQ(output.cols, 128);

    // Attention weights should be [1.0] for single token
    const Matrix& attn_weights = mha.get_attention_weights();
    EXPECT_NEAR(attn_weights(0, 0), 1.0f, 1e-5f);
}

TEST(MultiHeadAttentionEdgeCaseTest, IdenticalInput) {
    MultiHeadAttention mha(64, 4);

    Matrix input(5, 64);
    for (int i = 0; i < 5; ++i) {
        for (int j = 0; j < 64; ++j) {
            input(i, j) = 1.0f;  // All identical
        }
    }

    Matrix output = mha.forward(input);

    EXPECT_EQ(output.rows, 5);
    EXPECT_EQ(output.cols, 64);

    // With identical inputs, attention should be uniform
    const Matrix& attn_weights = mha.get_attention_weights();
    float expected_weight = 1.0f / 5.0f;

    for (int i = 0; i < 5; ++i) {
        for (int j = 0; j < 5; ++j) {
            EXPECT_NEAR(attn_weights(i, j), expected_weight, 0.05f);
        }
    }
}

// ============================================================================
// Optimizer Integration Tests
// ============================================================================

TEST(MultiHeadAttentionOptimizerTest, SetOptimizerBasic) {
    MultiHeadAttention mha(128, 4);
    Optimizer optimizer(OptimizerType::ADAM, 0.001f);

    // Should not throw
    EXPECT_NO_THROW(mha.set_optimizer(&optimizer));
}

TEST(MultiHeadAttentionOptimizerTest, SetOptimizerNullptr) {
    MultiHeadAttention mha(128, 4);

    // Setting nullptr should work (revert to simple gradient descent)
    EXPECT_NO_THROW(mha.set_optimizer(nullptr));
}

TEST(MultiHeadAttentionOptimizerTest, UpdateWithOptimizer) {
    MultiHeadAttention mha(64, 4);
    Optimizer optimizer(OptimizerType::ADAM, 0.01f);
    optimizer.set_betas(0.9f, 0.999f);

    mha.set_optimizer(&optimizer);

    // Create input and perform forward/backward pass
    Matrix input(5, 64);
    for (int i = 0; i < 5; ++i) {
        for (int j = 0; j < 64; ++j) {
            input(i, j) = 0.1f * (i + j);
        }
    }

    Matrix output = mha.forward(input);

    Matrix grad_output(5, 64);
    for (int i = 0; i < 5; ++i) {
        for (int j = 0; j < 64; ++j) {
            grad_output(i, j) = 0.01f;
        }
    }

    mha.backward(grad_output);

    // Update should use optimizer->step()
    EXPECT_NO_THROW(mha.update_weights());
}

TEST(MultiHeadAttentionOptimizerTest, UpdateWithoutOptimizer) {
    MultiHeadAttention mha(64, 4);
    mha.learning_rate = 0.01f;

    // No optimizer set - should use simple gradient descent
    Matrix input(5, 64);
    for (int i = 0; i < 5; ++i) {
        for (int j = 0; j < 64; ++j) {
            input(i, j) = 0.1f * (i + j);
        }
    }

    Matrix output = mha.forward(input);

    Matrix grad_output(5, 64);
    for (int i = 0; i < 5; ++i) {
        for (int j = 0; j < 64; ++j) {
            grad_output(i, j) = 0.01f;
        }
    }

    mha.backward(grad_output);

    // Update should use apply_gradients fallback
    EXPECT_NO_THROW(mha.update_weights());
}

TEST(MultiHeadAttentionOptimizerTest, OptimizerVsSimpleGradientDescent) {
    // Create two identical models
    MultiHeadAttention mha_with_opt(64, 4);
    MultiHeadAttention mha_without_opt(64, 4);

    // Set up optimizer for first model
    Optimizer optimizer(OptimizerType::ADAM, 0.01f);
    mha_with_opt.set_optimizer(&optimizer);

    // Set same learning rate for second model
    mha_without_opt.learning_rate = 0.01f;

    // Create identical input
    Matrix input(5, 64);
    for (int i = 0; i < 5; ++i) {
        for (int j = 0; j < 64; ++j) {
            input(i, j) = 0.1f * (i + j);
        }
    }

    // Forward pass for both
    Matrix output1 = mha_with_opt.forward(input);
    Matrix output2 = mha_without_opt.forward(input);

    // Create gradient
    Matrix grad_output(5, 64);
    for (int i = 0; i < 5; ++i) {
        for (int j = 0; j < 64; ++j) {
            grad_output(i, j) = 0.01f;
        }
    }

    // Backward pass for both
    mha_with_opt.backward(grad_output);
    mha_without_opt.backward(grad_output);

    // Update weights
    mha_with_opt.update_weights();
    mha_without_opt.update_weights();

    // With default Adam settings (beta1=0.9, beta2=0.999), updates will be different
    // This test just verifies both code paths work without errors
    EXPECT_TRUE(true);  // If we got here, both paths worked
}

TEST(MultiHeadAttentionOptimizerTest, MultipleUpdatesWithOptimizer) {
    MultiHeadAttention mha(64, 4);
    Optimizer optimizer(OptimizerType::ADAM, 0.01f);
    optimizer.set_betas(0.9f, 0.999f);

    mha.set_optimizer(&optimizer);

    Matrix input(5, 64);
    for (int i = 0; i < 5; ++i) {
        for (int j = 0; j < 64; ++j) {
            input(i, j) = 0.1f * (i + j);
        }
    }

    Matrix grad_output(5, 64);
    for (int i = 0; i < 5; ++i) {
        for (int j = 0; j < 64; ++j) {
            grad_output(i, j) = 0.01f;
        }
    }

    // Perform multiple training steps
    for (int step = 0; step < 10; ++step) {
        Matrix output = mha.forward(input);
        mha.backward(grad_output);
        mha.update_weights();
    }

    // Should complete without errors
    EXPECT_TRUE(true);
}

TEST(MultiHeadAttentionOptimizerTest, SwitchOptimizer) {
    MultiHeadAttention mha(64, 4);

    Optimizer optimizer1(OptimizerType::ADAM, 0.01f);
    mha.set_optimizer(&optimizer1);

    // Perform some training
    Matrix input(5, 64);
    for (int i = 0; i < 5; ++i) {
        for (int j = 0; j < 64; ++j) {
            input(i, j) = 0.1f;
        }
    }

    Matrix output = mha.forward(input);
    Matrix grad_output(5, 64);
    for (int i = 0; i < 5; ++i) {
        for (int j = 0; j < 64; ++j) {
            grad_output(i, j) = 0.01f;
        }
    }

    mha.backward(grad_output);
    mha.update_weights();

    // Switch to different optimizer
    Optimizer optimizer2(OptimizerType::ADAM, 0.001f);
    optimizer2.set_betas(0.95f, 0.999f);
    mha.set_optimizer(&optimizer2);

    // Continue training with new optimizer
    output = mha.forward(input);
    mha.backward(grad_output);

    EXPECT_NO_THROW(mha.update_weights());
}

TEST(MultiHeadAttentionOptimizerTest, OptimizerWithDifferentLearningRates) {
    MultiHeadAttention mha(64, 4);
    Optimizer optimizer(OptimizerType::ADAM, 0.1f);  // High learning rate

    mha.set_optimizer(&optimizer);

    Matrix input(5, 64);
    for (int i = 0; i < 5; ++i) {
        for (int j = 0; j < 64; ++j) {
            input(i, j) = 0.1f;
        }
    }

    Matrix output1 = mha.forward(input);

    Matrix grad_output(5, 64);
    for (int i = 0; i < 5; ++i) {
        for (int j = 0; j < 64; ++j) {
            grad_output(i, j) = 0.1f;
        }
    }

    mha.backward(grad_output);
    mha.update_weights();

    // After update, output should be different
    Matrix output2 = mha.forward(input);

    bool outputs_different = false;
    for (int i = 0; i < 5; ++i) {
        for (int j = 0; j < 64; ++j) {
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

TEST(MultiHeadAttentionOptimizerTest, OptimizerAfterSaveLoad) {
    MultiHeadAttention mha(64, 4);
    Optimizer optimizer(OptimizerType::ADAM, 0.01f);

    mha.set_optimizer(&optimizer);

    // Perform some training
    Matrix input(5, 64);
    for (int i = 0; i < 5; ++i) {
        for (int j = 0; j < 64; ++j) {
            input(i, j) = 0.1f;
        }
    }

    Matrix output = mha.forward(input);
    Matrix grad_output(5, 64);
    for (int i = 0; i < 5; ++i) {
        for (int j = 0; j < 64; ++j) {
            grad_output(i, j) = 0.01f;
        }
    }

    mha.backward(grad_output);
    mha.update_weights();

    // Save weights
    mha.save_weights("test_mha_optimizer.bin");

    // Create new model and load weights
    MultiHeadAttention mha_loaded(64, 4);
    mha_loaded.load_weights("test_mha_optimizer.bin");

    // Set optimizer on loaded model
    Optimizer optimizer2(OptimizerType::ADAM, 0.01f);
    mha_loaded.set_optimizer(&optimizer2);

    // Should work without issues
    Matrix output2 = mha_loaded.forward(input);
    mha_loaded.backward(grad_output);

    EXPECT_NO_THROW(mha_loaded.update_weights());

    // Clean up
    std::remove("test_mha_optimizer.bin");
}

TEST(MultiHeadAttentionOptimizerTest, RegisterParametersExplicit) {
    MultiHeadAttention mha(64, 4);
    Optimizer optimizer(OptimizerType::ADAM, 0.01f);

    mha.set_optimizer(&optimizer);

    // register_parameters() should be called automatically by set_optimizer()
    // but we can call it again without issues
    EXPECT_NO_THROW(mha.register_parameters());
}

TEST(MultiHeadAttentionOptimizerTest, GradientClippingWithOptimizer) {
    MultiHeadAttention mha(64, 4);
    Optimizer optimizer(OptimizerType::ADAM, 0.1f);
    // Note: Gradient clipping can be done via mha.clip_gradients() before update

    mha.set_optimizer(&optimizer);

    Matrix input(5, 64);
    for (int i = 0; i < 5; ++i) {
        for (int j = 0; j < 64; ++j) {
            input(i, j) = 10.0f;  // Large values
        }
    }

    Matrix output = mha.forward(input);

    Matrix grad_output(5, 64);
    for (int i = 0; i < 5; ++i) {
        for (int j = 0; j < 64; ++j) {
            grad_output(i, j) = 10.0f;  // Large gradients
        }
    }

    mha.backward(grad_output);

    // Update should handle gradient clipping via optimizer
    EXPECT_NO_THROW(mha.update_weights());
}

// ============================================================================
// Configuration Display Test
// ============================================================================

TEST(MultiHeadAttentionConfigTest, PrintConfig) {
    MultiHeadAttention mha(512, 8);

    testing::internal::CaptureStdout();
    mha.print_config();
    std::string output = testing::internal::GetCapturedStdout();

    EXPECT_TRUE(output.find("512") != std::string::npos);
    EXPECT_TRUE(output.find("8") != std::string::npos);
    EXPECT_TRUE(output.find("64") != std::string::npos);  // d_k
}

TEST(MultiHeadAttentionConfigTest, PrintConfigCustomName) {
    MultiHeadAttention mha(256, 4);

    testing::internal::CaptureStdout();
    mha.print_config("MyAttentionLayer");
    std::string output = testing::internal::GetCapturedStdout();

    EXPECT_TRUE(output.find("MyAttentionLayer") != std::string::npos);
}

// ============================================================================
// Attention Hook Tests  (TD-013)
// ============================================================================

TEST(AttentionHookTest, HookFiresOnForward) {
    MultiHeadAttention mha(64, 4);
    int call_count = 0;
    mha.set_attention_hook([&](const Matrix&) { ++call_count; });

    Matrix input(5, 64);
    input.randomize(0.1f);
    mha.forward(input);

    EXPECT_EQ(call_count, 1);
}

TEST(AttentionHookTest, HookReceivesCorrectShape) {
    MultiHeadAttention mha(64, 4);
    int hook_rows = -1, hook_cols = -1;
    mha.set_attention_hook([&](const Matrix& w) {
        hook_rows = w.rows;
        hook_cols = w.cols;
    });

    Matrix input(7, 64);
    input.randomize(0.1f);
    mha.forward(input);

    EXPECT_EQ(hook_rows, 7);  // seq_len × seq_len
    EXPECT_EQ(hook_cols, 7);
}

TEST(AttentionHookTest, ClearedHookDoesNotFire) {
    MultiHeadAttention mha(64, 4);
    int call_count = 0;
    mha.set_attention_hook([&](const Matrix&) { ++call_count; });
    mha.clear_attention_hook();

    Matrix input(5, 64);
    input.randomize(0.1f);
    mha.forward(input);

    EXPECT_EQ(call_count, 0);
}

TEST(AttentionHookTest, ReplacedHookOverridesPrevious) {
    MultiHeadAttention mha(64, 4);
    int old_count = 0, new_count = 0;
    mha.set_attention_hook([&](const Matrix&) { ++old_count; });
    mha.set_attention_hook([&](const Matrix&) { ++new_count; });

    Matrix input(5, 64);
    input.randomize(0.1f);
    mha.forward(input);

    EXPECT_EQ(old_count, 0);
    EXPECT_EQ(new_count, 1);
}

TEST(AttentionHookTest, WeightsAreProbabilityDistribution) {
    // Each row of the attention weight matrix should sum to 1.0 (it's a softmax)
    MultiHeadAttention mha(64, 4);
    bool all_rows_sum_to_one = false;
    mha.set_attention_hook([&](const Matrix& w) {
        all_rows_sum_to_one = true;
        for (int r = 0; r < w.rows; ++r) {
            float row_sum = 0.0f;
            for (int c = 0; c < w.cols; ++c)
                row_sum += w(r, c);
            if (std::abs(row_sum - 1.0f) > 1e-4f) {
                all_rows_sum_to_one = false;
                break;
            }
        }
    });

    Matrix input(6, 64);
    input.randomize(0.1f);
    mha.forward(input);

    EXPECT_TRUE(all_rows_sum_to_one);
}

TEST(AttentionHookTest, EntropyIsNonNegative) {
    // Shannon entropy of a probability distribution is always >= 0
    MultiHeadAttention mha(64, 4);
    float computed_entropy = -999.0f;
    mha.set_attention_hook([&](const Matrix& w) {
        float total = 0.0f;
        const int seq_len = w.rows;
        for (int i = 0; i < seq_len; ++i) {
            float row_h = 0.0f;
            for (int j = 0; j < w.cols; ++j) {
                float a = w(i, j);
                if (a > 0.0f)
                    row_h -= a * std::log(a + 1e-10f);
            }
            total += row_h;
        }
        computed_entropy = total / static_cast<float>(seq_len);
    });

    Matrix input(5, 64);
    input.randomize(0.1f);
    mha.forward(input);

    EXPECT_GE(computed_entropy, 0.0f);
}

TEST(AttentionHookTest, HookCalledOncePerForwardPass) {
    MultiHeadAttention mha(64, 4);
    int call_count = 0;
    mha.set_attention_hook([&](const Matrix&) { ++call_count; });

    Matrix input(5, 64);
    input.randomize(0.1f);
    mha.forward(input);
    mha.forward(input);
    mha.forward(input);

    EXPECT_EQ(call_count, 3);
}

#ifdef ADAI_ENABLE_GPU
// ============================================================================
// GPU Attention-Stats Hook Tests (gpu_forward() path — the fix for the
// attention_entropy-always--1.0 bug: these hooks fire from gpu_forward(),
// which set_attention_hook()'s CPU-only hook never sees).
// ============================================================================

TEST(MultiHeadAttentionGPUStatsHookTest, HookFiresOnGpuForward) {
    MultiHeadAttention mha(64, 4);

    bool hook_called = false;
    mha.set_gpu_attention_stats_hook([&hook_called](float) { hook_called = true; });

    adai::gpu::GPUMatrix input(6, 64);
    input.zero();
    mha.gpu_forward(input);

    EXPECT_TRUE(hook_called);
}

TEST(MultiHeadAttentionGPUStatsHookTest, ClearedHookDoesNotFire) {
    MultiHeadAttention mha(64, 4);

    bool hook_called = false;
    mha.set_gpu_attention_stats_hook([&hook_called](float) { hook_called = true; });
    mha.clear_gpu_attention_stats_hook();

    adai::gpu::GPUMatrix input(6, 64);
    input.zero();
    mha.gpu_forward(input);

    EXPECT_FALSE(hook_called);
}

TEST(MultiHeadAttentionGPUStatsHookTest, EntropyIsNonNegative) {
    // Shannon entropy of a probability distribution (post-softmax attention
    // weights) is always >= 0 — matches the CPU-hook EntropyIsNonNegative test.
    MultiHeadAttention mha(64, 4);

    float reported = -999.0f;
    mha.set_gpu_attention_stats_hook([&reported](float avg_entropy) { reported = avg_entropy; });

    adai::gpu::GPUMatrix input(5, 64);
    input.zero();
    mha.gpu_forward(input);

    EXPECT_GE(reported, 0.0f);
}

TEST(MultiHeadAttentionGPUStatsHookTest, HookCalledOncePerForwardPass) {
    MultiHeadAttention mha(64, 4);

    int call_count = 0;
    mha.set_gpu_attention_stats_hook([&call_count](float) { ++call_count; });

    adai::gpu::GPUMatrix input(5, 64);
    input.zero();
    mha.gpu_forward(input);
    mha.gpu_forward(input);
    mha.gpu_forward(input);

    EXPECT_EQ(call_count, 3);
}
#endif  // ADAI_ENABLE_GPU

// ============================================================================
// Main function
// ============================================================================

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
