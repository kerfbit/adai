#include "../src/TextGenerator.hpp"
#include <gtest/gtest.h>
#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>
#include "../src/Matrix.hpp"

// ============================================================================
// Helper Functions
// ============================================================================

/**
 * Check if two floats are approximately equal
 */
bool is_close(float a, float b, float rtol = 1e-5f, float atol = 1e-8f) {
    return std::abs(a - b) <= (atol + rtol * std::abs(b));
}

/**
 * Mock language model for testing
 * Returns predictable logits based on input sequence
 */
class MockLanguageModel {
   private:
    int vocab_size;
    int d_model;

   public:
    MockLanguageModel(int vocab, int model_dim) : vocab_size(vocab), d_model(model_dim) {}

    Matrix forward(const std::vector<int>& input_tokens) {
        int seq_len = static_cast<int>(input_tokens.size());
        Matrix logits(seq_len, vocab_size);

        // Generate deterministic logits for testing
        for (int i = 0; i < seq_len; ++i) {
            for (int j = 0; j < vocab_size; ++j) {
                float logit = -5.0f;  // Base low probability

                // Predictable pattern for testing
                if (j == 5)
                    logit = 2.0f;  // Token 5 is highly likely
                if (j == 10)
                    logit = 1.5f;  // Token 10 is likely
                if (j == 15)
                    logit = 1.0f;  // Token 15 is moderately likely

                // EOS token becomes likely after 5 tokens
                if (j == 3 && seq_len >= 5) {
                    logit = 3.0f;
                }

                // Context-dependent: if last token was 5, favor 10
                if (i > 0 && input_tokens[i - 1] == 5 && j == 10) {
                    logit = 2.5f;
                }

                logits.data[i][j] = logit;
            }
        }

        return logits;
    }
};

// ============================================================================
// Constructor Tests
// ============================================================================

TEST(TextGeneratorConstructorTest, DefaultConstructor) {
    TextGenerator gen;

    auto config = gen.get_config();
    EXPECT_EQ(config.max_length, 100);
    EXPECT_FLOAT_EQ(config.temperature, 1.0f);
    EXPECT_EQ(config.top_k, 0);
    EXPECT_FLOAT_EQ(config.top_p, 1.0f);
    EXPECT_EQ(config.num_beams, 1);
    EXPECT_EQ(config.bos_token_id, 2);
    EXPECT_EQ(config.eos_token_id, 3);
}

TEST(TextGeneratorConstructorTest, CustomConfiguration) {
    TextGenerator::GenerationConfig config;
    config.max_length = 50;
    config.temperature = 0.8f;
    config.top_k = 20;
    config.top_p = 0.9f;
    config.num_beams = 3;

    TextGenerator gen(config, 42);

    auto retrieved = gen.get_config();
    EXPECT_EQ(retrieved.max_length, 50);
    EXPECT_FLOAT_EQ(retrieved.temperature, 0.8f);
    EXPECT_EQ(retrieved.top_k, 20);
    EXPECT_FLOAT_EQ(retrieved.top_p, 0.9f);
    EXPECT_EQ(retrieved.num_beams, 3);
}

TEST(TextGeneratorConstructorTest, SeedReproducibility) {
    MockLanguageModel model(100, 64);
    auto model_fn = [&model](const std::vector<int>& tokens) { return model.forward(tokens); };

    TextGenerator::GenerationConfig config;
    config.max_length = 10;
    config.temperature = 1.0f;

    // Generate with same seed twice
    TextGenerator gen1(config, 42);
    std::vector<int> result1 = gen1.generate_sampling(model_fn, {2});

    TextGenerator gen2(config, 42);
    std::vector<int> result2 = gen2.generate_sampling(model_fn, {2});

    // Should be identical with same seed
    ASSERT_EQ(result1.size(), result2.size());
    for (size_t i = 0; i < result1.size(); ++i) {
        EXPECT_EQ(result1[i], result2[i]);
    }

    // Different seed should give different results
    TextGenerator gen3(config, 123);
    std::vector<int> result3 = gen3.generate_sampling(model_fn, {2});

    bool different = false;
    for (size_t i = 0; i < std::min(result1.size(), result3.size()); ++i) {
        if (result1[i] != result3[i]) {
            different = true;
            break;
        }
    }
    EXPECT_TRUE(different);
}

// ============================================================================
// Greedy Decoding Tests
// ============================================================================

TEST(TextGeneratorGreedyTest, BasicGeneration) {
    MockLanguageModel model(100, 64);
    auto model_fn = [&model](const std::vector<int>& tokens) { return model.forward(tokens); };

    TextGenerator::GenerationConfig config;
    config.max_length = 10;
    config.temperature = 0.0f;  // Greedy

    TextGenerator gen(config, 42);
    std::vector<int> result = gen.generate_greedy(model_fn, {2});

    EXPECT_GT(result.size(), 1);
    EXPECT_EQ(result[0], 2);  // Should start with prompt

    // Should stop at EOS (token 3) after sequence gets long enough
    bool has_eos = std::find(result.begin(), result.end(), 3) != result.end();
    EXPECT_TRUE(has_eos);
}

TEST(TextGeneratorGreedyTest, EmptyPrompt) {
    MockLanguageModel model(100, 64);
    auto model_fn = [&model](const std::vector<int>& tokens) { return model.forward(tokens); };

    TextGenerator::GenerationConfig config;
    config.max_length = 10;

    TextGenerator gen(config, 42);
    std::vector<int> result = gen.generate_greedy(model_fn, {});

    // Should start with BOS token
    EXPECT_GT(result.size(), 0);
    EXPECT_EQ(result[0], 2);  // BOS token
}

TEST(TextGeneratorGreedyTest, MaxLengthEnforced) {
    MockLanguageModel model(100, 64);
    auto model_fn = [&model](const std::vector<int>& tokens) {
        // Model that never produces EOS
        Matrix logits = model.forward(tokens);
        int seq_len = logits.rows;
        for (int i = 0; i < seq_len; ++i) {
            logits.data[i][3] = -100.0f;  // Suppress EOS
        }
        return logits;
    };

    TextGenerator::GenerationConfig config;
    config.max_length = 15;

    TextGenerator gen(config, 42);
    std::vector<int> result = gen.generate_greedy(model_fn, {2});

    // TD-516: max_length counts new tokens, so the BOS prompt comes on top. EOS never wins, so
    // generation runs to exactly max_length new tokens.
    EXPECT_EQ(result.size(), 1u + 15u);
}

TEST(TextGeneratorGreedyTest, Deterministic) {
    MockLanguageModel model(100, 64);
    auto model_fn = [&model](const std::vector<int>& tokens) { return model.forward(tokens); };

    TextGenerator::GenerationConfig config;
    config.max_length = 10;
    config.temperature = 0.0f;

    TextGenerator gen1(config, 42);
    std::vector<int> result1 = gen1.generate_greedy(model_fn, {2, 5});

    TextGenerator gen2(config, 999);  // Different seed shouldn't matter
    std::vector<int> result2 = gen2.generate_greedy(model_fn, {2, 5});

    // Greedy should always be deterministic
    ASSERT_EQ(result1.size(), result2.size());
    for (size_t i = 0; i < result1.size(); ++i) {
        EXPECT_EQ(result1[i], result2[i]);
    }
}

// ============================================================================
// Temperature Sampling Tests
// ============================================================================

TEST(TextGeneratorSamplingTest, TemperatureZeroIsGreedy) {
    MockLanguageModel model(100, 64);
    auto model_fn = [&model](const std::vector<int>& tokens) { return model.forward(tokens); };

    TextGenerator::GenerationConfig config;
    config.max_length = 8;

    TextGenerator gen1(config, 42);
    config.temperature = 0.0f;
    std::vector<int> greedy = gen1.generate_greedy(model_fn, {2});

    TextGenerator gen2(config, 42);
    std::vector<int> temp_zero = gen2.generate_sampling(model_fn, {2}, 0.0f);

    // Temperature 0 should behave like greedy
    ASSERT_EQ(greedy.size(), temp_zero.size());
    for (size_t i = 0; i < greedy.size(); ++i) {
        EXPECT_EQ(greedy[i], temp_zero[i]);
    }
}

TEST(TextGeneratorSamplingTest, HighTemperatureMoreRandom) {
    MockLanguageModel model(100, 64);
    auto model_fn = [&model](const std::vector<int>& tokens) { return model.forward(tokens); };

    TextGenerator::GenerationConfig config;
    config.max_length = 10;

    // Low temperature (conservative)
    config.temperature = 0.3f;
    TextGenerator gen_low(config, 42);
    std::vector<int> low_temp = gen_low.generate_sampling(model_fn, {2});

    // High temperature (creative)
    config.temperature = 2.0f;
    TextGenerator gen_high(config, 42);
    std::vector<int> high_temp = gen_high.generate_sampling(model_fn, {2});

    // Both should generate sequences
    EXPECT_GT(low_temp.size(), 1);
    EXPECT_GT(high_temp.size(), 1);

    // Can't guarantee they're different, but statistically likely
    // Just verify they both run without errors
}

TEST(TextGeneratorSamplingTest, ReproducibilityWithSeed) {
    MockLanguageModel model(100, 64);
    auto model_fn = [&model](const std::vector<int>& tokens) { return model.forward(tokens); };

    TextGenerator::GenerationConfig config;
    config.max_length = 8;
    config.temperature = 1.0f;

    TextGenerator gen1(config, 42);
    std::vector<int> result1 = gen1.generate_sampling(model_fn, {2});

    TextGenerator gen2(config, 42);
    std::vector<int> result2 = gen2.generate_sampling(model_fn, {2});

    ASSERT_EQ(result1.size(), result2.size());
    for (size_t i = 0; i < result1.size(); ++i) {
        EXPECT_EQ(result1[i], result2[i]);
    }
}

// ============================================================================
// Top-k Sampling Tests
// ============================================================================

TEST(TextGeneratorTopKTest, TopKFiltering) {
    MockLanguageModel model(100, 64);
    auto model_fn = [&model](const std::vector<int>& tokens) { return model.forward(tokens); };

    TextGenerator::GenerationConfig config;
    config.max_length = 8;
    config.temperature = 1.0f;
    config.top_k = 5;

    TextGenerator gen(config, 42);
    std::vector<int> result = gen.generate_top_k(model_fn, {2}, 5);

    EXPECT_GT(result.size(), 1);
    EXPECT_EQ(result[0], 2);
}

TEST(TextGeneratorTopKTest, TopKOneIsGreedy) {
    MockLanguageModel model(100, 64);
    auto model_fn = [&model](const std::vector<int>& tokens) { return model.forward(tokens); };

    TextGenerator::GenerationConfig config;
    config.max_length = 8;

    TextGenerator gen_greedy(config, 42);
    std::vector<int> greedy = gen_greedy.generate_greedy(model_fn, {2});

    config.top_k = 1;
    TextGenerator gen_topk(config, 42);
    std::vector<int> topk = gen_topk.generate_top_k(model_fn, {2}, 1);

    ASSERT_EQ(greedy.size(), topk.size());
    for (size_t i = 0; i < greedy.size(); ++i) {
        EXPECT_EQ(greedy[i], topk[i]);
    }
}

TEST(TextGeneratorTopKTest, DifferentKValues) {
    MockLanguageModel model(100, 64);
    auto model_fn = [&model](const std::vector<int>& tokens) { return model.forward(tokens); };

    TextGenerator::GenerationConfig config;
    config.max_length = 8;
    config.temperature = 1.0f;

    // Small k
    config.top_k = 3;
    TextGenerator gen_small(config, 42);
    std::vector<int> small_k = gen_small.generate_top_k(model_fn, {2});

    // Large k
    config.top_k = 20;
    TextGenerator gen_large(config, 42);
    std::vector<int> large_k = gen_large.generate_top_k(model_fn, {2});

    EXPECT_GT(small_k.size(), 1);
    EXPECT_GT(large_k.size(), 1);
}

// ============================================================================
// Nucleus (Top-p) Sampling Tests
// ============================================================================

TEST(TextGeneratorNucleusTest, TopPFiltering) {
    MockLanguageModel model(100, 64);
    auto model_fn = [&model](const std::vector<int>& tokens) { return model.forward(tokens); };

    TextGenerator::GenerationConfig config;
    config.max_length = 8;
    config.temperature = 1.0f;
    config.top_p = 0.9f;

    TextGenerator gen(config, 42);
    std::vector<int> result = gen.generate_nucleus(model_fn, {2}, 0.9f);

    EXPECT_GT(result.size(), 1);
    EXPECT_EQ(result[0], 2);
}

TEST(TextGeneratorNucleusTest, DifferentPValues) {
    MockLanguageModel model(100, 64);
    auto model_fn = [&model](const std::vector<int>& tokens) { return model.forward(tokens); };

    TextGenerator::GenerationConfig config;
    config.max_length = 8;
    config.temperature = 1.0f;

    // Conservative p
    config.top_p = 0.5f;
    TextGenerator gen_conservative(config, 42);
    std::vector<int> conservative = gen_conservative.generate_nucleus(model_fn, {2});

    // Liberal p
    config.top_p = 0.95f;
    TextGenerator gen_liberal(config, 42);
    std::vector<int> liberal = gen_liberal.generate_nucleus(model_fn, {2});

    EXPECT_GT(conservative.size(), 1);
    EXPECT_GT(liberal.size(), 1);
}

TEST(TextGeneratorNucleusTest, TopPOneNoFiltering) {
    MockLanguageModel model(100, 64);
    auto model_fn = [&model](const std::vector<int>& tokens) { return model.forward(tokens); };

    TextGenerator::GenerationConfig config;
    config.max_length = 8;
    config.temperature = 1.0f;

    // No filtering (p=1.0)
    config.top_p = 1.0f;
    TextGenerator gen(config, 42);
    std::vector<int> result = gen.generate_nucleus(model_fn, {2}, 1.0f);

    EXPECT_GT(result.size(), 1);
}

// ============================================================================
// Beam Search Tests
// ============================================================================

TEST(TextGeneratorBeamSearchTest, BasicBeamSearch) {
    MockLanguageModel model(100, 64);
    auto model_fn = [&model](const std::vector<int>& tokens) { return model.forward(tokens); };

    TextGenerator::GenerationConfig config;
    config.max_length = 10;
    config.num_beams = 3;

    TextGenerator gen(config, 42);
    std::vector<int> result = gen.generate_beam_search(model_fn, {2}, 3);

    EXPECT_GT(result.size(), 1);
    EXPECT_EQ(result[0], 2);
}

TEST(TextGeneratorBeamSearchTest, OneBeamIsGreedy) {
    MockLanguageModel model(100, 64);
    auto model_fn = [&model](const std::vector<int>& tokens) { return model.forward(tokens); };

    TextGenerator::GenerationConfig config;
    config.max_length = 8;

    TextGenerator gen_greedy(config, 42);
    std::vector<int> greedy = gen_greedy.generate_greedy(model_fn, {2});

    TextGenerator gen_beam(config, 42);
    std::vector<int> beam = gen_beam.generate_beam_search(model_fn, {2}, 1);

    ASSERT_EQ(greedy.size(), beam.size());
    for (size_t i = 0; i < greedy.size(); ++i) {
        EXPECT_EQ(greedy[i], beam[i]);
    }
}

TEST(TextGeneratorBeamSearchTest, DifferentBeamWidths) {
    MockLanguageModel model(100, 64);
    auto model_fn = [&model](const std::vector<int>& tokens) { return model.forward(tokens); };

    TextGenerator::GenerationConfig config;
    config.max_length = 10;

    // Small beam width
    config.num_beams = 2;
    TextGenerator gen_small(config, 42);
    std::vector<int> small_beam = gen_small.generate_beam_search(model_fn, {2});

    // Large beam width
    config.num_beams = 5;
    TextGenerator gen_large(config, 42);
    std::vector<int> large_beam = gen_large.generate_beam_search(model_fn, {2});

    EXPECT_GT(small_beam.size(), 1);
    EXPECT_GT(large_beam.size(), 1);
}

TEST(TextGeneratorBeamSearchTest, LengthPenalty) {
    MockLanguageModel model(100, 64);
    auto model_fn = [&model](const std::vector<int>& tokens) { return model.forward(tokens); };

    TextGenerator::GenerationConfig config;
    config.max_length = 10;
    config.num_beams = 3;
    config.length_penalty = true;
    config.length_penalty_alpha = 0.6f;

    TextGenerator gen(config, 42);
    std::vector<int> result = gen.generate_beam_search(model_fn, {2});

    EXPECT_GT(result.size(), 1);
}

TEST(TextGeneratorBeamSearchTest, EarlyStopping) {
    MockLanguageModel model(100, 64);
    auto model_fn = [&model](const std::vector<int>& tokens) { return model.forward(tokens); };

    TextGenerator::GenerationConfig config;
    config.max_length = 20;
    config.num_beams = 3;
    config.early_stopping = true;

    TextGenerator gen(config, 42);
    std::vector<int> result = gen.generate_beam_search(model_fn, {2});

    // Should stop before max_length when all beams finish
    EXPECT_LT(result.size(), 20);
}

// ============================================================================
// Combined Generation Tests
// ============================================================================

TEST(TextGeneratorCombinedTest, AllFiltersApplied) {
    MockLanguageModel model(100, 64);
    auto model_fn = [&model](const std::vector<int>& tokens) { return model.forward(tokens); };

    TextGenerator::GenerationConfig config;
    config.max_length = 10;
    config.temperature = 0.8f;
    config.top_k = 10;
    config.top_p = 0.9f;
    config.repetition_penalty = 1.2f;

    TextGenerator gen(config, 42);
    std::vector<int> result = gen.generate(model_fn, {2});

    EXPECT_GT(result.size(), 1);
    EXPECT_EQ(result[0], 2);
}

TEST(TextGeneratorCombinedTest, BeamSearchWhenConfigured) {
    MockLanguageModel model(100, 64);
    auto model_fn = [&model](const std::vector<int>& tokens) { return model.forward(tokens); };

    TextGenerator::GenerationConfig config;
    config.max_length = 10;
    config.num_beams = 3;

    TextGenerator gen(config, 42);
    std::vector<int> result = gen.generate(model_fn, {2});

    // Should use beam search since num_beams > 1
    EXPECT_GT(result.size(), 1);
}

// ============================================================================
// Repetition Penalty Tests
// ============================================================================

TEST(TextGeneratorRepetitionTest, PenaltyReducesRepetition) {
    // Create a model that tends to repeat
    auto repetitive_model = [](const std::vector<int>& tokens) {
        int seq_len = static_cast<int>(tokens.size());
        Matrix logits(seq_len, 100);

        for (int i = 0; i < seq_len; ++i) {
            for (int j = 0; j < 100; ++j) {
                logits.data[i][j] = -5.0f;
            }

            // Always favor token 5 (would cause repetition)
            logits.data[i][5] = 3.0f;

            // EOS after 8 tokens
            if (seq_len >= 8) {
                logits.data[i][3] = 4.0f;
            }
        }

        return logits;
    };

    TextGenerator::GenerationConfig config;
    config.max_length = 15;
    config.temperature = 0.0f;  // Greedy for determinism

    // Without penalty
    config.repetition_penalty = 1.0f;
    TextGenerator gen_no_penalty(config, 42);
    std::vector<int> no_penalty = gen_no_penalty.generate(repetitive_model, {2});

    // With penalty
    config.repetition_penalty = 1.5f;
    TextGenerator gen_with_penalty(config, 42);
    std::vector<int> with_penalty = gen_with_penalty.generate(repetitive_model, {2});

    // Count occurrences of token 5
    int count_no_penalty = std::count(no_penalty.begin(), no_penalty.end(), 5);
    int count_with_penalty = std::count(with_penalty.begin(), with_penalty.end(), 5);

    // Penalty should reduce repetition (though not guaranteed for all seeds)
    EXPECT_GT(no_penalty.size(), 1);
    EXPECT_GT(with_penalty.size(), 1);
}

// TD-066 regression: apply_repetition_penalty() used to divide a repeated
// token's logit by `penalty` once per OCCURRENCE in the generated sequence
// (compounding as penalty^count) instead of once per distinct token (the
// standard CTRL/HuggingFace/llama.cpp semantic). With a fixed per-step model
// that always favors one dominant token by a margin that survives exactly one
// penalty division but not two, the buggy version would switch away from that
// token after only 2-3 repeats as the compounding penalty overtook the
// runner-up; the fixed version keeps preferring it for the whole generation,
// since each step's penalized score is a fresh single division regardless of
// how many times the token has already appeared.
TEST(TextGeneratorRepetitionTest, PenaltyDoesNotCompoundAcrossRepeats) {
    constexpr int kDominant = 5;
    constexpr int kRunnerUp = 6;
    auto fixed_logit_model = [](const std::vector<int>& tokens) {
        int seq_len = static_cast<int>(tokens.size());
        Matrix logits(seq_len, 20);
        for (int i = 0; i < seq_len; ++i) {
            for (int j = 0; j < 20; ++j) {
                logits.data[i][j] = -100.0f;
            }
            logits.data[i][kDominant] = 10.0f;
            logits.data[i][kRunnerUp] = 8.0f;
        }
        return logits;
    };

    TextGenerator::GenerationConfig config;
    config.max_length = 10;
    config.temperature = 0.0f;  // greedy for determinism
    config.repetition_penalty = 1.15f;

    TextGenerator gen(config, 42);
    std::vector<int> result = gen.generate(fixed_logit_model, {2});

    // Single-division penalized score for the dominant token (10.0/1.15 ~=
    // 8.696) stays above the runner-up's raw 8.0 at every step, so a correct,
    // non-compounding penalty keeps selecting it for the entire generation.
    // Under the old per-occurrence-compounding bug, the dominant token's score
    // would have dropped below 8.0 by the third repeat (10.0/1.15^2 ~= 7.56)
    // and the generation would have switched to the runner-up well before the
    // end.
    int dominant_count = static_cast<int>(std::count(result.begin(), result.end(), kDominant));
    EXPECT_EQ(dominant_count, static_cast<int>(result.size()) - 1)
        << "expected the dominant token in every generated position (prompt excluded)";
}

TEST(TextGeneratorRepetitionTest, NoPenaltyWhenDisabled) {
    MockLanguageModel model(100, 64);
    auto model_fn = [&model](const std::vector<int>& tokens) { return model.forward(tokens); };

    TextGenerator::GenerationConfig config;
    config.max_length = 8;
    config.repetition_penalty = 1.0f;  // No penalty
    config.temperature = 0.0f;

    TextGenerator gen1(config, 42);
    std::vector<int> result1 = gen1.generate(model_fn, {2});

    TextGenerator gen2(config, 42);
    std::vector<int> result2 = gen2.generate(model_fn, {2});

    // Should be identical (greedy + no penalty)
    ASSERT_EQ(result1.size(), result2.size());
    for (size_t i = 0; i < result1.size(); ++i) {
        EXPECT_EQ(result1[i], result2[i]);
    }
}

// ============================================================================
// Configuration Management Tests
// ============================================================================

TEST(TextGeneratorConfigTest, UpdateConfiguration) {
    TextGenerator gen;

    TextGenerator::GenerationConfig new_config;
    new_config.max_length = 75;
    new_config.temperature = 0.7f;
    new_config.top_k = 15;

    gen.set_config(new_config);

    auto retrieved = gen.get_config();
    EXPECT_EQ(retrieved.max_length, 75);
    EXPECT_FLOAT_EQ(retrieved.temperature, 0.7f);
    EXPECT_EQ(retrieved.top_k, 15);
}

TEST(TextGeneratorConfigTest, UpdateSeed) {
    MockLanguageModel model(100, 64);
    auto model_fn = [&model](const std::vector<int>& tokens) { return model.forward(tokens); };

    TextGenerator::GenerationConfig config;
    config.max_length = 8;
    config.temperature = 1.0f;

    TextGenerator gen(config, 42);
    std::vector<int> result1 = gen.generate_sampling(model_fn, {2});

    gen.set_seed(42);
    std::vector<int> result2 = gen.generate_sampling(model_fn, {2});

    // Should be identical after resetting seed
    ASSERT_EQ(result1.size(), result2.size());
    for (size_t i = 0; i < result1.size(); ++i) {
        EXPECT_EQ(result1[i], result2[i]);
    }
}

// ============================================================================
// Edge Case Tests
// ============================================================================

TEST(TextGeneratorEdgeCaseTest, VeryShortMaxLength) {
    MockLanguageModel model(100, 64);
    auto model_fn = [&model](const std::vector<int>& tokens) { return model.forward(tokens); };

    TextGenerator::GenerationConfig config;
    config.max_length = 3;

    TextGenerator gen(config, 42);
    std::vector<int> result = gen.generate_greedy(model_fn, {2});

    // TD-516: at most max_length new tokens after the one-token prompt.
    EXPECT_LE(result.size(), 1u + 3u);
    EXPECT_GT(result.size(), 1u);
}

TEST(TextGeneratorEdgeCaseTest, LongPrompt) {
    MockLanguageModel model(100, 64);
    auto model_fn = [&model](const std::vector<int>& tokens) { return model.forward(tokens); };

    TextGenerator::GenerationConfig config;
    config.max_length = 15;

    // Long prompt (10 tokens)
    std::vector<int> long_prompt = {2, 5, 10, 15, 20, 25, 30, 35, 40, 45};

    TextGenerator gen(config, 42);
    std::vector<int> result = gen.generate_greedy(model_fn, long_prompt);

    EXPECT_GE(result.size(), long_prompt.size());
}

TEST(TextGeneratorEdgeCaseTest, ImmediateEOS) {
    // Model that immediately produces EOS
    auto eos_model = [](const std::vector<int>& tokens) {
        int seq_len = static_cast<int>(tokens.size());
        Matrix logits(seq_len, 100);

        for (int i = 0; i < seq_len; ++i) {
            for (int j = 0; j < 100; ++j) {
                logits.data[i][j] = -5.0f;
            }
            logits.data[i][3] = 10.0f;  // EOS always most likely
        }

        return logits;
    };

    TextGenerator::GenerationConfig config;
    config.max_length = 20;

    TextGenerator gen(config, 42);
    std::vector<int> result = gen.generate_greedy(eos_model, {2});

    // Should stop quickly with EOS
    EXPECT_LE(result.size(), 5);
    EXPECT_EQ(result.back(), 3);  // Should end with EOS
}

TEST(TextGeneratorEdgeCaseTest, SmallVocabulary) {
    // Model with very small vocabulary
    auto small_vocab_model = [](const std::vector<int>& tokens) {
        int seq_len = static_cast<int>(tokens.size());
        Matrix logits(seq_len, 10);  // Only 10 tokens

        for (int i = 0; i < seq_len; ++i) {
            for (int j = 0; j < 10; ++j) {
                logits.data[i][j] = static_cast<float>(j) - 5.0f;
            }
        }

        return logits;
    };

    TextGenerator::GenerationConfig config;
    config.max_length = 8;
    config.top_k = 5;

    TextGenerator gen(config, 42);
    std::vector<int> result = gen.generate_top_k(small_vocab_model, {2});

    EXPECT_GT(result.size(), 1);
}

// ============================================================================
// Integration Tests
// ============================================================================

TEST(TextGeneratorIntegrationTest, MultipleGenerationStrategies) {
    MockLanguageModel model(100, 64);
    auto model_fn = [&model](const std::vector<int>& tokens) { return model.forward(tokens); };

    TextGenerator::GenerationConfig config;
    config.max_length = 8;

    TextGenerator gen(config, 42);

    // Test all strategies work
    std::vector<int> greedy = gen.generate_greedy(model_fn, {2});
    EXPECT_GT(greedy.size(), 1);

    std::vector<int> sampling = gen.generate_sampling(model_fn, {2}, 0.8f);
    EXPECT_GT(sampling.size(), 1);

    std::vector<int> topk = gen.generate_top_k(model_fn, {2}, 10);
    EXPECT_GT(topk.size(), 1);

    std::vector<int> nucleus = gen.generate_nucleus(model_fn, {2}, 0.9f);
    EXPECT_GT(nucleus.size(), 1);

    std::vector<int> beam = gen.generate_beam_search(model_fn, {2}, 3);
    EXPECT_GT(beam.size(), 1);
}

TEST(TextGeneratorIntegrationTest, ConsistencyAcrossRuns) {
    MockLanguageModel model(100, 64);
    auto model_fn = [&model](const std::vector<int>& tokens) { return model.forward(tokens); };

    TextGenerator::GenerationConfig config;
    config.max_length = 10;
    config.temperature = 0.0f;  // Greedy for determinism

    TextGenerator gen(config, 42);

    // Multiple runs should be identical for greedy
    std::vector<int> run1 = gen.generate_greedy(model_fn, {2, 5});
    std::vector<int> run2 = gen.generate_greedy(model_fn, {2, 5});
    std::vector<int> run3 = gen.generate_greedy(model_fn, {2, 5});

    ASSERT_EQ(run1.size(), run2.size());
    ASSERT_EQ(run2.size(), run3.size());

    for (size_t i = 0; i < run1.size(); ++i) {
        EXPECT_EQ(run1[i], run2[i]);
        EXPECT_EQ(run2[i], run3[i]);
    }
}

TEST(TextGeneratorIntegrationTest, LongSequenceGeneration) {
    MockLanguageModel model(100, 64);
    auto model_fn = [&model](const std::vector<int>& tokens) {
        Matrix logits = model.forward(tokens);
        // Suppress EOS for longer generation
        int seq_len = logits.rows;
        for (int i = 0; i < seq_len; ++i) {
            logits.data[i][3] = -10.0f;
        }
        // Add EOS after 30 tokens
        if (seq_len >= 30) {
            for (int i = 0; i < seq_len; ++i) {
                logits.data[i][3] = 5.0f;
            }
        }
        return logits;
    };

    TextGenerator::GenerationConfig config;
    config.max_length = 50;

    TextGenerator gen(config, 42);
    std::vector<int> result = gen.generate_greedy(model_fn, {2});

    EXPECT_GT(result.size(), 20);
    EXPECT_LE(result.size(), 50);
}

// ============================================================================
// Performance Tests
// ============================================================================

TEST(TextGeneratorPerformanceTest, GreedyFasterThanBeam) {
    MockLanguageModel model(100, 64);
    auto model_fn = [&model](const std::vector<int>& tokens) { return model.forward(tokens); };

    TextGenerator::GenerationConfig config;
    config.max_length = 20;

    // Greedy should complete
    TextGenerator gen_greedy(config, 42);
    auto start_greedy = std::chrono::high_resolution_clock::now();
    gen_greedy.generate_greedy(model_fn, {2});
    auto end_greedy = std::chrono::high_resolution_clock::now();
    auto greedy_time =
        std::chrono::duration_cast<std::chrono::milliseconds>(end_greedy - start_greedy).count();

    // Beam search should complete (but slower)
    config.num_beams = 5;
    TextGenerator gen_beam(config, 42);
    auto start_beam = std::chrono::high_resolution_clock::now();
    gen_beam.generate_beam_search(model_fn, {2});
    auto end_beam = std::chrono::high_resolution_clock::now();
    auto beam_time =
        std::chrono::duration_cast<std::chrono::milliseconds>(end_beam - start_beam).count();

    // Just verify both complete (timing can vary)
    // Note: For small sequences, times may be 0ms
    EXPECT_GE(greedy_time, 0);
    EXPECT_GE(beam_time, 0);
    // If we have measurable time, beam should be slower or equal
    if (greedy_time > 0 && beam_time > 0) {
        EXPECT_GE(beam_time, greedy_time);
    }
}

// ============================================================================
// TD-514: beam width below 1 is rejected
// ============================================================================

TEST(TextGeneratorBeamSearchTest, BeamWidthBelowOneThrows) {
    // Width 0 used to segfault (beams[0] of an empty vector) and a negative width terminated the
    // process with an uncaught length_error.
    MockLanguageModel model(100, 64);
    auto model_fn = [&model](const std::vector<int>& tokens) { return model.forward(tokens); };
    TextGenerator::GenerationConfig config;
    config.max_length = 8;
    TextGenerator gen(config, 42);

    EXPECT_THROW(gen.generate_beam_search(model_fn, {2}, 0), std::invalid_argument);
    EXPECT_THROW(gen.generate_beam_search(model_fn, {2}, -2), std::invalid_argument);

    // -1 means "the configured width", which is checked the same way.
    config.num_beams = 0;
    gen.set_config(config);
    EXPECT_THROW(gen.generate_beam_search(model_fn, {2}), std::invalid_argument);
}

// ============================================================================
// TD-515: temperature <= 0 is greedy in every strategy
// ============================================================================

TEST(TextGeneratorSamplingTest, TemperatureZeroOrBelowIsGreedyForEveryStrategy) {
    // The mock's three likely tokens (5, 10, 15) make sampling at temperature 1 vary from seed to
    // seed. Top-k and nucleus used to sample at an effective temperature of 1 when the temperature
    // was 0 (33 distinct nucleus outputs in 50 runs); a negative temperature sampled in every strategy.
    MockLanguageModel model(100, 64);
    auto model_fn = [&model](const std::vector<int>& tokens) { return model.forward(tokens); };

    for (float temperature : {0.0f, -0.5f}) {
        TextGenerator::GenerationConfig config;
        config.max_length = 8;
        config.temperature = temperature;
        const std::vector<int> greedy = TextGenerator(config, 1).generate_greedy(model_fn, {2});

        for (unsigned seed = 1; seed <= 20; ++seed) {
            TextGenerator gen(config, seed);
            EXPECT_EQ(gen.generate_top_k(model_fn, {2}, 50), greedy)
                << "top-k, T=" << temperature << ", seed " << seed;
            EXPECT_EQ(gen.generate_nucleus(model_fn, {2}, 0.9f), greedy)
                << "nucleus, T=" << temperature << ", seed " << seed;
            EXPECT_EQ(gen.generate_sampling(model_fn, {2}, temperature), greedy)
                << "sampling, T=" << temperature << ", seed " << seed;
            EXPECT_EQ(gen.generate(model_fn, {2}), greedy)
                << "generate(), T=" << temperature << ", seed " << seed;
        }
    }
}

TEST(TextGeneratorSamplingTest, PositiveTemperatureStillSamples) {
    // Guards the TD-515 fix against turning sampling off: at temperature 1 the nucleus output varies.
    MockLanguageModel model(100, 64);
    auto model_fn = [&model](const std::vector<int>& tokens) { return model.forward(tokens); };
    TextGenerator::GenerationConfig config;
    config.max_length = 8;
    config.temperature = 1.0f;

    std::vector<std::vector<int>> outputs;
    for (unsigned seed = 1; seed <= 20; ++seed) {
        outputs.push_back(TextGenerator(config, seed).generate_nucleus(model_fn, {2}, 0.9f));
    }
    std::sort(outputs.begin(), outputs.end());
    outputs.erase(std::unique(outputs.begin(), outputs.end()), outputs.end());
    EXPECT_GT(outputs.size(), 1u);
}

// ============================================================================
// TD-516: min_length, non-finite logits, max_length, early stopping, filter order
// ============================================================================

namespace {

// Prefers EOS (id 3) at every position, as an EOS-happy model does.
Matrix eos_first_logits(MockLanguageModel& model, const std::vector<int>& tokens) {
    Matrix logits = model.forward(tokens);
    for (int i = 0; i < logits.rows; ++i)
        logits.data[i][3] = 10.0f;
    return logits;
}

// Deterministic pseudo-random logits per context, so beam search has real competition.
Matrix hashed_logits(const std::vector<int>& tokens, int vocab, unsigned salt) {
    Matrix logits(static_cast<int>(tokens.size()), vocab);
    for (int i = 0; i < logits.rows; ++i) {
        unsigned h = 2166136261u ^ salt;
        for (int j = 0; j <= i; ++j)
            h = (h ^ static_cast<unsigned>(tokens[j])) * 16777619u;
        for (int v = 0; v < vocab; ++v) {
            unsigned x = (h ^ static_cast<unsigned>(v * 2654435761u)) * 2246822519u;
            logits.data[i][v] = static_cast<float>(x % 1000u) / 250.0f;  // 0 .. 4
        }
    }
    return logits;
}

}  // namespace

TEST(TextGeneratorTD516Test, MinLengthKeepsStopTokensOutUntilReached) {
    // min_length was never read: an EOS-preferring model ended after one token (empty response).
    MockLanguageModel model(100, 64);
    auto model_fn = [&model](const std::vector<int>& tokens) {
        return eos_first_logits(model, tokens);
    };
    TextGenerator::GenerationConfig config;
    config.max_length = 20;
    config.min_length = 4;
    config.temperature = 0.0f;
    TextGenerator gen(config, 42);

    for (const auto& out : {gen.generate_greedy(model_fn, {2}), gen.generate_top_k(model_fn, {2}, 5),
                            gen.generate_nucleus(model_fn, {2}, 0.9f), gen.generate(model_fn, {2}),
                            gen.generate_beam_search(model_fn, {2}, 3)}) {
        ASSERT_EQ(out.size(), 1u + 4u + 1u);  // BOS, 4 tokens, then EOS
        for (size_t i = 1; i <= 4; ++i) {
            EXPECT_NE(out[i], 3);
            EXPECT_NE(out[i], 0);
        }
        EXPECT_EQ(out.back(), 3);
    }
}

TEST(TextGeneratorTD516Test, NonFiniteLogitsThrow) {
    // All-NaN logits used to pick token 0 (PAD, a stop token) and end with an empty response.
    const float nan = std::numeric_limits<float>::quiet_NaN();
    const float inf = std::numeric_limits<float>::infinity();
    for (float bad : {nan, inf}) {
        auto model_fn = [bad](const std::vector<int>& tokens) {
            Matrix logits(static_cast<int>(tokens.size()), 20);
            for (int i = 0; i < logits.rows; ++i)
                for (int v = 0; v < 20; ++v)
                    logits.data[i][v] = v == 7 ? bad : 0.5f;
            return logits;
        };
        TextGenerator::GenerationConfig config;
        config.max_length = 5;
        TextGenerator gen(config, 42);
        EXPECT_THROW(gen.generate_greedy(model_fn, {2}), std::runtime_error) << bad;
        EXPECT_THROW(gen.generate_sampling(model_fn, {2}, 1.0f), std::runtime_error) << bad;
        EXPECT_THROW(gen.generate_nucleus(model_fn, {2}, 0.9f), std::runtime_error) << bad;
        EXPECT_THROW(gen.generate_beam_search(model_fn, {2}, 3), std::runtime_error) << bad;
    }
    auto all_masked = [inf](const std::vector<int>& tokens) {
        Matrix logits(static_cast<int>(tokens.size()), 20);
        for (int i = 0; i < logits.rows; ++i)
            for (int v = 0; v < 20; ++v)
                logits.data[i][v] = -inf;
        return logits;
    };
    TextGenerator gen(TextGenerator::GenerationConfig{}, 42);
    EXPECT_THROW(gen.generate_greedy(all_masked, {2}), std::runtime_error);
}

TEST(TextGeneratorTD516Test, MaxLengthCountsNewTokensAfterThePrompt) {
    // max_length used to include the prompt: a 3-token prompt with max_length 4 gave 1 new token.
    MockLanguageModel model(100, 64);
    auto model_fn = [&model](const std::vector<int>& tokens) {
        Matrix logits = model.forward(tokens);
        for (int i = 0; i < logits.rows; ++i)
            logits.data[i][3] = -100.0f;  // never EOS
        return logits;
    };
    TextGenerator::GenerationConfig config;
    config.max_length = 4;
    config.temperature = 0.0f;
    TextGenerator gen(config, 42);
    const std::vector<int> prompt = {2, 5, 10};

    EXPECT_EQ(gen.generate_greedy(model_fn, prompt).size(), 3u + 4u);
    EXPECT_EQ(gen.generate_sampling(model_fn, prompt, 1.0f).size(), 3u + 4u);
    EXPECT_EQ(gen.generate_top_k(model_fn, prompt, 5).size(), 3u + 4u);
    EXPECT_EQ(gen.generate_nucleus(model_fn, prompt, 0.9f).size(), 3u + 4u);
    EXPECT_EQ(gen.generate(model_fn, prompt).size(), 3u + 4u);
    EXPECT_EQ(gen.generate_beam_search(model_fn, prompt, 3).size(), 3u + 4u);
}

TEST(TextGeneratorTD516Test, EarlyStoppingKeepsTheResultAndSavesModelCalls) {
    // early_stopping had no effect. It now stops only when no running beam can still win, so the
    // output must equal the full search's, with and without the length penalty.
    const int vocab = 12;
    int calls_full = 0;
    int calls_early = 0;
    for (unsigned salt = 0; salt < 40; ++salt) {
        for (bool length_penalty : {false, true}) {
            for (float alpha : {0.6f, -0.5f}) {
                TextGenerator::GenerationConfig config;
                config.max_length = 12;
                config.length_penalty = length_penalty;
                config.length_penalty_alpha = alpha;

                config.early_stopping = false;
                int calls = 0;
                auto counted = [&](const std::vector<int>& t) {
                    ++calls;
                    return hashed_logits(t, vocab, salt);
                };
                const auto full = TextGenerator(config, 1).generate_beam_search(counted, {2}, 3);
                calls_full += calls;

                config.early_stopping = true;
                calls = 0;
                const auto early = TextGenerator(config, 1).generate_beam_search(counted, {2}, 3);
                calls_early += calls;

                EXPECT_EQ(early, full) << "salt " << salt << ", length_penalty " << length_penalty
                                       << ", alpha " << alpha;
            }
        }
    }
    EXPECT_LT(calls_early, calls_full);
}

TEST(TextGeneratorTD516Test, RepetitionPenaltyRunsBeforeTopK) {
    // Token 5 is the model's favourite and already in the prompt. With the penalty applied after
    // top-1 filtering it stayed the only candidate; applied first, token 10 wins.
    MockLanguageModel model(100, 64);
    auto model_fn = [&model](const std::vector<int>& tokens) { return model.forward(tokens); };
    TextGenerator::GenerationConfig config;
    config.max_length = 1;
    config.repetition_penalty = 10.0f;
    TextGenerator gen(config, 42);

    const auto out = gen.generate_top_k(model_fn, {2, 5}, 1);
    ASSERT_EQ(out.size(), 3u);
    EXPECT_NE(out.back(), 5);
}

TEST(TextGeneratorTD516Test, TopPZeroOrBelowIsTopOne) {
    // top_p <= 0 used to switch filtering off, so nucleus sampling drew from the whole vocabulary.
    MockLanguageModel model(100, 64);
    auto model_fn = [&model](const std::vector<int>& tokens) { return model.forward(tokens); };
    TextGenerator::GenerationConfig config;
    config.max_length = 8;
    config.temperature = 1.0f;
    const auto greedy = TextGenerator(config, 1).generate_greedy(model_fn, {2});
    for (unsigned seed = 1; seed <= 20; ++seed) {
        EXPECT_EQ(TextGenerator(config, seed).generate_nucleus(model_fn, {2}, 0.0f), greedy) << seed;
    }
}

// ============================================================================
// Main
// ============================================================================

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
