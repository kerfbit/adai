/**
 * Unit tests for SpeculativeDecoding
 * Tests: SpeculativeDecodingConfig, TokenProposal, expected_tokens_per_target_pass,
 *        SpeculativeDecoder lifecycle, stats, config access, and functional generation.
 *
 * Strategy:
 *  - TextGenerator instances are set up with an EOS-returning model function
 *    (logit for token EOS = 100.0f ≫ all others).  This makes every proposal
 *    be EOS so generation terminates on the first step, without real model weights.
 *  - Default-constructed BPETokenizer provides special tokens {PAD,UNK,BOS,EOS}.
 */

#include "../src/SpeculativeDecoding.hpp"
#include <gtest/gtest.h>
#include "../src/BPETokenizer.hpp"
#include "../src/Matrix.hpp"
#include "../src/SpecialTokens.hpp"
#include "../src/TextGenerator.hpp"

#include <cmath>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

// ============================================================================
// Helpers
// ============================================================================

static constexpr int kVocabSize = 10;

/// Model forward function: always makes EOS (id=3) the highest logit.
static Matrix eos_model(const std::vector<int>& /*tokens*/) {
    Matrix logits(1, kVocabSize);
    logits(0, adai::SpecialTokenIDs::EOS) = 100.0f;
    return logits;
}

/// Build a TextGenerator configured to always emit EOS.
static std::unique_ptr<TextGenerator> make_eos_generator(BPETokenizer* tok) {
    TextGenerator::GenerationConfig cfg;
    cfg.max_length = 50;
    cfg.temperature = 1.0f;
    auto gen = std::make_unique<TextGenerator>(cfg, 42);
    gen->set_model_fn(eos_model);
    gen->set_tokenizer(tok);
    return gen;
}

// ============================================================================
// SpeculativeDecodingConfig Tests
// ============================================================================

TEST(SpeculativeDecodingConfigTest, DefaultNumCandidates) {
    SpeculativeDecodingConfig cfg;
    EXPECT_EQ(cfg.num_candidates, 4);
}

TEST(SpeculativeDecodingConfigTest, DefaultTemperature) {
    SpeculativeDecodingConfig cfg;
    EXPECT_FLOAT_EQ(cfg.temperature, 1.0f);
}

TEST(SpeculativeDecodingConfigTest, DefaultMaxLength) {
    SpeculativeDecodingConfig cfg;
    EXPECT_EQ(cfg.max_length, 100);
}

TEST(SpeculativeDecodingConfigTest, DefaultUseGreedy) {
    SpeculativeDecodingConfig cfg;
    EXPECT_FALSE(cfg.use_greedy);
}

TEST(SpeculativeDecodingConfigTest, CustomValues) {
    SpeculativeDecodingConfig cfg;
    cfg.num_candidates = 8;
    cfg.temperature = 0.7f;
    cfg.max_length = 200;
    cfg.use_greedy = true;

    EXPECT_EQ(cfg.num_candidates, 8);
    EXPECT_FLOAT_EQ(cfg.temperature, 0.7f);
    EXPECT_EQ(cfg.max_length, 200);
    EXPECT_TRUE(cfg.use_greedy);
}

TEST(SpeculativeDecodingConfigTest, DefaultConstructible) {
    SpeculativeDecodingConfig* cfg = new SpeculativeDecodingConfig();
    EXPECT_EQ(cfg->num_candidates, 4);
    delete cfg;
}

// ============================================================================
// TokenProposal Tests
// ============================================================================

TEST(TokenProposalTest, DefaultConstruction) {
    TokenProposal p;
    EXPECT_EQ(p.token_id, 0);
    EXPECT_FLOAT_EQ(p.draft_prob, 0.0f);
    EXPECT_FLOAT_EQ(p.target_prob, 0.0f);
    EXPECT_FALSE(p.accepted);
}

TEST(TokenProposalTest, ConstructionWithIdAndProb) {
    TokenProposal p(5, 0.8f);
    EXPECT_EQ(p.token_id, 5);
    EXPECT_FLOAT_EQ(p.draft_prob, 0.8f);
    EXPECT_FLOAT_EQ(p.target_prob, 0.0f);
    EXPECT_FALSE(p.accepted);
}

TEST(TokenProposalTest, SetTargetProb) {
    TokenProposal p(3, 0.5f);
    p.target_prob = 0.9f;
    EXPECT_FLOAT_EQ(p.target_prob, 0.9f);
}

TEST(TokenProposalTest, SetAccepted) {
    TokenProposal p(2, 0.3f);
    p.accepted = true;
    EXPECT_TRUE(p.accepted);
}

TEST(TokenProposalTest, CopyConstructible) {
    TokenProposal p(7, 0.6f);
    p.target_prob = 0.4f;
    p.accepted = true;

    TokenProposal q = p;
    EXPECT_EQ(q.token_id, 7);
    EXPECT_FLOAT_EQ(q.draft_prob, 0.6f);
    EXPECT_FLOAT_EQ(q.target_prob, 0.4f);
    EXPECT_TRUE(q.accepted);
}

// ============================================================================
// Free Function: expected_tokens_per_target_pass (TD-508)
// ============================================================================
//
// Replaces calculate_theoretical_speedup(), whose K·α/(K + 1) was below 1 for every acceptance rate.

TEST(ExpectedTokensPerTargetPassTest, NoCandidatesOrNoAcceptanceIsOne) {
    EXPECT_FLOAT_EQ(expected_tokens_per_target_pass(0, 0.8f), 1.0f);
    EXPECT_FLOAT_EQ(expected_tokens_per_target_pass(-1, 0.5f), 1.0f);
    EXPECT_FLOAT_EQ(expected_tokens_per_target_pass(4, 0.0f), 1.0f);
}

TEST(ExpectedTokensPerTargetPassTest, FullAcceptanceIsKPlusOne) {
    EXPECT_FLOAT_EQ(expected_tokens_per_target_pass(4, 1.0f), 5.0f);
}

TEST(ExpectedTokensPerTargetPassTest, MatchesTheGeometricSeries) {
    // (1 − α^(K+1)) / (1 − α) = 1 + α + … + α^K
    for (int K : {1, 2, 4, 8}) {
        for (float alpha : {0.3f, 0.5f, 0.8f}) {
            float series = 0.0f;
            for (int i = 0; i <= K; ++i)
                series += std::pow(alpha, static_cast<float>(i));
            EXPECT_NEAR(expected_tokens_per_target_pass(K, alpha), series, 1e-4f) << K << " " << alpha;
            EXPECT_GE(expected_tokens_per_target_pass(K, alpha), 1.0f);
        }
    }
}

TEST(ExpectedTokensPerTargetPassTest, MonotonicInKAndAlpha) {
    EXPECT_GT(expected_tokens_per_target_pass(8, 0.8f), expected_tokens_per_target_pass(4, 0.8f));
    EXPECT_GT(expected_tokens_per_target_pass(4, 0.9f), expected_tokens_per_target_pass(4, 0.3f));
}

// ============================================================================
// SpeculativeDecoder Construction Tests
// ============================================================================

class SpeculativeDecoderTest : public ::testing::Test {
   protected:
    void SetUp() override {
        tokenizer_ = std::make_unique<BPETokenizer>();
        draft_ = make_eos_generator(tokenizer_.get());
        target_ = make_eos_generator(tokenizer_.get());
    }

    std::unique_ptr<BPETokenizer> tokenizer_;
    std::unique_ptr<TextGenerator> draft_;
    std::unique_ptr<TextGenerator> target_;
};

TEST_F(SpeculativeDecoderTest, ConstructionThrowsOnNullDraft) {
    EXPECT_THROW(SpeculativeDecoder(nullptr, target_.get()), std::invalid_argument);
}

TEST_F(SpeculativeDecoderTest, ConstructionThrowsOnNullTarget) {
    EXPECT_THROW(SpeculativeDecoder(draft_.get(), nullptr), std::invalid_argument);
}

TEST_F(SpeculativeDecoderTest, ConstructionThrowsOnBothNull) {
    EXPECT_THROW(SpeculativeDecoder(nullptr, nullptr), std::invalid_argument);
}

TEST_F(SpeculativeDecoderTest, ConstructionSucceedsWithValidGenerators) {
    EXPECT_NO_THROW(SpeculativeDecoder(draft_.get(), target_.get()));
}

TEST_F(SpeculativeDecoderTest, ConstructionWithCustomConfig) {
    SpeculativeDecodingConfig cfg;
    cfg.num_candidates = 6;
    cfg.temperature = 0.8f;
    cfg.use_greedy = true;

    EXPECT_NO_THROW(SpeculativeDecoder(draft_.get(), target_.get(), cfg));
}

// ============================================================================
// SpeculativeDecoder Config Access Tests
// ============================================================================

TEST_F(SpeculativeDecoderTest, GetConfigReturnsConstructedConfig) {
    SpeculativeDecodingConfig cfg;
    cfg.num_candidates = 7;
    cfg.max_length = 50;
    cfg.temperature = 0.5f;

    SpeculativeDecoder decoder(draft_.get(), target_.get(), cfg);
    const auto& got = decoder.get_config();
    EXPECT_EQ(got.num_candidates, 7);
    EXPECT_EQ(got.max_length, 50);
    EXPECT_FLOAT_EQ(got.temperature, 0.5f);
}

TEST_F(SpeculativeDecoderTest, SetConfigUpdatesConfig) {
    SpeculativeDecoder decoder(draft_.get(), target_.get());

    SpeculativeDecodingConfig newcfg;
    newcfg.num_candidates = 10;
    newcfg.use_greedy = true;
    decoder.set_config(newcfg);

    EXPECT_EQ(decoder.get_config().num_candidates, 10);
    EXPECT_TRUE(decoder.get_config().use_greedy);
}

TEST_F(SpeculativeDecoderTest, SetConfigPreservesAllFields) {
    SpeculativeDecoder decoder(draft_.get(), target_.get());

    SpeculativeDecodingConfig cfg;
    cfg.num_candidates = 3;
    cfg.temperature = 1.5f;
    cfg.max_length = 30;
    cfg.use_greedy = false;
    decoder.set_config(cfg);

    const auto& got = decoder.get_config();
    EXPECT_EQ(got.num_candidates, 3);
    EXPECT_FLOAT_EQ(got.temperature, 1.5f);
    EXPECT_EQ(got.max_length, 30);
    EXPECT_FALSE(got.use_greedy);
}

TEST_F(SpeculativeDecoderTest, FewerThanOneCandidateIsRejected) {
    // K = 0 used to propose nothing and end generation with zero tokens.
    SpeculativeDecodingConfig cfg;
    cfg.num_candidates = 0;
    EXPECT_THROW(SpeculativeDecoder(draft_.get(), target_.get(), cfg), std::invalid_argument);
    SpeculativeDecoder decoder(draft_.get(), target_.get());
    EXPECT_THROW(decoder.set_config(cfg), std::invalid_argument);
}

// ============================================================================
// SpeculativeDecoder Initial Statistics Tests
// ============================================================================

TEST_F(SpeculativeDecoderTest, AcceptanceRateInitiallyZero) {
    SpeculativeDecoder decoder(draft_.get(), target_.get());
    EXPECT_FLOAT_EQ(decoder.get_acceptance_rate(), 0.0f);
}

TEST_F(SpeculativeDecoderTest, TokensPerTargetPassInitiallyZero) {
    SpeculativeDecoder decoder(draft_.get(), target_.get());
    EXPECT_FLOAT_EQ(decoder.get_tokens_per_target_pass(), 0.0f);
}

TEST_F(SpeculativeDecoderTest, ResetStatsDoesNotCrash) {
    SpeculativeDecoder decoder(draft_.get(), target_.get());
    EXPECT_NO_THROW(decoder.reset_stats());
}

TEST_F(SpeculativeDecoderTest, ResetStatsPreservesZeroState) {
    SpeculativeDecoder decoder(draft_.get(), target_.get());
    decoder.reset_stats();
    EXPECT_FLOAT_EQ(decoder.get_acceptance_rate(), 0.0f);
    EXPECT_FLOAT_EQ(decoder.get_tokens_per_target_pass(), 0.0f);
}

TEST_F(SpeculativeDecoderTest, LogStatsDoesNotCrash) {
    SpeculativeDecoder decoder(draft_.get(), target_.get());
    EXPECT_NO_THROW(decoder.log_stats());
}

// ============================================================================
// TextGenerator get_next_token_probs Tests
// ============================================================================

TEST(TextGeneratorExtensionTest, GetNextTokenProbsThrowsWithoutModelFn) {
    BPETokenizer tok;
    TextGenerator gen;
    gen.set_tokenizer(&tok);
    // No model fn set → should throw
    EXPECT_THROW(gen.get_next_token_probs({1, 2, 3}), std::runtime_error);
}

TEST(TextGeneratorExtensionTest, GetNextTokenProbsReturnsProbabilityVector) {
    BPETokenizer tok;
    TextGenerator::GenerationConfig cfg;
    TextGenerator gen(cfg, 0);
    gen.set_model_fn([](const std::vector<int>&) {
        Matrix m(1, kVocabSize);
        m(0, 0) = 1.0f;
        m(0, 1) = 2.0f;
        m(0, 2) = 3.0f;
        return m;
    });
    gen.set_tokenizer(&tok);

    auto probs = gen.get_next_token_probs({1});
    EXPECT_EQ(static_cast<int>(probs.size()), kVocabSize);

    // Probabilities should sum to ~1.0
    float sum = 0.0f;
    for (float p : probs)
        sum += p;
    EXPECT_NEAR(sum, 1.0f, 1e-4f);
}

TEST(TextGeneratorExtensionTest, GetNextTokenProbsNonNegative) {
    BPETokenizer tok;
    TextGenerator::GenerationConfig cfg;
    TextGenerator gen(cfg, 0);
    gen.set_model_fn(eos_model);
    gen.set_tokenizer(&tok);

    auto probs = gen.get_next_token_probs({1, 2});
    for (float p : probs) {
        EXPECT_GE(p, 0.0f);
    }
}

TEST(TextGeneratorExtensionTest, GetTokenizerReturnsSetPointer) {
    BPETokenizer tok;
    TextGenerator gen;
    EXPECT_EQ(gen.get_tokenizer(), nullptr);  // Initially null
    gen.set_tokenizer(&tok);
    EXPECT_EQ(gen.get_tokenizer(), &tok);
}

TEST(TextGeneratorExtensionTest, EosModelProducesHighProbForEos) {
    BPETokenizer tok;
    TextGenerator gen;
    gen.set_model_fn(eos_model);
    gen.set_tokenizer(&tok);

    auto probs = gen.get_next_token_probs({1});
    int eos_id = adai::SpecialTokenIDs::EOS;
    // EOS should dominate with logit=100
    EXPECT_GT(probs[eos_id], 0.99f);
}

// ============================================================================
// SpeculativeDecoder Functional Tests (EOS model stops generation immediately)
// ============================================================================

class SpeculativeDecoderFunctionalTest : public ::testing::Test {
   protected:
    void SetUp() override {
        tokenizer_ = std::make_unique<BPETokenizer>();

        SpeculativeDecodingConfig cfg;
        cfg.num_candidates = 2;
        cfg.max_length = 10;
        cfg.use_greedy = true;  // Deterministic for testing
        config_ = cfg;

        draft_ = make_eos_generator(tokenizer_.get());
        target_ = make_eos_generator(tokenizer_.get());
    }

    std::unique_ptr<BPETokenizer> tokenizer_;
    std::unique_ptr<TextGenerator> draft_;
    std::unique_ptr<TextGenerator> target_;
    SpeculativeDecodingConfig config_;
};

TEST_F(SpeculativeDecoderFunctionalTest, GenerateTokensDoesNotCrash) {
    SpeculativeDecoder decoder(draft_.get(), target_.get(), config_);
    EXPECT_NO_THROW(decoder.generate_tokens({adai::SpecialTokenIDs::BOS}));
}

TEST_F(SpeculativeDecoderFunctionalTest, GenerateTokensReturnsVector) {
    SpeculativeDecoder decoder(draft_.get(), target_.get(), config_);
    auto tokens = decoder.generate_tokens({adai::SpecialTokenIDs::BOS});
    // EOS model → first proposal = EOS → should stop
    EXPECT_GE(tokens.size(), 1u);
}

TEST_F(SpeculativeDecoderFunctionalTest, GenerateTokensRespectsMaxNewTokens) {
    SpeculativeDecoder decoder(draft_.get(), target_.get(), config_);
    auto tokens = decoder.generate_tokens({adai::SpecialTokenIDs::BOS}, 0);
    EXPECT_TRUE(tokens.empty());
}

TEST_F(SpeculativeDecoderFunctionalTest, GenerateTokensWithEmptyPrompt) {
    SpeculativeDecoder decoder(draft_.get(), target_.get(), config_);
    EXPECT_NO_THROW(decoder.generate_tokens({}));
}

TEST_F(SpeculativeDecoderFunctionalTest, GenerateDoesNotCrash) {
    SpeculativeDecoder decoder(draft_.get(), target_.get(), config_);
    EXPECT_NO_THROW(decoder.generate("hello"));
}

TEST_F(SpeculativeDecoderFunctionalTest, GenerateReturnsString) {
    SpeculativeDecoder decoder(draft_.get(), target_.get(), config_);
    std::string result = decoder.generate("test");
    // Result is a string — may be empty or decoded tokens; just shouldn't throw
    EXPECT_NO_THROW((void)result.size());
}

TEST_F(SpeculativeDecoderFunctionalTest, GenerateEmptyPromptThrowsTokenizerError) {
    // BPETokenizer::encode() rejects empty strings — expected behavior.
    SpeculativeDecoder decoder(draft_.get(), target_.get(), config_);
    EXPECT_THROW(decoder.generate(""), std::exception);
}

TEST_F(SpeculativeDecoderFunctionalTest, ResetStatsAfterGeneration) {
    SpeculativeDecoder decoder(draft_.get(), target_.get(), config_);
    decoder.generate_tokens({adai::SpecialTokenIDs::BOS});
    decoder.reset_stats();
    EXPECT_FLOAT_EQ(decoder.get_acceptance_rate(), 0.0f);
    EXPECT_FLOAT_EQ(decoder.get_tokens_per_target_pass(), 0.0f);
}

TEST_F(SpeculativeDecoderFunctionalTest, AcceptanceRateInRange) {
    SpeculativeDecoder decoder(draft_.get(), target_.get(), config_);
    decoder.generate_tokens({adai::SpecialTokenIDs::BOS});
    float rate = decoder.get_acceptance_rate();
    EXPECT_GE(rate, 0.0f);
    EXPECT_LE(rate, 1.0f);
}

TEST_F(SpeculativeDecoderFunctionalTest, TokensPerTargetPassAfterGeneration) {
    // eos_model returns only its last row, which the first target pass discovers; the proposal's
    // own row then takes a second pass: one token (EOS) for two target passes.
    SpeculativeDecoder decoder(draft_.get(), target_.get(), config_);
    EXPECT_EQ(decoder.generate_tokens({adai::SpecialTokenIDs::BOS}),
              std::vector<int>{adai::SpecialTokenIDs::EOS});
    EXPECT_EQ(decoder.get_target_passes(), 2);
    EXPECT_FLOAT_EQ(decoder.get_tokens_per_target_pass(), 0.5f);
}

TEST_F(SpeculativeDecoderFunctionalTest, MultipleGenerateCallsDoNotCrash) {
    SpeculativeDecoder decoder(draft_.get(), target_.get(), config_);
    for (int i = 0; i < 3; ++i) {
        EXPECT_NO_THROW(decoder.generate_tokens({adai::SpecialTokenIDs::BOS}));
    }
}

// ============================================================================
// TD-507: target distribution, greedy equivalence, one target pass per round
// ============================================================================

namespace {

constexpr int kFixedVocab = 8;  // ids 0..3 special (EOS = 3, never likely), 4..6 the real choices

// Logits giving probabilities `probs` over tokens 4, 5, 6 at every position (a row per position).
TextGenerator::ModelForwardFn fixed_distribution(std::vector<float> probs, int* calls = nullptr) {
    return [probs, calls](const std::vector<int>& tokens) {
        if (calls)
            ++*calls;
        Matrix logits(static_cast<int>(tokens.size()), kFixedVocab);
        for (int r = 0; r < logits.rows; ++r) {
            for (int v = 0; v < kFixedVocab; ++v)
                logits.data[r][v] = -1e9f;
            for (int i = 0; i < 3; ++i)
                logits.data[r][4 + i] = std::log(probs[i]);
        }
        return logits;
    };
}

// Deterministic context-dependent logits; `salt` makes the draft and target disagree.
TextGenerator::ModelForwardFn hashed_model(unsigned salt, int vocab, int* calls = nullptr) {
    return [salt, vocab, calls](const std::vector<int>& tokens) {
        if (calls)
            ++*calls;
        Matrix logits(static_cast<int>(tokens.size()), vocab);
        for (int r = 0; r < logits.rows; ++r) {
            unsigned h = 2166136261u;
            for (int j = 0; j <= r; ++j)
                h = (h ^ static_cast<unsigned>(tokens[j])) * 16777619u;
            for (int v = 0; v < vocab; ++v) {
                unsigned x = (h ^ (static_cast<unsigned>(v) * 2654435761u) ^ salt) * 2246822519u;
                logits.data[r][v] = static_cast<float>(x % 1000u) / 200.0f;
            }
            logits.data[r][adai::SpecialTokenIDs::EOS] = -1e9f;  // keep generations long
            logits.data[r][adai::SpecialTokenIDs::PAD] = -1e9f;
        }
        return logits;
    };
}

}  // namespace

TEST(SpeculativeDecodingTD507Test, FirstTokenFollowsTheTargetDistribution) {
    // The rejection residual used to subtract only the rejected token's draft probability: target
    // (0.5, 0.3, 0.2) with draft (0.2, 0.3, 0.5) came out (0.40, 0.40, 0.20).
    TextGenerator draft, target;
    draft.set_model_fn(fixed_distribution({0.2f, 0.3f, 0.5f}));
    target.set_model_fn(fixed_distribution({0.5f, 0.3f, 0.2f}));
    SpeculativeDecodingConfig cfg;
    cfg.num_candidates = 3;
    SpeculativeDecoder decoder(&draft, &target, cfg, 7);

    constexpr int kRuns = 20000;
    std::vector<int> counts(kFixedVocab, 0);
    for (int i = 0; i < kRuns; ++i)
        ++counts[decoder.generate_tokens({adai::SpecialTokenIDs::BOS}, 1).at(0)];
    EXPECT_NEAR(counts[4] / static_cast<float>(kRuns), 0.5f, 0.015f);
    EXPECT_NEAR(counts[5] / static_cast<float>(kRuns), 0.3f, 0.015f);
    EXPECT_NEAR(counts[6] / static_cast<float>(kRuns), 0.2f, 0.015f);
}

TEST(SpeculativeDecodingTD507Test, GreedyOutputEqualsTheTargetsGreedyOutput) {
    // Greedy mode used to accept randomly (41% of first tokens were the draft's argmax).
    constexpr int kVocab = 16;
    for (unsigned salt = 1; salt <= 10; ++salt) {
        TextGenerator draft, target;
        draft.set_model_fn(hashed_model(salt * 7919u, kVocab));
        target.set_model_fn(hashed_model(0u, kVocab));

        TextGenerator::GenerationConfig gcfg;
        gcfg.max_length = 24;
        const std::vector<int> greedy =
            TextGenerator(gcfg, 1).generate_greedy(hashed_model(0u, kVocab), {2});
        const std::vector<int> expected(greedy.begin() + 1, greedy.end());

        SpeculativeDecodingConfig cfg;
        cfg.num_candidates = 4;
        cfg.use_greedy = true;
        SpeculativeDecoder decoder(&draft, &target, cfg, salt);
        EXPECT_EQ(decoder.generate_tokens({2}, 24), expected) << "salt " << salt;
    }
}

TEST(SpeculativeDecodingTD507Test, OneTargetPassPerRoundAndABonusToken) {
    // With the target as its own draft every proposal is accepted, so each round yields K + 1
    // tokens for one target pass. It used to cost one target pass per token.
    constexpr int kVocab = 16;
    int target_calls = 0;
    TextGenerator draft, target;
    draft.set_model_fn(hashed_model(0u, kVocab));
    target.set_model_fn(hashed_model(0u, kVocab, &target_calls));
    SpeculativeDecodingConfig cfg;
    cfg.num_candidates = 4;
    cfg.use_greedy = true;
    SpeculativeDecoder decoder(&draft, &target, cfg, 1);

    const auto tokens = decoder.generate_tokens({2}, 20);
    EXPECT_EQ(tokens.size(), 20u);
    EXPECT_EQ(target_calls, 4);  // 20 tokens / (K + 1)
    EXPECT_FLOAT_EQ(decoder.get_tokens_per_target_pass(), 5.0f);
    EXPECT_FLOAT_EQ(decoder.get_acceptance_rate(), 1.0f);
}

TEST(SpeculativeDecodingTD507Test, OutputNeverExceedsMaxNewTokens) {
    // generate_tokens(…, 40) used to return 41 tokens.
    constexpr int kVocab = 16;
    TextGenerator draft, target;
    draft.set_model_fn(hashed_model(0u, kVocab));
    target.set_model_fn(hashed_model(0u, kVocab));
    SpeculativeDecodingConfig cfg;
    cfg.num_candidates = 4;
    for (int max_new : {1, 3, 7, 40}) {
        SpeculativeDecoder decoder(&draft, &target, cfg, 3);
        EXPECT_EQ(decoder.generate_tokens({2}, max_new).size(), static_cast<size_t>(max_new));
    }
}

TEST(SpeculativeDecodingTD507Test, TemperatureAppliesToBothModels) {
    // The draft sample used to be temperature-scaled while acceptance and the residual used unscaled
    // probabilities. At T = 0.5 the output must follow softmax(target logits / T), which for target
    // (0.5, 0.3, 0.2) is p² normalized: (0.658, 0.237, 0.105).
    TextGenerator draft, target;
    draft.set_model_fn(fixed_distribution({0.2f, 0.3f, 0.5f}));
    target.set_model_fn(fixed_distribution({0.5f, 0.3f, 0.2f}));
    SpeculativeDecodingConfig cfg;
    cfg.num_candidates = 3;
    cfg.temperature = 0.5f;
    SpeculativeDecoder decoder(&draft, &target, cfg, 9);

    constexpr int kRuns = 20000;
    std::vector<int> counts(kFixedVocab, 0);
    for (int i = 0; i < kRuns; ++i)
        ++counts[decoder.generate_tokens({adai::SpecialTokenIDs::BOS}, 1).at(0)];
    EXPECT_NEAR(counts[4] / static_cast<float>(kRuns), 0.25f / 0.38f, 0.015f);
    EXPECT_NEAR(counts[5] / static_cast<float>(kRuns), 0.09f / 0.38f, 0.015f);
    EXPECT_NEAR(counts[6] / static_cast<float>(kRuns), 0.04f / 0.38f, 0.015f);
}

TEST(SpeculativeDecodingTD507Test, LastRowOnlyTargetGivesTheSameGreedyOutput) {
    // A target that returns only its last row is scored once per proposal: same output, more passes.
    constexpr int kVocab = 16;
    auto full = hashed_model(0u, kVocab);
    int calls = 0;
    auto last_row_only = [&full, &calls](const std::vector<int>& tokens) {
        ++calls;
        Matrix all = full(tokens);
        Matrix last(1, all.cols);
        last.data[0] = all.data[all.rows - 1];
        return last;
    };
    TextGenerator draft, target_full, target_last;
    draft.set_model_fn(hashed_model(99u, kVocab));
    target_full.set_model_fn(full);
    target_last.set_model_fn(last_row_only);
    SpeculativeDecodingConfig cfg;
    cfg.num_candidates = 3;
    cfg.use_greedy = true;
    const auto a = SpeculativeDecoder(&draft, &target_full, cfg, 5).generate_tokens({2}, 15);
    const auto b = SpeculativeDecoder(&draft, &target_last, cfg, 5).generate_tokens({2}, 15);
    EXPECT_EQ(a, b);
    EXPECT_GT(calls, 0);
}

TEST(SpeculativeDecodingTD508Test, SameSeedSameOutput) {
    // The RNG used to be seeded from std::random_device only.
    constexpr int kVocab = 16;
    TextGenerator draft, target;
    draft.set_model_fn(hashed_model(5u, kVocab));
    target.set_model_fn(hashed_model(0u, kVocab));
    SpeculativeDecodingConfig cfg;
    cfg.num_candidates = 3;
    const auto a = SpeculativeDecoder(&draft, &target, cfg, 1234).generate_tokens({2}, 25);
    const auto b = SpeculativeDecoder(&draft, &target, cfg, 1234).generate_tokens({2}, 25);
    EXPECT_EQ(a, b);
}
