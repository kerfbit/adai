/**
 * Unit tests for SpeculativeDecoding
 * Tests: SpeculativeDecodingConfig, TokenProposal, calculate_theoretical_speedup,
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

TEST(SpeculativeDecodingConfigTest, DefaultAcceptanceThreshold) {
    SpeculativeDecodingConfig cfg;
    EXPECT_FLOAT_EQ(cfg.acceptance_threshold, 0.0f);
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
    cfg.acceptance_threshold = 0.1f;
    cfg.use_greedy = true;

    EXPECT_EQ(cfg.num_candidates, 8);
    EXPECT_FLOAT_EQ(cfg.temperature, 0.7f);
    EXPECT_EQ(cfg.max_length, 200);
    EXPECT_FLOAT_EQ(cfg.acceptance_threshold, 0.1f);
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
// Free Function: calculate_theoretical_speedup
// ============================================================================

TEST(TheoreticalSpeedupTest, ZeroCandidatesReturnsOne) {
    EXPECT_FLOAT_EQ(calculate_theoretical_speedup(0, 0.8f), 1.0f);
}

TEST(TheoreticalSpeedupTest, ZeroAcceptanceRateReturnsOne) {
    EXPECT_FLOAT_EQ(calculate_theoretical_speedup(4, 0.0f), 1.0f);
}

TEST(TheoreticalSpeedupTest, NegativeCandidatesReturnsOne) {
    EXPECT_FLOAT_EQ(calculate_theoretical_speedup(-1, 0.5f), 1.0f);
}

TEST(TheoreticalSpeedupTest, Formula_K4_Alpha1) {
    // K=4, α=1.0 → speedup = 4*1/(4+1) = 0.8
    float expected = (4.0f * 1.0f) / 5.0f;
    EXPECT_NEAR(calculate_theoretical_speedup(4, 1.0f), expected, 1e-5f);
}

TEST(TheoreticalSpeedupTest, Formula_K8_Alpha07) {
    // K=8, α=0.7 → speedup = 8*0.7/9 = 0.6222...
    float expected = (8.0f * 0.7f) / 9.0f;
    EXPECT_NEAR(calculate_theoretical_speedup(8, 0.7f), expected, 1e-5f);
}

TEST(TheoreticalSpeedupTest, Formula_K2_Alpha05) {
    // K=2, α=0.5 → speedup = 2*0.5/3 = 0.3333...
    float expected = (2.0f * 0.5f) / 3.0f;
    EXPECT_NEAR(calculate_theoretical_speedup(2, 0.5f), expected, 1e-5f);
}

TEST(TheoreticalSpeedupTest, SpeedupAlwaysPositive) {
    for (int K : {1, 4, 8, 16}) {
        for (float alpha : {0.1f, 0.5f, 0.9f}) {
            EXPECT_GT(calculate_theoretical_speedup(K, alpha), 0.0f);
        }
    }
}

TEST(TheoreticalSpeedupTest, SpeedupMonotonicInK) {
    // More candidates → higher theoretical speedup (same alpha)
    float s4 = calculate_theoretical_speedup(4, 0.8f);
    float s8 = calculate_theoretical_speedup(8, 0.8f);
    EXPECT_GT(s8, s4);
}

TEST(TheoreticalSpeedupTest, SpeedupMonotonicInAlpha) {
    // Higher acceptance rate → higher speedup
    float sLow = calculate_theoretical_speedup(4, 0.3f);
    float sHigh = calculate_theoretical_speedup(4, 0.9f);
    EXPECT_GT(sHigh, sLow);
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
    cfg.acceptance_threshold = 0.2f;
    cfg.use_greedy = false;
    decoder.set_config(cfg);

    const auto& got = decoder.get_config();
    EXPECT_EQ(got.num_candidates, 3);
    EXPECT_FLOAT_EQ(got.temperature, 1.5f);
    EXPECT_EQ(got.max_length, 30);
    EXPECT_FLOAT_EQ(got.acceptance_threshold, 0.2f);
    EXPECT_FALSE(got.use_greedy);
}

// ============================================================================
// SpeculativeDecoder Initial Statistics Tests
// ============================================================================

TEST_F(SpeculativeDecoderTest, AcceptanceRateInitiallyZero) {
    SpeculativeDecoder decoder(draft_.get(), target_.get());
    EXPECT_FLOAT_EQ(decoder.get_acceptance_rate(), 0.0f);
}

TEST_F(SpeculativeDecoderTest, SpeedupInitiallyZero) {
    SpeculativeDecoder decoder(draft_.get(), target_.get());
    EXPECT_FLOAT_EQ(decoder.get_speedup(), 0.0f);
}

TEST_F(SpeculativeDecoderTest, ResetStatsDoesNotCrash) {
    SpeculativeDecoder decoder(draft_.get(), target_.get());
    EXPECT_NO_THROW(decoder.reset_stats());
}

TEST_F(SpeculativeDecoderTest, ResetStatsPreservesZeroState) {
    SpeculativeDecoder decoder(draft_.get(), target_.get());
    decoder.reset_stats();
    EXPECT_FLOAT_EQ(decoder.get_acceptance_rate(), 0.0f);
    EXPECT_FLOAT_EQ(decoder.get_speedup(), 0.0f);
}

TEST_F(SpeculativeDecoderTest, PrintStatsDoesNotCrash) {
    SpeculativeDecoder decoder(draft_.get(), target_.get());
    // Redirect stdout to /dev/null during print
    std::streambuf* old_buf = std::cout.rdbuf();
    std::ostringstream devnull;
    std::cout.rdbuf(devnull.rdbuf());
    EXPECT_NO_THROW(decoder.print_stats());
    std::cout.rdbuf(old_buf);
}

// ============================================================================
// SpeculativeDecoder Speedup Formula Tests
// ============================================================================

TEST_F(SpeculativeDecoderTest, SpeedupFormulaMatchesHelper) {
    SpeculativeDecodingConfig cfg;
    cfg.num_candidates = 4;
    SpeculativeDecoder decoder(draft_.get(), target_.get(), cfg);

    // Before any proposals: rate = 0, speedup = 0 (K*0/(K+1) = 0)
    float k = static_cast<float>(cfg.num_candidates);
    float expected = (k * decoder.get_acceptance_rate()) / (k + 1.0f);
    EXPECT_FLOAT_EQ(decoder.get_speedup(), expected);
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
    EXPECT_FLOAT_EQ(decoder.get_speedup(), 0.0f);
}

TEST_F(SpeculativeDecoderFunctionalTest, AcceptanceRateInRange) {
    SpeculativeDecoder decoder(draft_.get(), target_.get(), config_);
    decoder.generate_tokens({adai::SpecialTokenIDs::BOS});
    float rate = decoder.get_acceptance_rate();
    EXPECT_GE(rate, 0.0f);
    EXPECT_LE(rate, 1.0f);
}

TEST_F(SpeculativeDecoderFunctionalTest, SpeedupNonNegativeAfterGeneration) {
    SpeculativeDecoder decoder(draft_.get(), target_.get(), config_);
    decoder.generate_tokens({adai::SpecialTokenIDs::BOS});
    EXPECT_GE(decoder.get_speedup(), 0.0f);
}

TEST_F(SpeculativeDecoderFunctionalTest, MultipleGenerateCallsDoNotCrash) {
    SpeculativeDecoder decoder(draft_.get(), target_.get(), config_);
    for (int i = 0; i < 3; ++i) {
        EXPECT_NO_THROW(decoder.generate_tokens({adai::SpecialTokenIDs::BOS}));
    }
}

// ============================================================================
// print_speedup_table Tests
// ============================================================================

TEST(PrintSpeedupTableTest, DoesNotCrash) {
    std::streambuf* old_buf = std::cout.rdbuf();
    std::ostringstream capture;
    std::cout.rdbuf(capture.rdbuf());
    EXPECT_NO_THROW(print_speedup_table());
    std::cout.rdbuf(old_buf);
}

TEST(PrintSpeedupTableTest, ProducesOutput) {
    std::ostringstream capture;
    std::streambuf* old_buf = std::cout.rdbuf(capture.rdbuf());
    print_speedup_table();
    std::cout.rdbuf(old_buf);
    EXPECT_GT(capture.str().size(), 0u);
}
