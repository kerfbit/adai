#include <gtest/gtest.h>
#include "GenerationQualityMetrics.hpp"
#include <cmath>
#include <string>
#include <vector>

// ============================================================================
// Tokenizer tests
// ============================================================================

TEST(GenerationQualityTokenizer, EmptyStringReturnsEmptyList) {
    auto tokens = GenerationQualityEvaluator::tokenize("");
    EXPECT_TRUE(tokens.empty());
}

TEST(GenerationQualityTokenizer, SingleWord) {
    auto tokens = GenerationQualityEvaluator::tokenize("hello");
    ASSERT_EQ(tokens.size(), 1u);
    EXPECT_EQ(tokens[0], "hello");
}

TEST(GenerationQualityTokenizer, LowercasesInput) {
    auto tokens = GenerationQualityEvaluator::tokenize("Hello WORLD");
    ASSERT_EQ(tokens.size(), 2u);
    EXPECT_EQ(tokens[0], "hello");
    EXPECT_EQ(tokens[1], "world");
}

TEST(GenerationQualityTokenizer, StripsPunctuationFromEnds) {
    auto tokens = GenerationQualityEvaluator::tokenize("hello, world!");
    ASSERT_EQ(tokens.size(), 2u);
    EXPECT_EQ(tokens[0], "hello");
    EXPECT_EQ(tokens[1], "world");
}

TEST(GenerationQualityTokenizer, PunctuationOnlyTokenDropped) {
    auto tokens = GenerationQualityEvaluator::tokenize("--- ,,, ...");
    EXPECT_TRUE(tokens.empty());
}

TEST(GenerationQualityTokenizer, MultipleSpacesSeparateWords) {
    auto tokens = GenerationQualityEvaluator::tokenize("a   b   c");
    ASSERT_EQ(tokens.size(), 3u);
    EXPECT_EQ(tokens[0], "a");
    EXPECT_EQ(tokens[2], "c");
}

// ============================================================================
// BLEU tests
// ============================================================================

TEST(GenerationQualityBLEU, IdenticalReferenceShouldScoreOne) {
    std::vector<std::string> refs = {"the cat sat on the mat"};
    std::vector<std::string> hyps = {"the cat sat on the mat"};
    GenerationQualityScore score = GenerationQualityEvaluator::evaluate(refs, hyps);
    // BLEU-1 should be 1.0 for identical strings
    EXPECT_NEAR(score.bleu1, 1.0f, 0.01f);
    // BLEU-4 may be slightly < 1 due to add-1 smoothing on a short sentence
    EXPECT_GT(score.bleu4, 0.8f);
}

TEST(GenerationQualityBLEU, CompletelyDifferentHypothesisScoresNearZero) {
    std::vector<std::string> refs = {"the cat sat on the mat"};
    std::vector<std::string> hyps = {"dogs run quickly"};
    GenerationQualityScore score = GenerationQualityEvaluator::evaluate(refs, hyps);
    EXPECT_LT(score.bleu1, 0.1f);
}

TEST(GenerationQualityBLEU, PartialOverlapScoresBetween) {
    std::vector<std::string> refs = {"the cat sat on the mat"};
    std::vector<std::string> hyps = {"the cat sat"};
    GenerationQualityScore score = GenerationQualityEvaluator::evaluate(refs, hyps);
    // Partial match should give a non-trivially bounded BLEU-1
    EXPECT_GT(score.bleu1, 0.0f);
    EXPECT_LT(score.bleu1, 1.0f);
}

TEST(GenerationQualityBLEU, BLEU1AtLeastBLEU2AtLeastBLEU4) {
    std::vector<std::string> refs = {"the cat sat on the mat"};
    std::vector<std::string> hyps = {"the cat sat on a rug"};
    GenerationQualityScore score = GenerationQualityEvaluator::evaluate(refs, hyps);
    EXPECT_GE(score.bleu1, score.bleu2 - 1e-5f);
    EXPECT_GE(score.bleu2, score.bleu4 - 1e-5f);
}

TEST(GenerationQualityBLEU, BrevityPenaltyApplied) {
    // Very short hypothesis relative to reference should have BP < 1
    std::vector<std::string> refs = {"the cat sat on the very long mat indeed"};
    std::vector<std::string> hyps = {"cat"};
    GenerationQualityScore score = GenerationQualityEvaluator::evaluate(refs, hyps);
    EXPECT_LT(score.bleu1, 0.5f);
}

TEST(GenerationQualityBLEU, CorpusLevelAggregatesMultipleSentences) {
    std::vector<std::string> refs = {"the cat sat", "a dog ran"};
    std::vector<std::string> hyps = {"the cat sat", "a dog ran"};
    GenerationQualityScore score = GenerationQualityEvaluator::evaluate(refs, hyps);
    EXPECT_NEAR(score.bleu1, 1.0f, 0.01f);
}

// ============================================================================
// ROUGE-N tests
// ============================================================================

TEST(GenerationQualityROUGE, IdenticalPairROUGE1IsOne) {
    std::vector<std::string> refs = {"the cat sat on the mat"};
    std::vector<std::string> hyps = {"the cat sat on the mat"};
    GenerationQualityScore score = GenerationQualityEvaluator::evaluate(refs, hyps);
    EXPECT_NEAR(score.rouge1, 1.0f, 1e-4f);
}

TEST(GenerationQualityROUGE, NullHypothesisScoresZero) {
    std::vector<std::string> refs = {"the cat sat on the mat"};
    std::vector<std::string> hyps = {""};
    GenerationQualityScore score = GenerationQualityEvaluator::evaluate(refs, hyps);
    EXPECT_FLOAT_EQ(score.rouge1, 0.0f);
    EXPECT_FLOAT_EQ(score.rouge2, 0.0f);
    EXPECT_FLOAT_EQ(score.rougeL, 0.0f);
}

TEST(GenerationQualityROUGE, ROUGE1AtLeastROUGE2) {
    std::vector<std::string> refs = {"the cat sat on the mat"};
    std::vector<std::string> hyps = {"the cat ran on the floor"};
    GenerationQualityScore score = GenerationQualityEvaluator::evaluate(refs, hyps);
    EXPECT_GE(score.rouge1, score.rouge2 - 1e-5f);
}

TEST(GenerationQualityROUGE, MacroAveragedAcrossSentences) {
    // First sentence perfect match, second no match → average F1 ≈ 0.5
    std::vector<std::string> refs = {"hello world", "foo bar"};
    std::vector<std::string> hyps = {"hello world", "xyz qrs"};
    GenerationQualityScore score = GenerationQualityEvaluator::evaluate(refs, hyps);
    EXPECT_NEAR(score.rouge1, 0.5f, 0.05f);
}

// ============================================================================
// ROUGE-L (LCS) tests
// ============================================================================

TEST(GenerationQualityROUGEL, IdenticalPairScoresOne) {
    std::vector<std::string> refs = {"the quick brown fox"};
    std::vector<std::string> hyps = {"the quick brown fox"};
    GenerationQualityScore score = GenerationQualityEvaluator::evaluate(refs, hyps);
    EXPECT_NEAR(score.rougeL, 1.0f, 1e-4f);
}

TEST(GenerationQualityROUGEL, ReorderedTokensScoreLower) {
    std::vector<std::string> refs = {"the quick brown fox"};
    std::vector<std::string> hyps = {"fox brown quick the"};
    GenerationQualityScore score = GenerationQualityEvaluator::evaluate(refs, hyps);
    // LCS of length 1 → F1 = 2 * (1/4) * (1/4) / (1/4 + 1/4) = 0.25
    EXPECT_LT(score.rougeL, 0.5f);
    EXPECT_GT(score.rougeL, 0.0f);
}

TEST(GenerationQualityROUGEL, SubsequenceGivesPartialCredit) {
    std::vector<std::string> refs = {"a b c d e"};
    std::vector<std::string> hyps = {"a c e"};
    GenerationQualityScore score = GenerationQualityEvaluator::evaluate(refs, hyps);
    // LCS = a c e (length 3): precision = 3/3 = 1.0, recall = 3/5 = 0.6, F1 = 0.75
    EXPECT_NEAR(score.rougeL, 0.75f, 0.01f);
}

// ============================================================================
// Edge cases
// ============================================================================

TEST(GenerationQualityEdge, EmptyInputReturnsAllNegativeOne) {
    std::vector<std::string> refs, hyps;
    GenerationQualityScore score = GenerationQualityEvaluator::evaluate(refs, hyps);
    EXPECT_FLOAT_EQ(score.bleu4,  -1.0f);
    EXPECT_FLOAT_EQ(score.rouge1, -1.0f);
    EXPECT_FLOAT_EQ(score.rouge2, -1.0f);
    EXPECT_FLOAT_EQ(score.rougeL, -1.0f);
}

TEST(GenerationQualityEdge, MismatchedSizesReturnsAllNegativeOne) {
    std::vector<std::string> refs = {"hello"};
    std::vector<std::string> hyps = {"hello", "world"};
    GenerationQualityScore score = GenerationQualityEvaluator::evaluate(refs, hyps);
    EXPECT_FLOAT_EQ(score.bleu4,  -1.0f);
    EXPECT_FLOAT_EQ(score.rouge1, -1.0f);
}

TEST(GenerationQualityEdge, SingleWordPairScoresCorrectly) {
    std::vector<std::string> refs = {"hello"};
    std::vector<std::string> hyps = {"hello"};
    GenerationQualityScore score = GenerationQualityEvaluator::evaluate(refs, hyps);
    EXPECT_NEAR(score.rouge1, 1.0f, 1e-4f);
    EXPECT_NEAR(score.rougeL, 1.0f, 1e-4f);
}

TEST(GenerationQualityEdge, AllScoresInValidRange) {
    std::vector<std::string> refs = {"the cat sat on the mat"};
    std::vector<std::string> hyps = {"a cat runs on a log"};
    GenerationQualityScore score = GenerationQualityEvaluator::evaluate(refs, hyps);
    EXPECT_GE(score.bleu1,  0.0f);  EXPECT_LE(score.bleu1,  1.0f);
    EXPECT_GE(score.bleu2,  0.0f);  EXPECT_LE(score.bleu2,  1.0f);
    EXPECT_GE(score.bleu4,  0.0f);  EXPECT_LE(score.bleu4,  1.0f);
    EXPECT_GE(score.rouge1, 0.0f);  EXPECT_LE(score.rouge1, 1.0f);
    EXPECT_GE(score.rouge2, 0.0f);  EXPECT_LE(score.rouge2, 1.0f);
    EXPECT_GE(score.rougeL, 0.0f);  EXPECT_LE(score.rougeL, 1.0f);
}
