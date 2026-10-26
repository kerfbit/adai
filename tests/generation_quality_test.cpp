#include <gtest/gtest.h>
#include <cmath>
#include <string>
#include <vector>
#include "GenerationQualityMetrics.hpp"

// TD-380: the scores follow sacreBLEU (13a tokens, exp smoothing) and rouge-score (lowercase [a-z0-9]
// tokens, no stemming), on a 0-100 scale. Every expected value below was produced by sacreBLEU 2.6.0
// (BLEU(max_ngram_order=n).corpus_score / sentence_bleu) and rouge-score (RougeScorer, use_stemmer=False,
// F-measure averaged over the pairs) on the same strings.

using E = GenerationQualityEvaluator;

namespace {
struct Expected {
    float b1, b2, b4, r1, r2, rl;
};

void expect_scores(const std::vector<std::string>& refs, const std::vector<std::string>& hyps,
                   const Expected& e) {
    const auto s = E::evaluate(refs, hyps);
    EXPECT_NEAR(s.bleu1, e.b1, 1e-3f);
    EXPECT_NEAR(s.bleu2, e.b2, 1e-3f);
    EXPECT_NEAR(s.bleu4, e.b4, 1e-3f);
    EXPECT_NEAR(s.rouge1, e.r1, 1e-3f);
    EXPECT_NEAR(s.rouge2, e.r2, 1e-3f);
    EXPECT_NEAR(s.rougeL, e.rl, 1e-3f);
}

std::string join(const std::vector<std::string>& tokens) {
    std::string out;
    for (const auto& t : tokens)
        out += (out.empty() ? "" : " ") + t;
    return out;
}
}  // namespace

// ============================================================================
// Tokenizers (outputs of sacreBLEU's Tokenizer13a and rouge-score's tokenize())
// ============================================================================

TEST(GenerationQualityTokenizer, Matches13a) {
    EXPECT_EQ(join(E::tokenize_13a("Hello, world! It's 3.14 - done.")),
              "Hello , world ! It's 3.14 - done .");
    EXPECT_EQ(join(E::tokenize_13a("1,000 people; 2.5x faster (really?) e-mail 3-4 \"q\" a.b,c")),
              "1,000 people ; 2.5x faster ( really ? ) e-mail 3 - 4 \" q \" a . b , c");
    EXPECT_EQ(join(E::tokenize_13a("Café déjà vu")), "Café déjà vu");
    EXPECT_EQ(join(E::tokenize_13a("A&amp;B &lt;tag&gt;")), "A & B < tag >");
    EXPECT_TRUE(E::tokenize_13a("").empty());
}

TEST(GenerationQualityTokenizer, MatchesRougeScore) {
    EXPECT_EQ(join(E::tokenize_rouge("Hello, world! It's 3.14 - done.")),
              "hello world it s 3 14 done");
    EXPECT_EQ(join(E::tokenize_rouge("1,000 people; 2.5x faster (really?) e-mail 3-4 \"q\" a.b,c")),
              "1 000 people 2 5x faster really e mail 3 4 q a b c");
    EXPECT_EQ(join(E::tokenize_rouge("Café déjà vu")), "caf d j vu");
    EXPECT_EQ(join(E::tokenize_rouge("A&amp;B &lt;tag&gt;")), "a amp b lt tag gt");
    EXPECT_TRUE(E::tokenize_rouge("--- ,,, ...").empty());
}

// ============================================================================
// Corpus scores
// ============================================================================

TEST(GenerationQualityReference, IdenticalScoresHundred) {
    expect_scores({"the cat sat on the mat"}, {"the cat sat on the mat"},
                  {100, 100, 100, 100, 100, 100});
}

TEST(GenerationQualityReference, PartialMatchWithPunctuation) {
    expect_scores({"The cat sat on the mat."}, {"The cat sat on a mat."},
                  {85.714286f, 75.592895f, 48.892302f, 83.333333f, 60.0f, 83.333333f});
}

TEST(GenerationQualityReference, UnrelatedAnswerScoresZero) {
    expect_scores({"the cat sat on the mat"}, {"dogs run very quickly home"}, {0, 0, 0, 0, 0, 0});
}

TEST(GenerationQualityReference, ScrambledWordsAreSmoothedNotZero) {
    // exp smoothing gives the zero-match orders a small precision (sacreBLEU's default).
    expect_scores({"a b c d e"}, {"e d c b a"}, {100, 35.355339f, 15.973578f, 100, 0, 20.0f});
}

TEST(GenerationQualityReference, ShortExactAnswerWithPeriod) {
    // "I am fine ." is four 13a tokens, so it has a 4-gram.
    expect_scores({"I am fine."}, {"I am fine."}, {100, 100, 100, 100, 100, 100});
}

TEST(GenerationQualityReference, BrevityPenalty) {
    expect_scores({"the cat sat on the mat"}, {"the cat sat on"},
                  {60.653066f, 60.653066f, 60.653066f, 80.0f, 75.0f, 80.0f});
}

TEST(GenerationQualityReference, BleuKeepsCaseRougeLowercases) {
    // Corpus BLEU-4 is 0: the 3-token hypothesis has no 4-gram (no effective order at corpus level).
    expect_scores({"The Cat sat"}, {"the cat sat"}, {33.333333f, 28.867513f, 0, 100, 100, 100});
    expect_scores({"Hello, World! It's 3.14 - done."}, {"hello world it's 3.14-done"},
                  {30.326533f, 27.124876f, 18.325568f, 100, 100, 100});
}

TEST(GenerationQualityReference, CorpusSumsCountsAcrossPairs) {
    expect_scores({"I am fine.", "The cat sat on the mat."}, {"I am fine.", "The cat sat on a mat."},
                  {90.909091f, 84.087497f, 63.404663f, 91.666667f, 80.0f, 91.666667f});
}

TEST(GenerationQualityReference, MoreCases) {
    expect_scores({"the cat sat on the mat"}, {""}, {0, 0, 0, 0, 0, 0});
    expect_scores({"the cat sat on the mat"}, {"the cat ran on the floor"},
                  {66.666667f, 51.639778f, 22.957488f, 66.666667f, 40.0f, 66.666667f});
    expect_scores({"hello world", "foo bar"}, {"hello world", "xyz qrs"}, {50, 50, 0, 50, 50, 50});
    expect_scores({"the quick brown fox"}, {"fox brown quick the"},
                  {100, 40.824829f, 22.590050f, 100, 0, 25.0f});
    expect_scores({"a b c d e"}, {"a c e"}, {51.341712f, 25.670856f, 0, 75.0f, 0, 75.0f});
    expect_scores({"hello"}, {"hello"}, {100, 0, 0, 100, 0, 100});
    expect_scores({"the cat sat on the mat"}, {"a cat runs on a log"},
                  {33.333333f, 18.257419f, 9.652435f, 33.333333f, 0, 33.333333f});
}

// ============================================================================
// Sentence BLEU (sacreBLEU's sentence_bleu: exp smoothing, effective order)
// ============================================================================

TEST(GenerationQualitySentenceBLEU, ReferenceValues) {
    EXPECT_NEAR(E::sentence_bleu("The cat sat on the mat.", "The cat sat on a mat."), 48.892302f,
                1e-3f);
    EXPECT_NEAR(E::sentence_bleu("a b c d e", "e d c b a"), 15.973578f, 1e-3f);
    EXPECT_NEAR(E::sentence_bleu("The Cat sat", "the cat sat"), 27.516060f, 1e-3f);
    EXPECT_NEAR(E::sentence_bleu("a b c d e", "a c e"), 20.374972f, 1e-3f);
    EXPECT_NEAR(E::sentence_bleu("hello", "hello"), 100.0f, 1e-3f) << "effective order";
    EXPECT_FLOAT_EQ(E::sentence_bleu("the cat sat on the mat", "dogs run very quickly home"), 0.0f);
    EXPECT_FLOAT_EQ(E::sentence_bleu("hello there", ""), 0.0f);
}

TEST(GenerationQualitySentenceBLEU, CorrectShortAnswerOutranksUnrelatedLongOne) {
    EXPECT_GT(E::sentence_bleu("i am fine", "i am fine"),
              E::sentence_bleu("the cat sat on the mat", "dogs run very quickly home"));
}

// ============================================================================
// Edge cases
// ============================================================================

TEST(GenerationQualityEdge, EmptyInputReturnsAllNegativeOne) {
    const auto score = E::evaluate({}, {});
    EXPECT_FLOAT_EQ(score.bleu4, -1.0f);
    EXPECT_FLOAT_EQ(score.rouge1, -1.0f);
    EXPECT_FLOAT_EQ(score.rouge2, -1.0f);
    EXPECT_FLOAT_EQ(score.rougeL, -1.0f);
}

TEST(GenerationQualityEdge, MismatchedSizesReturnsAllNegativeOne) {
    const auto score = E::evaluate({"hello"}, {"hello", "world"});
    EXPECT_FLOAT_EQ(score.bleu4, -1.0f);
    EXPECT_FLOAT_EQ(score.rouge1, -1.0f);
}

TEST(GenerationQualityEdge, ScoresAreOnTheHundredScale) {
    const auto score = E::evaluate({"the cat sat on the mat"}, {"a cat runs on a log"});
    for (float v : {score.bleu1, score.bleu2, score.bleu4, score.rouge1, score.rouge2, score.rougeL}) {
        EXPECT_GE(v, 0.0f);
        EXPECT_LE(v, 100.0f);
    }
    EXPECT_GT(score.bleu1, 1.0f) << "0-100, not 0-1";
}
