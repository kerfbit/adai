#pragma once

// @adai-status: stable
// @adai-version: 1.1.0
// @adai-reviewed: 2026-10-25


#include <map>
#include <string>
#include <vector>

/**
 * @brief BLEU/ROUGE generation quality scores for a batch of reference/hypothesis pairs.
 *
 * All fields default to -1.0 (not computed).
 */
struct GenerationQualityScore {
    float bleu1 = -1.0f;   ///< Corpus BLEU-1  (unigram modified precision)
    float bleu2 = -1.0f;   ///< Corpus BLEU-2  (bigram  modified precision)
    float bleu4 = -1.0f;   ///< Corpus BLEU-4  (1- to 4-grams, standard metric)
    float rouge1 = -1.0f;  ///< Macro-averaged ROUGE-1 F1  (unigram overlap)
    float rouge2 = -1.0f;  ///< Macro-averaged ROUGE-2 F1  (bigram  overlap)
    float rougeL = -1.0f;  ///< Macro-averaged ROUGE-L F1  (LCS-based)
};

/**
 * @brief Lightweight BLEU and ROUGE evaluator.
 *
 * No external dependencies.  All methods are static.
 *
 * BLEU:  evaluate() reports unsmoothed corpus BLEU-N (Papineni et al. 2002): n-gram matches and
 *        counts are summed over every pair, then BP · exp(mean log p_n) over orders 1..N. Any order
 *        with no matches makes the score 0, as in the original definition; a corpus of several pairs
 *        rarely hits that. Brevity penalty is applied at corpus level.
 *        sentence_bleu() is the explicitly smoothed single-pair score (see its comment).
 *        TD-378: add-1 smoothing used to be applied to every order, unigrams included, so an unrelated
 *        5-word answer scored BLEU-4 0.23 and an exactly correct 3-word answer 0.0067.
 *
 * ROUGE: Macro-averaged per-sentence F1 across all reference/hypothesis pairs.
 *        ROUGE-L uses a rolling 2-row DP LCS to bound memory use.
 */
class GenerationQualityEvaluator {
   public:
    /**
     * @brief Evaluate corpus-level BLEU/ROUGE metrics.
     * @param references  List of reference (target) strings.
     * @param hypotheses  List of generated (hypothesis) strings.
     *                    Must be the same length as @p references.
     * @return GenerationQualityScore with all fields populated;
     *         returns default struct (all -1) on empty or mismatched input.
     */
    static GenerationQualityScore evaluate(const std::vector<std::string>& references,
                                           const std::vector<std::string>& hypotheses);

    /**
     * @brief Smoothed sentence-level BLEU-N for one reference/hypothesis pair (TD-378).
     *
     * Unsmoothed BLEU of a single short sentence is almost always 0, so this uses the two standard
     * sentence-level adjustments sacreBLEU applies (smooth_method="floor", effective_order=True):
     *  - Chen & Cherry (2014) method 1: an order with no matches counts as 0.1 matches;
     *  - effective order: orders the hypothesis is too short to have are left out, so an exactly
     *    correct 3-word answer scores 1.0.
     * Unrelated answers score near 0 (0.025 for 5 unrelated words against 6). Returns 0 for an empty
     * hypothesis.
     */
    static float sentence_bleu(const std::string& reference, const std::string& hypothesis,
                               int max_n = 4);

    /**
     * @brief Tokenize a string into lowercase word tokens.
     *
     * Splits on whitespace then strips leading/trailing punctuation from each
     * token and lowercases it.  Empty tokens after stripping are discarded.
     */
    // TODO: See TD-380 in TECHNICAL_DEBT.md - differs from sacreBLEU/rouge-score tokenization; scores aren't
    // comparable with published numbers
    static std::vector<std::string> tokenize(const std::string& text);

   private:
    using TokenList = std::vector<std::string>;
    using NGram = std::vector<std::string>;
    using Counts = std::map<NGram, int>;

    static Counts count_ngrams(const TokenList& tokens, int n);

    // Clipped n-gram matches and hypothesis n-gram counts per order, and total lengths, summed over
    // every pair.
    struct BleuStats {
        std::vector<int> matches;
        std::vector<int> totals;
        int ref_len = 0;
        int hyp_len = 0;
    };
    static BleuStats bleu_stats(const std::vector<TokenList>& refs,
                                const std::vector<TokenList>& hyps, int max_n);
    static float brevity_penalty(int ref_len, int hyp_len);

    static float compute_corpus_bleu(const std::vector<TokenList>& refs,
                                     const std::vector<TokenList>& hyps, int max_n);

    static float compute_corpus_rouge_n(const std::vector<TokenList>& refs,
                                        const std::vector<TokenList>& hyps, int n);

    static float compute_corpus_rouge_l(const std::vector<TokenList>& refs,
                                        const std::vector<TokenList>& hyps);

    static int lcs_length(const TokenList& a, const TokenList& b);
};
