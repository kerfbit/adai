#pragma once

// @adai-status: stable
// @adai-version: 2.0.0
// @adai-reviewed: 2026-10-25


#include <map>
#include <string>
#include <vector>

/**
 * @brief BLEU/ROUGE generation quality scores for a batch of reference/hypothesis pairs.
 *
 * Scores are on a 0-100 scale (TD-380; 0-1 before October 25, 2026). All fields default to -1.0
 * (not computed).
 */
struct GenerationQualityScore {
    float bleu1 = -1.0f;   ///< Corpus BLEU-1  (sacreBLEU, 0-100)
    float bleu2 = -1.0f;   ///< Corpus BLEU-2  (sacreBLEU, 0-100)
    float bleu4 = -1.0f;   ///< Corpus BLEU-4  (sacreBLEU's default, the standard metric, 0-100)
    float rouge1 = -1.0f;  ///< Macro-averaged ROUGE-1 F1 (rouge-score, 0-100)
    float rouge2 = -1.0f;  ///< Macro-averaged ROUGE-2 F1 (rouge-score, 0-100)
    float rougeL = -1.0f;  ///< Macro-averaged ROUGE-L F1 (LCS-based, rouge-score, 0-100)
};

/**
 * @brief Lightweight BLEU and ROUGE evaluator.
 *
 * No external dependencies.  All methods are static.
 *
 * TD-380: the scores match the standard tools, so they can be compared with published numbers:
 * BLEU:  sacreBLEU's defaults: 13a tokenization (punctuation split into tokens, case kept), corpus
 *        BLEU-N with "exp" smoothing, 0-100. sentence_bleu() is sacreBLEU's sentence_bleu() (adds
 *        effective order). TD-378 replaced the old add-1-on-every-order smoothing.
 * ROUGE: rouge-score's defaults (lowercase, [a-z0-9] tokens, no stemming): per-pair F1, averaged
 *        over the pairs, 0-100.
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
     * @brief sacreBLEU's sentence BLEU for one pair, 0-100 (TD-380): 13a tokens, "exp" smoothing and
     *        effective order (orders longer than the hypothesis are left out, so an exactly correct
     *        short answer scores 100). Returns 0 when no unigram matches.
     */
    static float sentence_bleu(const std::string& reference, const std::string& hypothesis,
                               int max_n = 4);

    /// TD-380: sacreBLEU's 13a tokenizer (mteval-v13a): ASCII punctuation becomes separate tokens,
    /// '.'/',' only when not between digits, '-' only after a digit; case is kept.
    static std::vector<std::string> tokenize_13a(const std::string& text);

    /// TD-380: rouge-score's default tokenizer: lowercase, tokens are runs of [a-z0-9].
    static std::vector<std::string> tokenize_rouge(const std::string& text);

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

    static float compute_bleu(const std::vector<TokenList>& refs,
                              const std::vector<TokenList>& hyps, int max_n, bool effective_order);

    static float compute_corpus_rouge_n(const std::vector<TokenList>& refs,
                                        const std::vector<TokenList>& hyps, int n);

    static float compute_corpus_rouge_l(const std::vector<TokenList>& refs,
                                        const std::vector<TokenList>& hyps);

    static int lcs_length(const TokenList& a, const TokenList& b);
};
