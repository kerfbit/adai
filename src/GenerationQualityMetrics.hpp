#pragma once

// @adai-status: stable
// @adai-version: 1.0.0
// @adai-reviewed: 2026-09-07


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
    float bleu4 = -1.0f;   ///< Corpus BLEU-4  (4-gram, standard metric)
    float rouge1 = -1.0f;  ///< Macro-averaged ROUGE-1 F1  (unigram overlap)
    float rouge2 = -1.0f;  ///< Macro-averaged ROUGE-2 F1  (bigram  overlap)
    float rougeL = -1.0f;  ///< Macro-averaged ROUGE-L F1  (LCS-based)
};

/**
 * @brief Lightweight BLEU and ROUGE evaluator.
 *
 * No external dependencies.  All methods are static.
 *
 * BLEU:  Corpus-level BLEU-N with add-1 smoothing (Lin & Och 2004, method 1).
 *        Brevity penalty is applied at corpus level.
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
     * @brief Tokenize a string into lowercase word tokens.
     *
     * Splits on whitespace then strips leading/trailing punctuation from each
     * token and lowercases it.  Empty tokens after stripping are discarded.
     */
    static std::vector<std::string> tokenize(const std::string& text);

   private:
    using TokenList = std::vector<std::string>;
    using NGram = std::vector<std::string>;
    using Counts = std::map<NGram, int>;

    static Counts count_ngrams(const TokenList& tokens, int n);

    static float compute_corpus_bleu(const std::vector<TokenList>& refs,
                                     const std::vector<TokenList>& hyps, int max_n);

    static float compute_corpus_rouge_n(const std::vector<TokenList>& refs,
                                        const std::vector<TokenList>& hyps, int n);

    static float compute_corpus_rouge_l(const std::vector<TokenList>& refs,
                                        const std::vector<TokenList>& hyps);

    static int lcs_length(const TokenList& a, const TokenList& b);
};
