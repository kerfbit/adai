// @adai-status: stable
// @adai-version: 1.1.0
// @adai-reviewed: 2026-10-25

#include "GenerationQualityMetrics.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <sstream>

// ─── tokenize ────────────────────────────────────────────────────────────────

std::vector<std::string> GenerationQualityEvaluator::tokenize(const std::string& text) {
    std::vector<std::string> tokens;
    std::istringstream iss(text);
    std::string word;
    while (iss >> word) {
        // Lowercase
        std::transform(word.begin(), word.end(), word.begin(),
                       [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
        // Strip leading punctuation
        size_t start = 0;
        while (start < word.size() && std::ispunct(static_cast<unsigned char>(word[start]))) {
            ++start;
        }
        // Strip trailing punctuation
        size_t end = word.size();
        while (end > start && std::ispunct(static_cast<unsigned char>(word[end - 1]))) {
            --end;
        }
        if (start < end) {
            tokens.push_back(word.substr(start, end - start));
        }
    }
    return tokens;
}

// ─── evaluate ────────────────────────────────────────────────────────────────

GenerationQualityScore GenerationQualityEvaluator::evaluate(
    const std::vector<std::string>& references, const std::vector<std::string>& hypotheses) {
    if (references.empty() || references.size() != hypotheses.size()) {
        return {};
    }

    std::vector<TokenList> ref_toks, hyp_toks;
    ref_toks.reserve(references.size());
    hyp_toks.reserve(hypotheses.size());
    for (size_t i = 0; i < references.size(); ++i) {
        ref_toks.push_back(tokenize(references[i]));
        hyp_toks.push_back(tokenize(hypotheses[i]));
    }

    GenerationQualityScore score;
    score.bleu1 = compute_corpus_bleu(ref_toks, hyp_toks, 1);
    score.bleu2 = compute_corpus_bleu(ref_toks, hyp_toks, 2);
    score.bleu4 = compute_corpus_bleu(ref_toks, hyp_toks, 4);
    score.rouge1 = compute_corpus_rouge_n(ref_toks, hyp_toks, 1);
    score.rouge2 = compute_corpus_rouge_n(ref_toks, hyp_toks, 2);
    score.rougeL = compute_corpus_rouge_l(ref_toks, hyp_toks);
    return score;
}

// ─── count_ngrams ─────────────────────────────────────────────────────────────

GenerationQualityEvaluator::Counts GenerationQualityEvaluator::count_ngrams(const TokenList& tokens,
                                                                            int n) {
    Counts c;
    int sz = static_cast<int>(tokens.size());
    for (int i = 0; i + n <= sz; ++i) {
        c[NGram(tokens.begin() + i, tokens.begin() + i + n)]++;
    }
    return c;
}

// ─── BLEU ─────────────────────────────────────────────────────────────────────

GenerationQualityEvaluator::BleuStats GenerationQualityEvaluator::bleu_stats(
    const std::vector<TokenList>& refs, const std::vector<TokenList>& hyps, int max_n) {
    BleuStats st;
    st.matches.assign(max_n, 0);
    st.totals.assign(max_n, 0);
    for (size_t idx = 0; idx < refs.size(); ++idx) {
        st.ref_len += static_cast<int>(refs[idx].size());
        st.hyp_len += static_cast<int>(hyps[idx].size());
        for (int n = 1; n <= max_n; ++n) {
            Counts ref_ng = count_ngrams(refs[idx], n);
            Counts hyp_ng = count_ngrams(hyps[idx], n);
            for (auto& [ngram, cnt] : hyp_ng) {
                st.totals[n - 1] += cnt;
                auto it = ref_ng.find(ngram);
                if (it != ref_ng.end()) {
                    st.matches[n - 1] += std::min(cnt, it->second);
                }
            }
        }
    }
    return st;
}

float GenerationQualityEvaluator::brevity_penalty(int ref_len, int hyp_len) {
    if (hyp_len <= 0)
        return 0.0f;
    if (hyp_len >= ref_len)
        return 1.0f;
    return std::exp(1.0f - static_cast<float>(ref_len) / static_cast<float>(hyp_len));
}

// TD-378: unsmoothed (Papineni et al.). The old add-1 on every order, unigrams included, plus a fixed
// log of -20 for an order the hypotheses were too short to have, scored an unrelated 5-word answer
// BLEU-4 0.23 and an exactly correct 3-word one 0.0067.
float GenerationQualityEvaluator::compute_corpus_bleu(const std::vector<TokenList>& refs,
                                                      const std::vector<TokenList>& hyps,
                                                      int max_n) {
    const BleuStats st = bleu_stats(refs, hyps, max_n);
    double log_sum = 0.0;
    for (int n = 0; n < max_n; ++n) {
        if (st.matches[n] == 0)
            return 0.0f;  // includes an order with no hypothesis n-grams at all
        log_sum += std::log(static_cast<double>(st.matches[n]) / st.totals[n]);
    }
    return brevity_penalty(st.ref_len, st.hyp_len) *
           static_cast<float>(std::exp(log_sum / max_n));
}

float GenerationQualityEvaluator::sentence_bleu(const std::string& reference,
                                                const std::string& hypothesis, int max_n) {
    const std::vector<TokenList> refs = {tokenize(reference)};
    const std::vector<TokenList> hyps = {tokenize(hypothesis)};
    const BleuStats st = bleu_stats(refs, hyps, max_n);

    constexpr double kFloor = 0.1;  // Chen & Cherry 2014 method 1 (sacreBLEU "floor")
    int effective_order = 0;
    double log_sum = 0.0;
    for (int n = 0; n < max_n && st.totals[n] > 0; ++n) {
        ++effective_order;
        const double matches = st.matches[n] > 0 ? st.matches[n] : kFloor;
        log_sum += std::log(matches / st.totals[n]);
    }
    if (effective_order == 0)
        return 0.0f;
    return brevity_penalty(st.ref_len, st.hyp_len) *
           static_cast<float>(std::exp(log_sum / effective_order));
}

// ─── compute_corpus_rouge_n ───────────────────────────────────────────────────

float GenerationQualityEvaluator::compute_corpus_rouge_n(const std::vector<TokenList>& refs,
                                                         const std::vector<TokenList>& hyps,
                                                         int n) {
    float total_f1 = 0.0f;
    for (size_t idx = 0; idx < refs.size(); ++idx) {
        Counts ref_ng = count_ngrams(refs[idx], n);
        Counts hyp_ng = count_ngrams(hyps[idx], n);

        int ref_total = 0, hyp_total = 0, match = 0;
        for (auto& [ng, cnt] : ref_ng) {
            ref_total += cnt;
        }
        for (auto& [ng, cnt] : hyp_ng) {
            hyp_total += cnt;
            auto it = ref_ng.find(ng);
            if (it != ref_ng.end()) {
                match += std::min(cnt, it->second);
            }
        }
        float prec =
            hyp_total > 0 ? static_cast<float>(match) / static_cast<float>(hyp_total) : 0.0f;
        float rec =
            ref_total > 0 ? static_cast<float>(match) / static_cast<float>(ref_total) : 0.0f;
        float f1 = (prec + rec > 0.0f) ? 2.0f * prec * rec / (prec + rec) : 0.0f;
        total_f1 += f1;
    }
    return refs.empty() ? 0.0f : total_f1 / static_cast<float>(refs.size());
}

// ─── compute_corpus_rouge_l ───────────────────────────────────────────────────

float GenerationQualityEvaluator::compute_corpus_rouge_l(const std::vector<TokenList>& refs,
                                                         const std::vector<TokenList>& hyps) {
    float total_f1 = 0.0f;
    for (size_t idx = 0; idx < refs.size(); ++idx) {
        int lcs = lcs_length(refs[idx], hyps[idx]);
        float prec = hyps[idx].empty()
                         ? 0.0f
                         : static_cast<float>(lcs) / static_cast<float>(hyps[idx].size());
        float rec = refs[idx].empty()
                        ? 0.0f
                        : static_cast<float>(lcs) / static_cast<float>(refs[idx].size());
        float f1 = (prec + rec > 0.0f) ? 2.0f * prec * rec / (prec + rec) : 0.0f;
        total_f1 += f1;
    }
    return refs.empty() ? 0.0f : total_f1 / static_cast<float>(refs.size());
}

// ─── lcs_length ───────────────────────────────────────────────────────────────

int GenerationQualityEvaluator::lcs_length(const TokenList& a, const TokenList& b) {
    int m = static_cast<int>(a.size());
    int n = static_cast<int>(b.size());
    if (m == 0 || n == 0) {
        return 0;
    }

    std::vector<int> prev(n + 1, 0), curr(n + 1, 0);
    for (int i = 1; i <= m; ++i) {
        for (int j = 1; j <= n; ++j) {
            curr[j] = (a[i - 1] == b[j - 1]) ? prev[j - 1] + 1 : std::max(prev[j], curr[j - 1]);
        }
        std::swap(prev, curr);
        std::fill(curr.begin(), curr.end(), 0);
    }
    return prev[n];
}
