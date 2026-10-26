// @adai-status: stable
// @adai-version: 2.0.0
// @adai-reviewed: 2026-10-25

#include "GenerationQualityMetrics.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <sstream>

// ─── tokenizers (TD-380) ──────────────────────────────────────────────────────

namespace {

void replace_all(std::string& s, const std::string& from, const std::string& to) {
    for (std::size_t pos = 0; (pos = s.find(from, pos)) != std::string::npos; pos += to.size())
        s.replace(pos, from.size(), to);
}

// The first rule of mteval-v13a: these ASCII punctuation characters become separate tokens. (Period,
// comma, hyphen and apostrophe aren't among them; the rules below handle the first three.)
bool is_13a_punct(unsigned char c) {
    return (c >= '{' && c <= '~') || (c >= '[' && c <= '`') || (c >= ' ' && c <= '&') ||
           (c >= '(' && c <= '+') || (c >= ':' && c <= '@') || c == '/';
}

bool is_digit(char c) {
    return c >= '0' && c <= '9';
}

std::vector<std::string> split_whitespace(const std::string& s) {
    std::vector<std::string> tokens;
    std::istringstream iss(s);
    std::string t;
    while (iss >> t)
        tokens.push_back(t);
    return tokens;
}

}  // namespace

std::vector<std::string> GenerationQualityEvaluator::tokenize_13a(const std::string& text) {
    std::string line = text;
    replace_all(line, "<skipped>", "");
    replace_all(line, "-\n", "");
    replace_all(line, "\n", " ");
    if (line.find('&') != std::string::npos) {
        replace_all(line, "&quot;", "\"");
        replace_all(line, "&amp;", "&");
        replace_all(line, "&lt;", "<");
        replace_all(line, "&gt;", ">");
    }

    // s/([\{-\~\[-\` -\&\(-\+\:-\@\/])/ $1 /g
    std::string a;
    for (unsigned char c : line) {
        if (is_13a_punct(c)) {
            a += ' ';
            a += static_cast<char>(c);
            a += ' ';
        } else {
            a += static_cast<char>(c);
        }
    }
    // The next three rules, each a left-to-right, non-overlapping replacement like re.sub:
    // s/([^0-9])([\.,])/$1 $2 /g, s/([\.,])([^0-9])/ $1 $2/g, s/([0-9])(-)/$1 $2 /g
    auto apply = [](const std::string& in, auto first_ok, auto second_ok, const char* pre,
                    const char* mid, const char* post) {
        std::string out;
        std::size_t i = 0;
        while (i < in.size()) {
            if (i + 1 < in.size() && first_ok(in[i]) && second_ok(in[i + 1])) {
                out += pre;
                out += in[i];
                out += mid;
                out += in[i + 1];
                out += post;
                i += 2;
            } else {
                out += in[i++];
            }
        }
        return out;
    };
    const auto not_digit = [](char c) { return !is_digit(c); };
    const auto dot_comma = [](char c) { return c == '.' || c == ','; };
    std::string b = apply(a, not_digit, dot_comma, "", " ", " ");
    std::string c = apply(b, dot_comma, not_digit, " ", " ", "");
    std::string d = apply(c, [](char ch) { return is_digit(ch); }, [](char ch) { return ch == '-'; },
                          "", " ", " ");
    return split_whitespace(d);
}

std::vector<std::string> GenerationQualityEvaluator::tokenize_rouge(const std::string& text) {
    // rouge-score's default tokenizer: lowercase, every run of characters other than [a-z0-9]
    // (non-ASCII letters included) becomes a separator, no stemming.
    std::string s;
    s.reserve(text.size());
    for (unsigned char c : text) {
        if (c >= 'A' && c <= 'Z')
            c = static_cast<unsigned char>(c - 'A' + 'a');
        s += ((c >= 'a' && c <= 'z') || (c >= '0' && c <= '9')) ? static_cast<char>(c) : ' ';
    }
    return split_whitespace(s);
}

// ─── evaluate ────────────────────────────────────────────────────────────────

GenerationQualityScore GenerationQualityEvaluator::evaluate(
    const std::vector<std::string>& references, const std::vector<std::string>& hypotheses) {
    if (references.empty() || references.size() != hypotheses.size()) {
        return {};
    }

    // TD-380: BLEU on sacreBLEU's 13a tokens, ROUGE on rouge-score's tokens, both on 0-100.
    std::vector<TokenList> bleu_refs, bleu_hyps, rouge_refs, rouge_hyps;
    for (size_t i = 0; i < references.size(); ++i) {
        bleu_refs.push_back(tokenize_13a(references[i]));
        bleu_hyps.push_back(tokenize_13a(hypotheses[i]));
        rouge_refs.push_back(tokenize_rouge(references[i]));
        rouge_hyps.push_back(tokenize_rouge(hypotheses[i]));
    }

    GenerationQualityScore score;
    score.bleu1 = compute_bleu(bleu_refs, bleu_hyps, 1, /*effective_order=*/false);
    score.bleu2 = compute_bleu(bleu_refs, bleu_hyps, 2, false);
    score.bleu4 = compute_bleu(bleu_refs, bleu_hyps, 4, false);
    score.rouge1 = 100.0f * compute_corpus_rouge_n(rouge_refs, rouge_hyps, 1);
    score.rouge2 = 100.0f * compute_corpus_rouge_n(rouge_refs, rouge_hyps, 2);
    score.rougeL = 100.0f * compute_corpus_rouge_l(rouge_refs, rouge_hyps);
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

// TD-380: sacreBLEU's BLEU (its compute_bleu() with the defaults): "exp" smoothing, an order with no
// matches counting 1 / (2^k · n-grams) for its k-th such order; 0 when no unigram matches; with
// effective_order (sentence BLEU) the orders the hypotheses are too short for are left out, otherwise
// such an order makes the score 0 (sacreBLEU's log(0) = -9999999999). Scale 0-100.
// (TD-378 had made the corpus score unsmoothed and the sentence score "floor"-smoothed; sacreBLEU's
// defaults replace both, so the numbers match published sacreBLEU scores.)
float GenerationQualityEvaluator::compute_bleu(const std::vector<TokenList>& refs,
                                               const std::vector<TokenList>& hyps, int max_n,
                                               bool effective_order) {
    const BleuStats st = bleu_stats(refs, hyps, max_n);
    std::vector<double> precisions(max_n, 0.0);
    int eff_order = max_n;
    double smooth_mteval = 1.0;
    for (int n = 0; n < max_n; ++n) {
        if (st.totals[n] == 0)
            break;
        if (effective_order)
            eff_order = n + 1;
        if (st.matches[n] == 0) {
            smooth_mteval *= 2.0;
            precisions[n] = 100.0 / (smooth_mteval * st.totals[n]);
        } else {
            precisions[n] = 100.0 * st.matches[n] / st.totals[n];
        }
    }
    if (st.matches[0] == 0)
        return 0.0f;

    double bp = 1.0;
    if (st.hyp_len < st.ref_len)
        bp = st.hyp_len > 0 ? std::exp(1.0 - static_cast<double>(st.ref_len) / st.hyp_len) : 0.0;
    double log_sum = 0.0;
    for (int n = 0; n < eff_order; ++n)
        log_sum += precisions[n] == 0.0 ? -9999999999.0 : std::log(precisions[n]);
    return static_cast<float>(bp * std::exp(log_sum / eff_order));
}

float GenerationQualityEvaluator::sentence_bleu(const std::string& reference,
                                                const std::string& hypothesis, int max_n) {
    const std::vector<TokenList> refs = {tokenize_13a(reference)};
    const std::vector<TokenList> hyps = {tokenize_13a(hypothesis)};
    return compute_bleu(refs, hyps, max_n, /*effective_order=*/true);
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
