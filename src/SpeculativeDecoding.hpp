#ifndef SPECULATIVE_DECODING_HPP
#define SPECULATIVE_DECODING_HPP

// @adai-status: stable
// @adai-version: 2.0.0
// @adai-reviewed: 2026-10-26


#include <algorithm>
#include <cmath>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>
#include "Logger.hpp"
#include "Matrix.hpp"
#include "TextGenerator.hpp"

/**
 * @file SpeculativeDecoding.hpp
 * @brief Speculative decoding (speculative sampling)
 *
 * A small, fast "draft" model proposes K tokens one at a time; the "target" model scores all of them
 * in one forward pass. Each proposal is accepted with probability min(1, p(x)/q(x)) (p = target,
 * q = draft); the first rejection is replaced by a sample from norm(max(0, p − q)), and when all K
 * are accepted a bonus token is sampled from the target's next distribution. The output is
 * distributed exactly as sampling from the target alone (Leviathan et al. 2023, Chen et al. 2023).
 * In greedy mode a proposal is accepted only when it is the target's argmax, so the output equals the
 * target's greedy output.
 *
 * Each round costs K draft passes and one target pass and yields between 1 and K + 1 tokens; with
 * acceptance rate α the expected yield is (1 − α^(K+1)) / (1 − α) tokens per target pass. Whether
 * that saves time depends on how much cheaper the draft is than the target.
 *
 * Model functions: a TextGenerator's model_fn returns logits whose last row predicts the token after
 * its input. The target's should return a row per position (at least the last K + 1), so one pass
 * scores every proposal; a target that returns only its last row is called once per prefix it needs
 * instead (the output is the same, the pass count higher).
 *
 * TD-507/TD-508 (October 26, 2026): rewritten. The old version sampled rejections from a residual that
 * subtracted only the rejected token's draft probability, accepted randomly in greedy mode, called
 * the target once per proposal, never sampled a bonus token, overshot max_new_tokens, compared a
 * temperature-scaled draft sample with unscaled probabilities, and reported K·α/(K + 1) (always
 * below 1) as a speedup.
 */

/**
 * @struct SpeculativeDecodingConfig
 * @brief Configuration for speculative decoding
 */
struct SpeculativeDecodingConfig {
    int num_candidates = 4;    // K: tokens the draft proposes per round (at least 1)
    float temperature = 1.0f;  // Applied to both models; <= 0 means greedy (as TextGenerator, TD-515)
    int max_length = 100;      // Default maximum number of new tokens
    bool use_greedy = false;   // Greedy: output equals the target's greedy output

    SpeculativeDecodingConfig() = default;
};

/**
 * @brief Token proposal from draft model
 */
struct TokenProposal {
    int token_id;       // Proposed token (replaced by the resampled token on rejection)
    float draft_prob;   // q(token) under the temperature-adjusted draft distribution
    float target_prob;  // p(token) under the temperature-adjusted target distribution
    bool accepted;      // Whether proposal was accepted

    TokenProposal(int id = 0, float dp = 0.0f)
        : token_id(id), draft_prob(dp), target_prob(0.0f), accepted(false) {}
};

/**
 * @brief Expected tokens per target pass for K proposals at acceptance rate α:
 * (1 − α^(K+1)) / (1 − α), which is K + 1 at α = 1 and 1 at α = 0.
 */
inline float expected_tokens_per_target_pass(int num_candidates, float acceptance_rate) {
    if (num_candidates <= 0 || acceptance_rate <= 0.0f)
        return 1.0f;
    const float alpha = std::min(acceptance_rate, 1.0f);
    if (alpha >= 1.0f)
        return static_cast<float>(num_candidates + 1);
    return (1.0f - std::pow(alpha, static_cast<float>(num_candidates + 1))) / (1.0f - alpha);
}

/**
 * @class SpeculativeDecoder
 * @brief Speculative decoding over two TextGenerators' model functions
 *
 * Example Usage:
 * @code
 * TextGenerator draft_gen, target_gen;
 * draft_gen.set_model_fn(draft_fn);    // last row = next-token logits
 * target_gen.set_model_fn(target_fn);  // a row per position (at least the last K + 1)
 * target_gen.set_tokenizer(&tokenizer);
 *
 * SpeculativeDecodingConfig config;
 * config.num_candidates = 4;
 * SpeculativeDecoder decoder(&draft_gen, &target_gen, config, 42);
 * std::vector<int> tokens = decoder.generate_tokens({bos_id}, 64);
 * float yield = decoder.get_tokens_per_target_pass();
 * @endcode
 */
class SpeculativeDecoder {
   private:
    TextGenerator* draft_generator_;   // Fast draft model generator
    TextGenerator* target_generator_;  // Accurate target model generator
    SpeculativeDecodingConfig config_;

    // Statistics
    int total_proposals_;
    int accepted_proposals_;
    int total_forward_passes_draft_;
    int total_forward_passes_target_;
    int total_generated_tokens_;

    std::mt19937 rng_;

    bool greedy() const {
        return config_.use_greedy || config_.temperature <= 0.0f;
    }

    static void check_config(const SpeculativeDecodingConfig& config) {
        if (config.num_candidates < 1) {
            throw std::invalid_argument("SpeculativeDecoder: num_candidates must be at least 1, got " +
                                        std::to_string(config.num_candidates));
        }
    }

    static int argmax(const std::vector<float>& values) {
        return static_cast<int>(std::max_element(values.begin(), values.end()) - values.begin());
    }

    // softmax(logits / T), the same temperature for both models.
    std::vector<float> distribution(const std::vector<float>& logits) const {
        std::vector<float> probs(logits.size());
        const float max_logit = *std::max_element(logits.begin(), logits.end());
        const float temperature = config_.temperature;
        float sum = 0.0f;
        for (size_t i = 0; i < logits.size(); ++i) {
            probs[i] = std::exp((logits[i] - max_logit) / temperature);
            sum += probs[i];
        }
        for (float& p : probs) {
            p /= sum;
        }
        return probs;
    }

    int sample(const std::vector<float>& probs) {
        std::discrete_distribution<int> dist(probs.begin(), probs.end());
        return dist(rng_);
    }

    static std::vector<float> row(const Matrix& logits, int r) {
        return std::vector<float>(logits.data[r].begin(), logits.data[r].end());
    }

    static void check_logits(const std::vector<float>& logits, const char* model) {
        for (float value : logits) {
            if (!std::isfinite(value) && !(std::isinf(value) && value < 0.0f)) {
                throw std::runtime_error(std::string("SpeculativeDecoder: the ") + model +
                                         " model returned non-finite logits");
            }
        }
    }

    int eos_token_id() const {
        const BPETokenizer* tokenizer = target_generator_->get_tokenizer();
        return tokenizer ? tokenizer->get_eos_token_id()
                         : target_generator_->get_config().eos_token_id;
    }

    // Whether the target's model_fn returns a row per position: unknown until its first call.
    enum class TargetRows { Unknown, PerPosition, LastOnly };
    TargetRows target_rows_mode_ = TargetRows::Unknown;

    /**
     * The target's logits for the token after each prefix `sequence[0 .. base + i]`, i = 0 .. count-1,
     * fetched on demand. A per-position target answers all of them from one pass; a last-row-only
     * target is called once per prefix actually needed.
     */
    class TargetRowSource {
       public:
        TargetRowSource(SpeculativeDecoder& decoder, const std::vector<int>& sequence, int count)
            : decoder_(decoder), sequence_(sequence), count_(count), rows_(count) {
            if (decoder_.target_rows_mode_ != TargetRows::LastOnly) {
                Matrix logits = decoder_.target_generator_->get_logits(sequence_);
                ++decoder_.total_forward_passes_target_;
                if (logits.rows >= count_) {
                    decoder_.target_rows_mode_ = TargetRows::PerPosition;
                    for (int i = 0; i < count_; ++i)
                        rows_[i] = row(logits, logits.rows - count_ + i);
                } else {
                    decoder_.target_rows_mode_ = TargetRows::LastOnly;
                    rows_[count_ - 1] = row(logits, logits.rows - 1);
                }
            }
        }

        const std::vector<float>& get(int i) {
            if (rows_[i].empty()) {
                const int drop = count_ - 1 - i;
                std::vector<int> prefix(sequence_.begin(), sequence_.end() - drop);
                Matrix logits = decoder_.target_generator_->get_logits(prefix);
                ++decoder_.total_forward_passes_target_;
                rows_[i] = row(logits, logits.rows - 1);
            }
            check_logits(rows_[i], "target");
            return rows_[i];
        }

       private:
        SpeculativeDecoder& decoder_;
        const std::vector<int>& sequence_;
        int count_;
        std::vector<std::vector<float>> rows_;
    };

    /**
     * One round: up to `budget` (<= K) draft proposals, one target verification, then the
     * resampled or bonus token. Returns the tokens to append (accepted proposals first).
     */
    std::vector<int> round(const std::vector<int>& context, int budget,
                           std::vector<TokenProposal>& proposals) {
        const int eos = eos_token_id();
        proposals.clear();

        // Step 1: draft proposals, each sampled from q = softmax(draft logits / T) (argmax if greedy).
        std::vector<std::vector<float>> draft_dists;
        std::vector<int> draft_context = context;
        for (int i = 0; i < budget; ++i) {
            Matrix logits = draft_generator_->get_logits(draft_context);
            ++total_forward_passes_draft_;
            std::vector<float> last = row(logits, logits.rows - 1);
            check_logits(last, "draft");
            int token;
            float q = 1.0f;
            if (greedy()) {
                token = argmax(last);
                draft_dists.emplace_back();
            } else {
                draft_dists.push_back(distribution(last));
                token = sample(draft_dists.back());
                q = draft_dists.back()[token];
            }
            proposals.emplace_back(token, q);
            draft_context.push_back(token);
            if (token == eos)
                break;
        }

        // Step 2: one target pass over context + proposals; row i predicts proposal i, row n the bonus.
        const int n = static_cast<int>(proposals.size());
        TargetRowSource rows(*this, draft_context, n + 1);
        if (!draft_dists.empty() && !draft_dists.front().empty() &&
            draft_dists.front().size() != rows.get(0).size()) {
            throw std::runtime_error("SpeculativeDecoder: draft and target vocabularies differ");
        }

        // Step 3: accept or replace each proposal in order.
        std::vector<int> out;
        for (int i = 0; i < n; ++i) {
            TokenProposal& proposal = proposals[i];
            ++total_proposals_;
            if (greedy()) {
                const int best = argmax(rows.get(i));
                proposal.target_prob = proposal.token_id == best ? 1.0f : 0.0f;
                if (proposal.token_id == best) {
                    proposal.accepted = true;
                } else {
                    proposal.token_id = best;
                }
            } else {
                const std::vector<float> p = distribution(rows.get(i));
                const std::vector<float>& q = draft_dists[i];
                const int token = proposal.token_id;
                proposal.target_prob = p[token];
                std::uniform_real_distribution<float> uniform(0.0f, 1.0f);
                if (uniform(rng_) < std::min(1.0f, p[token] / std::max(q[token], 1e-30f))) {
                    proposal.accepted = true;
                } else {
                    // Residual norm(max(0, p − q)) over the whole draft distribution.
                    std::vector<float> residual(p.size());
                    float sum = 0.0f;
                    for (size_t j = 0; j < p.size(); ++j) {
                        residual[j] = std::max(0.0f, p[j] - q[j]);
                        sum += residual[j];
                    }
                    proposal.token_id = sum > 0.0f ? sample(residual) : sample(p);
                }
            }
            out.push_back(proposal.token_id);
            if (!proposal.accepted) {
                return out;
            }
            ++accepted_proposals_;
            if (proposal.token_id == eos) {
                return out;
            }
        }

        // All accepted: the target's next distribution gives a bonus token for free.
        out.push_back(greedy() ? argmax(rows.get(n)) : sample(distribution(rows.get(n))));
        return out;
    }

   public:
    /**
     * @brief Construct speculative decoder
     *
     * @param draft_generator Generator using draft (fast) model
     * @param target_generator Generator using target (accurate) model
     * @param config Speculative decoding configuration (num_candidates >= 1)
     * @param seed Seed for acceptance and sampling (TD-508: was always random)
     */
    SpeculativeDecoder(TextGenerator* draft_generator, TextGenerator* target_generator,
                       const SpeculativeDecodingConfig& config = SpeculativeDecodingConfig(),
                       unsigned int seed = std::random_device{}())
        : draft_generator_(draft_generator),
          target_generator_(target_generator),
          config_(config),
          total_proposals_(0),
          accepted_proposals_(0),
          total_forward_passes_draft_(0),
          total_forward_passes_target_(0),
          total_generated_tokens_(0),
          rng_(seed) {
        if (!draft_generator || !target_generator) {
            throw std::invalid_argument("Generators cannot be null");
        }
        check_config(config_);
    }

    /**
     * @brief Generate text using speculative decoding
     *
     * @param prompt Input prompt text (tokenized with the target's tokenizer, special tokens added)
     * @return Generated text
     */
    std::string generate(const std::string& prompt) {
        auto tokenizer = target_generator_->get_tokenizer();
        if (!tokenizer) {
            throw std::runtime_error("SpeculativeDecoder: the target generator has no tokenizer");
        }
        return tokenizer->decode(generate_tokens(tokenizer->encode(prompt), config_.max_length));
    }

    /**
     * @brief Generate tokens (returns the new token IDs only)
     *
     * @param prompt_tokens Input prompt as token IDs; empty starts from the target's BOS
     * @param max_new_tokens Maximum new tokens (-1 = config max_length); the output never exceeds it
     * @return Generated token IDs, ending at EOS if one was generated
     */
    std::vector<int> generate_tokens(const std::vector<int>& prompt_tokens, int max_new_tokens = -1) {
        if (max_new_tokens < 0) {
            max_new_tokens = config_.max_length;
        }

        std::vector<int> context = prompt_tokens;
        if (context.empty()) {
            context.push_back(target_generator_->get_config().bos_token_id);
        }
        const int eos = eos_token_id();

        std::vector<int> output_tokens;
        std::vector<TokenProposal> proposals;
        while (static_cast<int>(output_tokens.size()) < max_new_tokens) {
            const int remaining = max_new_tokens - static_cast<int>(output_tokens.size());
            // A round yields at most budget + 1 tokens; the cut below keeps the bonus from overshooting.
            const int budget = std::min(config_.num_candidates, remaining);
            std::vector<int> tokens = round(context, budget, proposals);
            for (int token : tokens) {
                if (static_cast<int>(output_tokens.size()) >= max_new_tokens)
                    break;
                output_tokens.push_back(token);
                context.push_back(token);
                ++total_generated_tokens_;
                if (token == eos)
                    return output_tokens;
            }
        }
        return output_tokens;
    }

    /**
     * @brief Fraction of proposals accepted
     */
    float get_acceptance_rate() const {
        if (total_proposals_ == 0)
            return 0.0f;
        return static_cast<float>(accepted_proposals_) / total_proposals_;
    }

    /**
     * @brief Generated tokens per target forward pass (1 for plain decoding; up to K + 1)
     *
     * TD-508: replaces get_speedup(), which returned K·α/(K + 1), below 1 for every rate. Time saved
     * also depends on the draft's cost relative to the target's.
     */
    float get_tokens_per_target_pass() const {
        if (total_forward_passes_target_ == 0)
            return 0.0f;
        return static_cast<float>(total_generated_tokens_) / total_forward_passes_target_;
    }

    int get_draft_passes() const {
        return total_forward_passes_draft_;
    }
    int get_target_passes() const {
        return total_forward_passes_target_;
    }
    int get_generated_tokens() const {
        return total_generated_tokens_;
    }

    /**
     * @brief Reset statistics
     */
    void reset_stats() {
        total_proposals_ = 0;
        accepted_proposals_ = 0;
        total_forward_passes_draft_ = 0;
        total_forward_passes_target_ = 0;
        total_generated_tokens_ = 0;
    }

    /**
     * @brief Log statistics (TD-508: through Logger, not std::cout)
     */
    void log_stats() const {
        adai::Logger::info(
            "Speculative decoding: {} proposals, {} accepted ({:.1f}%), {} tokens, {} target passes "
            "({:.2f} tokens/pass), {} draft passes",
            total_proposals_, accepted_proposals_, get_acceptance_rate() * 100.0f,
            total_generated_tokens_, total_forward_passes_target_, get_tokens_per_target_pass(),
            total_forward_passes_draft_);
    }

    /**
     * @brief Get configuration
     */
    const SpeculativeDecodingConfig& get_config() const {
        return config_;
    }

    /**
     * @brief Update configuration (num_candidates >= 1)
     */
    void set_config(const SpeculativeDecodingConfig& config) {
        check_config(config);
        config_ = config;
    }
};

#endif  // SPECULATIVE_DECODING_HPP
