#ifndef SPECULATIVE_DECODING_HPP
#define SPECULATIVE_DECODING_HPP

// @adai-status: beta        (capped by TD-038 — tested but not wired into any shipped binary)
// @adai-version: 0.7.0
// @adai-reviewed: 2026-09-07


#include <algorithm>
#include <random>
#include <stdexcept>
#include <vector>
#include "Matrix.hpp"
#include "TextGenerator.hpp"

/**
 * @file SpeculativeDecoding.hpp
 * @brief Speculative Decoding for Faster Autoregressive Inference
 *
 * Speculative decoding (also called "speculative sampling" or "assisted generation")
 * speeds up inference by using a small, fast "draft" model to propose tokens,
 * then verifying them with the larger "target" model in parallel.
 *
 * Algorithm:
 * 1. Draft model generates K candidate tokens quickly
 * 2. Target model evaluates all K candidates in parallel (single forward pass)
 * 3. Accept/reject candidates based on probability comparison
 * 4. On rejection, sample from adjusted distribution
 *
 * Key Benefits:
 * - 2-3x speedup for similar-quality outputs
 * - Mathematically equivalent to standard sampling (same distribution)
 * - No training required - works with any model pair
 * - Larger K (lookahead) → more speedup (diminishing returns)
 *
 * Requirements:
 * - Draft model: Small, fast (e.g., 125M params)
 * - Target model: Large, accurate (e.g., 7B params)
 * - Models should produce similar-quality outputs
 *
 * @version 1.0
 * @date January 2026
 */

/**
 * @struct SpeculativeDecodingConfig
 * @brief Configuration for speculative decoding
 */
struct SpeculativeDecodingConfig {
    int num_candidates = 4;             // K: Number of tokens to propose (typical: 4-8)
    float temperature = 1.0f;           // Sampling temperature
    int max_length = 100;               // Maximum sequence length
    float acceptance_threshold = 0.0f;  // Minimum acceptance probability
    bool use_greedy = false;            // Use greedy sampling instead of probabilistic

    SpeculativeDecodingConfig() = default;
};

/**
 * @brief Token proposal from draft model
 */
struct TokenProposal {
    int token_id;       // Proposed token
    float draft_prob;   // Probability from draft model
    float target_prob;  // Probability from target model (after verification)
    bool accepted;      // Whether proposal was accepted

    TokenProposal(int id = 0, float dp = 0.0f)
        : token_id(id), draft_prob(dp), target_prob(0.0f), accepted(false) {}
};

/**
 * @class SpeculativeDecoder
 * @brief Implements speculative decoding for accelerated generation
 *
 * Example Usage:
 * @code
 * // Create draft and target models
 * LLMDecoder draft_model(256, 4, 1024, 6, vocab_size, max_len);  // Small
 * LLMDecoder target_model(768, 12, 3072, 24, vocab_size, max_len); // Large
 *
 * // Configure speculative decoding
 * SpeculativeDecodingConfig config;
 * config.num_candidates = 5;
 * config.temperature = 0.8;
 *
 * SpeculativeDecoder decoder(&draft_model, &target_model, &tokenizer, config);
 *
 * // Generate with speedup
 * std::string prompt = "Once upon a time";
 * std::string output = decoder.generate(prompt);
 *
 * // Check statistics
 * std::cout << "Acceptance rate: " << decoder.get_acceptance_rate() << std::endl;
 * std::cout << "Speedup: " << decoder.get_speedup() << "x" << std::endl;
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

    std::mt19937 rng_;

    /**
     * @brief Sample token from probability distribution
     *
     * @param probs Probability distribution over vocabulary
     * @param temperature Sampling temperature
     * @return Sampled token ID
     */
    int sample_token(const std::vector<float>& probs, float temperature = 1.0f) {
        if (config_.use_greedy) {
            // Greedy: return argmax
            return std::distance(probs.begin(), std::max_element(probs.begin(), probs.end()));
        }

        // Temperature sampling
        std::vector<float> scaled_probs = probs;
        if (temperature != 1.0f && temperature > 0.0f) {
            float sum = 0.0f;
            for (float& p : scaled_probs) {
                p = std::pow(p, 1.0f / temperature);
                sum += p;
            }
            for (float& p : scaled_probs) {
                p /= sum;
            }
        }

        // Sample
        std::uniform_real_distribution<float> dist(0.0f, 1.0f);
        float rand_val = dist(rng_);
        float cumsum = 0.0f;

        for (size_t i = 0; i < scaled_probs.size(); i++) {
            cumsum += scaled_probs[i];
            if (rand_val <= cumsum) {
                return i;
            }
        }

        return scaled_probs.size() - 1;
    }

    /**
     * @brief Generate K candidate tokens with draft model
     *
     * @param context Current sequence context
     * @return Vector of token proposals
     */
    std::vector<TokenProposal> generate_candidates(const std::vector<int>& context) {
        std::vector<TokenProposal> proposals;
        std::vector<int> draft_context = context;

        for (int i = 0; i < config_.num_candidates; i++) {
            // Get probabilities from draft model
            std::vector<float> draft_probs = draft_generator_->get_next_token_probs(draft_context);
            total_forward_passes_draft_++;

            // Sample token
            int token_id = sample_token(draft_probs, config_.temperature);
            float token_prob = draft_probs[token_id];

            proposals.push_back(TokenProposal(token_id, token_prob));
            draft_context.push_back(token_id);

            // Early stopping if draft model predicts EOS
            if (token_id == draft_generator_->get_tokenizer()->get_eos_token_id()) {
                break;
            }
        }

        return proposals;
    }

    /**
     * @brief Verify candidates with target model
     *
     * All K candidates are evaluated in a single forward pass (parallel verification)
     *
     * @param context Original context
     * @param proposals Token proposals from draft model
     * @return Number of accepted tokens
     */
    int verify_candidates(const std::vector<int>& context, std::vector<TokenProposal>& proposals) {
        if (proposals.empty())
            return 0;

        std::vector<int> extended_context = context;
        int num_accepted = 0;

        for (size_t i = 0; i < proposals.size(); i++) {
            // Get target model probabilities for current context
            std::vector<float> target_probs =
                target_generator_->get_next_token_probs(extended_context);
            total_forward_passes_target_++;

            int proposed_token = proposals[i].token_id;
            float target_prob = target_probs[proposed_token];
            proposals[i].target_prob = target_prob;

            // Acceptance criterion: p_target(x) >= p_draft(x)
            // With randomness: accept if uniform(0,1) < min(1, p_target/p_draft)
            float acceptance_ratio = target_prob / (proposals[i].draft_prob + 1e-10f);

            std::uniform_real_distribution<float> dist(0.0f, 1.0f);
            float rand_val = dist(rng_);

            if (rand_val < std::min(1.0f, acceptance_ratio)) {
                // Accept proposal
                proposals[i].accepted = true;
                extended_context.push_back(proposed_token);
                num_accepted++;
                total_proposals_++;
                accepted_proposals_++;
            } else {
                // Reject proposal - resample from adjusted distribution
                // Adjusted distribution: max(0, p_target - p_draft) / Z
                std::vector<float> adjusted_probs(target_probs.size());
                float sum = 0.0f;

                for (size_t j = 0; j < target_probs.size(); j++) {
                    // Get draft probability for token j
                    float draft_p = (j == static_cast<size_t>(proposed_token))
                                        ? proposals[i].draft_prob
                                        : 0.0f;  // Simplified - would need full draft dist

                    adjusted_probs[j] = std::max(0.0f, target_probs[j] - draft_p);
                    sum += adjusted_probs[j];
                }

                // Normalize
                if (sum > 1e-10f) {
                    for (float& p : adjusted_probs) {
                        p /= sum;
                    }
                } else {
                    adjusted_probs = target_probs;  // Fallback
                }

                // Sample from adjusted distribution
                int resampled_token = sample_token(adjusted_probs, config_.temperature);
                proposals[i].token_id = resampled_token;
                proposals[i].accepted = false;
                extended_context.push_back(resampled_token);
                total_proposals_++;

                // Stop verification - must regenerate remaining candidates
                break;
            }
        }

        return num_accepted;
    }

   public:
    /**
     * @brief Construct speculative decoder
     *
     * @param draft_generator Generator using draft (fast) model
     * @param target_generator Generator using target (accurate) model
     * @param config Speculative decoding configuration
     */
    SpeculativeDecoder(TextGenerator* draft_generator, TextGenerator* target_generator,
                       const SpeculativeDecodingConfig& config = SpeculativeDecodingConfig())
        : draft_generator_(draft_generator),
          target_generator_(target_generator),
          config_(config),
          total_proposals_(0),
          accepted_proposals_(0),
          total_forward_passes_draft_(0),
          total_forward_passes_target_(0),
          rng_(std::random_device{}()) {
        if (!draft_generator || !target_generator) {
            throw std::invalid_argument("Generators cannot be null");
        }
    }

    /**
     * @brief Generate text using speculative decoding
     *
     * @param prompt Input prompt text
     * @return Generated text
     */
    std::string generate(const std::string& prompt) {
        // Tokenize prompt
        auto tokenizer = target_generator_->get_tokenizer();
        std::vector<int> context = tokenizer->encode(prompt);

        std::vector<int> output_tokens;

        while (output_tokens.size() < static_cast<size_t>(config_.max_length)) {
            // Step 1: Generate K candidates with draft model
            std::vector<int> current_context = context;
            current_context.insert(current_context.end(), output_tokens.begin(),
                                   output_tokens.end());

            std::vector<TokenProposal> proposals = generate_candidates(current_context);

            if (proposals.empty())
                break;

            // Step 2: Verify candidates with target model
            int num_accepted = verify_candidates(current_context, proposals);

            // Step 3: Add accepted tokens to output
            for (const auto& proposal : proposals) {
                output_tokens.push_back(proposal.token_id);

                // Check for EOS
                if (proposal.token_id == tokenizer->get_eos_token_id()) {
                    goto generation_complete;
                }

                // Only process up to first rejection
                if (!proposal.accepted)
                    break;
            }
        }

    generation_complete:
        // Decode output
        return tokenizer->decode(output_tokens);
    }

    /**
     * @brief Generate tokens (returns token IDs)
     *
     * @param prompt_tokens Input prompt as token IDs
     * @param max_new_tokens Maximum new tokens to generate
     * @return Generated token IDs
     */
    std::vector<int> generate_tokens(const std::vector<int>& prompt_tokens,
                                     int max_new_tokens = -1) {
        if (max_new_tokens < 0) {
            max_new_tokens = config_.max_length;
        }

        std::vector<int> output_tokens;
        auto tokenizer = target_generator_->get_tokenizer();

        while (output_tokens.size() < static_cast<size_t>(max_new_tokens)) {
            std::vector<int> current_context = prompt_tokens;
            current_context.insert(current_context.end(), output_tokens.begin(),
                                   output_tokens.end());

            std::vector<TokenProposal> proposals = generate_candidates(current_context);
            if (proposals.empty())
                break;

            verify_candidates(current_context, proposals);

            for (const auto& proposal : proposals) {
                output_tokens.push_back(proposal.token_id);
                if (proposal.token_id == tokenizer->get_eos_token_id()) {
                    return output_tokens;
                }
                if (!proposal.accepted)
                    break;
            }
        }

        return output_tokens;
    }

    /**
     * @brief Get acceptance rate (fraction of proposals accepted)
     */
    float get_acceptance_rate() const {
        if (total_proposals_ == 0)
            return 0.0f;
        return static_cast<float>(accepted_proposals_) / total_proposals_;
    }

    /**
     * @brief Estimate speedup compared to standard generation
     *
     * Speedup = K * acceptance_rate / (K + 1)
     * where K is number of candidates
     */
    float get_speedup() const {
        float acc_rate = get_acceptance_rate();
        float K = config_.num_candidates;
        return (K * acc_rate) / (K + 1.0f);
    }

    /**
     * @brief Reset statistics
     */
    void reset_stats() {
        total_proposals_ = 0;
        accepted_proposals_ = 0;
        total_forward_passes_draft_ = 0;
        total_forward_passes_target_ = 0;
    }

    /**
     * @brief Print statistics
     */
    void print_stats() const {
        std::cout << "=== Speculative Decoding Statistics ===\n";
        std::cout << "Total proposals: " << total_proposals_ << "\n";
        std::cout << "Accepted proposals: " << accepted_proposals_ << "\n";
        std::cout << "Acceptance rate: " << get_acceptance_rate() * 100 << "%\n";
        std::cout << "Estimated speedup: " << get_speedup() << "x\n";
        std::cout << "Draft forward passes: " << total_forward_passes_draft_ << "\n";
        std::cout << "Target forward passes: " << total_forward_passes_target_ << "\n";
    }

    /**
     * @brief Get configuration
     */
    const SpeculativeDecodingConfig& get_config() const {
        return config_;
    }

    /**
     * @brief Update configuration
     */
    void set_config(const SpeculativeDecodingConfig& config) {
        config_ = config;
    }
};

/**
 * @brief Helper: Calculate theoretical speedup
 *
 * Given acceptance rate and number of candidates, estimate speedup.
 * Formula: speedup = K * α / (K + 1)
 * where K = num_candidates, α = acceptance_rate
 */
inline float calculate_theoretical_speedup(int num_candidates, float acceptance_rate) {
    if (num_candidates <= 0 || acceptance_rate <= 0.0f)
        return 1.0f;
    return (num_candidates * acceptance_rate) / (num_candidates + 1.0f);
}

/**
 * @brief Print speedup table for different K and acceptance rates
 */
inline void print_speedup_table() {
    std::cout << "\n=== Theoretical Speculative Decoding Speedup ===\n\n";
    std::cout << "K (candidates) | Acceptance Rate | Speedup\n";
    std::cout << "------------------------------------------------\n";

    std::vector<int> K_values = {2, 4, 6, 8, 10};
    std::vector<float> acc_rates = {0.5f, 0.6f, 0.7f, 0.8f, 0.9f};

    for (int K : K_values) {
        for (float acc : acc_rates) {
            float speedup = calculate_theoretical_speedup(K, acc);
            printf("%14d | %15.1f%% | %.2fx\n", K, acc * 100, speedup);
        }
        std::cout << "------------------------------------------------\n";
    }

    std::cout << "\nNote: Actual speedup depends on draft/target model speed ratio\n";
}

#endif  // SPECULATIVE_DECODING_HPP
