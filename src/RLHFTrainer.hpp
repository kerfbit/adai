#ifndef RLHF_TRAINER_HPP
#define RLHF_TRAINER_HPP

// @adai-status: experimental        (TD-038 — new class, first real RLHF integration; see TECHNICAL_DEBT.md)
// @adai-version: 0.1.0
// @adai-reviewed: 2026-09-13


#include <string>
#include <vector>
#include "EncoderDecoderModel.hpp"
#include "Optimizer.hpp"
#include "PPOOptimizer.hpp"
#include "RewardModel.hpp"

/**
 * @file RLHFTrainer.hpp
 * @brief TD-038: the actual RLHF integration RewardModel/PPOOptimizer were missing.
 *
 * Both RewardModel and PPOOptimizer had real, tested logic but nothing in the codebase ever
 * drove them against a real EncoderDecoderModel policy. Two gaps specifically had to be closed
 * here, neither of which existed anywhere else in the tree:
 *
 * 1. **An encoding bridge.** RewardModel takes fixed-width `std::vector<float>` encodings, not
 *    raw token sequences. This trainer builds them by mean-pooling `LLMEncoder::encode_with_mask()`'s
 *    per-token output across positions — the same encoder representations the policy model
 *    itself already learns, reused rather than inventing a second embedding scheme.
 *
 * 2. **Applying the policy gradient.** PPOOptimizer::update() computes a real clipped-ratio
 *    policy loss and trains its own internal ValueFunction, but — as its own doc comment
 *    states — it has no policy reference of its own and never touches any model's weights.
 *    This class builds the actual advantage-weighted gradient at the logits level (the same
 *    `softmax(logits) - one_hot(action)` shape `EncoderDecoderModel::compute_loss_gradient()`
 *    already uses for ordinary cross-entropy training, generalized by a per-position GAE
 *    advantage weight — advantage ≡ 1 everywhere reduces to that exact formula, which is how
 *    the sign convention here was checked against already-tested code) and applies it via
 *    `EncoderDecoderModel::backward_pass()` + a real `Optimizer::step()`, the same
 *    zero_grad → forward → backward_pass → step shape `ChatbotTrainer` already uses in
 *    production (deliberately NOT `EncoderDecoderModel::update_weights()`, which — per its own
 *    comment — never actually updates the encoder's weights at all; only the external-optimizer
 *    path updates every component).
 *
 * Scope note: this applies exactly one advantage-weighted gradient step per rollout batch, using
 * the rollout-time policy's own log-probs as `old_log_probs`. That makes PPOOptimizer's
 * clipped-ratio term inert on the very rollout it was computed from (ratio ≡ 1, since nothing has
 * changed the policy yet) — mathematically this is the well-known "one gradient step per
 * rollout" limit of PPO, equivalent to REINFORCE with a learned (GAE) baseline. The clipping
 * genuinely activates across iterations, once a later rollout is scored against a policy that
 * diverged from whichever one generated it. A caller wanting multiple gradient epochs per
 * rollout (real intra-rollout PPO clipping) would need to re-run `forward()` between epochs to
 * get genuinely updated log-probs — not implemented here; see TECHNICAL_DEBT.md.
 */

/** One human-labeled preference example, as raw text — this class handles encoding it. */
struct PreferenceExample {
    std::string prompt;
    std::string chosen;
    std::string rejected;
};

struct RLHFConfig {
    PPOConfig ppo;
    std::vector<int> reward_model_layer_dims = {256, 128, 1};
    float reward_model_learning_rate = 1e-4f;
    OptimizerType policy_optimizer_type = OptimizerType::ADAM;
    float policy_learning_rate = 1e-5f;
    int max_response_length = 64;
    std::string generation_strategy = "sampling";
    float temperature = 1.0f;

    RLHFConfig() = default;
};

/** Diagnostics from one run_iteration() call. */
struct RLHFStepResult {
    float mean_reward = 0.0f;
    // The REAL loss actually backpropagated into the policy this iteration: mean over response
    // tokens of -advantage * log_prob(action taken). This is what run_iteration() optimizes.
    float policy_gradient_loss = 0.0f;
    // PPOOptimizer::update()'s own return value -- ValueFunction-training diagnostics /
    // clipped-ratio bookkeeping. Does NOT reflect the policy_gradient_loss above (see this
    // header's own top comment for why PPOOptimizer never touches the policy at all).
    float ppo_diagnostic_loss = 0.0f;
    int num_rollouts = 0;
    int num_response_tokens = 0;
};

/**
 * @class RLHFTrainer
 * @brief Drives one real reward-model-training or PPO-policy-update pass against a live
 * EncoderDecoderModel policy.
 *
 * Does not own the policy model -- constructed against an existing, already-initialized
 * EncoderDecoderModel (with a real tokenizer already set via set_tokenizer()), matching how
 * ChatbotTrainer/IncrementalTrainer already hold their own model.
 */
class RLHFTrainer {
   public:
    RLHFTrainer(EncoderDecoderModel& model, const RLHFConfig& config = RLHFConfig());

    /**
     * @brief Trains the internal RewardModel on preference data for `epochs` passes.
     * @return The final epoch's average Bradley-Terry loss.
     */
    float train_reward_model_on_batch(const std::vector<PreferenceExample>& examples,
                                      int epochs = 1);

    /**
     * @brief Runs one full rollout-and-update cycle: generates a response per prompt with the
     * current policy, scores it with the reward model, and applies one real advantage-weighted
     * gradient step to the policy's weights (see this header's own top comment for the exact
     * mechanism and its scope).
     */
    RLHFStepResult run_iteration(const std::vector<std::string>& prompts);

    /**
     * @brief The core mechanism run_iteration() applies per rollout, exposed directly so it can
     * be exercised (and tested) with caller-supplied advantages, independent of this trainer's
     * own reward-model-scoring and GAE machinery -- see this header's own top comment for the
     * exact gradient formula. `advantages.size()` must be >= `response_tokens.size()`; only the
     * first `response_tokens.size()` entries are used.
     *
     * Calls model.zero_grad()/backward_pass()/optimizer step() itself -- a single, complete
     * gradient step for this one (prompt, response) pair. Does NOT call generate/score/GAE.
     *
     * @return The mean per-token policy-gradient loss actually applied (-advantage * log_prob,
     *   averaged over response_tokens.size()).
     */
    float apply_policy_gradient(const std::vector<int>& prompt_tokens,
                                const std::vector<int>& response_tokens,
                                const std::vector<float>& advantages);

    RewardModel& reward_model() {
        return reward_model_;
    }
    PPOOptimizer& ppo() {
        return ppo_;
    }

   private:
    /** Mean-pools LLMEncoder::encode_with_mask()'s per-token output into one d_model-wide
     * vector. MUST NOT be called between EncoderDecoderModel::forward() and
     * EncoderDecoderModel::backward_pass() -- see this header's own top comment: every encoder
     * call (this one included) overwrites EncoderBlock's own internal activation caches
     * unconditionally, which forward()'s matching backward_pass() depends on. */
    std::vector<float> encode_to_vector(const std::vector<int>& tokens);

    /** Concatenates encode_to_vector(prompt) ++ encode_to_vector(response) and scores it with
     * the reward model. Same encoder-call ordering constraint as encode_to_vector() itself. */
    float score_response(const std::string& prompt, const std::string& response);

    EncoderDecoderModel& model_;
    RLHFConfig config_;
    RewardModel reward_model_;
    PPOOptimizer ppo_;
    Optimizer optimizer_;
};

#endif  // RLHF_TRAINER_HPP
