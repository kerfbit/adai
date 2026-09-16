// @adai-status: experimental        (TD-038 — new class, first real RLHF integration; see TECHNICAL_DEBT.md; run_iteration() now checks for an empty generated response before encode() instead of after, closing a real uncaught-exception crash)
// @adai-version: 0.1.1
// @adai-reviewed: 2026-09-16

#include "RLHFTrainer.hpp"
#include <algorithm>
#include <cmath>
#include <stdexcept>

RLHFTrainer::RLHFTrainer(EncoderDecoderModel& model, const RLHFConfig& config)
    : model_(model),
      config_(config),
      // RewardModel's own input_dim is prompt_encoding.size() + chosen/rejected_encoding.size()
      // concatenated (see RewardModel::compute_loss()) -- both sides are one d_model-wide
      // mean-pooled encoder vector each, so the model's real input width is 2 * d_model.
      reward_model_(2 * model.get_d_model(), config.reward_model_layer_dims),
      // PPOOptimizer's own reward_model_ pointer is stored but never read anywhere in that
      // class (verified directly) -- the real reward model instance lives here, and this
      // trainer scores rollouts with it itself before ever touching PPOOptimizer.
      ppo_(&reward_model_, config.ppo, model.get_d_model()),
      optimizer_(config.policy_optimizer_type, config.policy_learning_rate) {
    model_.register_parameters(optimizer_);
}

std::vector<float> RLHFTrainer::encode_to_vector(const std::vector<int>& tokens) {
    const int d_model = model_.get_d_model();
    if (tokens.empty()) {
        return std::vector<float>(d_model, 0.0f);
    }

    const int n = static_cast<int>(tokens.size());
    // No masking -- matches EncoderDecoderModel::forward()'s own encoder_mask construction
    // exactly (an all-ones [n, n] matrix), so this reuses the identical encoder behavior the
    // policy's own training already exercises.
    Matrix mask(n, n);
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            mask(i, j) = 1.0f;
        }
    }

    Matrix encoded = model_.get_encoder()->encode_with_mask(tokens, mask);  // [n, d_model]

    std::vector<float> pooled(encoded.cols, 0.0f);
    for (int i = 0; i < encoded.rows; ++i) {
        for (int j = 0; j < encoded.cols; ++j) {
            pooled[j] += encoded(i, j);
        }
    }
    for (float& v : pooled) {
        v /= static_cast<float>(encoded.rows);
    }
    return pooled;
}

float RLHFTrainer::score_response(const std::string& prompt, const std::string& response) {
    BPETokenizer* tok = model_.get_tokenizer();
    if (!tok) {
        throw std::runtime_error("RLHFTrainer: model has no tokenizer set (call set_tokenizer() first)");
    }

    std::vector<float> combined = encode_to_vector(tok->encode(prompt, true));
    std::vector<float> response_vec = encode_to_vector(tok->encode(response, true));
    combined.insert(combined.end(), response_vec.begin(), response_vec.end());
    return reward_model_.predict_reward(combined);
}

float RLHFTrainer::train_reward_model_on_batch(const std::vector<PreferenceExample>& examples,
                                               int epochs) {
    if (examples.empty()) {
        throw std::invalid_argument("RLHFTrainer::train_reward_model_on_batch: no examples given");
    }
    BPETokenizer* tok = model_.get_tokenizer();
    if (!tok) {
        throw std::runtime_error("RLHFTrainer: model has no tokenizer set (call set_tokenizer() first)");
    }

    std::vector<PreferencePair> pairs;
    pairs.reserve(examples.size());
    for (const auto& ex : examples) {
        pairs.emplace_back(encode_to_vector(tok->encode(ex.prompt, true)),
                           encode_to_vector(tok->encode(ex.chosen, true)),
                           encode_to_vector(tok->encode(ex.rejected, true)));
    }

    float last_loss = 0.0f;
    for (int e = 0; e < epochs; ++e) {
        last_loss = reward_model_.train_on_batch(pairs, config_.reward_model_learning_rate);
    }
    return last_loss;
}

float RLHFTrainer::apply_policy_gradient(const std::vector<int>& prompt_tokens,
                                         const std::vector<int>& response_tokens,
                                         const std::vector<float>& advantages) {
    if (response_tokens.empty()) {
        return 0.0f;
    }
    if (advantages.size() < response_tokens.size()) {
        throw std::invalid_argument(
            "RLHFTrainer::apply_policy_gradient: advantages shorter than response_tokens");
    }

    model_.set_training(true);  // backward_pass() below is a no-op unless requires_grad is on
    model_.zero_grad();

    // THE authoritative forward pass: re-derives this exact response's logits under the
    // model's current weights via teacher forcing. Nothing may call into model_.get_encoder()
    // again (directly or via encode_to_vector()/score_response()) until backward_pass() below
    // has run -- see this class's header comment.
    Matrix logits = model_.forward(prompt_tokens, response_tokens);
    const int vocab_size = model_.get_vocab_size();
    const int steps = std::min(static_cast<int>(response_tokens.size()), logits.rows);

    // Numerically-stable softmax per position, same formula
    // EncoderDecoderModel::compute_loss_gradient() already uses.
    Matrix grad(logits.rows, vocab_size);
    float pg_loss_sum = 0.0f;
    for (int t = 0; t < steps; ++t) {
        float max_logit = logits(t, 0);
        for (int v = 1; v < vocab_size; ++v) {
            max_logit = std::max(max_logit, logits(t, v));
        }
        float sum_exp = 0.0f;
        std::vector<float> probs(vocab_size);
        for (int v = 0; v < vocab_size; ++v) {
            probs[v] = std::exp(logits(t, v) - max_logit);
            sum_exp += probs[v];
        }
        for (int v = 0; v < vocab_size; ++v) {
            probs[v] /= sum_exp;
        }

        int action = response_tokens[t];
        // The REAL policy gradient: advantage-weighted (softmax(logits) - one_hot(action)) --
        // EncoderDecoderModel::compute_loss_gradient()'s exact cross-entropy gradient shape,
        // generalized by a per-position advantage weight (advantage ≡ 1 recovers that formula
        // exactly, which is how the sign convention here was checked).
        for (int v = 0; v < vocab_size; ++v) {
            grad(t, v) = advantages[t] * probs[v];
        }
        grad(t, action) -= advantages[t];

        float log_prob = std::log(probs[action] + 1e-10f);
        pg_loss_sum += -advantages[t] * log_prob;
    }

    model_.backward_pass(grad);
    optimizer_.step();

    return steps > 0 ? pg_loss_sum / static_cast<float>(steps) : 0.0f;
}

RLHFStepResult RLHFTrainer::run_iteration(const std::vector<std::string>& prompts) {
    RLHFStepResult result;
    if (prompts.empty()) {
        return result;
    }

    BPETokenizer* tok = model_.get_tokenizer();
    if (!tok) {
        throw std::runtime_error("RLHFTrainer: model has no tokenizer set (call set_tokenizer() first)");
    }

    float total_reward = 0.0f;
    float total_pg_loss = 0.0f;
    float total_ppo_diag_loss = 0.0f;
    int total_response_tokens = 0;

    for (const auto& prompt : prompts) {
        std::vector<int> prompt_tokens = tok->encode(prompt, true);

        // 1. Roll out a response from the CURRENT policy. Safe to call before the encoder-call
        //    ordering constraint kicks in -- generation's own internal caches are fully
        //    superseded by apply_policy_gradient()'s own authoritative forward() call below.
        std::string response = model_.generate_response_with_strategy(
            prompt, config_.max_response_length, config_.generation_strategy, config_.temperature);
        // Checked BEFORE encode(), not after: BPETokenizer::encode() unconditionally throws
        // TokenizerInputError on an empty string (BPETokenizer.cpp's own validate_input()), so an
        // empty response -- which real generation genuinely produces sometimes, e.g. greedy/
        // low-temperature decoding immediately emitting EOS -- used to crash this whole method
        // uncaught instead of being skipped the way the comment already intended. Confirmed via a
        // real, reproduced flake in this exact shape (tests/rlhftrainer_test.cpp's own
        // investigation notes on RunIterationChangesPolicyWeights /
        // PositiveAdvantageIncreasesLogProbNegativeAdvantageDecreasesIt).
        if (response.empty()) {
            continue;  // nothing to learn from an empty response
        }
        std::vector<int> response_tokens = tok->encode(response, true);
        if (response_tokens.empty()) {
            continue;  // encode() can still tokenize a non-empty string to nothing in principle
        }

        // 2. Score the pair. Must happen before the authoritative forward() inside
        //    apply_policy_gradient() below -- see encode_to_vector()'s own doc comment on why
        //    nothing may touch the encoder between forward() and backward_pass().
        float reward = score_response(prompt, response);
        total_reward += reward;

        // 3. Per-step value-function states and old (pre-update) log-probs, both needed to
        //    build the Trajectory. Also must happen before step 5.
        std::vector<std::vector<float>> states;
        states.reserve(response_tokens.size());
        std::vector<int> growing_context = prompt_tokens;
        for (int action : response_tokens) {
            growing_context.push_back(action);
            states.push_back(encode_to_vector(growing_context));
        }

        std::vector<float> old_log_probs(response_tokens.size());
        {
            // A read-only forward() purely to record this rollout's own log-probs before any
            // update -- safe to call here since nothing has touched the encoder in between (the
            // states loop above already finished), and apply_policy_gradient() below will
            // re-derive its own authoritative forward() pass anyway.
            Matrix logits = model_.forward(prompt_tokens, response_tokens);
            const int vocab_size = model_.get_vocab_size();
            const int steps = std::min(static_cast<int>(response_tokens.size()), logits.rows);
            for (int t = 0; t < steps; ++t) {
                float max_logit = logits(t, 0);
                for (int v = 1; v < vocab_size; ++v) {
                    max_logit = std::max(max_logit, logits(t, v));
                }
                float sum_exp = 0.0f;
                for (int v = 0; v < vocab_size; ++v) {
                    sum_exp += std::exp(logits(t, v) - max_logit);
                }
                float log_prob = (logits(t, response_tokens[t]) - max_logit) - std::log(sum_exp);
                old_log_probs[t] = log_prob;
            }
        }

        // 4. Build the Trajectory. The reward-model score applies to the whole response, so it's
        //    assigned entirely to the final token -- standard RLHF practice; GAE (via
        //    PPOOptimizer::compute_advantages() below) propagates credit backward across the
        //    response from there.
        Trajectory traj;
        const int steps = static_cast<int>(response_tokens.size());
        for (int t = 0; t < steps; ++t) {
            float value = ppo_.estimate_value(states[t]);
            float step_reward = (t + 1 == steps) ? reward : 0.0f;
            traj.add_step(states[t], response_tokens[t], step_reward, old_log_probs[t], value);
        }

        // 5. PPOOptimizer::update() trains its own ValueFunction and reports a diagnostic loss;
        //    it cannot and does not touch the policy's weights -- see this class's header
        //    comment. The callback answers with the SAME log-prob computed in step 3 for each
        //    trajectory position, in the same order update() iterates them (index-correlated by
        //    call count) -- correct because nothing has changed the policy between step 3 and
        //    this call, so ratio == 1 for this rollout's own data; real ratio-based clipping
        //    activates once a later rollout is scored against an already-diverged policy.
        size_t log_prob_call_count = 0;
        auto log_prob_fn = [&](const std::vector<float>&, int) -> float {
            size_t idx = steps > 0 ? (log_prob_call_count % static_cast<size_t>(steps)) : 0;
            ++log_prob_call_count;
            return old_log_probs[idx];
        };
        total_ppo_diag_loss += ppo_.update(traj, log_prob_fn);

        // 6. The real policy update -- see apply_policy_gradient()'s own doc comment.
        std::vector<float> advantages = ppo_.compute_advantages(traj);
        total_pg_loss += apply_policy_gradient(prompt_tokens, response_tokens, advantages) *
                        static_cast<float>(steps);
        total_response_tokens += steps;
        result.num_rollouts++;
    }

    if (result.num_rollouts > 0) {
        result.mean_reward = total_reward / static_cast<float>(result.num_rollouts);
        result.ppo_diagnostic_loss = total_ppo_diag_loss / static_cast<float>(result.num_rollouts);
    }
    if (total_response_tokens > 0) {
        result.policy_gradient_loss = total_pg_loss / static_cast<float>(total_response_tokens);
    }
    result.num_response_tokens = total_response_tokens;
    return result;
}
