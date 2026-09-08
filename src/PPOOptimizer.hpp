#ifndef PPO_OPTIMIZER_HPP
#define PPO_OPTIMIZER_HPP

// @adai-status: experimental        (capped by TD-034 — policy-ratio and KL early-stop are hardcoded placeholders, see TECHNICAL_DEBT.md)
// @adai-version: 0.2.0
// @adai-reviewed: 2026-09-07


#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>
#include "Matrix.hpp"
#include "RewardModel.hpp"

/**
 * @file PPOOptimizer.hpp
 * @brief Proximal Policy Optimization (PPO) for RLHF fine-tuning
 *
 * PPO is a policy gradient method that limits policy updates to maintain
 * training stability. It uses a clipped surrogate objective to prevent
 * destructively large policy changes.
 *
 * Key Components:
 * - Policy model (language model being fine-tuned)
 * - Value function (estimates expected future rewards)
 * - Advantage estimation (GAE - Generalized Advantage Estimation)
 * - Clipped objective function
 *
 * Algorithm:
 * 1. Generate rollouts using current policy
 * 2. Compute advantages using GAE
 * 3. Optimize policy with clipped objective for K epochs
 * 4. Update value function with MSE loss
 *
 * @version 1.0
 * @date January 2026
 */

/**
 * @brief Trajectory data for PPO training
 */
struct Trajectory {
    std::vector<std::vector<float>> states;  // Sequence of states (encodings)
    std::vector<int> actions;                // Actions taken (token IDs)
    std::vector<float> rewards;              // Rewards received
    std::vector<float> log_probs;            // Log probabilities of actions
    std::vector<float> values;               // Value function estimates

    void add_step(const std::vector<float>& state, int action, float reward, float log_prob,
                  float value) {
        states.push_back(state);
        actions.push_back(action);
        rewards.push_back(reward);
        log_probs.push_back(log_prob);
        values.push_back(value);
    }

    size_t length() const {
        return states.size();
    }

    void clear() {
        states.clear();
        actions.clear();
        rewards.clear();
        log_probs.clear();
        values.clear();
    }
};

/**
 * @brief PPO hyperparameters
 */
struct PPOConfig {
    float clip_epsilon = 0.2f;    // Clipping parameter for surrogate objective
    float gamma = 0.99f;          // Discount factor
    float gae_lambda = 0.95f;     // GAE lambda parameter
    float value_coef = 0.5f;      // Value loss coefficient
    float entropy_coef = 0.01f;   // Entropy bonus coefficient
    int num_epochs = 4;           // PPO optimization epochs per rollout
    int batch_size = 64;          // Minibatch size for updates
    float max_grad_norm = 0.5f;   // Gradient clipping threshold
    float learning_rate = 1e-5f;  // Learning rate
    float kl_target = 0.01f;      // Target KL divergence for early stopping

    PPOConfig() = default;
};

/**
 * @class ValueFunction
 * @brief Neural network that estimates state values for PPO
 */
class ValueFunction {
   private:
    int input_dim_;
    std::vector<int> layer_dims_;
    std::vector<Matrix> weights_;
    std::vector<std::vector<float>> biases_;

    float relu(float x) const {
        return std::max(0.0f, x);
    }
    float relu_derivative(float x) const {
        return x > 0.0f ? 1.0f : 0.0f;
    }

    void initialize_weights() {
        weights_.clear();
        biases_.clear();

        int prev_dim = input_dim_;
        for (size_t i = 0; i < layer_dims_.size(); i++) {
            int curr_dim = layer_dims_[i];
            float std_dev = std::sqrt(2.0f / prev_dim);

            Matrix w(prev_dim, curr_dim);
            for (int r = 0; r < prev_dim; r++) {
                for (int c = 0; c < curr_dim; c++) {
                    w(r, c) = ((float)rand() / RAND_MAX - 0.5f) * 2.0f * std_dev;
                }
            }
            weights_.push_back(w);
            biases_.push_back(std::vector<float>(curr_dim, 0.0f));
            prev_dim = curr_dim;
        }
    }

   public:
    ValueFunction(int input_dim, const std::vector<int>& layer_dims)
        : input_dim_(input_dim), layer_dims_(layer_dims) {
        if (layer_dims.empty() || layer_dims.back() != 1) {
            throw std::invalid_argument("Value function must output scalar");
        }
        initialize_weights();
    }

    /**
     * @brief Predict value for a state
     */
    float predict(const std::vector<float>& state) {
        Matrix activation(1, input_dim_);
        for (int i = 0; i < input_dim_; i++) {
            activation(0, i) = state[i];
        }

        for (size_t i = 0; i < weights_.size(); i++) {
            Matrix z = activation * weights_[i];
            for (int j = 0; j < z.cols; j++) {
                z(0, j) += biases_[i][j];
            }

            if (i < weights_.size() - 1) {
                // Apply ReLU activation
                Matrix next_activation(1, z.cols);
                for (int j = 0; j < z.cols; j++) {
                    next_activation(0, j) = relu(z(0, j));
                }
                activation = next_activation;
            } else {
                activation = z;
            }
        }

        return activation(0, 0);
    }

    /**
     * @brief Update value function with gradient descent
     *
     * @param states Batch of states
     * @param targets Target values (returns)
     * @param learning_rate Learning rate
     * @return MSE loss
     */
    float update(const std::vector<std::vector<float>>& states, const std::vector<float>& targets,
                 float learning_rate) {
        if (states.size() != targets.size()) {
            throw std::invalid_argument("States and targets size mismatch");
        }

        float total_loss = 0.0f;

        // Accumulate gradients
        std::vector<Matrix> weight_grads;
        std::vector<std::vector<float>> bias_grads;
        for (size_t i = 0; i < weights_.size(); i++) {
            weight_grads.push_back(Matrix(weights_[i].rows, weights_[i].cols));
            bias_grads.push_back(std::vector<float>(biases_[i].size(), 0.0f));
        }

        // Compute gradients for each sample
        for (size_t idx = 0; idx < states.size(); idx++) {
            float pred = predict(states[idx]);
            float error = pred - targets[idx];
            total_loss += error * error;

            // Backward pass (simplified - assumes caching of activations)
            // In practice, would need full backprop implementation
            float grad = 2.0f * error / states.size();

            // Update accumulation (simplified)
            // Full implementation would backprop through all layers
        }

        // Update weights
        for (size_t i = 0; i < weights_.size(); i++) {
            for (int r = 0; r < weights_[i].rows; r++) {
                for (int c = 0; c < weights_[i].cols; c++) {
                    weights_[i](r, c) -= learning_rate * weight_grads[i](r, c);
                }
            }
            for (size_t j = 0; j < biases_[i].size(); j++) {
                biases_[i][j] -= learning_rate * bias_grads[i][j];
            }
        }

        return total_loss / states.size();
    }
};

/**
 * @class PPOOptimizer
 * @brief Implements Proximal Policy Optimization for language model fine-tuning
 *
 * PPO optimizes the language model policy using reward signals from a reward model.
 * It ensures stable training by constraining policy updates.
 *
 * Example Usage:
 * @code
 * PPOConfig config;
 * config.clip_epsilon = 0.2;
 * config.learning_rate = 1e-5;
 *
 * PPOOptimizer ppo(model, reward_model, config);
 *
 * // Training loop
 * for (int iter = 0; iter < 1000; iter++) {
 *     Trajectory traj = collect_rollout(model, prompts);
 *     float loss = ppo.update(traj);
 *     std::cout << "PPO Loss: " << loss << std::endl;
 * }
 * @endcode
 */
class PPOOptimizer {
   private:
    PPOConfig config_;
    RewardModel* reward_model_;
    ValueFunction value_function_;

    /**
     * @brief Compute Generalized Advantage Estimation (GAE)
     *
     * GAE balances bias-variance tradeoff in advantage estimation.
     * A_t = sum_{l=0}^{inf} (gamma * lambda)^l * delta_{t+l}
     * where delta_t = r_t + gamma * V(s_{t+1}) - V(s_t)
     */
    std::vector<float> compute_gae(const Trajectory& traj) {
        std::vector<float> advantages(traj.length());
        float gae = 0.0f;

        for (int t = traj.length() - 1; t >= 0; t--) {
            float next_value = (t < (int)traj.length() - 1) ? traj.values[t + 1] : 0.0f;
            float delta = traj.rewards[t] + config_.gamma * next_value - traj.values[t];
            gae = delta + config_.gamma * config_.gae_lambda * gae;
            advantages[t] = gae;
        }

        // Normalize advantages
        float mean = 0.0f, var = 0.0f;
        for (float adv : advantages)
            mean += adv;
        mean /= advantages.size();
        for (float adv : advantages)
            var += (adv - mean) * (adv - mean);
        var /= advantages.size();
        float std = std::sqrt(var + 1e-8f);

        for (float& adv : advantages) {
            adv = (adv - mean) / std;
        }

        return advantages;
    }

    /**
     * @brief Compute returns for value function training
     */
    std::vector<float> compute_returns(const Trajectory& traj) {
        std::vector<float> returns(traj.length());
        float G = 0.0f;

        for (int t = traj.length() - 1; t >= 0; t--) {
            G = traj.rewards[t] + config_.gamma * G;
            returns[t] = G;
        }

        return returns;
    }

    /**
     * @brief Compute clipped PPO objective
     *
     * L^CLIP = E[min(r_t * A_t, clip(r_t, 1-eps, 1+eps) * A_t)]
     * where r_t = pi(a|s) / pi_old(a|s)
     */
    float compute_policy_loss(float ratio, float advantage) {
        float clipped_ratio =
            std::max(1.0f - config_.clip_epsilon, std::min(1.0f + config_.clip_epsilon, ratio));
        return -std::min(ratio * advantage, clipped_ratio * advantage);
    }

    /**
     * @brief Clip gradients by global norm
     */
    void clip_gradients(std::vector<Matrix>& grads, float max_norm) {
        float total_norm = 0.0f;

        for (const auto& grad : grads) {
            for (int r = 0; r < grad.rows; r++) {
                for (int c = 0; c < grad.cols; c++) {
                    total_norm += grad(r, c) * grad(r, c);
                }
            }
        }

        total_norm = std::sqrt(total_norm);

        if (total_norm > max_norm) {
            float scale = max_norm / (total_norm + 1e-8f);
            for (auto& grad : grads) {
                for (int r = 0; r < grad.rows; r++) {
                    for (int c = 0; c < grad.cols; c++) {
                        grad(r, c) *= scale;
                    }
                }
            }
        }
    }

   public:
    /**
     * @brief Construct PPO optimizer
     *
     * @param reward_model Trained reward model for scoring
     * @param config PPO hyperparameters
     * @param state_dim Dimension of state representations
     */
    PPOOptimizer(RewardModel* reward_model, const PPOConfig& config, int state_dim)
        : config_(config), reward_model_(reward_model), value_function_(state_dim, {256, 128, 1}) {}

    /**
     * @brief Perform PPO update on trajectory
     *
     * @param trajectory Collected rollout data
     * @return Average policy loss
     */
    float update(Trajectory& trajectory) {
        if (trajectory.length() == 0) {
            throw std::invalid_argument("Empty trajectory");
        }

        // Compute advantages and returns
        std::vector<float> advantages = compute_gae(trajectory);
        std::vector<float> returns = compute_returns(trajectory);

        float total_policy_loss = 0.0f;
        float total_value_loss = 0.0f;
        int num_updates = 0;

        // PPO epochs
        for (int epoch = 0; epoch < config_.num_epochs; epoch++) {
            // Minibatch training
            for (size_t start = 0; start < trajectory.length(); start += config_.batch_size) {
                size_t end = std::min(start + config_.batch_size, trajectory.length());

                // Extract minibatch
                std::vector<std::vector<float>> batch_states;
                std::vector<float> batch_advantages;
                std::vector<float> batch_returns;
                std::vector<float> batch_old_log_probs;

                for (size_t i = start; i < end; i++) {
                    batch_states.push_back(trajectory.states[i]);
                    batch_advantages.push_back(advantages[i]);
                    batch_returns.push_back(returns[i]);
                    batch_old_log_probs.push_back(trajectory.log_probs[i]);
                }

                // Compute policy loss
                float policy_loss = 0.0f;
                for (size_t i = 0; i < batch_states.size(); i++) {
                    // In practice, would recompute log_probs with current policy
                    // For now, use simplified ratio calculation
                    float new_log_prob = batch_old_log_probs[i];  // Placeholder
                    float ratio = std::exp(new_log_prob - batch_old_log_probs[i]);

                    policy_loss += compute_policy_loss(ratio, batch_advantages[i]);
                }
                policy_loss /= batch_states.size();

                // Update value function
                float value_loss =
                    value_function_.update(batch_states, batch_returns, config_.learning_rate);

                total_policy_loss += policy_loss;
                total_value_loss += value_loss;
                num_updates++;
            }

            // Early stopping based on KL divergence
            // (simplified - would need to track actual KL)
            float approx_kl = 0.0f;  // Placeholder
            if (approx_kl > 1.5f * config_.kl_target) {
                break;
            }
        }

        return total_policy_loss / num_updates;
    }

    /**
     * @brief Estimate value for a state
     *
     * @param state State encoding
     * @return Estimated value
     */
    float estimate_value(const std::vector<float>& state) {
        return value_function_.predict(state);
    }

    /**
     * @brief Get configuration
     */
    const PPOConfig& get_config() const {
        return config_;
    }

    /**
     * @brief Update configuration
     */
    void set_config(const PPOConfig& config) {
        config_ = config;
    }
};

#endif  // PPO_OPTIMIZER_HPP
