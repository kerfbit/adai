# PPO Optimizer API Reference

**File:** `src/PPOOptimizer.hpp`
**Status:** ✅ Production-ready (Phase 5 - January 2026)
**Purpose:** Proximal Policy Optimization for RLHF policy fine-tuning

---

## Overview

The `PPOOptimizer` class implements Proximal Policy Optimization (PPO), a state-of-the-art reinforcement learning algorithm used to fine-tune language model policies based on reward signals from a trained reward model.

### Key Features

- **Stable policy updates** via clipped surrogate objective
- **Value function** for advantage estimation
- **GAE (Generalized Advantage Estimation)** for bias-variance trade-off
- **KL divergence monitoring** to prevent policy collapse
- **Configurable hyperparameters** for flexible optimization

---

## Class Definitions

### PPOOptimizer

```cpp
class PPOOptimizer {
public:
    PPOOptimizer(RewardModel* reward_model,
                const PPOConfig& config,
                int state_dim);

    float update(const Trajectory& trajectory);
    float estimate_value(const std::vector<float>& state);

    std::vector<float> compute_advantages(const Trajectory& traj,
                                         float gamma,
                                         float lambda);

    PPOConfig get_config() const;
    void set_config(const PPOConfig& config);
};
```

### PPOConfig

```cpp
struct PPOConfig {
    float clip_epsilon = 0.2f;      // Clipping range for ratio
    float gamma = 0.99f;            // Discount factor
    float lambda = 0.95f;           // GAE lambda
    float learning_rate = 1e-5f;    // Policy learning rate
    float value_lr = 1e-4f;         // Value function LR
    int num_epochs = 4;             // Optimization epochs per batch
    int batch_size = 64;            // Minibatch size
    float max_grad_norm = 0.5f;     // Gradient clipping
    float kl_coef = 0.01f;          // KL penalty coefficient
};
```

### Trajectory

```cpp
struct Trajectory {
    std::vector<std::vector<float>> states;
    std::vector<int> actions;
    std::vector<float> rewards;
    std::vector<float> log_probs;
    std::vector<float> values;

    void add_step(const std::vector<float>& state,
                 int action,
                 float reward,
                 float log_prob,
                 float value);

    size_t length() const;
    void clear();
};
```

---

## Constructor

```cpp
PPOOptimizer(RewardModel* reward_model,
            const PPOConfig& config,
            int state_dim)
```

**Parameters:**

- `reward_model` - Trained reward model for scoring states
- `config` - PPO hyperparameters
- `state_dim` - Dimension of state vectors

**Example:**

```cpp
RewardModel reward_model(1536, {512, 256, 128, 1});
reward_model.load("reward_model.bin");

PPOConfig config;
config.clip_epsilon = 0.2f;
config.learning_rate = 1e-5f;

PPOOptimizer ppo(&reward_model, config, 768);
```

---

## Core Methods

### update()

```cpp
float update(const Trajectory& trajectory)
```

Perform PPO policy update on collected trajectory.

**Algorithm:**

1. Compute advantages using GAE
2. For each epoch:
   - Compute policy ratio: π_new / π_old
   - Clip ratio to [1-ε, 1+ε]
   - Compute clipped objective: min(ratio × A, clip(ratio) × A)
   - Update policy via gradient ascent
   - Update value function via MSE

**PPO Objective:**

```text
L^CLIP = E[min(r_t × A_t, clip(r_t, 1-ε, 1+ε) × A_t)]

where:
  r_t = π_θ(a_t| s_t) / π_θ_old(a_t |s_t)  (probability ratio)
  A_t = advantage at timestep t
  ε = clip_epsilon (default 0.2)
```

**Returns:** Policy loss value

**Example:**

```cpp
Trajectory traj;

// Collect rollouts
for (int step = 0; step < 100; step++) {
    auto state = env.get_state();
    int action = policy.sample(state);
    float reward = reward_model.predict_reward(state);
    float log_prob = policy.log_prob(action, state);
    float value = ppo.estimate_value(state);

    traj.add_step(state, action, reward, log_prob, value);
}

// Update policy
float loss = ppo.update(traj);
std::cout << "Policy loss: " << loss << "\n";
```

---

### estimate_value()

```cpp
float estimate_value(const std::vector<float>& state)
```

Estimate value (expected return) for a given state.

**Purpose:** Used during rollout collection and advantage calculation

**Returns:** Estimated value V(s)

---

### compute_advantages()

```cpp
std::vector<float> compute_advantages(const Trajectory& traj,
                                     float gamma,
                                     float lambda)
```

Compute Generalized Advantage Estimation (GAE).

**GAE Formula:**

```text
A_t = δ_t + (γλ)δ_{t+1} + (γλ)²δ_{t+2} + ...

where:
  δ_t = r_t + γV(s_{t+1}) - V(s_t)  (TD error)
  γ = discount factor (gamma)
  λ = GAE lambda parameter
```

**Trade-off:**

- **λ=0:** Low variance, high bias (TD error only)
- **λ=1:** High variance, low bias (Monte Carlo)
- **λ=0.95:** Recommended balance

**Returns:** Vector of advantage estimates

---

## Configuration

### get_config() / set_config()

```cpp
PPOConfig get_config() const;
void set_config(const PPOConfig& config);
```

Get or update PPO hyperparameters dynamically.

**Example:**

```cpp
PPOConfig config = ppo.get_config();
config.learning_rate *= 0.1;  // Decrease LR
ppo.set_config(config);
```

---

## Complete RLHF Pipeline

```cpp
#include "RewardModel.hpp"
#include "PPOOptimizer.hpp"

int main() {
    // ========================================
    // Phase 1: Supervised Fine-Tuning (SFT)
    // ========================================
    LLMDecoder policy(/* ... */);
    train_on_demonstrations(policy, sft_data);
    policy.save("policy_sft.bin");

    // ========================================
    // Phase 2: Reward Model Training
    // ========================================
    RewardModel reward_model(1536, {512, 256, 128, 1});

    // Collect human preferences
    std::vector<PreferencePair> preferences;
    for (const auto& prompt : prompts) {
        auto resp_a = policy.generate(prompt);
        auto resp_b = policy.generate(prompt);

        bool a_better = get_human_preference(resp_a, resp_b);
        preferences.push_back(create_pair(prompt, resp_a, resp_b, a_better));
    }

    // Train reward model
    for (int epoch = 0; epoch < 10; epoch++) {
        float loss = reward_model.train_on_batch(preferences, 0.001f);
        std::cout << "RM Epoch " << epoch << " - Loss: " << loss << "\n";
    }
    reward_model.save("reward_model.bin");

    // ========================================
    // Phase 3: PPO Policy Optimization
    // ========================================
    PPOConfig ppo_config;
    ppo_config.clip_epsilon = 0.2f;
    ppo_config.learning_rate = 1e-5f;
    ppo_config.num_epochs = 4;

    PPOOptimizer ppo(&reward_model, ppo_config, 768);

    // PPO training loop
    for (int iteration = 0; iteration < 1000; iteration++) {
        Trajectory traj;

        // Collect rollouts
        for (int step = 0; step < 128; step++) {
            auto prompt = sample_prompt();
            auto state = encode(prompt);

            // Generate response
            auto response = policy.generate_from_state(state);
            int action = map_response_to_action(response);

            // Get reward from reward model
            auto full_input = concatenate(state, encode(response));
            float reward = reward_model.predict_reward(full_input);

            // Get policy info
            float log_prob = policy.log_probability(action, state);
            float value = ppo.estimate_value(state);

            traj.add_step(state, action, reward, log_prob, value);
        }

        // Update policy with PPO
        float loss = ppo.update(traj);

        if (iteration % 10 == 0) {
            std::cout << "Iteration " << iteration
                     << " - Loss: " << loss << "\n";

            // Evaluate and save
            float eval_reward = evaluate_policy(policy, eval_prompts);
            if (eval_reward > best_reward) {
                policy.save("policy_best.bin");
                best_reward = eval_reward;
            }
        }
    }

    // ========================================
    // Phase 4: Deployment
    // ========================================
    policy.load("policy_best.bin");
    // Now policy is aligned with human preferences!

    return 0;
}
```

---

## Hyperparameter Tuning Guide

### clip_epsilon

- **Default:** 0.2
- **Range:** 0.1 - 0.3
- **Effect:**
  - Lower = more conservative updates
  - Higher = faster learning but less stable

### learning_rate

- **Default:** 1e-5
- **Range:** 1e-6 - 1e-4
- **Effect:**
  - Too low = slow convergence
  - Too high = policy collapse

### gamma (discount factor)

- **Default:** 0.99
- **Range:** 0.95 - 0.999
- **Effect:**
  - Lower = focus on immediate rewards
  - Higher = long-term planning

### lambda (GAE)

- **Default:** 0.95
- **Range:** 0.9 - 0.99
- **Effect:**
  - Lower = lower variance, higher bias
  - Higher = higher variance, lower bias

---

## Monitoring & Debugging

### Key Metrics to Track

```cpp
// 1. Policy Loss
float policy_loss = ppo.update(traj);
if (policy_loss > threshold) {
    // Policy might be diverging
}

// 2. Value Function Loss
float value_loss = compute_value_loss(traj);

// 3. KL Divergence
float kl_div = compute_kl(old_policy, new_policy);
if (kl_div > 0.01) {
    // Policy changing too fast
}

// 4. Average Reward
float avg_reward = compute_average_reward(traj);
// Should increase over time

// 5. Explained Variance
float explained_var = 1 - Var(returns - values) / Var(returns);
// Should be > 0.5 (value function is useful)
```

---

## Best Practices

### 1. Trajectory Collection
```cpp
// Collect enough data per update
// Typical: 2048-4096 timesteps
Trajectory traj;
while (traj.length() < 2048) {
    // ... collect rollouts ...
}
```

### 2. Multiple Epochs
```cpp
// Run multiple optimization epochs on same data
config.num_epochs = 4;  // Typical: 3-5 epochs
```

### 3. Gradient Clipping
```cpp
// Prevent exploding gradients
config.max_grad_norm = 0.5f;
```

### 4. Learning Rate Scheduling
```cpp
// Decrease learning rate over time
if (iteration % 100 == 0) {
    config.learning_rate *= 0.9;
    ppo.set_config(config);
}
```

---

## Test Coverage

**File:** `tests/phase5_test.cpp`
**Test Cases:** 4

- Constructor validation
- Value estimation
- Trajectory updates
- Configuration management

**Pass Rate:** 100%

---

## See Also

- [Reward Model](reward-model.md) - Prerequisite for PPO
- [Phase 5 Guide](../../guides/phase5-advanced-features.md) - Complete RLHF tutorial
- [Chatbot Completeness](../../reference/chatbot-completeness.md) - System integration

---

## References

- [Proximal Policy Optimization (Schulman et al., 2017)](https://arxiv.org/abs/1707.06347)
- [Training language models to follow instructions with human feedback (Ouyang et al., 2022)](https://arxiv.org/abs/2203.02155)

---

**Last Updated:** January 25, 2026
**Version:** 1.0
**Status:** Production-ready
