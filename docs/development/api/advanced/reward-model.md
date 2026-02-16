# Reward Model API Reference

**File:** `src/RewardModel.hpp`
**Status:** ✅ Production-ready (Phase 5 - January 2026)
**Purpose:** Learn human preferences for Reinforcement Learning from Human Feedback (RLHF)

---

## Overview

The `RewardModel` class implements a neural network that learns to predict human preferences between response pairs. It's a critical component of RLHF training pipelines, used to align language models with human values and preferences.

### Key Concepts

- **Bradley-Terry Model:** Statistical model for pairwise comparisons
- **Preference Learning:** Train on (chosen, rejected) response pairs
- **Reward Scoring:** Predict scalar reward for any response
- **Integration:** Works with PPOOptimizer for policy fine-tuning

---

## Class Definition

```cpp
class RewardModel {
private:
    int input_dim_;
    std::vector<int> layer_dims_;
    std::vector<Matrix> weights_;
    std::vector<std::vector<float>> biases_;

public:
    RewardModel(int input_dim, const std::vector<int>& layer_dims);

    float forward(const std::vector<float>& input);
    float predict_reward(const std::vector<float>& encoding);
    float compute_loss(const PreferencePair& pair);
    float train_on_batch(const std::vector<PreferencePair>& pairs, float learning_rate);

    void save(const std::string& filepath) const;
    void load(const std::string& filepath);
};
```

---

## Constructor

```cpp
RewardModel(int input_dim, const std::vector<int>& layer_dims)
```

**Parameters:**

- `input_dim` - Dimension of input encodings (typically 2× encoding_dim for prompt+response)
- `layer_dims` - Hidden layer dimensions, last must be 1 for scalar reward

**Example:**

```cpp
// For 768-dim encoder with prompt+response concatenation
RewardModel reward_model(1536, {512, 256, 128, 1});
```

**Throws:**

- `std::invalid_argument` if `layer_dims` is empty or final layer ≠ 1

---

## Core Methods

### forward()

```cpp
float forward(const std::vector<float>& input)
```

Compute reward score for a given input encoding.

**Parameters:**

- `input` - Concatenated prompt + response encoding

**Returns:** Scalar reward value (higher = better response)

**Throws:**

- `std::invalid_argument` if input dimension mismatch

**Implementation:** Multi-layer perceptron with ReLU activations (hidden) and linear output

---

### predict_reward()

```cpp
float predict_reward(const std::vector<float>& encoding)
```

Convenience wrapper for forward pass.

**Parameters:**

- `encoding` - Response encoding to score

**Returns:** Predicted reward

---

### compute_loss()

```cpp
float compute_loss(const PreferencePair& pair)
```

Compute Bradley-Terry loss for a preference pair.

**Loss Formula:**

```text
L = -log(σ(r_chosen - r_rejected))
```

where σ is sigmoid function.

**Parameters:**

- `pair` - PreferencePair containing prompt, chosen, and rejected encodings

**Returns:** Loss value (lower = model better matches preference)

---

### train_on_batch()

```cpp
float train_on_batch(const std::vector<PreferencePair>& pairs,
                     float learning_rate)
```

Train reward model on a batch of preference pairs.

**Parameters:**

- `pairs` - Vector of preference pairs (chosen vs rejected)
- `learning_rate` - Gradient descent learning rate (typical: 1e-4 to 1e-3)

**Returns:** Average loss across batch

**Algorithm:**

1. Accumulate gradients across all pairs
2. Average gradients
3. Update weights via gradient descent

**Example:**

```cpp
std::vector<PreferencePair> preferences;
// ... populate preferences ...

for (int epoch = 0; epoch < 10; epoch++) {
    float loss = reward_model.train_on_batch(preferences, 0.001f);
    std::cout << "Epoch " << epoch << " - Loss: " << loss << "\n";
}
```

---

## Persistence Methods

### save()

```cpp
void save(const std::string& filepath) const
```

Save trained reward model to binary file.

**Format:** Binary format with architecture + weights

**Example:**

```cpp
reward_model.save("reward_model.bin");
```

---

### load()

```cpp
void load(const std::string& filepath)
```

Load trained reward model from file.

**Example:**

```cpp
RewardModel loaded_model(1536, {512, 256, 128, 1});
loaded_model.load("reward_model.bin");
```

---

## Supporting Structures

### PreferencePair

```cpp
struct PreferencePair {
    std::vector<float> prompt_encoding;
    std::vector<float> chosen_encoding;    // Better response
    std::vector<float> rejected_encoding;  // Worse response

    PreferencePair(const std::vector<float>& prompt,
                   const std::vector<float>& chosen,
                   const std::vector<float>& rejected);
};
```

---

## Complete Example

```cpp
#include "RewardModel.hpp"
#include <iostream>

int main() {
    // 1. Create reward model (768-dim encoder → 1536 input)
    RewardModel reward_model(1536, {512, 256, 128, 1});

    // 2. Prepare training data
    std::vector<PreferencePair> preferences;

    for (int i = 0; i < 100; i++) {
        auto prompt = create_encoding(768);
        auto good_response = create_encoding(768);
        auto bad_response = create_encoding(768);

        preferences.push_back(PreferencePair(prompt, good_response, bad_response));
    }

    // 3. Train reward model
    std::cout << "Training reward model...\n";
    for (int epoch = 0; epoch < 10; epoch++) {
        float loss = reward_model.train_on_batch(preferences, 0.001f);
        std::cout << "Epoch " << epoch + 1 << " - Loss: " << loss << "\n";
    }

    // 4. Save model
    reward_model.save("reward_model.bin");

    // 5. Use for inference
    auto test_prompt = create_encoding(768);
    auto test_response = create_encoding(768);

    // Concatenate prompt + response
    std::vector<float> full_input = test_prompt;
    full_input.insert(full_input.end(),
                     test_response.begin(),
                     test_response.end());

    float reward = reward_model.predict_reward(full_input);
    std::cout << "Response reward: " << reward << "\n";

    return 0;
}
```

---

## Usage Patterns

### RLHF Training Pipeline

```cpp
// Step 1: Supervised Fine-Tuning (SFT)
// Train base model on high-quality demonstrations
train_base_model(demonstrations);

// Step 2: Reward Model Training
RewardModel reward_model(input_dim, {512, 256, 128, 1});
reward_model.train_on_batch(human_preferences, 0.001f);
reward_model.save("reward_model.bin");

// Step 3: Policy Optimization
PPOOptimizer ppo(&reward_model, config, state_dim);
// ... PPO training loop ...
```

### Preference Data Collection

```cpp
// Collect human preferences
for (const auto& prompt : prompts) {
    auto response_a = model.generate(prompt);
    auto response_b = model.generate(prompt);

    // Human annotator chooses better response
    bool a_is_better = get_human_preference(response_a, response_b);

    if (a_is_better) {
        preferences.push_back(PreferencePair(
            encode(prompt),
            encode(response_a),  // chosen
            encode(response_b)   // rejected
        ));
    } else {
        preferences.push_back(PreferencePair(
            encode(prompt),
            encode(response_b),  // chosen
            encode(response_a)   // rejected
        ));
    }
}
```

---

## Performance Characteristics

- **Training:** O(N × D × H) per batch, where N=pairs, D=dimensions, H=hidden size
- **Inference:** O(D × H) per prediction
- **Memory:** ~4 bytes × (D×H1 + H1×H2 + ... + Hn×1) for weights

### Typical Configuration

```cpp
// Standard configuration for 768-dim transformer
RewardModel reward_model(
    1536,                    // Input: prompt (768) + response (768)
    {512, 256, 128, 1}      // 3 hidden layers + output
);

// Training: ~15 seconds per epoch on 100 pairs (CPU)
// Inference: <1ms per prediction
```

---

## Best Practices

### 1. Input Preparation
```cpp
// Always concatenate prompt + response
std::vector<float> input = prompt_encoding;
input.insert(input.end(), response_encoding.begin(), response_encoding.end());
```

### 2. Learning Rate Selection

- Start with 1e-3
- Decrease if loss oscillates
- Increase if convergence too slow

### 3. Data Quality

- Need 100-1000+ preference pairs for good model
- Ensure diverse prompts
- Multiple annotators reduce bias

### 4. Monitoring
```cpp
// Track loss convergence
if (current_loss < best_loss) {
    best_loss = current_loss;
    reward_model.save("best_reward_model.bin");
}
```

---

## Integration with PPO

```cpp
// After training reward model
RewardModel reward_model(1536, {512, 256, 128, 1});
reward_model.load("trained_reward_model.bin");

// Use in PPO for policy optimization
PPOConfig config;
config.learning_rate = 1e-5f;
config.clip_epsilon = 0.2f;

PPOOptimizer ppo(&reward_model, config, 768);

// Generate trajectories and optimize policy
Trajectory traj = collect_rollouts(policy, reward_model);
float policy_loss = ppo.update(traj);
```

---

## Test Coverage

**File:** `tests/phase5_test.cpp`
**Test Cases:** 5

- `RewardModelTest.Constructor` - Architecture validation
- `RewardModelTest.ForwardPass` - Inference correctness
- `RewardModelTest.PreferencePairLoss` - Bradley-Terry loss
- `RewardModelTest.TrainOnBatch` - Training convergence
- `RewardModelTest.SaveLoad` - Persistence

**Pass Rate:** 100%

---

## See Also

- [PPO Optimizer](ppo-optimizer.md) - Policy optimization
- [Phase 5 Advanced Features Guide](../../guides/phase5-advanced-features.md) - Complete RLHF guide
- [Chatbot Completeness](../../reference/chatbot-completeness.md) - Architecture overview

---

**Last Updated:** January 25, 2026
**Version:** 1.0
**Status:** Production-ready
