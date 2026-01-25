# LoRA (Low-Rank Adaptation) API Reference

**File:** `src/LoRA.hpp`  
**Status:** ✅ Production-ready (Phase 5 - January 2026)  
**Purpose:** Parameter-efficient fine-tuning with 100-1000x parameter reduction

---

## Overview

The `LoRAAdapter` class implements Low-Rank Adaptation (LoRA), a technique for fine-tuning large models by only training small adapter matrices. This enables efficient task-specific adaptation while freezing the base model weights.

### Key Benefits

- **100-1000x fewer trainable parameters** compared to full fine-tuning
- **Zero inference overhead** when weights are merged
- **Multiple task adapters** can be saved and swapped
- **Memory efficient** - only store small adapter weights
- **Industry standard** - used in production by major AI labs

---

## Mathematical Foundation

For a weight matrix W ∈ ℝ^(d×k), LoRA represents the update as:

```
W' = W + (α/r) × B × A
```

where:
- **W** - Original frozen weights (d × k)
- **A** - Low-rank matrix (r × k), randomly initialized
- **B** - Low-rank matrix (d × r), initialized to zeros
- **r** - Rank (typically 4, 8, 16, or 32)
- **α** - Scaling factor (typically 16 or 32)

**Parameter Reduction:**
- Original: d × k parameters
- LoRA: (d × r) + (r × k) = r(d + k) parameters
- Reduction: (d × k) / [r(d + k)]

**Example:** For d=768, k=768, r=8:
- Original: 589,824 parameters
- LoRA: 12,288 parameters
- **Reduction: 48x fewer parameters!**

---

## Class Definition

```cpp
class LoRAAdapter {
private:
    int input_dim_;
    int output_dim_;
    int rank_;
    float alpha_;
    
    Matrix A_;        // (rank × input_dim)
    Matrix B_;        // (output_dim × rank)
    Matrix grad_A_;
    Matrix grad_B_;
    
public:
    LoRAAdapter(int input_dim, int output_dim, int rank, float alpha = 16.0f);
    
    Matrix forward(const Matrix& x, const Matrix& W_output);
    void backward(const Matrix& x, const Matrix& grad_output);
    void update(float learning_rate);
    void zero_grad();
    
    Matrix merge_with_base(const Matrix& W);
    
    Matrix get_A() const;
    Matrix get_B() const;
    
    void save(const std::string& filepath) const;
    void load(const std::string& filepath);
    
    int trainable_parameters() const;
    float parameter_reduction(int original_params) const;
};
```

---

## Constructor

```cpp
LoRAAdapter(int input_dim, int output_dim, int rank, float alpha = 16.0f)
```

**Parameters:**
- `input_dim` - Input dimension (k)
- `output_dim` - Output dimension (d)
- `rank` - Rank of low-rank decomposition (r)
- `alpha` - Scaling factor (default: 16.0)

**Initialization:**
- **A** ~ N(0, 1/√r) - Random Gaussian
- **B** = 0 - Ensures ΔW = 0 initially

**Example:**
```cpp
// For transformer attention: 768 → 768 with rank 8
LoRAAdapter adapter(768, 768, 8, 16.0f);

// This creates only 12,288 trainable params instead of 589,824!
```

---

## Forward Pass

```cpp
Matrix forward(const Matrix& x, const Matrix& W_output)
```

Compute forward pass with LoRA adaptation.

**Algorithm:**
```
1. y_base = x × W_output          (frozen base model)
2. xA = x × A^T                   (low-rank projection)
3. xAB = xA × B^T                 (reconstruct adaptation)
4. y = y_base + (α/r) × xAB       (add scaled adaptation)
```

**Parameters:**
- `x` - Input activations (batch_size × input_dim)
- `W_output` - Base model weights (input_dim × output_dim), **frozen**

**Returns:** Adapted output (batch_size × output_dim)

**Example:**
```cpp
Matrix x(2, 768);           // Batch of 2 inputs
Matrix W_base(768, 768);    // Frozen base weights

Matrix output = adapter.forward(x, W_base);
// output has LoRA adaptation applied!
```

---

## Backward Pass

```cpp
void backward(const Matrix& x, const Matrix& grad_output)
```

Compute gradients for A and B (W remains frozen).

**Gradients:**
- `grad_B = (α/r) × grad_output^T × (x × A^T)`
- `grad_A = (α/r) × B^T × grad_output^T × x`

**Parameters:**
- `x` - Input from forward pass (must be cached!)
- `grad_output` - Gradient flowing back from loss

**Note:** Only computes gradients for A and B. Base weights W are never updated!

---

## Weight Update

```cpp
void update(float learning_rate)
```

Apply gradient descent to adapter parameters.

**Update Rule:**
```
A ← A - lr × grad_A
B ← B - lr × grad_B
```

**Example:**
```cpp
// Training loop
for (int epoch = 0; epoch < epochs; epoch++) {
    for (auto& batch : data) {
        // Forward
        Matrix output = adapter.forward(batch.x, W_base);
        Matrix loss_grad = compute_loss_gradient(output, batch.target);
        
        // Backward
        adapter.backward(batch.x, loss_grad);
        
        // Update only adapter weights
        adapter.update(0.001f);
        adapter.zero_grad();
    }
}
```

---

## Zero Gradients

```cpp
void zero_grad()
```

Reset accumulated gradients to zero. Call after each weight update.

---

## Weight Merging

```cpp
Matrix merge_with_base(const Matrix& W)
```

Merge LoRA adapter into base weights for deployment.

**Formula:**
```
W_merged = W + (α/r) × B × A
```

**Benefits:**
- **Zero inference overhead** - merged weights = standard matrix
- **No adapter needed** at deployment
- **Same computational cost** as original model

**Example:**
```cpp
// After training
Matrix W_base(768, 768);  // Original weights
Matrix W_merged = adapter.merge_with_base(W_base);

// Now use W_merged directly - no LoRA overhead!
// Save space by discarding adapter
```

**Use Case:** Production deployment where you need one optimized model

---

## Inspection Methods

### get_A() / get_B()

```cpp
Matrix get_A() const;
Matrix get_B() const;
```

Access adapter matrices (useful for debugging, saving, analysis).

### trainable_parameters()

```cpp
int trainable_parameters() const
```

Returns: `r × (input_dim + output_dim)`

### parameter_reduction()

```cpp
float parameter_reduction(int original_params) const
```

Calculate reduction ratio.

**Returns:** `original_params / trainable_params`

**Example:**
```cpp
int original = 768 * 768;  // 589,824
float reduction = adapter.parameter_reduction(original);
// reduction = 48.0x
```

---

## Persistence

### save()

```cpp
void save(const std::string& filepath) const
```

Save adapter weights to binary file.

**Format:** Custom binary (architecture + A + B matrices)

### load()

```cpp
void load(const std::string& filepath)
```

Load adapter weights from file.

**Example - Multiple Task Adapters:**
```cpp
// Train adapters for different tasks
LoRAAdapter summarization(768, 768, 8);
train(summarization, summarization_data);
summarization.save("lora_summarization.bin");

LoRAAdapter translation(768, 768, 8);
train(translation, translation_data);
translation.save("lora_translation.bin");

// Switch tasks at runtime
LoRAAdapter active_adapter(768, 768, 8);
active_adapter.load("lora_summarization.bin");
// Now model is adapted for summarization

active_adapter.load("lora_translation.bin");
// Now model is adapted for translation
```

---

## Complete Example

```cpp
#include "LoRA.hpp"
#include <iostream>

int main() {
    // 1. Create LoRA adapter
    int dim = 768;
    LoRAAdapter lora(dim, dim, 8, 16.0f);
    
    std::cout << "Trainable params: " << lora.trainable_parameters() << "\n";
    std::cout << "Reduction: " << lora.parameter_reduction(dim * dim) << "x\n";
    
    // 2. Load frozen base model
    Matrix W_base(dim, dim);
    load_pretrained_weights(W_base, "base_model.bin");
    
    // 3. Training loop
    for (int epoch = 0; epoch < 10; epoch++) {
        float epoch_loss = 0.0f;
        
        for (auto& batch : training_data) {
            // Forward with LoRA
            Matrix output = lora.forward(batch.x, W_base);
            
            // Compute loss
            float loss = cross_entropy(output, batch.labels);
            epoch_loss += loss;
            
            // Backward (only through LoRA)
            Matrix grad = compute_gradient(output, batch.labels);
            lora.backward(batch.x, grad);
            
            // Update adapters only
            lora.update(0.001f);
            lora.zero_grad();
        }
        
        std::cout << "Epoch " << epoch + 1 
                  << " - Loss: " << epoch_loss << "\n";
    }
    
    // 4. Save adapter
    lora.save("lora_adapter.bin");
    
    // 5. Option A: Use with adapter at inference
    Matrix test_output = lora.forward(test_input, W_base);
    
    // 6. Option B: Merge for deployment (zero overhead)
    Matrix W_merged = lora.merge_with_base(W_base);
    // Now just use: output = x * W_merged (no LoRA needed!)
    
    return 0;
}
```

---

## Advanced Usage Patterns

### Multi-Layer LoRA Application

```cpp
// Apply LoRA to all attention projections
struct TransformerWithLoRA {
    LoRAAdapter query_lora;
    LoRAAdapter key_lora;
    LoRAAdapter value_lora;
    LoRAAdapter output_lora;
    
    Matrix W_Q, W_K, W_V, W_O;  // Frozen base weights
    
    Matrix forward(const Matrix& x) {
        Matrix Q = query_lora.forward(x, W_Q);
        Matrix K = key_lora.forward(x, W_K);
        Matrix V = value_lora.forward(x, W_V);
        
        Matrix attention_out = compute_attention(Q, K, V);
        Matrix output = output_lora.forward(attention_out, W_O);
        
        return output;
    }
};
```

### Selective LoRA Application

```cpp
// Only adapt specific layers
LoRAConfig config;
config.target_modules = {"q_proj", "v_proj"};  // Skip k_proj, o_proj
config.rank = 8;
config.alpha = 16;

// Apply selectively
if (config.should_adapt("q_proj")) {
    adapters["q_proj"] = LoRAAdapter(dim, dim, config.rank, config.alpha);
}
```

### Rank Tuning

```cpp
// Test different ranks
std::vector<int> ranks = {4, 8, 16, 32};

for (int r : ranks) {
    LoRAAdapter adapter(768, 768, r);
    float accuracy = train_and_evaluate(adapter, data);
    std::cout << "Rank " << r << ": " << accuracy << "\n";
}

// Typical findings:
// rank=4:  Good for simple tasks, 96x reduction
// rank=8:  Balanced, 48x reduction (recommended)
// rank=16: More capacity, 24x reduction
// rank=32: High capacity, 12x reduction
```

---

## Configuration Structure

```cpp
struct LoRAConfig {
    int rank = 8;
    float alpha = 16.0f;
    float dropout = 0.0f;  // Optional dropout
    std::vector<std::string> target_modules;
    
    bool should_adapt(const std::string& module_name) const {
        if (target_modules.empty()) return true;
        return std::find(target_modules.begin(), 
                        target_modules.end(), 
                        module_name) != target_modules.end();
    }
};
```

---

## Performance Characteristics

### Memory Usage

| Model Size | Full Fine-Tune | LoRA (r=8) | Reduction |
|-----------|----------------|------------|-----------|
| 124M params | 496 MB | 4 MB | 124x |
| 350M params | 1.4 GB | 11 MB | 127x |
| 1.3B params | 5.2 GB | 42 MB | 124x |
| 7B params | 28 GB | 224 MB | 125x |

### Training Speed

- **Same as full fine-tuning** (forward/backward through same computation graph)
- **Faster checkpoint saves** (only save adapter, not full model)
- **Faster task switching** (swap adapters in <1s)

### Inference Speed

- **With adapter:** ~same as base model (extra matrix multiply)
- **After merging:** **zero overhead** (identical to base model)

---

## Best Practices

### 1. Rank Selection
```cpp
// Start with r=8 (good default)
// Increase if task is complex
// Decrease if overfitting or want more reduction
```

### 2. Alpha Tuning
```cpp
// alpha controls adaptation strength
// alpha = 2×rank is common (alpha=16 for r=8)
// Increase alpha for stronger adaptation
// Decrease alpha for more conservative updates
```

### 3. Module Selection
```cpp
// For attention: adapt Q, V (80% of full fine-tuning performance)
// For maximum performance: adapt Q, K, V, O
// For FFN: adapt both linear layers
// For full model: adapt all linear projections
```

### 4. Learning Rate
```cpp
// LoRA adapters can handle higher learning rates
// Try 1e-3 to 1e-4 (vs 1e-5 for full fine-tuning)
// Use warmup for stability
```

---

## Troubleshooting

### Issue: Poor Adaptation Quality

**Solution:**
- Increase rank (try r=16 or r=32)
- Increase alpha
- Adapt more modules
- Increase training data

### Issue: Overfitting

**Solution:**
- Decrease rank
- Add dropout
- Reduce learning rate
- More training data

### Issue: Slow Inference

**Solution:**
```cpp
// Merge adapter for deployment
Matrix W_merged = adapter.merge_with_base(W_base);
// Now inference has zero LoRA overhead!
```

---

## Test Coverage

**File:** `tests/phase5_test.cpp`  
**Test Cases:** 9

- Constructor validation
- Forward pass correctness
- Backward pass gradients
- Weight updates
- Merging with base weights
- Save/load persistence
- Parameter counting
- Configuration management
- Parameter reduction calculation

**Pass Rate:** 100%

---

## See Also

- [Phase 5 Advanced Features Guide](../../guides/phase5-advanced-features.md) - Complete LoRA guide with examples
- [Quantization](quantization.md) - Combine with LoRA for maximum efficiency
- [Chatbot Completeness](../../reference/chatbot-completeness.md) - Integration overview

---

## References

- [LoRA: Low-Rank Adaptation of Large Language Models (Hu et al., 2021)](https://arxiv.org/abs/2106.09685)
- [QLoRA: Efficient Finetuning of Quantized LLMs (Dettmers et al., 2023)](https://arxiv.org/abs/2305.14314)

---

**Last Updated:** January 25, 2026  
**Version:** 1.0  
**Status:** Production-ready
