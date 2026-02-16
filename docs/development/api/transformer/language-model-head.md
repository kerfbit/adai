# LanguageModelHead Component - Context Documentation

**Version:** 1.1
**Last Updated:** January 24, 2026
**Status:** Active - Optimizer Integration Complete

## Overview

**File**: `src/LanguageModelHead.hpp`, `src/LanguageModelHead.cpp`
**Tests**: `tests/languagemodelhead_test.cpp` (42/42 tests passing)
**Purpose**: Final projection layer that maps decoder hidden states to vocabulary logits for next-token prediction
**Role in Decoder**: Output layer that enables the model to generate probability distributions over the vocabulary
**Dependencies**: `Matrix.hpp`, `Activation.hpp`, `Optimizer.hpp`

The LanguageModelHead is the final component in a transformer decoder that converts the model's internal representations (d_model dimension) into scores for each token in the vocabulary (vocab_size dimension). This is essential for language modeling tasks where the model needs to predict the next token.

---

## Architecture

### Component Structure

```text
Input: [seq_len, d_model]
    ↓
Linear Projection: input * W_output
    ↓
Add Bias: + bias (broadcasted)
    ↓
Output Logits: [seq_len, vocab_size]
    ↓
(Optional) Softmax → Probabilities
```

### Mathematical Formulation

**Forward Pass**:

```text
logits = input × W_output + bias

Where:
  input     ∈ ℝ^(seq_len × d_model)
  W_output  ∈ ℝ^(d_model × vocab_size)
  bias      ∈ ℝ^(1 × vocab_size)
  logits    ∈ ℝ^(seq_len × vocab_size)
```

**Probability Distribution** (optional):

```text
probabilities = softmax(logits)

For position i:
  P(token_j | context) = exp(logits[i,j]) / Σ_k exp(logits[i,k])
```

**Backward Pass**:

```text
Gradients:
  ∂L/∂W_output = input^T × ∂L/∂logits
  ∂L/∂bias     = Σ_i ∂L/∂logits[i,:]  (sum over sequence)
  ∂L/∂input    = ∂L/∂logits × W_output^T
```

---

## Class Interface

### Constructor

```cpp
LanguageModelHead(int d_model, int vocab_size)
```

**Parameters**:

- `d_model`: Model dimension (input feature size, typically 512 or 768)
- `vocab_size`: Size of vocabulary (output dimension, e.g., 50000)

**Initialization**:

- **Weights**: Xavier/Glorot initialization

```text
  scale = sqrt(2.0 / (d_model + vocab_size))
  W_output ~ Uniform(-scale, scale)
  ```

- **Bias**: Zero initialization
- **Gradients**: Zero initialization

**Example**:

```cpp
// For GPT-2 small: d_model=768, vocab_size=50257
LanguageModelHead lm_head(768, 50257);
```

### Public Methods

#### Forward Pass

```cpp
Matrix forward(const Matrix& input)
```

**Input**:

- `input`: Decoder output `[seq_len, d_model]`

**Output**:

- Logits `[seq_len, vocab_size]`

**Process**:

1. Cache input for backward pass
2. Compute `logits = input × W_output`
3. Add bias (broadcast to all sequence positions)
4. Return unnormalized scores

**Example**:

```cpp
Matrix decoder_output(10, 768);  // 10 tokens, 768 dimensions
Matrix logits = lm_head.forward(decoder_output);
// logits.shape = [10, 50257]
```

#### Get Probabilities

```cpp
std::vector<float> get_probabilities(const std::vector<float>& logits)
```

**Input**:

- `logits`: Raw scores for a single position `[vocab_size]`

**Output**:

- Probability distribution `[vocab_size]` summing to 1.0

**Process**:

1. Convert vector to Matrix
2. Apply softmax normalization
3. Convert back to vector

**Use Case**: Next-token prediction during text generation

**Example**:

```cpp
// Get logits for last position
std::vector<float> last_logits(vocab_size);
for (int i = 0; i < vocab_size; ++i) {
    last_logits[i] = logits(seq_len-1, i);
}

// Convert to probabilities
std::vector<float> probs = lm_head.get_probabilities(last_logits);

// Sample next token
int next_token = sample_from_distribution(probs);
```

#### Backward Pass

```cpp
Matrix backward(const Matrix& grad_output)
```

**Input**:

- `grad_output`: Gradient from loss `[seq_len, vocab_size]`

**Output**:

- Gradient w.r.t. input `[seq_len, d_model]`

**Process**:

1. Compute weight gradient: `grad_W = input^T × grad_output`
2. Accumulate into `W_output_grad`
3. Compute bias gradient: sum `grad_output` over sequence dimension
4. Accumulate into `bias_grad`
5. Compute input gradient: `grad_input = grad_output × W_output^T`

**Example**:

```cpp
// Assuming cross-entropy loss gradient
Matrix grad_from_loss(seq_len, vocab_size);
// grad_from_loss = predicted_probs - target_distribution

Matrix grad_to_decoder = lm_head.backward(grad_from_loss);
// grad_to_decoder.shape = [seq_len, d_model]
```

#### Weight Update

```cpp
void update_weights()
```

**Process**:

1. If optimizer is set, calls `optimizer->step()`
2. Otherwise, applies simple gradient descent:
   - `W_output -= learning_rate × W_output_grad`
   - `bias -= learning_rate × bias_grad`
3. Automatically zeros gradients

**Typical Usage**:

```cpp
// Training loop
for (int step = 0; step < num_steps; ++step) {
    Matrix logits = lm_head.forward(decoder_output);
    Matrix grad = compute_loss_gradient(logits, targets);
    lm_head.backward(grad);
    lm_head.update_weights();  // Uses optimizer if set
}
```

#### Optimizer Integration (New in v1.1)

```cpp
void set_optimizer(Optimizer* opt)
void register_parameters()
```

**Purpose**: Enable advanced optimization algorithms (Adam, AdamW, SGD with momentum)

**Usage**:

```cpp
LanguageModelHead lm_head(768, 50257);
Optimizer adam(OptimizerType::ADAM, 0.001f);
lm_head.set_optimizer(&adam);

// Parameters automatically registered:
//   - W_output and W_output_grad
//   - bias and bias_grad
```

**Backward Compatibility**: When optimizer is `nullptr`, uses simple gradient descent with `learning_rate`

#### Gradient Zeroing

```cpp
void zero_grad()
```

**Purpose**: Reset gradient accumulators to zero before new forward/backward pass

**Called By**: Automatically called by `update_weights()`, or manually before accumulating gradients

#### Model Persistence

```cpp
void save(const std::string& filepath)
void load(const std::string& filepath)
```

**Save Format** (binary):

1. Dimensions: `d_model`, `vocab_size` (2 × int)
2. W_output matrix: `d_model × vocab_size` floats
3. Bias vector: `vocab_size` floats

**Example**:

```cpp
lm_head.save("lm_head_weights.bin");

// Later, load into new instance
LanguageModelHead loaded_head(768, 50257);
loaded_head.load("lm_head_weights.bin");
```

---

## Implementation Details

### Memory Layout

**Parameters**:

- `W_output`: `d_model × vocab_size` = 768 × 50257 ≈ **154M floats** (617 MB)
- `bias`: `vocab_size` = 50257 floats (201 KB)
- **Total**: ~617 MB for GPT-2 scale

**Gradients** (same size as parameters):

- `W_output_grad`: ~617 MB
- `bias_grad`: ~201 KB

**Cache**:

- `cached_input`: `seq_len × d_model` (varies by sequence length)

### Initialization Strategy

**Xavier Initialization** for weights:

```cpp
float scale = std::sqrt(2.0f / (d_model + vocab_size));
W_output.randomize(scale);
```

**Why Xavier?**

- Maintains variance of activations across layers
- Prevents vanishing/exploding gradients
- Formula: `Var(W) = 2 / (n_in + n_out)`

**Bias Initialization**:

```cpp
bias.fill(0.0f);
```

- Standard practice: start with zero bias
- Model learns appropriate biases during training

### Gradient Computation

**Weight Gradient**:

```cpp
// grad_W = input^T × grad_output
Matrix grad_W = cached_input.transpose() * grad_output;

// Accumulate (supports mini-batching)
W_output_grad += grad_W;
```

**Bias Gradient**:

```cpp
// Sum over sequence dimension (each position contributes)
for (int j = 0; j < vocab_size; ++j) {
    float sum = 0.0f;
    for (int i = 0; i < seq_len; ++i) {
        sum += grad_output(i, j);
    }
    bias_grad(0, j) += sum;
}
```

**Input Gradient** (for decoder backprop):

```cpp
Matrix grad_input = grad_output * W_output.transpose();
```

---

## Integration with Decoder

### Typical Usage in Transformer Decoder

```cpp
// 1. Build decoder
LLMDecoder decoder(vocab_size, d_model, num_layers, num_heads, d_ff);

// 2. Add language model head
LanguageModelHead lm_head(d_model, vocab_size);

// 3. Forward pass
Matrix input_ids = tokenize(input_text);
Matrix decoder_output = decoder.forward(input_ids, causal_mask);
Matrix logits = lm_head.forward(decoder_output);

// 4. Compute loss (cross-entropy)
Matrix loss_grad = compute_cross_entropy_grad(logits, target_ids);

// 5. Backward pass
Matrix grad_to_decoder = lm_head.backward(loss_grad);
decoder.backward(grad_to_decoder);

// 6. Update weights
lm_head.update_weights();
decoder.update_weights();
```

### Text Generation Loop

```cpp
// Autoregressive generation
std::vector<int> generated_tokens;
for (int step = 0; step < max_length; ++step) {
    // Forward pass
    Matrix decoder_output = decoder.forward(generated_tokens);
    Matrix logits = lm_head.forward(decoder_output);

    // Get probabilities for last position
    std::vector<float> last_logits = extract_last_position(logits);
    std::vector<float> probs = lm_head.get_probabilities(last_logits);

    // Sample next token
    int next_token = sample_with_temperature(probs, temperature);
    generated_tokens.push_back(next_token);

    // Stop if EOS token
    if (next_token == EOS_TOKEN) break;
}
```

---

## Loss Functions

### Cross-Entropy Loss

**For Next-Token Prediction**:

```cpp
// Given logits [seq_len, vocab_size] and targets [seq_len]
Matrix compute_cross_entropy_grad(const Matrix& logits,
                                   const std::vector<int>& targets) {
    Matrix grad(logits.rows, logits.cols);

    for (int i = 0; i < logits.rows; ++i) {
        // Compute softmax probabilities
        std::vector<float> probs = softmax_row(logits, i);

        // Gradient: predicted_prob - 1.0 for correct token,
        //          predicted_prob - 0.0 for other tokens
        for (int j = 0; j < logits.cols; ++j) {
            grad(i, j) = probs[j];
            if (j == targets[i]) {
                grad(i, j) -= 1.0f;  // Subtract 1 for true token
            }
        }
    }

    return grad;
}
```

**Loss Value**:

```text
L = -Σ_i log(P(y_i | x_i))

Where y_i is the target token at position i
```

---

## Performance Characteristics

### Computational Complexity

**Forward Pass**:

- Matrix multiplication: `O(seq_len × d_model × vocab_size)`
- Bias addition: `O(seq_len × vocab_size)`
- **Total**: `O(seq_len × d_model × vocab_size)`

**Backward Pass**:

- Weight gradient: `O(seq_len × d_model × vocab_size)`
- Input gradient: `O(seq_len × d_model × vocab_size)`
- Bias gradient: `O(seq_len × vocab_size)`
- **Total**: `O(seq_len × d_model × vocab_size)`

### Memory Usage

**For GPT-2 Small (d_model=768, vocab_size=50257)**:

- Parameters: ~617 MB
- Gradients: ~617 MB
- Activations (seq_len=1024): ~50 MB
- **Total**: ~1.3 GB

**Optimization Opportunity**:

- Weight tying: Share embeddings with token embeddings to reduce parameters by ~50%
- Quantization: Use int8/int16 to reduce memory

### Bottlenecks

1. **Large Vocabulary**:
   - For vocab_size=50K and d_model=768, W_output has 38M parameters
   - This is often the largest single layer in the model

2. **Softmax Computation**:
   - Computing probabilities over large vocabulary is expensive
   - Use sampling strategies (top-k, nucleus) to reduce computation

---

## Design Patterns

### 1. Weight Tying (Optional Enhancement)

**Concept**: Share LanguageModelHead weights with TokenEmbedding layer

```cpp
// Instead of separate weights:
class LanguageModelHead {
    Matrix W_output;  // [d_model, vocab_size]
};

class TokenEmbedding {
    Matrix embedding_table;  // [vocab_size, d_model]
};

// Use same weights (transposed):
Matrix logits = input * embedding_table.transpose();
```

**Benefits**:

- Reduces parameters by ~50% (for typical models)
- Often improves generalization
- Used in GPT-2, GPT-3, BERT

### 2. Adaptive Softmax (For Very Large Vocabularies)

**Concept**: Cluster rare tokens into groups, compute probabilities hierarchically

**Benefits**:

- Reduces computation for vocab_size > 100K
- Speeds up both training and inference

### 3. Temperature Scaling

**Applied During Generation**:

```cpp
std::vector<float> get_probabilities_with_temp(
    const std::vector<float>& logits, float temperature) {

    std::vector<float> scaled_logits(logits.size());
    for (size_t i = 0; i < logits.size(); ++i) {
        scaled_logits[i] = logits[i] / temperature;
    }

    return get_probabilities(scaled_logits);
}
```

**Temperature Effects**:

- `T < 1.0`: Sharper distribution (more confident, less diverse)
- `T = 1.0`: Standard softmax
- `T > 1.0`: Flatter distribution (less confident, more diverse)

---

## Testing and Validation

### Unit Test Coverage

#### Test 1: Forward Pass Dimensions

```cpp
LanguageModelHead lm_head(512, 10000);
Matrix input(20, 512);  // 20 tokens
Matrix logits = lm_head.forward(input);
assert(logits.rows == 20);
assert(logits.cols == 10000);
```

#### Test 2: Probability Normalization

```cpp
std::vector<float> logits = {1.0, 2.0, 3.0, 4.0, 5.0};
std::vector<float> probs = lm_head.get_probabilities(logits);

float sum = 0.0f;
for (float p : probs) sum += p;
assert(abs(sum - 1.0f) < 1e-5);  // Should sum to 1.0
```

#### Test 3: Gradient Check

```cpp
// Numerical gradient vs analytical gradient
float epsilon = 1e-5f;
Matrix numerical_grad = compute_numerical_gradient(lm_head, input, epsilon);
Matrix analytical_grad = lm_head.backward(grad_output);

float max_diff = max_absolute_difference(numerical_grad, analytical_grad);
assert(max_diff < 1e-3);  // Should match closely
```

#### Test 4: Save/Load Consistency

```cpp
lm_head.save("test.bin");
LanguageModelHead loaded(512, 10000);
loaded.load("test.bin");

Matrix output1 = lm_head.forward(input);
Matrix output2 = loaded.forward(input);
assert(matrices_equal(output1, output2, 1e-6));
```

### Integration Tests

#### Test 5: Full Decoder Pipeline

```cpp
// Decoder → LanguageModelHead → Loss → Backward
Matrix decoder_out = decoder.forward(input_ids);
Matrix logits = lm_head.forward(decoder_out);
float loss = cross_entropy_loss(logits, targets);
Matrix grad = lm_head.backward(loss_gradient);
decoder.backward(grad);

// Check gradient flow
assert(decoder.has_gradients());
assert(lm_head.has_gradients());
```

---

## Common Pitfalls and Solutions

### Issue 1: Exploding Gradients in Large Vocabulary

**Problem**: Vocabulary size (50K+) causes large gradient magnitudes

**Solution**:

```cpp
// Gradient clipping
void clip_gradients(float max_norm) {
    float grad_norm = compute_gradient_norm();
    if (grad_norm > max_norm) {
        float scale = max_norm / grad_norm;
        W_output_grad = W_output_grad.scale(scale);
        bias_grad = bias_grad.scale(scale);
    }
}
```

### Issue 2: Numerical Instability in Softmax

**Problem**: `exp(logit)` overflows for large logits

**Solution**: LogSumExp trick (already handled in Activation::softmax)

```cpp
// Subtract max before exp
float max_logit = max(logits);
for (int i = 0; i < n; ++i) {
    stable_exp = exp(logits[i] - max_logit);
}
```

### Issue 3: Slow Inference with Large Vocabulary

**Problem**: Computing full softmax over 50K tokens is slow

**Solutions**:

1. **Top-k sampling**: Only consider top-k most likely tokens
2. **Nucleus (top-p) sampling**: Consider smallest set with cumulative prob > p
3. **Beam search**: Track only top-b candidates

### Issue 4: Memory Issues with Large Batch

**Problem**: Activations `[batch × seq_len × vocab_size]` don't fit in memory

**Solution**:

```cpp
// Process in chunks
for (int start = 0; start < seq_len; start += chunk_size) {
    int end = min(start + chunk_size, seq_len);
    Matrix chunk = input.slice(start, end);
    Matrix chunk_logits = lm_head.forward(chunk);
    // Process chunk...
}
```

---

## Comparison with Existing Components

### vs. FeedForward Layer

| Aspect | LanguageModelHead | FeedForward |
| -------- | ------------------- | ------------- |
| Purpose | Project to vocabulary | Non-linear transformation |
| Layers | 1 linear | 2 linear + activation |
| Output size | vocab_size | d_model |
| Activation | None (logits) | ReLU/GELU |
| Position | Final layer | Middle layers |

### vs. TokenEmbedding Layer

| Aspect | LanguageModelHead | TokenEmbedding |
| -------- | ------------------- | ---------------- |
| Direction | d_model → vocab | vocab → d_model |
| Input | Hidden states | Token IDs |
| Output | Logits | Embeddings |
| Weights | Can be tied | Can be tied |
| Gradient | Always computed | Only for trainable tokens |

---

## Future Enhancements

### 1. Weight Tying Support

```cpp
class LanguageModelHead {
    Matrix* W_output;  // Pointer to shared embedding table
    bool owns_weights;

    void tie_weights(Matrix& embedding_table) {
        W_output = &embedding_table;
        owns_weights = false;
    }
};
```

### 2. Label Smoothing

```cpp
Matrix compute_loss_with_smoothing(const Matrix& logits,
                                    const std::vector<int>& targets,
                                    float smoothing = 0.1f) {
    // Distribute smoothing probability to all tokens
    // Instead of P(correct) = 1.0, use P(correct) = 1.0 - smoothing
    // Remaining smoothing / (vocab_size - 1) to other tokens
}
```

### 3. Mixed Precision Support

```cpp
// Use fp16 for forward pass, fp32 for gradients
Matrix forward_fp16(const Matrix& input) {
    Matrix input_fp16 = input.to_half();
    Matrix logits_fp16 = compute_in_half_precision();
    return logits_fp16.to_float();
}
```

### 4. Vocabulary Projection Caching

```cpp
// Cache frequent projections
std::unordered_map<int, std::vector<float>> projection_cache;

std::vector<float> get_token_projection(int token_id) {
    if (projection_cache.count(token_id)) {
        return projection_cache[token_id];
    }
    // Compute and cache
}
```

---

## Usage Examples

### Example 1: Training Loop

```cpp
LanguageModelHead lm_head(768, 50000);
lm_head.learning_rate = 0.0001f;

for (int epoch = 0; epoch < num_epochs; ++epoch) {
    for (auto& batch : training_data) {
        lm_head.zero_grad();

        Matrix decoder_out = decoder.forward(batch.input);
        Matrix logits = lm_head.forward(decoder_out);

        Matrix grad = cross_entropy_gradient(logits, batch.targets);
        lm_head.backward(grad);
        lm_head.update_weights();
    }
}
```

### Example 2: Greedy Decoding

```cpp
std::vector<int> greedy_decode(LLMDecoder& decoder,
                               LanguageModelHead& lm_head,
                               const std::vector<int>& prompt,
                               int max_length) {
    std::vector<int> tokens = prompt;

    for (int i = 0; i < max_length; ++i) {
        Matrix decoder_out = decoder.forward(tokens);
        Matrix logits = lm_head.forward(decoder_out);

        // Get last position logits
        std::vector<float> last_logits(lm_head.vocab_size);
        for (int j = 0; j < lm_head.vocab_size; ++j) {
            last_logits[j] = logits(logits.rows - 1, j);
        }

        // Select token with highest logit
        int next_token = argmax(last_logits);
        tokens.push_back(next_token);

        if (next_token == EOS_TOKEN) break;
    }

    return tokens;
}
```

### Example 3: Top-k Sampling

```cpp
int sample_top_k(const std::vector<float>& logits, int k, float temperature) {
    // Get probabilities with temperature
    std::vector<float> scaled_logits(logits.size());
    for (size_t i = 0; i < logits.size(); ++i) {
        scaled_logits[i] = logits[i] / temperature;
    }

    // Get top-k indices
    auto top_k_indices = get_top_k_indices(scaled_logits, k);

    // Compute probabilities only for top-k
    std::vector<float> top_k_probs(k);
    float sum = 0.0f;
    for (int i = 0; i < k; ++i) {
        top_k_probs[i] = exp(scaled_logits[top_k_indices[i]]);
        sum += top_k_probs[i];
    }

    // Normalize
    for (int i = 0; i < k; ++i) {
        top_k_probs[i] /= sum;
    }

    // Sample from top-k
    int sampled_idx = sample_categorical(top_k_probs);
    return top_k_indices[sampled_idx];
}
```

---

## References

### Related Components

- **Decoder.hpp**: Stacks DecoderBlocks to process sequences
- **TokenEmbedding.hpp**: Converts token IDs to embeddings (inverse operation)
- **Activation.hpp**: Provides softmax for probability computation
- **Matrix.hpp**: Core tensor operations

### Design Documents

- **DECODER_DESIGN.md**: Overall decoder architecture
- **DECODER_IMPLEMENTATION_GUIDE.md**: Implementation patterns
- **ENCODER_DECODER_COMPARISON.md**: Differences from encoder

### Academic References

- "Attention Is All You Need" (Vaswani et al., 2017)
  - Section 3.1: Scaled Dot-Product Attention
  - Section 3.3: Position-wise Feed-Forward Networks
- "Language Models are Unsupervised Multitask Learners" (GPT-2)
  - Weight tying between embedding and output layers
- "Deep Learning" (Goodfellow et al., 2016)
  - Chapter 6.2.2: Softmax function
  - Chapter 8.7: Optimization strategies

---

## Summary

The **LanguageModelHead** component is the critical final layer that transforms decoder hidden states into actionable vocabulary predictions. Key characteristics:

✅ **Simple but Essential**: Single linear projection + bias
✅ **Scalable**: Handles vocabularies from 1K to 100K+ tokens
✅ **Flexible**: Supports both training (with gradients) and inference (with sampling)
✅ **Efficient**: Direct matrix operations with minimal overhead
✅ **Integration-Ready**: Works seamlessly with decoder and loss functions
✅ **Advanced Optimization**: Supports Adam, AdamW, SGD with momentum (v1.1)

**Performance**: O(seq_len × d_model × vocab_size) complexity makes this the most expensive layer for large vocabularies, but optimizations like weight tying and adaptive softmax can mitigate costs.

**Best Practice**: Start with standard implementation, add weight tying if memory is tight, use sampling strategies (top-k/nucleus) for faster generation.

---

## Recent Updates

### Version 1.1 (January 24, 2026)

#### Major Feature: Optimizer Integration

Added support for advanced optimization algorithms through the `Optimizer` class:

**New Functionality:**

- `set_optimizer(Optimizer* opt)` - Register optimizer and parameters
- `register_parameters()` - Explicitly register parameter groups
- Enhanced `update_weights()` - Uses optimizer when available, falls back to gradient descent

**Parameter Management:**

- Registers 2 parameter groups: W_output, bias
- Each group linked with corresponding gradient matrix
- Supports Adam, AdamW, SGD with momentum

**Backward Compatibility:**

- Optimizer pointer defaults to `nullptr`
- Existing code works without modification
- `learning_rate` still used when optimizer not set
- Simple gradient descent fallback maintained

**Benefits:**

- Better convergence with adaptive learning rates
- Built-in weight decay and momentum support
- Learning rate scheduling capability
- Unified optimization across model components
- Especially beneficial for large vocabulary sizes

**Test Coverage:**

- Added 12 comprehensive optimizer integration tests
- Total test suite: 42/42 tests passing
- Tests cover all optimizer types and edge cases

**Migration:**

```cpp
// Old code (still works):
LanguageModelHead lm_head(768, 50257);
lm_head.learning_rate = 0.001f;
lm_head.update_weights();

// New code (with optimizer):
LanguageModelHead lm_head(768, 50257);
Optimizer opt(OptimizerType::ADAM, 0.001f);
lm_head.set_optimizer(&opt);
lm_head.update_weights();  // Uses Adam
```

**Documentation Updates:**

- Added Optimizer Integration section
- Updated Weight Update documentation
- Added backward compatibility notes
- Included optimizer usage examples

---

End of Documentation
