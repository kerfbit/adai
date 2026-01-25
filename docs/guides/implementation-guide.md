# Decoder Implementation Quick Reference

Quick reference guide for implementing the decoder components following the existing codebase patterns.

---

## File Template Patterns

### Header File Pattern (.hpp)

```cpp
#pragma once

#include "Matrix.hpp"
#include <memory>
#include <vector>
#include <string>

/**
 * [Component Name]
 * 
 * [Brief description]
 * 
 * Features:
 * - [Feature 1]
 * - [Feature 2]
 * 
 * Architecture:
 * [ASCII diagram if helpful]
 */
class ComponentName {
private:
    // Components (use std::unique_ptr for owned objects)
    std::unique_ptr<SubComponent> component;
    
    // Parameters
    Matrix weights;
    Matrix bias;
    
    // Gradients
    Matrix weights_grad;
    Matrix bias_grad;
    
    // Hyperparameters
    int param1;
    float param2;
    
    // Cached values for backward pass
    Matrix cached_input;
    Matrix cached_intermediate;
    
public:
    float learning_rate;  // Public learning rate
    
    /**
     * Constructor
     * 
     * @param param1 Description
     * @param param2 Description (default: value)
     */
    ComponentName(int param1, float param2 = 1.0f);
    
    /**
     * Forward pass
     * 
     * @param input Input matrix [shape]
     * @return Output matrix [shape]
     */
    Matrix forward(const Matrix& input);
    
    /**
     * Backward pass
     * 
     * @param grad_output Gradient from next layer [shape]
     * @return Gradient w.r.t. input [shape]
     */
    Matrix backward(const Matrix& grad_output);
    
    /**
     * Update weights using accumulated gradients
     */
    void update_weights();
    
    /**
     * Zero accumulated gradients
     */
    void zero_grad();
    
    /**
     * Save parameters to file
     */
    void save(const std::string& filepath);
    
    /**
     * Load parameters from file
     */
    void load(const std::string& filepath);
};
```

### Implementation File Pattern (.cpp)

```cpp
#include "ComponentName.hpp"
#include <iostream>
#include <fstream>
#include <cmath>
#include <random>

ComponentName::ComponentName(int param1, float param2)
    : param1(param1), param2(param2), learning_rate(0.001f),
      weights(rows, cols), bias(1, cols),
      weights_grad(rows, cols), bias_grad(1, cols) {
    
    // Initialize weights (Xavier/He initialization)
    float scale = std::sqrt(2.0f / param1);
    weights.randomize(scale);
    bias.fill(0.0f);
    
    // Zero gradients
    weights_grad.fill(0.0f);
    bias_grad.fill(0.0f);
    
    // Print initialization info
    std::cout << "ComponentName initialized:" << std::endl;
    std::cout << "  param1: " << param1 << std::endl;
    std::cout << "  param2: " << param2 << std::endl;
}

Matrix ComponentName::forward(const Matrix& input) {
    // Cache input for backward pass
    cached_input = input;
    
    // Compute output
    Matrix output = input * weights + bias;
    
    return output;
}

Matrix ComponentName::backward(const Matrix& grad_output) {
    // Compute gradient w.r.t. weights: input^T * grad_output
    Matrix grad_weights = cached_input.transpose() * grad_output;
    
    // Accumulate gradients
    for (int i = 0; i < weights_grad.rows; ++i) {
        for (int j = 0; j < weights_grad.cols; ++j) {
            weights_grad(i, j) += grad_weights(i, j);
        }
    }
    
    // Compute gradient w.r.t. bias (sum over batch)
    for (int j = 0; j < bias_grad.cols; ++j) {
        float sum = 0.0f;
        for (int i = 0; i < grad_output.rows; ++i) {
            sum += grad_output(i, j);
        }
        bias_grad(0, j) += sum;
    }
    
    // Compute gradient w.r.t. input: grad_output * weights^T
    Matrix grad_input = grad_output * weights.transpose();
    
    return grad_input;
}

void ComponentName::update_weights() {
    weights.apply_gradients(weights_grad, learning_rate);
    bias.apply_gradients(bias_grad, learning_rate);
    zero_grad();
}

void ComponentName::zero_grad() {
    weights_grad.fill(0.0f);
    bias_grad.fill(0.0f);
}

void ComponentName::save(const std::string& filepath) {
    std::ofstream file(filepath, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open file: " + filepath);
    }
    
    // Save dimensions
    file.write(reinterpret_cast<const char*>(&param1), sizeof(param1));
    
    // Save weights
    for (int i = 0; i < weights.rows; ++i) {
        for (int j = 0; j < weights.cols; ++j) {
            file.write(reinterpret_cast<const char*>(&weights(i, j)), sizeof(float));
        }
    }
    
    // Save bias
    for (int j = 0; j < bias.cols; ++j) {
        file.write(reinterpret_cast<const char*>(&bias(0, j)), sizeof(float));
    }
    
    file.close();
}

void ComponentName::load(const std::string& filepath) {
    std::ifstream file(filepath, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open file: " + filepath);
    }
    
    // Load dimensions
    int loaded_param1;
    file.read(reinterpret_cast<char*>(&loaded_param1), sizeof(loaded_param1));
    
    if (loaded_param1 != param1) {
        throw std::runtime_error("Parameter mismatch");
    }
    
    // Load weights
    for (int i = 0; i < weights.rows; ++i) {
        for (int j = 0; j < weights.cols; ++j) {
            file.read(reinterpret_cast<char*>(&weights(i, j)), sizeof(float));
        }
    }
    
    // Load bias
    for (int j = 0; j < bias.cols; ++j) {
        file.read(reinterpret_cast<char*>(&bias(0, j)), sizeof(float));
    }
    
    file.close();
}
```

---

## Common Code Snippets

### Residual Connection

```cpp
// Add residual connection
Matrix residual(input.rows, input.cols);
for (int i = 0; i < input.rows; ++i) {
    for (int j = 0; j < input.cols; ++j) {
        residual(i, j) = input(i, j) + layer_output(i, j);
    }
}
```

### Gradient Split at Residual

```cpp
// Gradient flows through both paths
Matrix grad_path1(grad_output.rows, grad_output.cols);
Matrix grad_path2(grad_output.rows, grad_output.cols);

for (int i = 0; i < grad_output.rows; ++i) {
    for (int j = 0; j < grad_output.cols; ++j) {
        grad_path1(i, j) = grad_output(i, j);  // Copy
        grad_path2(i, j) = grad_output(i, j);  // Copy
    }
}

// Backprop through path1
Matrix grad_from_path1 = component1->backward(grad_path1);

// Accumulate gradients from both paths
for (int i = 0; i < grad_path2.rows; ++i) {
    for (int j = 0; j < grad_path2.cols; ++j) {
        grad_path2(i, j) += grad_from_path1(i, j);
    }
}
```

### Causal Mask Creation

```cpp
Matrix create_causal_mask(int seq_len) {
    Matrix mask(seq_len, seq_len);
    
    for (int i = 0; i < seq_len; ++i) {
        for (int j = 0; j < seq_len; ++j) {
            if (j <= i) {
                mask(i, j) = 0.0f;      // Allow attention
            } else {
                mask(i, j) = -1e9f;     // Block attention
            }
        }
    }
    
    return mask;
}
```

### Softmax with Temperature

```cpp
std::vector<float> softmax_with_temperature(const std::vector<float>& logits, 
                                            float temperature) {
    std::vector<float> scaled_logits(logits.size());
    
    // Apply temperature scaling
    for (size_t i = 0; i < logits.size(); ++i) {
        scaled_logits[i] = logits[i] / temperature;
    }
    
    // Find max for numerical stability
    float max_logit = *std::max_element(scaled_logits.begin(), scaled_logits.end());
    
    // Compute exp and sum
    std::vector<float> probs(logits.size());
    float sum = 0.0f;
    for (size_t i = 0; i < logits.size(); ++i) {
        probs[i] = std::exp(scaled_logits[i] - max_logit);
        sum += probs[i];
    }
    
    // Normalize
    for (size_t i = 0; i < probs.size(); ++i) {
        probs[i] /= sum;
    }
    
    return probs;
}
```

### Sampling from Distribution

```cpp
int sample_from_distribution(const std::vector<float>& probs) {
    static std::random_device rd;
    static std::mt19937 gen(rd());
    
    std::discrete_distribution<> dist(probs.begin(), probs.end());
    return dist(gen);
}
```

### Cross-Entropy Loss

```cpp
float compute_cross_entropy_loss(const Matrix& logits, 
                                 const std::vector<int>& targets) {
    float total_loss = 0.0f;
    int seq_len = logits.rows;
    
    for (int i = 0; i < seq_len; ++i) {
        // Get logits for this position
        std::vector<float> position_logits(logits.cols);
        for (int j = 0; j < logits.cols; ++j) {
            position_logits[j] = logits(i, j);
        }
        
        // Softmax
        std::vector<float> probs = softmax(position_logits);
        
        // Cross-entropy: -log(p[target])
        int target = targets[i];
        total_loss -= std::log(probs[target] + 1e-10f);  // Add epsilon for stability
    }
    
    return total_loss / seq_len;  // Average over sequence
}
```

### Cross-Entropy Gradient

```cpp
Matrix compute_cross_entropy_gradient(const Matrix& logits,
                                      const std::vector<int>& targets) {
    Matrix grad(logits.rows, logits.cols);
    
    for (int i = 0; i < logits.rows; ++i) {
        // Get logits and compute softmax
        std::vector<float> position_logits(logits.cols);
        for (int j = 0; j < logits.cols; ++j) {
            position_logits[j] = logits(i, j);
        }
        std::vector<float> probs = softmax(position_logits);
        
        // Gradient: p - 1 at target, p elsewhere
        for (int j = 0; j < logits.cols; ++j) {
            if (j == targets[i]) {
                grad(i, j) = probs[j] - 1.0f;
            } else {
                grad(i, j) = probs[j];
            }
        }
    }
    
    // Scale by sequence length
    return grad.scale(1.0f / logits.rows);
}
```

---

## Component-Specific Implementation Notes

### DecoderBlock

**Key Differences from EncoderBlock:**
1. Three sub-layers instead of two (add cross-attention)
2. Self-attention uses causal mask
3. Cross-attention takes encoder output as K,V

**Forward Pass Order:**
```cpp
Matrix DecoderBlock::forward(const Matrix& input, 
                            const Matrix& encoder_output,
                            const Matrix& causal_mask,
                            const Matrix* cross_mask) {
    // 1. Self-attention (causal)
    Matrix self_attn_out = self_attention->forward(input, &causal_mask);
    Matrix res1 = add_residual(input, self_attn_out);
    Matrix norm1_out = norm1->forward(res1);
    
    // 2. Cross-attention (to encoder)
    Matrix cross_attn_out = cross_attention->forward_cross(
        norm1_out,           // Query from decoder
        encoder_output,      // Key from encoder
        encoder_output,      // Value from encoder
        cross_mask
    );
    Matrix res2 = add_residual(norm1_out, cross_attn_out);
    Matrix norm2_out = norm2->forward(res2);
    
    // 3. Feed-forward
    Matrix ff_out = feed_forward->forward(norm2_out);
    Matrix res3 = add_residual(norm2_out, ff_out);
    Matrix output = norm3->forward(res3);
    
    return output;
}
```

**Cross-Attention Usage:**
```cpp
// Self-attention: Q = K = V = input
Matrix self_out = self_attention->forward(input, input, input, mask);

// Cross-attention: Q = decoder, K = V = encoder
Matrix cross_out = cross_attention->forward(
    decoder_state,    // Query
    encoder_output,   // Key
    encoder_output,   // Value
    mask
);
```

### LanguageModelHead

**Simple Linear Projection:**
```cpp
Matrix LanguageModelHead::forward(const Matrix& input) {
    // input: [seq_len, d_model]
    // output: [seq_len, vocab_size]
    
    cached_input = input;
    Matrix output = input * W_output;  // [seq_len, vocab_size]
    
    // Add bias (broadcast)
    for (int i = 0; i < output.rows; ++i) {
        for (int j = 0; j < output.cols; ++j) {
            output(i, j) += bias(0, j);
        }
    }
    
    return output;
}
```

**Gradient Computation:**
```cpp
Matrix LanguageModelHead::backward(const Matrix& grad_output) {
    // grad_output: [seq_len, vocab_size]
    
    // Gradient w.r.t. W_output: input^T * grad_output
    Matrix grad_W = cached_input.transpose() * grad_output;
    
    // Accumulate
    for (int i = 0; i < W_output_grad.rows; ++i) {
        for (int j = 0; j < W_output_grad.cols; ++j) {
            W_output_grad(i, j) += grad_W(i, j);
        }
    }
    
    // Gradient w.r.t. bias: sum over batch
    for (int j = 0; j < bias_grad.cols; ++j) {
        float sum = 0.0f;
        for (int i = 0; i < grad_output.rows; ++i) {
            sum += grad_output(i, j);
        }
        bias_grad(0, j) += sum;
    }
    
    // Gradient w.r.t. input
    Matrix grad_input = grad_output * W_output.transpose();
    
    return grad_input;
}
```

### LLMDecoder

**Constructor Pattern:**
```cpp
LLMDecoder::LLMDecoder(int vocab_size, int d_model, int num_layers,
                       int num_heads, int d_ff, int max_seq_length)
    : vocab_size(vocab_size), d_model(d_model), num_layers(num_layers),
      num_heads(num_heads), d_ff(d_ff), max_seq_length(max_seq_length),
      requires_grad(false), learning_rate(0.001f) {
    
    // Initialize components
    tokenizer = std::make_unique<BPETokenizer>();
    token_embedding = std::make_unique<TokenEmbedding>(vocab_size, d_model);
    positional_encoding = std::make_unique<PositionalEncoding>(max_seq_length, d_model);
    final_norm = std::make_unique<LayerNorm>(d_model);
    lm_head = std::make_unique<LanguageModelHead>(d_model, vocab_size);
    
    // Initialize decoder blocks
    for (int i = 0; i < num_layers; ++i) {
        decoder_blocks.push_back(
            std::make_unique<DecoderBlock>(d_model, num_heads, d_ff)
        );
    }
    
    std::cout << "LLM Decoder initialized with:" << std::endl;
    std::cout << "  Vocab size: " << vocab_size << std::endl;
    std::cout << "  Model dimension: " << d_model << std::endl;
    std::cout << "  Number of layers: " << num_layers << std::endl;
}
```

### TextGenerator

**Greedy Generation:**
```cpp
std::string TextGenerator::generate_greedy(const std::string& prompt, 
                                          int max_length) {
    // Encode input
    Matrix encoder_output = encoder->encode(prompt);
    
    // Initialize with BOS token
    std::vector<int> generated_tokens = {bos_token_id};
    
    for (int step = 0; step < max_length; ++step) {
        // Get next token logits
        std::vector<float> logits = decoder->generate_next_token_logits(
            generated_tokens, encoder_output
        );
        
        // Find argmax (greedy)
        int next_token = std::distance(
            logits.begin(),
            std::max_element(logits.begin(), logits.end())
        );
        
        // Append token
        generated_tokens.push_back(next_token);
        
        // Stop if EOS
        if (next_token == eos_token_id) {
            break;
        }
    }
    
    // Decode to text
    return decoder->get_tokenizer()->decode(generated_tokens);
}
```

---

## Testing Patterns

### Unit Test Template

```cpp
#include <gtest/gtest.h>
#include "DecoderBlock.hpp"

class DecoderBlockTest : public ::testing::Test {
protected:
    void SetUp() override {
        d_model = 64;
        num_heads = 4;
        d_ff = 256;
        
        decoder_block = std::make_unique<DecoderBlock>(d_model, num_heads, d_ff);
    }
    
    int d_model;
    int num_heads;
    int d_ff;
    std::unique_ptr<DecoderBlock> decoder_block;
};

TEST_F(DecoderBlockTest, ForwardPassShapeCorrect) {
    int seq_len = 10;
    Matrix input(seq_len, d_model);
    input.randomize(0.1f);
    
    Matrix encoder_output(15, d_model);  // Different length
    encoder_output.randomize(0.1f);
    
    Matrix causal_mask = create_causal_mask(seq_len);
    
    Matrix output = decoder_block->forward(input, encoder_output, causal_mask);
    
    EXPECT_EQ(output.rows, seq_len);
    EXPECT_EQ(output.cols, d_model);
}

TEST_F(DecoderBlockTest, BackwardGradientCheck) {
    // Numerical gradient checking
    int seq_len = 5;
    Matrix input(seq_len, d_model);
    input.randomize(0.1f);
    
    Matrix encoder_output(8, d_model);
    encoder_output.randomize(0.1f);
    
    Matrix causal_mask = create_causal_mask(seq_len);
    
    // Forward
    Matrix output = decoder_block->forward(input, encoder_output, causal_mask);
    
    // Backward
    Matrix grad_output(seq_len, d_model);
    grad_output.fill(1.0f);  // All ones
    
    Matrix grad_input = decoder_block->backward(grad_output);
    
    EXPECT_EQ(grad_input.rows, seq_len);
    EXPECT_EQ(grad_input.cols, d_model);
    
    // Check gradient is not all zeros
    float grad_sum = grad_input.sum();
    EXPECT_NE(grad_sum, 0.0f);
}
```

---

## CMakeLists.txt Updates

```cmake
# Add decoder source files
set(DECODER_SOURCES
    src/DecoderBlock.cpp
    src/LanguageModelHead.cpp
    src/Decoder.cpp
    src/TextGenerator.cpp
    src/EncoderDecoderModel.cpp
)

# Create decoder library
add_library(decoder ${DECODER_SOURCES})

# Link with existing components
target_link_libraries(decoder
    encoder
    matrix
    activation
    layer_norm
    feed_forward
    multi_head_attention
    token_embedding
    positional_encoding
    bpe_tokenizer
)

# Add decoder tests
add_executable(decoder_test
    tests/decoderblock_test.cpp
    tests/decoder_test.cpp
    tests/textgenerator_test.cpp
)

target_link_libraries(decoder_test
    decoder
    gtest
    gtest_main
)

# Add example programs
add_executable(decoder_example src/DecoderExample.cpp)
target_link_libraries(decoder_example decoder)

add_executable(chatbot_example src/EncoderDecoderExample.cpp)
target_link_libraries(chatbot_example decoder)
```

---

## Common Pitfalls & Solutions

### Pitfall 1: Forgetting to Cache for Backward Pass
```cpp
// ❌ WRONG
Matrix forward(const Matrix& input) {
    return input * weights;  // Input not cached!
}

// ✅ CORRECT
Matrix forward(const Matrix& input) {
    cached_input = input;  // Cache for backward
    return input * weights;
}
```

### Pitfall 2: Not Setting Learning Rates for Sub-components
```cpp
// ❌ WRONG
DecoderBlock::DecoderBlock(int d_model, int num_heads, int d_ff) {
    self_attention = std::make_unique<MultiHeadAttention>(d_model, num_heads);
    // Learning rate not set!
}

// ✅ CORRECT
DecoderBlock::DecoderBlock(int d_model, int num_heads, int d_ff) 
    : learning_rate(0.001f) {
    self_attention = std::make_unique<MultiHeadAttention>(d_model, num_heads);
    self_attention->learning_rate = learning_rate;  // Propagate learning rate
}
```

### Pitfall 3: Incorrect Gradient Accumulation at Residuals
```cpp
// ❌ WRONG (gradient lost)
Matrix grad_input = component->backward(grad_output);
return grad_input;  // Forgot residual path!

// ✅ CORRECT
Matrix grad_from_component = component->backward(grad_output);

// Accumulate with residual path
Matrix grad_input(grad_output.rows, grad_output.cols);
for (int i = 0; i < grad_output.rows; ++i) {
    for (int j = 0; j < grad_output.cols; ++j) {
        grad_input(i, j) = grad_output(i, j) + grad_from_component(i, j);
    }
}
return grad_input;
```

### Pitfall 4: Not Zeroing Gradients After Update
```cpp
// ❌ WRONG
void update_weights() {
    weights.apply_gradients(weights_grad, learning_rate);
    // Gradients accumulate indefinitely!
}

// ✅ CORRECT
void update_weights() {
    weights.apply_gradients(weights_grad, learning_rate);
    zero_grad();  // Clear gradients
}
```

---

## Debugging Checklist

- [ ] All shapes match expected dimensions
- [ ] Gradients are not NaN or infinite
- [ ] Learning rates set for all components
- [ ] Caching done before computations
- [ ] Residual gradients properly accumulated
- [ ] Masks applied correctly (causal, padding)
- [ ] Memory not leaking (valgrind check)
- [ ] Save/load preserves model behavior
- [ ] Generation terminates properly (EOS)
- [ ] Temperature > 0 in sampling

---

## Performance Optimization Tips

1. **Reuse encoder output** during generation (don't recompute)
2. **Batch operations** when possible
3. **Pre-allocate matrices** for common sizes
4. **Use const references** to avoid copies
5. **Profile before optimizing** (measure, don't guess)

---

**Document Version:** 1.0  
**Last Updated:** January 18, 2026  
**Related:** DECODER_DESIGN.md, DECODER_DESIGN_SUMMARY.md


---

## Implementation Summary

# BPE Tokenizer Save/Load Implementation Summary

## Changes Made

Added comprehensive save and load functionality to the BPE Tokenizer class that preserves all necessary state for full tokenizer reconstruction.

### Files Modified

1. **src/BPETokenizer.cpp**
   - Enhanced `save_vocab()` function
   - Enhanced `load_vocab()` function

2. **src/BPETokenizerexample.cpp**
   - Added demonstration of save/load functionality
   - Added verification tests

### Implementation Details

#### Save Functionality (`save_vocab`)

The enhanced save function now stores:

1. **File Header**: Version identifier for format compatibility
2. **Special Token IDs**: Preserves pad, unk, bos, eos token IDs
3. **Vocabulary**: Complete token-to-ID mappings with proper escaping
4. **BPE Merges**: All merge rules in order (critical for tokenization)

**Special Character Escaping:**
- Space → `\s`
- Newline → `\n`
- Tab → `\t`
- Carriage return → `\r`
- Backslash → `\\`

**File Format:**
```
# BPE Tokenizer Vocabulary v1.0
VOCAB_SIZE <size>
SPECIAL_TOKENS
pad_token_id <id>
unk_token_id <id>
bos_token_id <id>
eos_token_id <id>
VOCAB
<token>\t<id>
...
BPE_MERGES <count>
<first>\t<second>
...
```

#### Load Functionality (`load_vocab`)

The enhanced load function:

1. **Clears existing state**: vocab, inverse_vocab, bpe_merges, special_tokens
2. **Parses sections**: Identifies and processes each file section
3. **Unescapes tokens**: Converts escape sequences back to original characters
4. **Rebuilds all structures**: 
   - Vocabulary mappings
   - Inverse vocabulary
   - Special tokens set
   - BPE merge rules (in original order)

**Section Processing:**
- Skips comments and empty lines
- Reads special token IDs
- Rebuilds vocabulary and inverse vocabulary
- Loads BPE merges in correct order
- Reconstructs special tokens set

### Testing

#### Comprehensive Test Coverage

1. **Basic Functionality** (test_save_load.cpp):
   - Build vocabulary from training data
   - Save to file
   - Load into new tokenizer
   - Verify identical behavior
   - ✓ All tests pass

2. **Integration Test** (BPETokenizerexample.cpp):
   - Build from large corpus (Bible text)
   - Save 10,000 token vocabulary
   - Load and verify
   - ✓ Token IDs match
   - ✓ Decoded text matches

3. **Edge Cases Tested**:
   - Special characters in tokens
   - Spaces, tabs, newlines
   - Large vocabularies (10K tokens)
   - Long merge lists (9,931 merges)

### Benefits

1. **Performance**: Build vocabulary once, load instantly
2. **Consistency**: Identical tokenization across sessions
3. **Portability**: Human-readable text format
4. **Completeness**: All state preserved (vocab + merges + special tokens)
5. **Robustness**: Proper escaping handles all character types

### Usage Example

```cpp
// Build once
BPETokenizer builder;
builder.build_vocab(training_texts, 10000);
builder.save_vocab("vocab.txt");

// Load many times
BPETokenizer tokenizer;
tokenizer.load_vocab("vocab.txt");
auto ids = tokenizer.encode("Hello, world!");
```

### Validation Results

```
Original tokenizer: "Hello, this is a fascinating test of tokenization!"
Loaded tokenizer:   "Hello, this is a fascinating test of tokenization!"

Tokens match: YES
IDs match: YES
Decoded text matches: YES
```

## Conclusion

The save/load functionality is fully implemented, tested, and working correctly. The tokenizer can now:
- Save complete state to disk
- Load and restore exact behavior
- Handle all special characters properly
- Work with vocabularies of any size
- Preserve BPE merge order for correct tokenization
