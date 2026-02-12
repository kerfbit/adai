# PositionalEncoding Class - Technical Context Documentation

## Table of Contents

1. [Overview](#overview)
2. [Mathematical Foundation](#mathematical-foundation)
3. [Class Architecture](#class-architecture)
4. [Implementation Details](#implementation-details)
5. [Usage Patterns](#usage-patterns)
6. [Performance Considerations](#performance-considerations)
7. [Comparison with Learned Positional Embeddings](#comparison-with-learned-positional-embeddings)
8. [Common Issues and Solutions](#common-issues-and-solutions)
9. [Integration Examples](#integration-examples)
10. [References](#references)

---

## Overview

### Purpose

The `PositionalEncoding` class implements **sinusoidal positional encodings** for Transformer architectures, as introduced in the seminal paper "Attention is All You Need" (Vaswani et al., 2017). Unlike recurrent neural networks that inherently process sequences in order, Transformers process all positions in parallel and require explicit positional information to understand token order.

Positional encoding solves this problem by adding position-dependent signals to input embeddings, enabling the model to:

- **Distinguish token positions** - Understand which token appears where in the sequence
- **Learn relative positions** - Attend to tokens based on their distance from each other
- **Generalize to longer sequences** - Work with sequences longer than those seen during training
- **Maintain permutation sensitivity** - Produce different outputs for different orderings of the same tokens

### Key Benefits

1. **Deterministic and Fixed** - No learnable parameters, reducing model complexity
2. **Length Generalization** - Can handle sequences longer than training examples
3. **Relative Position Encoding** - The sinusoidal pattern allows the model to learn relative positions
4. **Efficient Computation** - Pre-computed once during initialization
5. **Mathematical Properties** - Linear relationships between positions enable attention to relative offsets

### Location

- **Header**: `src/PositionalEncoding.hpp`
- **Implementation**: `src/PositionalEncoding.cpp`
- **Dependencies**: `Matrix.hpp`, `<cmath>`, `<vector>`, `<iostream>`

---

## Mathematical Foundation

### Sinusoidal Positional Encoding Formula

For a given position $\text{pos}$ in the sequence and dimension index $i$ in the embedding:

$$\text{PE}(\text{pos}, 2i) = \sin\left(\frac{\text{pos}}{10000^{2i/d_{\text{model}}}}\right)$$

$$\text{PE}(\text{pos}, 2i+1) = \cos\left(\frac{\text{pos}}{10000^{2i/d_{\text{model}}}}\right)$$

Where:

- $\text{pos}$ is the position in the sequence (0-indexed)
- $i$ is the dimension index ($0 \leq i < d_{\text{model}}/2$)
- $d_{\text{model}}$ is the embedding dimension
- Even indices (0, 2, 4, ...) use sine function
- Odd indices (1, 3, 5, ...) use cosine function

### Why Sinusoidal Functions?

The choice of sinusoidal functions provides several important properties:

#### 1. **Bounded Range**

All positional encoding values lie in $[-1, 1]$, preventing encoding values from dominating the input embeddings.

#### 2. **Unique Encoding for Each Position**

Each position has a distinct encoding pattern across all dimensions.

#### 3. **Relative Position Representation**

For any fixed offset $k$, the encoding at position $\text{pos} + k$ can be represented as a linear function of the encoding at position $\text{pos}$:

$$\text{PE}(\text{pos} + k) = f(\text{PE}(\text{pos}), k)$$

This linear relationship allows the model to easily learn to attend by relative positions.

Mathematically:
$$\sin(\alpha + \beta) = \sin(\alpha)\cos(\beta) + \cos(\alpha)\sin(\beta)$$
$$\cos(\alpha + \beta) = \cos(\alpha)\cos(\beta) - \sin(\alpha)\sin(\beta)$$

#### 4. **Wavelength Spectrum**

Different dimensions encode position at different wavelengths:

- **Lower dimensions** (small $i$): High frequency, change rapidly with position
  - Wavelength $\approx 2\pi$ (useful for distinguishing adjacent tokens)

- **Higher dimensions** (large $i$): Low frequency, change slowly with position
  - Wavelength $\approx 2\pi \times 10000$ (useful for long-range dependencies)

The wavelength for dimension $2i$ is:
$$\lambda_i = 2\pi \times 10000^{2i/d_{\text{model}}}$$

This creates a geometric progression from $2\pi$ to $2\pi \times 10000$.

### Visualization of Encoding Pattern

For a 512-dimensional embedding:

- Dimensions 0-1: Wavelength ≈ 6.28, completes ~16 cycles over 100 positions
- Dimensions 510-511: Wavelength ≈ 62,832, barely changes over 100 positions

This multi-scale representation enables the model to attend to both local and global positional relationships.

### Mathematical Intuition

The encoding can be viewed as a **binary-like position representation** in continuous space:

- Binary representation uses powers of 2: $2^0, 2^1, 2^2, ...$
- Sinusoidal encoding uses continuous wavelengths: $\lambda_0, \lambda_1, \lambda_2, ...$

Just as each bit in binary representation changes at different rates (least significant bit changes every position, most significant bit changes slowly), each dimension in positional encoding changes at different frequencies.

---

## Class Architecture

### Class Declaration

```cpp
class PositionalEncoding {
private:
    Matrix pos_encoding;  // Pre-computed positional encodings [max_len, d_model]
    int max_len;          // Maximum sequence length
    int d_model;          // Embedding dimension

public:
    PositionalEncoding(int max_len, int d_model);
    Matrix forward(const Matrix& input);

    const Matrix& get_encoding() const;
    int get_max_len() const;
    int get_d_model() const;

    std::vector<float> get_position_encoding(int pos) const;
    void print_config(const std::string& name = "PositionalEncoding") const;
    void visualize(int num_positions = 10, int num_dims = 8) const;
};
```

### Member Variables

#### Private Members

- **`pos_encoding`** (Matrix [max_len, d_model]): Pre-computed sinusoidal encodings
- **`max_len`** (int): Maximum sequence length supported
- **`d_model`** (int): Embedding dimension

**Note**: No learnable parameters - the class is entirely deterministic.

### Public Interface

#### Constructor
```cpp
PositionalEncoding(int max_len, int d_model)
```

Pre-computes all positional encodings up to `max_len` positions.

**Parameters**:

- `max_len`: Maximum sequence length (e.g., 512, 1024, 2048)
- `d_model`: Embedding dimension (must match token embeddings)

**Initialization**:

- Allocates matrix of size [max_len, d_model]
- Computes sinusoidal values for all position-dimension pairs
- One-time computation during construction

#### Forward Pass
```cpp
Matrix forward(const Matrix& input)
```

Adds positional encodings to input embeddings.

**Parameters**:

- `input`: Token embeddings [sequence_length, d_model]

**Returns**:

- Matrix [sequence_length, d_model] with positional information added

**Process**:

```text
output[i][j] = input[i][j] + pos_encoding[i][j]
```

#### Accessors
```cpp
const Matrix& get_encoding() const;        // Get full encoding matrix
int get_max_len() const;                   // Get max sequence length
int get_d_model() const;                   // Get embedding dimension
std::vector<float> get_position_encoding(int pos) const;  // Get specific position
```

#### Utilities
```cpp
void print_config(const std::string& name = "PositionalEncoding") const;
void visualize(int num_positions = 10, int num_dims = 8) const;
```

---

## Implementation Details

### Constructor Implementation

```cpp
PositionalEncoding::PositionalEncoding(int max_len, int d_model)
    : pos_encoding(max_len, d_model), max_len(max_len), d_model(d_model) {

    for (int pos = 0; pos < max_len; ++pos) {
        for (int i = 0; i < d_model; ++i) {
            // Compute angle for this position-dimension pair
            float angle = pos / std::pow(10000.0f, (2.0f * (i / 2)) / static_cast<float>(d_model));

            if (i % 2 == 0) {
                // Even indices: sine
                pos_encoding(pos, i) = std::sin(angle);
            } else {
                // Odd indices: cosine
                pos_encoding(pos, i) = std::cos(angle);
            }
        }
    }
}
```

**Key Implementation Details**:

1. **Angle Computation**: `(i / 2)` ensures adjacent sine-cosine pairs use the same wavelength
2. **Type Casting**: `static_cast<float>(d_model)` ensures floating-point division
3. **Pre-computation**: All values computed once, stored for reuse
4. **Memory Allocation**: Matrix automatically initialized with correct dimensions

**Time Complexity**: O(max_len × d_model) - one-time during construction
**Space Complexity**: O(max_len × d_model)

### Forward Pass Implementation

```cpp
Matrix PositionalEncoding::forward(const Matrix& input) {
    Matrix result = input;
    int seq_len = std::min(input.rows, max_len);

    for (int i = 0; i < seq_len; ++i) {
        for (int j = 0; j < input.cols; ++j) {
            result(i, j) += pos_encoding(i, j);
        }
    }

    if (input.rows > max_len) {
        std::cerr << "Warning: Input sequence length exceeds max_len..." << std::endl;
    }

    return result;
}
```

**Key Features**:

1. **Element-wise Addition**: Simple addition of pre-computed encodings
2. **Length Safety**: Uses `std::min` to handle sequences longer than max_len
3. **Warning System**: Alerts user if sequence exceeds maximum length
4. **Non-destructive**: Creates new matrix, preserves original input

**Time Complexity**: O(seq_len × d_model)
**Space Complexity**: O(seq_len × d_model) for result matrix

### Visualization Implementation

```cpp
void PositionalEncoding::visualize(int num_positions, int num_dims) const {
    num_positions = std::min(num_positions, max_len);
    num_dims = std::min(num_dims, d_model);

    // Print header
    std::cout << std::setw(6) << "Pos";
    for (int j = 0; j < num_dims; ++j) {
        std::cout << std::setw(10) << ("Dim" + std::to_string(j));
    }

    // Print values
    for (int i = 0; i < num_positions; ++i) {
        std::cout << std::setw(6) << i;
        for (int j = 0; j < num_dims; ++j) {
            std::cout << std::setw(10) << std::fixed << std::setprecision(4)
                      << pos_encoding(i, j);
        }
        std::cout << std::endl;
    }
}
```

**Purpose**: Debug tool to visualize the sinusoidal pattern and verify correct implementation.

---

## Usage Patterns

### Basic Usage

```cpp
#include "PositionalEncoding.hpp"
#include "Matrix.hpp"

int main() {
    // Create positional encoding for max 512 positions, 256 dimensions
    PositionalEncoding pe(512, 256);

    // Token embeddings from embedding layer
    Matrix token_embeddings(10, 256);  // Sequence of 10 tokens
    // ... fill with actual embeddings ...

    // Add positional information
    Matrix positioned_embeddings = pe.forward(token_embeddings);

    // Now feed to transformer encoder
    // encoder.process(positioned_embeddings);

    return 0;
}
```

### BERT-style Configuration

```cpp
// BERT-base configuration
PositionalEncoding pe_bert(512, 768);  // max_len=512, d_model=768

// BERT-large configuration
PositionalEncoding pe_bert_large(512, 1024);  // max_len=512, d_model=1024
```

### GPT-style Configuration

```cpp
// GPT-2 small
PositionalEncoding pe_gpt2(1024, 768);  // max_len=1024, d_model=768

// GPT-2 large
PositionalEncoding pe_gpt2_large(1024, 1280);  // max_len=1024, d_model=1280
```

### Custom Transformer

```cpp
// Custom configuration for long sequences
PositionalEncoding pe_long(2048, 512);  // Support up to 2048 tokens

Matrix input(1500, 512);  // Actual sequence length: 1500
// ... fill input ...

Matrix output = pe_long.forward(input);  // Works fine (1500 < 2048)
```

### Inspecting Specific Positions

```cpp
PositionalEncoding pe(100, 512);

// Get encoding for position 5
std::vector<float> pos_5_encoding = pe.get_position_encoding(5);

std::cout << "Position 5 encoding (first 4 dims):" << std::endl;
for (int i = 0; i < 4; ++i) {
    std::cout << "  Dim " << i << ": " << pos_5_encoding[i] << std::endl;
}

// Compare adjacent positions
std::vector<float> pos_4 = pe.get_position_encoding(4);
std::vector<float> pos_6 = pe.get_position_encoding(6);

// Positions 4 and 6 are equidistant from position 5
```

### Configuration Inspection

```cpp
PositionalEncoding pe(512, 768);

// Print configuration
pe.print_config("BERT Positional Encoding");

// Output:
// BERT Positional Encoding Configuration:
//   Maximum Sequence Length: 512
//   Embedding Dimension (d_model): 768
//   Encoding Type: Sinusoidal (fixed, not learned)
//   Memory Usage: 1536 KB
```

### Debugging with Visualization

```cpp
PositionalEncoding pe(100, 512);

// Visualize first 10 positions, 8 dimensions
pe.visualize(10, 8);

// Shows the sinusoidal pattern:
// Pos      Dim0      Dim1      Dim2      Dim3 ...
//   0    0.0000    1.0000    0.0000    1.0000 ...
//   1    0.8415    0.5403    0.8219    0.5697 ...
//   2    0.9093   -0.4161    0.9364   -0.3509 ...
//  ...
```

### Integration in Transformer

```cpp
class TransformerEncoder {
private:
    TokenEmbedding token_emb;
    PositionalEncoding pos_enc;
    std::vector<EncoderBlock> layers;

public:
    TransformerEncoder(int vocab_size, int d_model, int max_len)
        : token_emb(vocab_size, d_model),
          pos_enc(max_len, d_model) {
        // Initialize encoder layers...
    }

    Matrix encode(const std::vector<int>& token_ids) {
        // 1. Convert tokens to embeddings
        Matrix embeddings = token_emb.forward(token_ids);

        // 2. Add positional information
        Matrix positioned = pos_enc.forward(embeddings);

        // 3. Process through encoder layers
        Matrix output = positioned;
        for (auto& layer : layers) {
            output = layer.forward(output);
        }

        return output;
    }
};
```

---

## Performance Considerations

### Time Complexity

| Operation | Complexity | Description |
| ----------- | ----------- | ------------- |
| Constructor | O(L × D) | L = max_len, D = d_model |
| Forward Pass | O(S × D) | S = sequence_length |
| get_position_encoding | O(D) | Copy one position's encoding |
| get_encoding | O(1) | Return reference |

### Space Complexity

| Component | Space | Notes |
| ----------- | ------- | ------- |
| pos_encoding Matrix | O(L × D) | Pre-computed encodings |
| Forward Pass Output | O(S × D) | New matrix created |
| **Total** | **O(L × D + S × D)** | Dominated by pre-computed matrix |

### Memory Usage Examples

| Configuration | max_len | d_model | Memory (MB) |
| -------------- | --------- | --------- | ------------- |
| Small | 128 | 256 | 0.125 |
| BERT-base | 512 | 768 | 1.5 |
| GPT-2 | 1024 | 768 | 3.0 |
| Long Context | 2048 | 512 | 4.0 |
| Very Long | 4096 | 1024 | 16.0 |

Formula: `Memory (MB) = (max_len × d_model × 4 bytes) / (1024 × 1024)`

### Optimization Opportunities

#### 1. **Memory Optimization for Inference**

If max_len is very large but typical sequences are short:

```cpp
// Instead of pre-computing all positions
// Compute on-demand for actual sequence length
Matrix forward_dynamic(const Matrix& input) {
    Matrix result = input;
    for (int pos = 0; pos < input.rows; ++pos) {
        for (int i = 0; i < input.cols; ++i) {
            float angle = pos / std::pow(10000.0f, (2.0f * (i / 2)) / d_model);
            result(pos, i) += (i % 2 == 0) ? std::sin(angle) : std::cos(angle);
        }
    }
    return result;
}
```

Trade-off: Saves memory but increases computation time.

#### 2. **SIMD Vectorization**

The element-wise addition in forward pass can be vectorized:

```cpp
#include <immintrin.h>  // For AVX instructions

// Vectorized addition (pseudocode)
for (int i = 0; i < seq_len; ++i) {
    for (int j = 0; j < d_model; j += 8) {
        __m256 input_vec = _mm256_load_ps(&input(i, j));
        __m256 pos_vec = _mm256_load_ps(&pos_encoding(i, j));
        __m256 result_vec = _mm256_add_ps(input_vec, pos_vec);
        _mm256_store_ps(&result(i, j), result_vec);
    }
}
```

#### 3. **In-Place Operation**

For memory-constrained scenarios:

```cpp
void forward_inplace(Matrix& input) {
    int seq_len = std::min(input.rows, max_len);
    for (int i = 0; i < seq_len; ++i) {
        for (int j = 0; j < input.cols; ++j) {
            input(i, j) += pos_encoding(i, j);
        }
    }
}
```

Saves memory by modifying input directly.

#### 4. **Caching Strategy**

For repeated sequences of the same length:

```cpp
class CachedPositionalEncoding {
private:
    std::map<int, Matrix> cache;  // Cache encodings by sequence length

public:
    Matrix forward(const Matrix& input) {
        int seq_len = input.rows;
        if (cache.find(seq_len) == cache.end()) {
            // Compute and cache
            cache[seq_len] = compute_for_length(seq_len);
        }
        return input + cache[seq_len];
    }
};
```

---

## Comparison with Learned Positional Embeddings

### Sinusoidal (Fixed) vs. Learned Positional Encodings

| Aspect | Sinusoidal (Fixed) | Learned |
| -------- | ------------------- | --------- |
| **Parameters** | None | O(max_len × d_model) |
| **Training** | Not needed | Requires training |
| **Generalization** | Handles any length | Limited to max_len |
| **Memory** | Same as learned | Same as sinusoidal |
| **Initialization** | Deterministic | Random or zeros |
| **Relative Position** | Built-in property | Must be learned |
| **Interpretability** | Clear mathematical meaning | Opaque learned values |

### When to Use Sinusoidal

✅ **Limited training data** - No parameters to overfit
✅ **Variable sequence lengths** - Generalizes beyond training
✅ **Transfer learning** - Position meaning is universal
✅ **Long sequences** - Can extend beyond max_len gracefully
✅ **Interpretability** - Clear understanding of position encoding

### When to Use Learned

✅ **Large training datasets** - Can learn task-specific patterns
✅ **Fixed sequence lengths** - All sequences same length
✅ **Domain-specific tasks** - Position meaning varies by task
✅ **Fine-tuning flexibility** - Can adapt to downstream tasks

### Empirical Comparisons

From "Attention is All You Need":
> "We chose the sinusoidal version because it may allow the model to extrapolate to sequence lengths longer than the ones encountered during training."

From BERT paper:
> "We use learned positional embeddings with supported sequence lengths up to 512 tokens."

**Observation**: Both approaches work well in practice. BERT uses learned embeddings but is limited to 512 tokens. T5 and many modern models return to sinusoidal for better generalization.

---

## Common Issues and Solutions

### Issue 1: Sequence Length Exceeds max_len

**Symptom**: Warning message during forward pass

```text
Warning: Input sequence length (600) exceeds max_len (512).
```

**Cause**: Input sequence longer than initialized max_len

**Solutions**:

```cpp
// Solution 1: Increase max_len during initialization
PositionalEncoding pe(2048, 768);  // Support longer sequences

// Solution 2: Truncate input sequences
Matrix truncated_input = input.slice(0, pe.get_max_len(), 0, input.cols);
Matrix output = pe.forward(truncated_input);

// Solution 3: Use dynamic computation (if implemented)
Matrix output = pe.forward_dynamic(input);  // Computes on-the-fly
```

### Issue 2: Dimension Mismatch

**Symptom**: Crash or incorrect results

```text
Assertion failed: input.cols == d_model
```

**Cause**: Input embedding dimension doesn't match positional encoding dimension

**Solutions**:

```cpp
// Verify dimensions match
TokenEmbedding token_emb(vocab_size, 512);
PositionalEncoding pos_enc(max_len, 512);  // Same dimension!

// Check before forward pass
assert(input.cols == pos_enc.get_d_model());

// Or add dimension validation in forward
if (input.cols != d_model) {
    throw std::invalid_argument("Input dimension mismatch");
}
```

### Issue 3: Encoding Values Dominating Input

**Symptom**: Model doesn't learn properly, embeddings overshadowed

**Cause**: Input embeddings have very small magnitudes compared to positional encodings ([-1, 1])

**Solutions**:

```cpp
// Solution 1: Scale positional encodings
Matrix forward_scaled(const Matrix& input, float scale = 0.1f) {
    Matrix result = input;
    for (int i = 0; i < seq_len; ++i) {
        for (int j = 0; j < d_model; ++j) {
            result(i, j) += scale * pos_encoding(i, j);
        }
    }
    return result;
}

// Solution 2: Scale input embeddings during initialization
// Use larger initialization for token embeddings
float emb_scale = std::sqrt(d_model);  // Common practice
embeddings = embeddings * emb_scale;

// Solution 3: Add as a separate channel (concatenation)
// Instead of addition, concatenate [embeddings, pos_encodings]
```

### Issue 4: Numerical Precision Issues

**Symptom**: Very small or very large angle values cause precision loss

**Cause**: For large positions or extreme dimension indices, angle computation may lose precision

**Solutions**:

```cpp
// Use double precision for angle computation
double angle = static_cast<double>(pos) /
               std::pow(10000.0, (2.0 * (i / 2)) / static_cast<double>(d_model));
pos_encoding(pos, i) = static_cast<float>(std::sin(angle));

// Or use log-space computation to avoid overflow
float log_angle = std::log(pos) -
                  (2.0f * (i / 2) / d_model) * std::log(10000.0f);
float angle = std::exp(log_angle);
```

### Issue 5: Memory Consumption Too High

**Symptom**: Out of memory for very long max_len

**Cause**: Large matrix allocation (e.g., max_len=100000, d_model=1024 ≈ 400MB)

**Solutions**:

```cpp
// Solution 1: Compute on-demand instead of pre-computing
// (Shown in optimization section)

// Solution 2: Use half-precision floats (if supported)
#include <half.hpp>
Matrix<half_float::half> pos_encoding;  // 16-bit floats, half memory

// Solution 3: Store only up to typical max length
// Compute additional positions if needed
const int PRECOMPUTE_LEN = 1024;
if (pos < PRECOMPUTE_LEN) {
    return precomputed[pos];
} else {
    return compute_position(pos);
}
```

---

## Integration Examples

### Example 1: Complete Transformer Encoder

```cpp
#include "PositionalEncoding.hpp"
#include "TokenEmbedding.hpp"
#include "EncoderBlock.hpp"

class SimpleTransformer {
private:
    TokenEmbedding token_emb;
    PositionalEncoding pos_enc;
    std::vector<EncoderBlock> layers;
    LayerNorm final_norm;

public:
    SimpleTransformer(int vocab_size, int d_model, int num_layers, int max_len)
        : token_emb(vocab_size, d_model),
          pos_enc(max_len, d_model),
          final_norm(d_model) {

        for (int i = 0; i < num_layers; ++i) {
            layers.emplace_back(d_model, 8, 2048);  // 8 heads, 2048 ff_dim
        }
    }

    Matrix forward(const std::vector<int>& token_ids) {
        // Token embedding
        Matrix x = token_emb.forward(token_ids);

        // Add positional encoding
        x = pos_enc.forward(x);

        // Pass through encoder layers
        for (auto& layer : layers) {
            x = layer.forward(x);
        }

        // Final layer norm
        x = final_norm.forward(x);

        return x;
    }
};
```

### Example 2: BERT-style Pre-training

```cpp
class BERTEncoder {
private:
    TokenEmbedding token_emb;
    TokenEmbedding segment_emb;  // Segment/sentence embeddings
    PositionalEncoding pos_enc;

public:
    BERTEncoder(int vocab_size, int d_model)
        : token_emb(vocab_size, d_model),
          segment_emb(2, d_model),  // Only 2 segments
          pos_enc(512, d_model) {}

    Matrix forward(const std::vector<int>& tokens,
                   const std::vector<int>& segments) {
        // Three types of embeddings
        Matrix tok_emb = token_emb.forward(tokens);
        Matrix seg_emb = segment_emb.forward(segments);
        Matrix pos_emb = pos_enc.forward(tok_emb);

        // Sum all three embeddings
        Matrix combined(tok_emb.rows, tok_emb.cols);
        for (int i = 0; i < tok_emb.rows; ++i) {
            for (int j = 0; j < tok_emb.cols; ++j) {
                combined(i, j) = tok_emb(i, j) + seg_emb(i, j) +
                                pos_emb(i, j) - tok_emb(i, j);  // pos already added
            }
        }

        return combined;
    }
};
```

### Example 3: GPT-style Decoder with Masked Attention

```cpp
class GPTDecoder {
private:
    TokenEmbedding token_emb;
    PositionalEncoding pos_enc;
    std::vector<DecoderBlock> layers;

    Matrix create_causal_mask(int seq_len) {
        // Lower triangular mask for autoregressive generation
        Matrix mask(seq_len, seq_len);
        for (int i = 0; i < seq_len; ++i) {
            for (int j = 0; j < seq_len; ++j) {
                mask(i, j) = (j <= i) ? 1.0f : 0.0f;
            }
        }
        return mask;
    }

public:
    GPTDecoder(int vocab_size, int d_model, int num_layers)
        : token_emb(vocab_size, d_model),
          pos_enc(1024, d_model) {

        for (int i = 0; i < num_layers; ++i) {
            layers.emplace_back(d_model);
        }
    }

    Matrix forward(const std::vector<int>& tokens) {
        // Embeddings + positions
        Matrix x = token_emb.forward(tokens);
        x = pos_enc.forward(x);

        // Create causal mask
        Matrix mask = create_causal_mask(tokens.size());

        // Pass through decoder layers with mask
        for (auto& layer : layers) {
            x = layer.forward(x, &mask);
        }

        return x;
    }
};
```

### Example 4: Relative Position Attention

```cpp
// Demonstrate how sinusoidal encodings enable relative position
class RelativePositionExample {
public:
    void demonstrate() {
        PositionalEncoding pe(100, 512);

        // Get encodings for positions
        std::vector<float> pos_10 = pe.get_position_encoding(10);
        std::vector<float> pos_20 = pe.get_position_encoding(20);
        std::vector<float> pos_30 = pe.get_position_encoding(30);

        // Compute similarity (dot product) between positions
        float sim_10_20 = dot_product(pos_10, pos_20);
        float sim_20_30 = dot_product(pos_20, pos_30);

        // Due to sinusoidal properties, similar relative distances
        // have similar dot products
        std::cout << "Similarity(10, 20): " << sim_10_20 << std::endl;
        std::cout << "Similarity(20, 30): " << sim_20_30 << std::endl;
        // These should be very close!
    }

    float dot_product(const std::vector<float>& a, const std::vector<float>& b) {
        float sum = 0.0f;
        for (size_t i = 0; i < a.size(); ++i) {
            sum += a[i] * b[i];
        }
        return sum;
    }
};
```

### Example 5: Dynamic Sequence Length Handling

```cpp
class AdaptivePositionalEncoding {
private:
    PositionalEncoding base_pe;
    int base_max_len;

public:
    AdaptivePositionalEncoding(int initial_max_len, int d_model)
        : base_pe(initial_max_len, d_model),
          base_max_len(initial_max_len) {}

    Matrix forward(const Matrix& input) {
        if (input.rows <= base_max_len) {
            // Fast path: use pre-computed encodings
            return base_pe.forward(input);
        } else {
            // Slow path: compute additional positions on-the-fly
            Matrix result = input;
            int d_model = base_pe.get_d_model();

            // Use pre-computed for first base_max_len positions
            for (int i = 0; i < base_max_len; ++i) {
                std::vector<float> enc = base_pe.get_position_encoding(i);
                for (int j = 0; j < d_model; ++j) {
                    result(i, j) += enc[j];
                }
            }

            // Compute remaining positions dynamically
            for (int pos = base_max_len; pos < input.rows; ++pos) {
                for (int i = 0; i < d_model; ++i) {
                    float angle = pos / std::pow(10000.0f,
                                  (2.0f * (i / 2)) / static_cast<float>(d_model));
                    result(pos, i) += (i % 2 == 0) ? std::sin(angle) : std::cos(angle);
                }
            }

            return result;
        }
    }
};
```

---

## Testing and Validation

### Unit Test Example

```cpp
#include <gtest/gtest.h>
#include "PositionalEncoding.hpp"

TEST(PositionalEncodingTest, EncodingRange) {
    PositionalEncoding pe(100, 128);

    const Matrix& encoding = pe.get_encoding();

    // All values should be in [-1, 1]
    for (int i = 0; i < 100; ++i) {
        for (int j = 0; j < 128; ++j) {
            EXPECT_GE(encoding(i, j), -1.0f);
            EXPECT_LE(encoding(i, j), 1.0f);
        }
    }
}

TEST(PositionalEncodingTest, Position0IsZeroForEvenDims) {
    PositionalEncoding pe(10, 64);

    std::vector<float> pos_0 = pe.get_position_encoding(0);

    // At position 0, sine of 0 is 0 (even dims)
    for (int i = 0; i < 64; i += 2) {
        EXPECT_NEAR(pos_0[i], 0.0f, 1e-5f);
    }

    // At position 0, cosine of 0 is 1 (odd dims)
    for (int i = 1; i < 64; i += 2) {
        EXPECT_NEAR(pos_0[i], 1.0f, 1e-5f);
    }
}

TEST(PositionalEncodingTest, RelativePositionProperty) {
    PositionalEncoding pe(100, 128);

    // Positions with same relative offset should have similar patterns
    auto pos_5 = pe.get_position_encoding(5);
    auto pos_15 = pe.get_position_encoding(15);
    auto pos_25 = pe.get_position_encoding(25);

    // Difference between (5,15) and (15,25) should be similar
    // This tests the linear offset property
    for (int i = 0; i < 128; ++i) {
        float diff1 = pos_15[i] - pos_5[i];
        float diff2 = pos_25[i] - pos_15[i];
        // Allow some numerical error
        EXPECT_NEAR(diff1, diff2, 0.1f);
    }
}

TEST(PositionalEncodingTest, ForwardAddsCorrectly) {
    PositionalEncoding pe(10, 4);

    Matrix input(3, 4);
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 4; ++j) {
            input(i, j) = 1.0f;
        }
    }

    Matrix output = pe.forward(input);

    // Output should be input + encoding
    for (int i = 0; i < 3; ++i) {
        auto pos_enc = pe.get_position_encoding(i);
        for (int j = 0; j < 4; ++j) {
            EXPECT_FLOAT_EQ(output(i, j), 1.0f + pos_enc[j]);
        }
    }
}
```

---

## References

### Academic Papers

1. **Original Paper - Attention is All You Need**:
   - Vaswani, A., Shazeer, N., Parmar, N., et al. (2017)
   - "Attention is All You Need"
   - NeurIPS 2017
   - https://arxiv.org/abs/1706.03762
   - **Key contribution**: Introduced sinusoidal positional encodings for Transformers

2. **BERT - Learned Positional Embeddings**:
   - Devlin, J., Chang, M. W., Lee, K., & Toutanova, K. (2018)
   - "BERT: Pre-training of Deep Bidirectional Transformers"
   - https://arxiv.org/abs/1810.04805
   - Uses learned positional embeddings instead of sinusoidal

3. **Position Information in Transformers**:
   - Ke, G., He, D., & Liu, T. Y. (2020)
   - "Rethinking Positional Encoding in Language Pre-training"
   - ICLR 2021
   - https://arxiv.org/abs/2006.15595

4. **Relative Position Representations**:
   - Shaw, P., Uszkoreit, J., & Vaswani, A. (2018)
   - "Self-Attention with Relative Position Representations"
   - NAACL 2018
   - https://arxiv.org/abs/1803.02155

### Implementation References

1. **Original Tensor2Tensor Implementation**:
   - https://github.com/tensorflow/tensor2tensor
   - Reference implementation by authors

2. **PyTorch Transformer**:
   - https://pytorch.org/docs/stable/generated/torch.nn.Transformer.html
   - Uses sinusoidal positional encoding

3. **Hugging Face Transformers**:
   - https://github.com/huggingface/transformers
   - Various positional encoding implementations

### Mathematical Background

1. **Fourier Analysis and Signal Processing**:
   - Understanding wavelengths and frequencies in positional encoding

2. **Trigonometric Identities**:
   - Sum and difference formulas enabling relative position learning

### Best Practices

1. **Choosing max_len**:
   - Set to maximum expected sequence length
   - Add buffer (e.g., 10-20%) for safety
   - BERT: 512, GPT-2: 1024, GPT-3: 2048

2. **Scaling Considerations**:
   - Original paper doesn't scale positional encodings
   - Some implementations scale embeddings by √d_model
   - Experiment to find best balance

3. **Alternative Approaches**:
   - Rotary Position Embedding (RoPE) - used in recent models
   - ALiBi (Attention with Linear Biases)
   - T5-style relative position biases

---

## Summary

The `PositionalEncoding` class provides a deterministic, parameter-free method for adding positional information to Transformer models through sinusoidal functions.

**Key Advantages**:
✅ **Zero learnable parameters** - Reduces model complexity
✅ **Length generalization** - Works with sequences longer than training data
✅ **Relative position learning** - Sinusoidal properties enable relative attention
✅ **Efficient** - Pre-computed once, reused for all sequences
✅ **Interpretable** - Clear mathematical meaning
✅ **Universal** - Position meaning is task-independent

**Core Formula**:

```text
PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
```

**When to Use**:

- Transformer-based architectures (BERT, GPT, T5, etc.)
- Variable-length sequences
- Transfer learning scenarios
- Limited training data
- Long-context applications

**Integration Pattern**:

```cpp
// 1. Initialize with max_len and d_model
PositionalEncoding pos_enc(max_len, d_model);

// 2. Get token embeddings
Matrix embeddings = token_embedding.forward(tokens);

// 3. Add positional information
Matrix positioned = pos_enc.forward(embeddings);

// 4. Feed to transformer
Matrix output = transformer.forward(positioned);
```

---

**Document Version**: 1.0
**Last Updated**: January 17, 2026
**Implementation Files**: `src/PositionalEncoding.hpp`, `src/PositionalEncoding.cpp`
**Total Lines of Documentation**: 1400+
**Coverage**: Complete class documentation with theory, implementation, usage, and examples
