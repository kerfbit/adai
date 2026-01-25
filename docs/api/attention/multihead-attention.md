# MultiHeadAttention Class - Technical Context Documentation

## Overview

The `MultiHeadAttention` class implements the multi-head self-attention mechanism introduced in "Attention is All You Need" (Vaswani et al., 2017). This is the core component of transformer architectures, enabling models to attend to different representation subspaces and capture diverse relationships in the input sequence.

**Files:**
- `src/MultiHeadAttention.hpp` - Header file with class declaration and interface
- `src/MultiHeadAttention.cpp` - Implementation file with all method definitions
- `src/MultiHeadAttentionExample.cpp` - Standalone example demonstrating usage

**Purpose:** Provide a fully-featured multi-head attention mechanism that allows neural networks to process sequences by computing weighted relationships between all positions, enabling the model to focus on relevant parts of the input while learning different types of relationships through multiple attention heads.

---

## Table of Contents
1. [Mathematical Foundation](#mathematical-foundation)
2. [Class Architecture](#class-architecture)
3. [Implementation Details](#implementation-details)
4. [Forward Pass](#forward-pass)
5. [Backward Pass](#backward-pass)
6. [Usage Patterns](#usage-patterns)
7. [Attention Masking](#attention-masking)
8. [Performance Considerations](#performance-considerations)
9. [Integration with Transformers](#integration-with-transformers)
10. [Debugging and Monitoring](#debugging-and-monitoring)

---

## Mathematical Foundation

### Core Concept: Attention Mechanism

The attention mechanism computes a weighted sum of values based on the similarity between queries and keys.

#### Scaled Dot-Product Attention

For a single head:

```
Attention(Q, K, V) = softmax(QK^T / √d_k) V
```

Where:
- **Q** (Queries): Matrix of shape `[seq_len, d_k]` - represents what we're looking for
- **K** (Keys): Matrix of shape `[seq_len, d_k]` - represents what each position offers
- **V** (Values): Matrix of shape `[seq_len, d_k]` - represents the actual content to aggregate
- **d_k**: Dimension per head (d_model / num_heads)
- **√d_k**: Scaling factor to prevent softmax saturation

#### Multi-Head Attention

Instead of computing a single attention function, multi-head attention projects Q, K, V into multiple subspaces and computes attention in parallel:

```
MultiHead(Q, K, V) = Concat(head₁, ..., head_h) W_o
```

Where each head is computed as:
```
head_i = Attention(Q W_q^i, K W_k^i, V W_v^i)
```

**Parameters:**
- **W_q**: Query projection matrix `[d_model, d_model]`
- **W_k**: Key projection matrix `[d_model, d_model]`
- **W_v**: Value projection matrix `[d_model, d_model]`
- **W_o**: Output projection matrix `[d_model, d_model]`

### Why Multi-Head?

Multiple heads allow the model to jointly attend to information from different representation subspaces:

1. **Diverse Relationships**: Different heads can learn different types of relationships (syntactic, semantic, positional, etc.)
2. **Robustness**: Multiple heads provide redundancy and robustness to individual head failures
3. **Rich Representations**: Each head captures different aspects of the input
4. **Parallel Processing**: All heads computed independently (parallelizable)

### Complexity Analysis

**Time Complexity:**
- Attention computation: O(seq_len² × d_model)
- Linear projections: O(seq_len × d_model²)
- Total: **O(seq_len² × d_model + seq_len × d_model²)**

**Space Complexity:**
- Weight matrices: O(4 × d_model²)
- Attention weights: O(seq_len²)
- Cached activations: O(seq_len × d_model)
- Total: **O(d_model² + seq_len² + seq_len × d_model)**

---

## Class Architecture

### Private Members

```cpp
// Model dimensions
int d_model;        // Model dimension (embedding size)
int num_heads;      // Number of attention heads
int d_k;            // Dimension per head (d_model / num_heads)

// Learnable weight matrices [d_model, d_model]
Matrix W_q;         // Query projection
Matrix W_k;         // Key projection
Matrix W_v;         // Value projection
Matrix W_o;         // Output projection

// Gradient matrices [d_model, d_model]
Matrix W_q_grad;    // Accumulated gradients for W_q
Matrix W_k_grad;    // Accumulated gradients for W_k
Matrix W_v_grad;    // Accumulated gradients for W_v
Matrix W_o_grad;    // Accumulated gradients for W_o

// Optimizer (optional, for advanced optimization algorithms)
Optimizer* optimizer;  // Pointer to optimizer (nullptr = use simple gradient descent)

// Cached values for backward pass
Matrix cached_input;              // Input to forward pass
Matrix cached_Q;                  // Projected queries
Matrix cached_K;                  // Projected keys
Matrix cached_V;                  // Projected values
Matrix cached_attention_weights;  // Softmax attention weights
Matrix cached_attention_output;   // Output after applying attention
Matrix cached_scores;             // Pre-softmax attention scores
```

### Public Members

```cpp
float learning_rate;  // Learning rate for weight updates (default: 0.001)
```

### Initialization

**Xavier/He Initialization:**
```cpp
scale = √(2 / d_model)
```

All weight matrices (W_q, W_k, W_v, W_o) are initialized with random values scaled by this factor to ensure stable gradients during early training.

---

## Implementation Details

### Constructor

```cpp
MultiHeadAttention(int d_model, int num_heads)
```

**Parameters:**
- `d_model`: Model dimension (must be divisible by num_heads)
- `num_heads`: Number of attention heads

**Validation:**
- Throws `std::invalid_argument` if d_model % num_heads ≠ 0

**Initialization Steps:**
1. Validates dimension compatibility
2. Computes d_k = d_model / num_heads
3. Initializes weight matrices with Xavier scaling
4. Zeros out gradient matrices
5. Sets default learning rate to 0.001
6. Initializes optimizer pointer to nullptr (no optimizer by default)

**Example:**
```cpp
// For 512-dimensional embeddings with 8 heads
MultiHeadAttention mha(512, 8);  // d_k = 64 per head
mha.learning_rate = 0.0001f;     // Adjust learning rate
```

**Common Configurations:**
- **Small models**: d_model=256, num_heads=4
- **Base models**: d_model=512, num_heads=8
- **Large models**: d_model=1024, num_heads=16
- **GPT-3 scale**: d_model=12288, num_heads=96

---

## Forward Pass

### Method Signature

```cpp
Matrix forward(const Matrix& input, const Matrix* mask = nullptr)
```

**Parameters:**
- `input`: Input matrix `[seq_len, d_model]`
- `mask`: Optional attention mask `[seq_len, seq_len]` (default: nullptr)

**Returns:** Attention output `[seq_len, d_model]`

### Algorithm Steps

#### 1. Input Validation

```cpp
if (input.cols != d_model) {
    throw std::invalid_argument("Input dimension must match d_model");
}
```

#### 2. Linear Projections

Compute queries, keys, and values:
```cpp
Q = input × W_q   // [seq_len, d_model]
K = input × W_k   // [seq_len, d_model]
V = input × W_v   // [seq_len, d_model]
```

#### 3. Compute Attention Scores

```cpp
scores = Q × K^T  // [seq_len, seq_len]
scores = scores / √d_k  // Scale to prevent gradient vanishing
```

**Why scaling?** Without scaling, dot products grow with √d_k, pushing softmax into regions with extremely small gradients.

#### 4. Apply Mask (Optional)

```cpp
if (mask != nullptr) {
    for each position (i, j):
        if mask(i, j) == 0:
            scores(i, j) = -1e9  // Large negative value
}
```

Masked positions get extremely negative scores, resulting in ~0 attention weight after softmax.

#### 5. Apply Softmax

```cpp
attention_weights = softmax(scores)  // [seq_len, seq_len]
```

Each row sums to 1.0, representing a probability distribution over all positions.

#### 6. Apply Attention to Values

```cpp
attention_output = attention_weights × V  // [seq_len, d_model]
```

Weighted sum of values based on attention weights.

#### 7. Output Projection

```cpp
output = attention_output × W_o  // [seq_len, d_model]
```

Final linear transformation to integrate information from all heads.

### Caching

The forward pass caches intermediate values for backward pass:
- `cached_input`: Original input
- `cached_Q, cached_K, cached_V`: Projected queries, keys, values
- `cached_scores`: Pre-softmax attention scores
- `cached_attention_weights`: Softmax attention weights
- `cached_attention_output`: Output before final projection

---

## Forward Pass with KV Cache ✨ NEW

### Method Signature

```cpp
Matrix forward_with_cache(const Matrix& input,
                         const Matrix* mask = nullptr,
                         KVCache* kv_cache = nullptr,
                         bool use_cache = true)
```

**Purpose**: Optimized forward pass using KV cache for autoregressive generation

**Parameters:**
- `input`: New token embeddings [num_new_tokens, d_model] (typically 1 during generation)
- `mask`: Optional attention mask [num_new_tokens, total_seq_len]
- `kv_cache`: Pointer to KVCache structure (nullptr = no caching)
- `use_cache`: Whether to update cache with new K/V pairs (default: true)

**Returns**: Attention output [num_new_tokens, d_model]

**Performance**: ~2-3x speedup for long sequences by avoiding redundant K/V computation

### How It Works

**KV Caching Concept**:
In autoregressive generation (e.g., text generation), we generate one token at a time. Without caching, we recompute K and V for all previous tokens every step. With caching, we:
1. Compute K/V only for the new token
2. Store ("cache") these K/V pairs
3. Reuse cached K/V from previous steps
4. Attention computed over all cached + new K/V pairs

**Cache Behavior**:

1. **First Call (Empty Cache)**:
   ```cpp
   // Input: [1, d_model] (first token)
   Q_new = input * W_q    // [1, d_model]
   K_new = input * W_k    // [1, d_model]
   V_new = input * W_v    // [1, d_model]
   
   cache.append(K_new, V_new)  // Cache now has 1 token
   K_full = cache.get_keys()    // [1, d_model]
   V_full = cache.get_values()  // [1, d_model]
   
   // Compute attention
   scores = Q_new * K_full.transpose()  // [1, 1]
   attention_weights = softmax(scores / sqrt(d_k))
   output = attention_weights * V_full * W_o
   ```

2. **Second Call (Cache Has 1 Token)**:
   ```cpp
   // Input: [1, d_model] (second token)
   Q_new = input * W_q    // [1, d_model] - only for new token
   K_new = input * W_k    // [1, d_model] - only for new token
   V_new = input * W_v    // [1, d_model] - only for new token
   
   cache.append(K_new, V_new)  // Cache now has 2 tokens
   K_full = cache.get_keys()    // [2, d_model] - reuses previous!
   V_full = cache.get_values()  // [2, d_model] - reuses previous!
   
   // Compute attention over ALL tokens (cached + new)
   scores = Q_new * K_full.transpose()  // [1, 2]
   attention_weights = softmax(scores / sqrt(d_k))
   output = attention_weights * V_full * W_o
   ```

3. **Nth Call (Cache Has N-1 Tokens)**:
   ```cpp
   // Only compute K/V for new token, reuse cached K/V from previous N-1 tokens
   K_full = [cached_K[0:N-1]; K_new]  // [N, d_model]
   V_full = [cached_V[0:N-1]; V_new]  // [N, d_model]
   ```

### Algorithm Steps

#### 1. Fallback Check

```cpp
if (!use_cache || kv_cache == nullptr) {
    return forward(input, mask);  // Use regular forward
}
```

#### 2. Compute Q, K, V for New Tokens Only

```cpp
Q_new = input * W_q   // [num_new_tokens, d_model]
K_new = input * W_k   // [num_new_tokens, d_model]
V_new = input * W_v   // [num_new_tokens, d_model]
```

**Key Insight**: We only compute K/V for new tokens, not for all previous tokens.

#### 3. Update Cache

```cpp
kv_cache->append(K_new, V_new);  // Append new K/V to cache
```

Cache grows: [1, d_model] → [2, d_model] → ... → [seq_len, d_model]

#### 4. Get Full K, V from Cache

```cpp
const Matrix& K_full = kv_cache->get_keys();    // [total_seq_len, d_model]
const Matrix& V_full = kv_cache->get_values();  // [total_seq_len, d_model]
```

Includes all previous + new K/V pairs.

#### 5. Compute Attention Scores

```cpp
scores = Q_new * K_full.transpose()  // [num_new_tokens, total_seq_len]
scores = scores / sqrt(d_k)          // Scale
```

**Note**: Attention computed over ALL tokens (new token attends to all previous + itself).

#### 6. Apply Mask (Adapted for Cache)

```cpp
if (mask != nullptr) {
    // Mask shape: [num_new_tokens, total_seq_len]
    // New tokens can attend to positions <= current position
    for (i, j):
        if mask(i, j) == 0:
            scores(i, j) = -1e9
}
```

**Mask Adaptation**: Mask now has shape [num_new_tokens, total_seq_len] instead of [seq_len, seq_len].

#### 7. Softmax and Apply to Values

```cpp
attention_weights = softmax(scores)  // [num_new_tokens, total_seq_len]
attention_output = attention_weights * V_full  // [num_new_tokens, d_model]
```

#### 8. Output Projection

```cpp
output = attention_output * W_o  // [num_new_tokens, d_model]
```

### Performance Analysis

**Without Cache (Inefficient)**:
```
Step 1: Compute K/V for token 0 (1 computation)
Step 2: Compute K/V for tokens 0, 1 (2 computations) 
Step 3: Compute K/V for tokens 0, 1, 2 (3 computations)
...
Step N: Compute K/V for tokens 0...N-1 (N computations)

Total: 1 + 2 + 3 + ... + N = N(N+1)/2 computations
For N=50: 1,275 K/V computations
Complexity: O(N²)
```

**With Cache (Efficient)**:
```
Step 1: Compute K/V for token 0, cache it (1 computation)
Step 2: Compute K/V for token 1, cache it, reuse cached K/V[0] (1 computation)
Step 3: Compute K/V for token 2, cache it, reuse cached K/V[0:1] (1 computation)
...
Step N: Compute K/V for token N-1, cache it, reuse cached K/V[0:N-2] (1 computation)

Total: N computations
For N=50: 50 K/V computations
Complexity: O(N)
Speedup: 1,275 / 50 = 25.5x
```

**Speedup Formula**: For N tokens, speedup ≈ N/2

### Typical Usage Pattern

```cpp
// Initialize attention and cache
MultiHeadAttention mha(512, 8);
KVCache kv_cache;

// Generate tokens autoregressively
std::vector<Matrix> generated;
Matrix current_token_emb = start_token_embedding;  // [1, 512]

for (int i = 0; i < max_gen_length; ++i) {
    // Create causal mask for current position
    int current_pos = i;
    int total_len = current_pos + 1;
    Matrix causal_mask(1, total_len);
    for (int j = 0; j < total_len; ++j) {
        causal_mask(0, j) = (j <= current_pos) ? 1.0f : 0.0f;
    }
    
    // Forward with cache (much faster!)
    Matrix attn_output = mha.forward_with_cache(
        current_token_emb,  // [1, 512] - only new token
        &causal_mask,       // [1, total_len]
        &kv_cache,          // Growing cache
        true                // Update cache
    );
    
    // Project to vocabulary and sample next token
    Matrix logits = lm_head.forward(attn_output);
    int next_token = sample_token(logits);
    current_token_emb = token_embedding(next_token);
    
    generated.push_back(current_token_emb);
    if (next_token == EOS_TOKEN) break;
}

// Clear cache for next sequence
kv_cache.clear();
```

### Performance Comparison Example

```cpp
// Without cache (slow - recomputes everything)
auto start = std::chrono::high_resolution_clock::now();
for (int i = 0; i < 50; ++i) {
    Matrix all_tokens_so_far(i + 1, 512);  // Grows each iteration
    Matrix output = mha.forward(all_tokens_so_far, &mask);
}
auto end = std::chrono::high_resolution_clock::now();
float time_no_cache = duration(end - start).count();
// Result: ~1,275 K/V computations

// With cache (fast - only new token)
KVCache cache;
start = std::chrono::high_resolution_clock::now();
for (int i = 0; i < 50; ++i) {
    Matrix new_token(1, 512);  // Always 1 token
    Matrix output = mha.forward_with_cache(new_token, &adapted_mask, &cache, true);
}
end = std::chrono::high_resolution_clock::now();
float time_with_cache = duration(end - start).count();
// Result: 50 K/V computations

float speedup = time_no_cache / time_with_cache;
// Expected: ~25x speedup for 50 tokens
```

### Key Differences from Regular Forward

| Aspect | `forward()` | `forward_with_cache()` |
|--------|-------------|------------------------|
| Input Size | Full sequence | Only new tokens |
| K/V Computation | All tokens | Only new tokens |
| K/V Source | Computed | Cached + new |
| Mask Shape | [seq_len, seq_len] | [num_new, total_len] |
| Complexity | O(seq_len²) | O(seq_len) per token |
| Use Case | Training, first pass | Inference, generation |
| Memory | Lower | Higher (cache) |
| Speed | Baseline | 2-3x faster |

### Important Notes

1. **Cache Management**:
   - Call `kv_cache.clear()` between sequences
   - Ensure cache capacity >= max_seq_length
   - Cache grows incrementally during generation

2. **Mask Adaptation**:
   - Regular forward: mask is [seq_len, seq_len]
   - Cached forward: mask is [num_new_tokens, total_seq_len]
   - Mask shape changes as cache grows

3. **Training vs Inference**:
   - Training: Use regular `forward()` (full sequences)
   - Inference: Use `forward_with_cache()` (incremental)

4. **Memory Trade-off**:
   - Cache storage: O(max_seq_len × d_model) per cache
   - Typical: ~1-2 MB for 512-dim, 100-token sequence
   - Worth it for generation >10 tokens

5. **Fallback Behavior**:
   - If `use_cache=false` or `kv_cache=nullptr`, falls back to regular `forward()`
   - Ensures backward compatibility

6. **Integration with Transformers**:
   - Used in DecoderBlock for self-attention caching
   - Each layer in decoder stack has its own KV cache
   - See [DecoderBlock](../transformer/decoder-block.md) for multi-layer usage

### When to Use

✅ **Use `forward_with_cache()` for**:
- Autoregressive text generation
- Beam search decoding  
- Sampling-based generation
- Long sequence generation (>10 tokens)
- Real-time chatbot responses

❌ **Use regular `forward()` for**:
- Training (process full batches)
- Single-pass inference
- Bi-directional attention
- Parallel sequence processing

---

## Backward Pass

### Method Signature

```cpp
Matrix backward(const Matrix& grad_output)
```

**Parameters:**
- `grad_output`: Gradient from upstream `[seq_len, d_model]`

**Returns:** Gradient w.r.t. input `[seq_len, d_model]`

**Side Effects:** Accumulates gradients in W_q_grad, W_k_grad, W_v_grad, W_o_grad

### Algorithm Steps

The backward pass implements the chain rule through the multi-head attention computation.

#### 1. Gradient w.r.t. W_o

```cpp
∂L/∂W_o = attention_output^T × grad_output
```

#### 2. Gradient w.r.t. Attention Output

```cpp
∂L/∂(attention_output) = grad_output × W_o^T
```

#### 3. Gradient w.r.t. V

```cpp
∂L/∂V = attention_weights^T × ∂L/∂(attention_output)
```

#### 4. Gradient w.r.t. Attention Weights

```cpp
∂L/∂(attention_weights) = ∂L/∂(attention_output) × V^T
```

#### 5. Gradient Through Softmax

This is the most complex step. For softmax, the Jacobian is:

```
∂softmax_i/∂x_j = softmax_i × (δ_ij - softmax_j)
```

Implemented as:
```cpp
for each row i:
    sum = Σ(attention_weights[i,k] × grad_attn_weights[i,k])
    for each col j:
        grad_scores[i,j] = attention_weights[i,j] × (grad_attn_weights[i,j] - sum)
```

#### 6. Scale Gradient

```cpp
∂L/∂scores = ∂L/∂(softmax_output) / √d_k
```

#### 7. Gradients w.r.t. Q and K

```cpp
∂L/∂Q = ∂L/∂scores × K
∂L/∂K = (∂L/∂scores)^T × Q
```

#### 8. Gradients w.r.t. Weight Matrices

```cpp
∂L/∂W_q = input^T × ∂L/∂Q
∂L/∂W_k = input^T × ∂L/∂K
∂L/∂W_v = input^T × ∂L/∂V
```

#### 9. Gradient w.r.t. Input

Sum gradients from all three projection paths:
```cpp
∂L/∂input = ∂L/∂Q × W_q^T + ∂L/∂K × W_k^T + ∂L/∂V × W_v^T
```

### Weight Update

```cpp
void update_weights()
```

Applies accumulated gradients using Matrix::apply_gradients():
```cpp
W_q.apply_gradients(W_q_grad, learning_rate);
W_k.apply_gradients(W_k_grad, learning_rate);
W_v.apply_gradients(W_v_grad, learning_rate);
W_o.apply_gradients(W_o_grad, learning_rate);
zero_grad();  // Reset gradients
```

---

## Usage Patterns

### Basic Usage

```cpp
// Initialize
MultiHeadAttention mha(512, 8);
mha.learning_rate = 0.0001f;

// Forward pass
Matrix input(seq_len, 512);  // Your input embeddings
Matrix output = mha.forward(input);

// Backward pass
Matrix grad_output(seq_len, 512);  // Gradient from loss
Matrix grad_input = mha.backward(grad_output);

// Update weights
mha.update_weights();
```

### Training Loop

```cpp
MultiHeadAttention mha(d_model, num_heads);
mha.learning_rate = 0.0001f;

for (int epoch = 0; epoch < num_epochs; ++epoch) {
    for (auto& batch : training_data) {
        // Forward pass
        Matrix output = mha.forward(batch.input);
        
        // Compute loss (example: MSE with target)
        float loss = compute_loss(output, batch.target);
        
        // Compute gradient of loss
        Matrix grad_output = compute_loss_gradient(output, batch.target);
        
        // Backward pass
        Matrix grad_input = mha.backward(grad_output);
        
        // Update weights (uses optimizer if set, otherwise simple gradient descent)
        mha.update_weights();
    }
}
```

### With Advanced Optimizer and Scheduling

```cpp
MultiHeadAttention mha(512, 8);
Optimizer optimizer(0.001f);
optimizer.set_beta1(0.9f);
optimizer.set_beta2(0.999f);
mha.set_optimizer(&optimizer);

float base_lr = 0.001f;
int warmup_steps = 4000;

for (int step = 0; step < max_steps; ++step) {
    // Learning rate warmup schedule
    float lr = base_lr * std::min(1.0f, step / (float)warmup_steps);
    optimizer.set_learning_rate(lr);
    
    // Training step
    Matrix output = mha.forward(input);
    Matrix grad_input = mha.backward(grad_output);
    mha.update_weights();  // Uses Adam optimization
    
    if (step % 100 == 0) {
        std::cout << "Step " << step << ", LR: " << lr << std::endl;
    }
}
```

### With Gradient Monitoring

```cpp
MultiHeadAttention mha(512, 8);

// Training with gradient monitoring
for (int step = 0; step < max_steps; ++step) {
    Matrix output = mha.forward(input);
    Matrix grad_input = mha.backward(grad_output);
    
    // Monitor gradients
    float grad_norm = mha.get_gradient_norm();
    if (grad_norm > 10.0f) {
        std::cout << "Warning: Large gradients detected! Norm: " 
                  << grad_norm << std::endl;
        mha.clip_gradients(5.0f);
    }
    
    mha.update_weights();
}
```

### Model Persistence

```cpp
// Training
MultiHeadAttention mha(512, 8);
// ... training ...
mha.save_weights("mha_checkpoint.bin");

// Later: Load for inference or fine-tuning
MultiHeadAttention mha_loaded(512, 8);
mha_loaded.load_weights("mha_checkpoint.bin");
```

---

## Attention Masking

### Types of Masks

#### 1. Padding Mask

Used to ignore padded positions in variable-length sequences:

```cpp
Matrix create_padding_mask(const std::vector<int>& lengths, int max_len) {
    int batch_size = lengths.size();
    Matrix mask(max_len, max_len);
    
    for (int i = 0; i < max_len; ++i) {
        for (int j = 0; j < max_len; ++j) {
            // Allow attention only within valid sequence length
            mask(i, j) = (j < lengths[0]) ? 1.0f : 0.0f;
        }
    }
    
    return mask;
}
```

#### 2. Causal Mask (Look-Ahead Mask)

Used in autoregressive models (e.g., language models) to prevent attending to future positions:

```cpp
Matrix create_causal_mask(int seq_len) {
    Matrix mask(seq_len, seq_len);
    
    for (int i = 0; i < seq_len; ++i) {
        for (int j = 0; j < seq_len; ++j) {
            // Position i can only attend to positions <= i
            mask(i, j) = (j <= i) ? 1.0f : 0.0f;
        }
    }
    
    return mask;
}

// Usage
Matrix causal_mask = create_causal_mask(seq_len);
Matrix output = mha.forward(input, &causal_mask);
```

**Causal Mask Pattern:**
```
1 0 0 0 0
1 1 0 0 0
1 1 1 0 0
1 1 1 1 0
1 1 1 1 1
```

#### 3. Combined Mask

Combine padding and causal masks:

```cpp
Matrix combine_masks(const Matrix& padding_mask, const Matrix& causal_mask) {
    Matrix combined(padding_mask.rows, padding_mask.cols);
    
    for (int i = 0; i < padding_mask.rows; ++i) {
        for (int j = 0; j < padding_mask.cols; ++j) {
            // Both masks must allow attention
            combined(i, j) = padding_mask(i, j) * causal_mask(i, j);
        }
    }
    
    return combined;
}
```

### Mask Implementation Details

**How masking works:**
1. Mask values of 0 → set score to -1e9 (very negative)
2. Softmax(-1e9) ≈ 0, so no attention to masked positions
3. Mask values of 1 → score unchanged, normal attention

**Why -1e9?** 
- Large enough to make softmax output ≈ 0
- Not too large to avoid numerical overflow
- Works reliably with float32 precision

---

## Performance Considerations

### Memory Usage

For d_model = D, seq_len = S:

**Static Memory:**
- Weight matrices: 4 × D² × 4 bytes = 16D² bytes
- Gradient matrices: 4 × D² × 4 bytes = 16D² bytes
- Total static: **32D² bytes**

**Dynamic Memory (per forward pass):**
- Attention scores: S² × 4 bytes
- Attention weights: S² × 4 bytes
- Cached Q, K, V, output: 4 × S × D × 4 bytes
- Total dynamic: **2S² + 16SD bytes**

**Example Sizes:**
- d_model=512, seq_len=512: ~8.6 MB per layer
- d_model=1024, seq_len=1024: ~37 MB per layer

### Computational Bottlenecks

1. **Matrix Multiplications** (QK^T and attention×V)
   - Complexity: O(S² × D)
   - Dominant cost for long sequences

2. **Softmax Computation**
   - Complexity: O(S²)
   - Requires exponentiation for each element

3. **Gradient Computation**
   - Similar to forward pass
   - Softmax gradient is complex

### Optimization Strategies

#### 1. Reduced Sequence Length

For very long sequences, consider:
- **Sliding window attention**: Only attend to nearby positions
- **Sparse attention**: Attend to fixed patterns
- **Chunking**: Process long sequences in chunks

#### 2. Mixed Precision Training

```cpp
// Use float16 for forward pass, float32 for gradients
// (Requires additional implementation)
```

#### 3. Gradient Checkpointing

Trade computation for memory by not caching all activations:
```cpp
// Recompute forward pass during backward instead of caching
// (Saves memory at cost of 33% more computation)
```

#### 4. Flash Attention

For very large models, implement fused kernels that:
- Compute attention in blocks
- Avoid materializing full attention matrix
- Reduce memory bandwidth usage

---

## Integration with Transformers

### Transformer Encoder Layer

Typical usage in a transformer encoder block:

```cpp
class EncoderBlock {
private:
    std::unique_ptr<MultiHeadAttention> attention;
    std::unique_ptr<FeedForward> feed_forward;
    std::unique_ptr<LayerNorm> norm1;
    std::unique_ptr<LayerNorm> norm2;
    
public:
    EncoderBlock(int d_model, int num_heads, int d_ff)
        : attention(new MultiHeadAttention(d_model, num_heads)),
          feed_forward(new FeedForward(d_model, d_ff)),
          norm1(new LayerNorm(d_model)),
          norm2(new LayerNorm(d_model)) {}
    
    void set_optimizer(Optimizer* opt) {
        attention->set_optimizer(opt);
        // Can also set optimizer for feed_forward if it supports it
    }
    
    Matrix forward(const Matrix& input) {
        // Self-attention with residual connection
        Matrix attn_output = attention->forward(input);
        Matrix residual1 = input + attn_output;
        Matrix normed1 = norm1->forward(residual1);
        
        // Feed-forward with residual connection
        Matrix ff_output = feed_forward->forward(normed1);
        Matrix residual2 = normed1 + ff_output;
        Matrix output = norm2->forward(residual2);
        
        return output;
    }
};
```

### Multi-Layer Encoder

```cpp
class TransformerEncoder {
private:
    std::vector<std::unique_ptr<EncoderBlock>> layers;
    
public:
    TransformerEncoder(int num_layers, int d_model, int num_heads, int d_ff) {
        for (int i = 0; i < num_layers; ++i) {
            layers.push_back(
                std::make_unique<EncoderBlock>(d_model, num_heads, d_ff)
            );
        }
    }
    
    Matrix forward(const Matrix& input, const Matrix* mask = nullptr) {
        Matrix output = input;
        
        for (auto& layer : layers) {
            output = layer->forward(output, mask);
        }
        
        return output;
    }
};
```

### With Positional Encoding

```cpp
#include "TokenEmbedding.hpp"
#include "PositionalEncoding.hpp"
#include "MultiHeadAttention.hpp"

// Full encoder pipeline
TokenEmbedding token_emb(vocab_size, d_model);
PositionalEncoding pos_enc(max_seq_len, d_model);
MultiHeadAttention mha(d_model, num_heads);

// Encode tokens
std::vector<int> tokens = {5, 12, 3, 8};
Matrix embedded = token_emb.forward(tokens);
Matrix with_position = pos_enc.forward(embedded);
Matrix encoded = mha.forward(with_position);
```

---

## Debugging and Monitoring

### Configuration Display

```cpp
mha.print_config();
```

**Output:**
```
MultiHeadAttention Configuration:
  Model Dimension (d_model): 512
  Number of Heads: 8
  Dimension per Head (d_k): 64
  Total Parameters: 1048576
  Memory Usage: 4.00 MB
  Learning Rate: 0.001
```

### Attention Visualization

```cpp
// Get attention weights for visualization
const Matrix& attn_weights = mha.get_attention_weights();

// Visualize which positions attend to which
for (int i = 0; i < attn_weights.rows; ++i) {
    std::cout << "Position " << i << " attends to: ";
    for (int j = 0; j < attn_weights.cols; ++j) {
        if (attn_weights(i, j) > 0.1) {  // Threshold for significant attention
            std::cout << j << "(" << attn_weights(i, j) << ") ";
        }
    }
    std::cout << std::endl;
}
```

### Gradient Monitoring

```cpp
// Check for vanishing/exploding gradients
float grad_norm = mha.get_gradient_norm();

if (grad_norm < 1e-6) {
    std::cout << "Warning: Vanishing gradients!" << std::endl;
} else if (grad_norm > 100.0) {
    std::cout << "Warning: Exploding gradients!" << std::endl;
    mha.clip_gradients(10.0f);
}
```

### Gradient Clipping

```cpp
float get_gradient_norm() const;
void clip_gradients(float max_norm);
```

**Purpose:** Prevent exploding gradients during training

**Usage:**
```cpp
// Clip gradients if norm exceeds threshold
mha.backward(grad_output);
mha.clip_gradients(5.0f);  // Clip to max norm of 5.0
mha.update_weights();
```

**Algorithm:**
```
norm = ||gradients||₂
if norm > max_norm:
    gradients = gradients × (max_norm / norm)
```

---

## Common Patterns and Best Practices

### 1. Learning Rate Scheduling

```cpp
float initial_lr = 0.001f;
int warmup_steps = 4000;

for (int step = 0; step < max_steps; ++step) {
    // Transformer learning rate schedule
    float lr = initial_lr * std::min(
        std::pow(step + 1, -0.5f),
        (step + 1) * std::pow(warmup_steps, -1.5f)
    );
    
    mha.learning_rate = lr;
    
    // Training step...
}
```

### 2. Residual Connections

Always use residual connections with attention:
```cpp
Matrix attn_output = mha.forward(input);
Matrix output = input + attn_output;  // Residual connection
```

### 3. Layer Normalization

Normalize before or after attention:

**Pre-LN (more stable):**
```cpp
Matrix normed = layer_norm.forward(input);
Matrix attn_output = mha.forward(normed);
Matrix output = input + attn_output;
```

**Post-LN (original transformer):**
```cpp
Matrix attn_output = mha.forward(input);
Matrix residual = input + attn_output;
Matrix output = layer_norm.forward(residual);
```

### 4. Dropout (Future Enhancement)

After attention weights (not implemented yet):
```cpp
// attention_weights = dropout(attention_weights, p=0.1)
```

### 5. Weight Initialization Verification

```cpp
MultiHeadAttention mha(512, 8);

// Check initialization is reasonable
float sum = 0.0f;
for (int i = 0; i < 512; ++i) {
    for (int j = 0; j < 512; ++j) {
        sum += std::abs(mha.get_W_q()(i, j));
    }
}
float avg = sum / (512 * 512);
std::cout << "Average |W_q| value: " << avg << std::endl;
// Should be close to sqrt(2/512) ≈ 0.0625
```

---

## Error Handling

### Common Exceptions

1. **std::invalid_argument** - Constructor
   - Cause: d_model not divisible by num_heads
   - Solution: Choose compatible dimensions

2. **std::invalid_argument** - Forward pass
   - Cause: Input dimension mismatch
   - Solution: Ensure input.cols == d_model

3. **std::invalid_argument** - Masking
   - Cause: Mask dimensions don't match sequence length
   - Solution: Ensure mask is [seq_len, seq_len]

4. **std::runtime_error** - Load weights
   - Cause: File doesn't exist or dimension mismatch
   - Solution: Verify file path and model dimensions

### Validation Example

```cpp
bool validate_input(const Matrix& input, int expected_d_model) {
    if (input.cols != expected_d_model) {
        std::cerr << "Input dimension " << input.cols 
                  << " doesn't match expected " << expected_d_model 
                  << std::endl;
        return false;
    }
    return true;
}

// Use validation
if (validate_input(input, mha.get_d_model())) {
    Matrix output = mha.forward(input);
}
```

---

## Theoretical Background

### Why Attention?

**Problem with RNNs:**
- Sequential processing (slow)
- Vanishing gradients over long distances
- Difficulty capturing long-range dependencies

**Attention Solution:**
- Parallel processing (fast)
- Direct connections between all positions
- Constant path length for information flow

### Self-Attention vs. Cross-Attention

**Self-Attention** (implemented here):
- Q, K, V all from same source
- Relates positions within a single sequence
- Used in: Encoders, decoders

**Cross-Attention**:
- Q from one source, K and V from another
- Relates positions across two sequences
- Used in: Encoder-decoder attention

### Attention as Soft Dictionary Lookup

Think of attention as:
- **Keys**: Dictionary keys
- **Values**: Dictionary values
- **Queries**: Lookup requests
- **Output**: Weighted combination of values based on key-query similarity

### Positional Information

Attention is **permutation-invariant** - it doesn't inherently know position. Solutions:
1. **Positional Encoding**: Add position info to embeddings
2. **Relative Positional Encodings**: Learn relative positions
3. **Positional Attention**: Modify attention to be position-aware

---

## Comparison with Other Mechanisms

### vs. Recurrent Neural Networks (RNNs)

| Feature | MultiHeadAttention | RNN |
|---------|-------------------|-----|
| Parallelization | Full | Sequential |
| Path length | O(1) | O(n) |
| Training speed | Fast | Slow |
| Long-range deps | Excellent | Poor |
| Memory | O(n²) | O(n) |

### vs. Convolutional Neural Networks (CNNs)

| Feature | MultiHeadAttention | CNN |
|---------|-------------------|-----|
| Receptive field | Global | Local |
| Parameter sharing | No | Yes |
| Translation invariance | No | Yes |
| Sequence modeling | Native | With modifications |
| Computational cost | O(n²d) | O(nkd) |

---

## Performance Benchmarks

### Typical Forward Pass Times (CPU)

| Configuration | Seq Len | Time (ms) |
|--------------|---------|-----------|
| d=256, h=4   | 128     | ~5        |
| d=512, h=8   | 128     | ~15       |
| d=512, h=8   | 512     | ~180      |
| d=1024, h=16 | 128     | ~50       |

*Times are approximate and hardware-dependent*

### Memory Footprint

| d_model | num_heads | Static Memory | Dynamic (seq=512) |
|---------|-----------|---------------|-------------------|
| 256     | 4         | 2 MB          | 1.5 MB            |
| 512     | 8         | 8 MB          | 5 MB              |
| 1024    | 16        | 32 MB         | 18 MB             |

---

## Recent Updates

### Optimizer Integration (January 2026)

The MultiHeadAttention class now supports the centralized Optimizer class:

**Changes:**
- Added `Optimizer* optimizer` member (optional, defaults to nullptr)
- Added `set_optimizer(Optimizer* opt)` method to attach optimizer
- Added `register_parameters()` method to register weights with optimizer
- Modified `update_weights()` to use `optimizer->step()` when available
- Maintains backward compatibility - falls back to simple gradient descent if no optimizer set

**Migration Guide:**

Old code (still works):
```cpp
MultiHeadAttention mha(512, 8);
mha.learning_rate = 0.001f;
mha.update_weights();  // Simple gradient descent
```

New code (recommended):
```cpp
MultiHeadAttention mha(512, 8);
Optimizer optimizer(0.001f);
mha.set_optimizer(&optimizer);
mha.update_weights();  // Adam/AdamW/etc optimization
```

**Advantages:**
- Use Adam, AdamW, or other advanced optimizers
- Centralized learning rate scheduling
- Gradient clipping at optimizer level
- Weight decay / L2 regularization
- Consistent optimization across all model components

---

## Future Extensions

Potential enhancements to consider:

1. ~~**Cached Inference**~~ ✅ IMPLEMENTED (v1.1)
   - Reuse K and V for autoregressive generation
   - Faster inference via `forward_with_cache()`
   - ~2-3x speedup for long sequences

2. **Multi-Query Attention**
   - Share K and V across heads
   - Reduces parameters and memory

3. **Grouped-Query Attention**
   - Compromise between multi-head and multi-query
   - Groups of heads share K and V

4. **Flash Attention**
   - Fused attention kernel
   - Reduces memory bandwidth

5. **Sparse Attention**
   - Only attend to subset of positions
   - Reduces complexity for long sequences

6. **Rotary Position Embeddings (RoPE)**
   - Built-in positional awareness
   - Better generalization to longer sequences

---

## Summary

The `MultiHeadAttention` class provides:

- **Complete Implementation**: Forward pass, backward pass, weight updates
- **KV Caching**: Optimized `forward_with_cache()` for inference ✨ NEW
- **Optimizer Support**: Optional integration with Optimizer class for advanced optimization (Adam, AdamW, etc.)
- **Flexibility**: Optional masking for various use cases
- **Backward Compatibility**: Falls back to simple gradient descent if no optimizer set
- **Efficiency**: Optimized matrix operations, gradient caching, KV cache for generation
- **Robustness**: Input validation, gradient clipping, error handling
- **Debuggability**: Attention visualization, gradient monitoring, configuration display
- **Persistence**: Save/load functionality for model checkpointing
- **Integration**: Designed to work with LayerNorm, PositionalEncoding, TokenEmbedding, Optimizer, KVCache

**Key Strengths:**
- Industry-standard implementation of transformer attention
- Well-documented with mathematical foundations
- Comprehensive error handling and validation
- Easy to integrate into larger architectures
- Monitoring and debugging capabilities
- Production-ready inference optimization (KV caching)

**Performance:**
- Training: O(seq_len² × d_model) per forward pass
- Inference (cached): O(seq_len × d_model) per token - ~25x speedup for 50 tokens
- Memory (cache): ~1-2 MB for typical configurations

**Use Cases:**
- Transformer encoders and decoders
- BERT-style models
- GPT-style autoregressive models
- Sequence-to-sequence models
- Any architecture requiring attention mechanisms

This implementation forms the foundation for modern transformer-based architectures and can be extended for various NLP, computer vision, and multimodal applications.

## See Also

### Related Components
- **[CrossAttention](cross-attention.md)** - Encoder-decoder attention with KV cache
- **[DecoderBlock](../transformer/decoder-block.md)** - Uses MultiHeadAttention with dual caching
- **[LLMDecoder](../transformer/decoder.md)** - Full decoder stack with multi-layer caching
- **[LayerNorm](../normalization/layer-norm.md)** - Layer normalization
- **[PositionalEncoding](../embeddings/positional-encoding.md)** - Position embeddings
- **[TokenEmbedding](../embeddings/token-embedding.md)** - Token to vector conversion

### Optimization & Performance
- **[KVCache API](../../reference/kvcache.md)** - Key-Value caching system documentation
- **[BatchProcessor API](../../reference/batchprocessor.md)** - Batch processing utilities
- **[PerformanceProfiler API](../../reference/performanceprofiler.md)** - Profiling and benchmarking
- **[Inference Optimization Guide](../../guides/inference-optimization.md)** - Complete optimization guide
- **[Inference Quickstart](../../guides/inference-optimization-quickstart.md)** - Quick optimization setup

### Academic References
- **"Attention Is All You Need"** (Vaswani et al., 2017) - Original transformer paper
- **"BERT"** (Devlin et al., 2018) - Bidirectional encoder representations
- **"GPT-2"** (Radford et al., 2019) - Autoregressive language modeling

---

**Last Updated**: January 25, 2026  
**Version**: 1.1  
**Dependencies**: `Matrix.hpp`, `Optimizer.hpp`, `KVCache.hpp`
