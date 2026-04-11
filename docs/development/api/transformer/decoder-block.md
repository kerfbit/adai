# DecoderBlock Component - Context Documentation

## Overview

**File**: `src/DecoderBlock.hpp`, `src/DecoderBlock.cpp`
**Purpose**: Single layer of transformer decoder with masked self-attention, cross-attention to encoder, and feed-forward network
**Role in Decoder**: Building block that processes sequential data autoregressively while attending to encoder context
**Dependencies**: `MultiHeadAttention.hpp`, `CrossAttention.hpp`, `FeedForward.hpp`, `LayerNorm.hpp`, `Matrix.hpp`

The DecoderBlock is a fundamental component in transformer-based decoder architectures. Unlike the EncoderBlock which only has self-attention, the DecoderBlock includes both **masked self-attention** (preventing future token access) and **cross-attention** (attending to encoder output for sequence-to-sequence tasks).

---

## Architecture

### Component Structure

```text
Input [seq_len, d_model]
    ↓
┌─────────────────────────────────┐
│ 1. Masked Self-Attention        │ ← Causal mask (no future peeking)
│    (Q=K=V=input)                │
└─────────────────────────────────┘
    ↓
┌─────────────────────────────────┐
│ 2. Residual + LayerNorm         │
│    normed1 = LN(input + attn)   │
└─────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────┐
│ 3. Cross-Attention                          │
│    (Q=normed1, K=V=encoder_output)         │ ← Attend to encoder
└─────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────┐
│ 4. Residual + LayerNorm         │
│    normed2 = LN(normed1 + attn) │
└─────────────────────────────────┘
    ↓
┌─────────────────────────────────┐
│ 5. Feed-Forward Network         │
│    FFN(x) = ReLU(xW1+b1)W2+b2  │
└─────────────────────────────────┘
    ↓
┌─────────────────────────────────┐
│ 6. Residual + LayerNorm         │
│    output = LN(normed2 + FFN)   │
└─────────────────────────────────┘
    ↓
Output [seq_len, d_model]
```

### Three Sub-Layers

#### Layer 1: Masked Self-Attention

- **Purpose**: Model dependencies between current and past tokens
- **Masking**: Causal mask prevents attending to future positions
- **Operation**: MultiHeadAttention(input, input, input, causal_mask)

#### Layer 2: Cross-Attention (Unique to decoder!)

- **Purpose**: Attend to encoder output for context
- **Query**: From decoder (current layer)
- **Key/Value**: From encoder output
- **Operation**: CrossAttention(decoder_state, encoder_output, mask)

#### Layer 3: Feed-Forward Network

- **Purpose**: Non-linear transformation for each position
- **Operation**: Two linear layers with ReLU activation
- **Parameters**: d_model → d_ff → d_model

### Residual Connections

Three residual connections ensure stable gradient flow:

```text
residual1 = input + self_attention(input)
residual2 = residual1_norm + cross_attention(residual1_norm, encoder)
residual3 = residual2_norm + feed_forward(residual2_norm)
```

---

## Mathematical Formulation

### Forward Pass

#### Step 1-2: Masked Self-Attention

```text
self_attn_out = MultiHeadAttention(input, input, input, causal_mask)
residual1 = input + self_attn_out
normed1 = LayerNorm(residual1)
```

#### Step 3-4: Cross-Attention to Encoder

```text
cross_attn_out = CrossAttention(
    query=normed1,           [tgt_len, d_model]
    key_value=encoder_output [src_len, d_model]
)
residual2 = normed1 + cross_attn_out
normed2 = LayerNorm(residual2)
```

#### Step 5-6: Feed-Forward Network

```text
ff_out = FeedForward(normed2)
residual3 = normed2 + ff_out
output = LayerNorm(residual3)
```

### Backward Pass

Gradients flow in reverse order through all components:

```text
∂L/∂output → LayerNorm₃ → Residual₃ → FeedForward
                                    ↓
                          → LayerNorm₂ → Residual₂ → CrossAttention
                                                   ↓
                                         → LayerNorm₁ → Residual₁ → SelfAttention
                                                                  ↓
                                                      → ∂L/∂input
```

**Residual Connection Gradients**:

```text
At each residual connection, gradient splits:
  grad_input_path = grad_residual
  grad_sublayer_path = grad_residual

Total gradient = grad_input_path + grad_from_sublayer
```

---

## Class Interface

### Constructor

```cpp
DecoderBlock(int d_model, int num_heads, int d_ff, float dropout = 0.1f)
```

**Parameters**:

- `d_model`: Model dimension (e.g., 512, 768)
- `num_heads`: Number of attention heads (must divide d_model)
- `d_ff`: Feed-forward hidden dimension (typically 4 × d_model)
- `dropout`: Dropout rate for regularization (default: 0.1)

**Initialization**:

- Creates MultiHeadAttention for masked self-attention
- Creates CrossAttention for encoder-decoder attention
- Creates FeedForward network
- Creates 3 LayerNorm layers
- Sets learning rates for all components

**Example**:

```cpp
// Standard configuration
DecoderBlock decoder_layer(512, 8, 2048, 0.1f);

// GPT-2 small configuration
DecoderBlock gpt2_layer(768, 12, 3072, 0.1f);
```

### API Methods

#### Forward Pass (Standard)

```cpp
Matrix forward(const Matrix& input,
               const Matrix& encoder_output,
               const Matrix& self_attn_mask,
               const Matrix* cross_attn_mask = nullptr)
```

**Inputs**:

- `input`: Decoder input `[tgt_len, d_model]`
- `encoder_output`: Encoder output `[src_len, d_model]`
- `self_attn_mask`: Causal mask `[tgt_len, tgt_len]`
  - `1.0` = attend, `0.0` = mask (set to -inf before softmax)
  - Lower triangular matrix for causal masking
- `cross_attn_mask`: Optional padding mask `[tgt_len, src_len]`

**Output**:

- Transformed representation `[tgt_len, d_model]`

**Process**:

1. Masked self-attention on decoder input
2. Residual connection + layer norm
3. Cross-attention to encoder output
4. Residual connection + layer norm
5. Feed-forward transformation
6. Residual connection + layer norm

**Example**:

```cpp
// Create inputs
Matrix decoder_input(10, 512);    // 10 target tokens
Matrix encoder_output(20, 512);   // 20 source tokens
Matrix causal_mask = create_causal_mask(10);  // Lower triangular

// Forward pass
Matrix output = decoder_layer.forward(
    decoder_input, encoder_output, causal_mask, nullptr
);
// output.shape = [10, 512]
```

#### Forward Pass with KV Cache ✨ NEW

```cpp
Matrix forward_with_cache(const Matrix& input,
                         const Matrix& encoder_output,
                         const Matrix& self_attn_mask,
                         KVCache* self_attn_cache,
                         KVCache* cross_attn_cache,
                         const Matrix* cross_attn_mask = nullptr,
                         bool use_cache = true)
```

**Purpose**: Optimized forward pass using dual KV caches for autoregressive generation

**Inputs**:

- `input`: New decoder tokens (typically 1 token) `[num_new_tokens, d_model]`
- `encoder_output`: Encoder output `[src_len, d_model]`
- `self_attn_mask`: Causal mask adapted for cache `[num_new_tokens, total_seq_len]`
- `self_attn_cache`: Cache for self-attention K/V pairs (grows with each token)
- `cross_attn_cache`: Cache for cross-attention K/V pairs (computed once from encoder)
- `cross_attn_mask`: Optional padding mask `[num_new_tokens, src_len]`
- `use_cache`: Whether to update caches (default: true)

**Output**:

- Transformed representation `[num_new_tokens, d_model]`

**Performance**: ~2-3x speedup for autoregressive generation by avoiding redundant computation

**How It Works**:

1. **Self-Attention Cache** (Decoder K/V):
   - Stores K/V pairs from previous decoder positions
   - Grows incrementally: cache[0], cache[0:1], cache[0:2], ...
   - New token attends to all cached positions + itself
   - Avoids recomputing K/V for previous tokens

2. **Cross-Attention Cache** (Encoder K/V):
   - Stores K/V pairs from encoder output
   - Computed once on first generation step
   - Reused for all subsequent decoder steps (encoder is constant)
   - Massive savings: no recomputation needed

**Dual Cache Structure**:

```cpp
// Self-attention cache (grows with generation)
KVCache self_attn_cache;  // Stores decoder K/V pairs
// - Step 1: [1, d_model]
// - Step 2: [2, d_model]
// - Step 3: [3, d_model]
// - ...

// Cross-attention cache (computed once)
KVCache cross_attn_cache; // Stores encoder K/V pairs
// - Computed on first call from encoder_output
// - Reused for all subsequent steps (constant)
```

**Typical Usage Pattern**:

```cpp
// Initialize caches
KVCache self_attn_cache;
KVCache cross_attn_cache;

// Optional: Encode input
Matrix encoder_output = encoder.encode(input_text);  // [src_len, 512]

// Generate tokens autoregressively
std::vector<Matrix> generated_embeddings;
Matrix current_embedding = start_token_embedding;  // [1, 512]

for (int i = 0; i < max_gen_length; ++i) {
    // Create causal mask for current position
    int current_pos = i;
    int total_len = current_pos + 1;
    Matrix causal_mask(1, total_len);  // New token can see all previous
    for (int j = 0; j < total_len; ++j) {
        causal_mask(0, j) = 1.0f;  // Attend to all positions up to current
    }

    // Forward with cache (much faster than reprocessing all tokens)
    Matrix output = decoder_layer.forward_with_cache(
        current_embedding,      // Only new token [1, 512]
        encoder_output,         // Encoder output [src_len, 512]
        causal_mask,           // Mask [1, total_len]
        &self_attn_cache,      // Grows each step
        &cross_attn_cache,     // Computed once
        nullptr,               // No cross-attention mask
        true                   // Update caches
    );

    // Project to vocabulary and sample next token
    Matrix logits = lm_head.forward(output);
    int next_token = sample_token(logits);
    current_embedding = token_embedding(next_token);

    if (next_token == EOS_TOKEN) break;
}

// Clear caches for next sequence
self_attn_cache.clear();
cross_attn_cache.clear();
```

**Performance Comparison**:

*Without Cache (Inefficient)*:

```cpp
// Generate 50 tokens - reprocesses everything each time
for (int i = 0; i < 50; ++i) {
    // all_tokens grows: [1], [2], [3], ..., [50]
    Matrix output = decoder_layer.forward(
        all_tokens_so_far,  // Grows from 1 to 50 tokens
        encoder_output,
        create_mask(all_tokens_so_far.rows)
    );
    // Computation: 1+2+3+...+50 = 1,275 attention operations
}
```

*With Cache (Efficient)*:

```cpp
// Generate 50 tokens - only new token each time
KVCache self_cache, cross_cache;
for (int i = 0; i < 50; ++i) {
    Matrix output = decoder_layer.forward_with_cache(
        new_token,          // Always [1, d_model]
        encoder_output,
        adapted_mask,
        &self_cache,       // Reuses previous K/V
        &cross_cache,      // Reuses encoder K/V
        nullptr, true
    );
    // Computation: 50 attention operations (one per token)
    // Speedup: 1,275 / 50 = 25.5x faster!
}
```

**Key Insights**:

1. **Self-Attention Optimization**:
   - Without cache: O(n²) for n tokens (recompute all K/V pairs)
   - With cache: O(n) for n tokens (compute only new K/V)
   - Speedup: ~n/2 for n tokens

2. **Cross-Attention Optimization**:
   - Without cache: Recompute encoder K/V every step
   - With cache: Compute once, reuse forever
   - Speedup: Near-infinite for long generation

3. **Memory Trade-off**:
   - Cache storage: O(seq_len × d_model) per cache
   - Typical overhead: ~2-4 MB for 512-dim, 100-token sequence
   - Worth it for generation >10 tokens

**Important Notes**:

1. **Cache Invalidation**: Clear caches when starting new sequence
2. **Mask Adaptation**: Mask shape changes from `[seq_len, seq_len]` to `[num_new, total_len]`
3. **Training vs Inference**: Use regular `forward()` for training, `forward_with_cache()` for inference
4. **Encoder Cache**: Cross-attention cache computed once from encoder, never changes
5. **Fallback**: If `use_cache=false` or cache is null, falls back to regular `forward()`

**Integration with LLMDecoder**:

The `LLMDecoder` class uses `forward_with_cache()` across multiple DecoderBlocks:

```cpp
// Inside LLMDecoder::forward_with_cache()
for (int layer_idx = 0; layer_idx < num_layers; ++layer_idx) {
    KVCache& self_cache = kv_cache.get_self_attention_cache(layer_idx);
    KVCache& cross_cache = kv_cache.get_cross_attention_cache(layer_idx);

    x = decoder_blocks[layer_idx]->forward_with_cache(
        x, encoder_output, causal_mask,
        &self_cache, &cross_cache, nullptr, use_cache
    );
}
```

See [LLMDecoder documentation](decoder.md) for multi-layer caching examples.

#### Backward Pass (Gradient Computation)

```cpp
Matrix backward(const Matrix& grad_output)
```

**Input**:

- `grad_output`: Gradient from next layer `[tgt_len, d_model]`

**Output**:

- Gradient w.r.t. decoder input `[tgt_len, d_model]`

**Process** (reverse order):

1. Backprop through LayerNorm₃
2. Split gradient at Residual₃
3. Backprop through FeedForward
4. Backprop through LayerNorm₂
5. Split gradient at Residual₂
6. Backprop through CrossAttention (produces 2 gradients!)
7. Backprop through LayerNorm₁
8. Split gradient at Residual₁
9. Backprop through SelfAttention

**Note**: CrossAttention produces two gradients:

- `grad_query_input`: For decoder path
- `grad_kv_input`: For encoder path (typically discarded in decoder training)

**Example**:

```cpp
Matrix grad_from_next_layer(10, 512);
Matrix grad_input = decoder_layer.backward(grad_from_next_layer);
// grad_input.shape = [10, 512]
```

#### Weight Management

```cpp
void update_weights()
void zero_grad()
void set_learning_rate(float lr)
```

**`update_weights()`**:

- Updates all sub-components (self-attention, cross-attention, FFN)
- LayerNorms update during their own backward pass

**`zero_grad()`**:

- Resets gradient accumulators in all components
- Call before each forward/backward cycle

**`set_learning_rate(float lr)`**:

- Propagates learning rate to all sub-components

**Example**:

```cpp
decoder_layer.set_learning_rate(0.0001f);
decoder_layer.zero_grad();

// Training step
output = decoder_layer.forward(input, encoder_output, mask);
grad = decoder_layer.backward(loss_gradient);
decoder_layer.update_weights();
```

#### Optimizer Integration

```cpp
void register_parameters_with_optimizer(Optimizer& optimizer)
```

**Purpose**: Register all decoder block parameters with an external optimizer

**Process**:

1. Register self-attention parameters (Q, K, V, output projections)
2. Register cross-attention parameters (Q, K, V, output projections)
3. Register feed-forward parameters (W1, W2, biases)
4. Register all three layer normalization parameters (gamma, beta)

**Example**:

```cpp
// Create decoder block
DecoderBlock block(d_model, num_heads, d_ff);

// Create and configure optimizer
Optimizer optimizer(OptimizerType::ADAMW, 0.001f);
optimizer.set_betas(0.9f, 0.999f);
optimizer.set_weight_decay(0.01f);
optimizer.set_max_grad_norm(1.0f);

// Register all parameters
block.register_parameters_with_optimizer(optimizer);

// Training loop
for (auto& batch : data) {
    optimizer.zero_grad();
    Matrix output = block.forward(batch.input, encoder_out, mask);
    Matrix grad = compute_gradient(output, batch.target);
    block.backward(grad);
    optimizer.clip_gradients();
    optimizer.step();
}
```

**Benefits**:

- Advanced optimization algorithms (Adam, AdamW)
- Automatic gradient clipping for training stability
- Weight decay regularization
- Centralized parameter management

**Note**: Preferred over `update_weights()` for production training

#### Model Persistence

```cpp
void save(const std::string& filepath)
void load(const std::string& filepath)
```

**Save Format**:

- Main file: Hyperparameters (d_model, num_heads, d_ff, dropout, lr)
- `filepath.self_attn`: Self-attention weights
- `filepath.cross_attn`: Cross-attention weights
- `filepath.ff`: Feed-forward weights
- LayerNorm parameters NOT saved (minimal impact, reinitialized on load)

**Example**:

```cpp
decoder_layer.save("decoder_layer_0.bin");

// Later...
DecoderBlock loaded_layer(512, 8, 2048);
loaded_layer.load("decoder_layer_0.bin");
```

#### Accessors

```cpp
FeedForward* get_feed_forward()
```

Returns a raw pointer to the internal `FeedForward` sublayer. Use it to register activation
hooks for saturation tracking (see [FeedForward — Activation Hook](feed-forward.md#activation-hook)).

```cpp
DecoderBlock layer(512, 8, 2048);
layer.get_feed_forward()->set_activation_hook([](const Matrix& act) {
    // inspect post-GELU activations
});
```

---

## Implementation Details

### Causal Masking

**Purpose**: Prevent decoder from attending to future tokens during training

**Causal Mask Structure** (for seq_len=5):

```text
Position:  0  1  2  3  4
       0 [ 1  0  0  0  0 ]  ← Position 0 only sees itself
       1 [ 1  1  0  0  0 ]  ← Position 1 sees 0-1
       2 [ 1  1  1  0  0 ]  ← Position 2 sees 0-2
       3 [ 1  1  1  1  0 ]  ← Position 3 sees 0-3
       4 [ 1  1  1  1  1 ]  ← Position 4 sees 0-4
```

**Creation**:

```cpp
Matrix create_causal_mask(int seq_len) {
    Matrix mask(seq_len, seq_len);
    for (int i = 0; i < seq_len; ++i) {
        for (int j = 0; j < seq_len; ++j) {
            mask(i, j) = (j <= i) ? 1.0f : 0.0f;
        }
    }
    return mask;
}
```

**Application**:

- Mask value `0.0` → replaced with `-1e9` before softmax
- Ensures `softmax(-1e9) ≈ 0` (no attention to future)

### Cross-Attention Mechanism

**Key Difference from Self-Attention**:

|Aspect|Self-Attention|Cross-Attention|
|--------|----------------|-----------------|
|Query source|Decoder input|Decoder input|
|Key source|Decoder input|**Encoder output**|
|Value source|Decoder input|**Encoder output**|
|Sequence lengths|Same (tgt_len)|**Different** (tgt_len × src_len)|
|Purpose|Model decoder context|**Access encoder information**|

**Cross-Attention Forward**:

```cpp
// Q from decoder, K/V from encoder
Matrix query = normed1;              // [tgt_len, d_model]
Matrix key_value = encoder_output;   // [src_len, d_model]

// CrossAttention handles different sequence lengths
Matrix cross_attn_out = cross_attention->forward(
    query,      // [tgt_len, d_model]
    key_value,  // [src_len, d_model]
    mask        // [tgt_len, src_len] optional
);
// Output: [tgt_len, d_model]
```

**Cross-Attention Backward**:

```cpp
// Produces TWO gradients
Matrix grad_query_input;     // [tgt_len, d_model]
Matrix grad_kv_input;         // [src_len, d_model]

cross_attention->backward(
    grad_output,        // Input gradient
    grad_query_input,   // Output: gradient for decoder
    grad_kv_input       // Output: gradient for encoder (usually ignored)
);
```

### Residual Connection Implementation

**Manual Addition** (element-wise):

```cpp
Matrix residual(input.rows, input.cols);
for (int i = 0; i < input.rows; ++i) {
    for (int j = 0; j < input.cols; ++j) {
        residual(i, j) = input(i, j) + sublayer_output(i, j);
    }
}
```

**Gradient Split**:

```cpp
// At residual: y = x + f(x)
// Gradient: ∂L/∂x = ∂L/∂y + ∂L/∂f(x)

Matrix grad_input(grad_residual.rows, grad_residual.cols);
Matrix grad_sublayer(grad_residual.rows, grad_residual.cols);

for (int i = 0; i < rows; ++i) {
    for (int j = 0; j < cols; ++j) {
        grad_input(i, j) = grad_residual(i, j);      // Direct path
        grad_sublayer(i, j) = grad_residual(i, j);   // Through sublayer
    }
}
```

### Parameter Count

**For d_model=512, num_heads=8, d_ff=2048**:

|Component|Parameters|
|-----------|------------|
|Self-Attention|4 × (512 × 512) = 1,048,576|
|Cross-Attention|4 × (512 × 512) = 1,048,576|
|FeedForward|512×2048 + 2048×512 = 2,097,152|
|LayerNorm (3×)|3 × (512 + 512) = 3,072|
|**Total**|**4,197,376 params**|

**Memory** (float32):

- Parameters: ~16.8 MB per layer
- Gradients: ~16.8 MB per layer
- Activations: Varies with batch size and sequence length

---

## Integration with Decoder

### Single Layer Usage

```cpp
// Initialize
DecoderBlock layer(d_model=512, num_heads=8, d_ff=2048);

// Prepare inputs
Matrix decoder_input(tgt_len, 512);
Matrix encoder_output(src_len, 512);
Matrix causal_mask = create_causal_mask(tgt_len);

// Forward
Matrix output = layer.forward(
    decoder_input, encoder_output, causal_mask, nullptr
);

// Backward
Matrix grad = layer.backward(loss_gradient);
layer.update_weights();
```

### Multi-Layer Stack (LLMDecoder)

```cpp
class LLMDecoder {
    std::vector<std::unique_ptr<DecoderBlock>> layers;

    LLMDecoder(int num_layers, int d_model, int num_heads, int d_ff) {
        for (int i = 0; i < num_layers; ++i) {
            layers.push_back(
                std::make_unique<DecoderBlock>(d_model, num_heads, d_ff)
            );
        }
    }

    Matrix forward(Matrix input, Matrix encoder_output, Matrix mask) {
        Matrix x = input;
        for (auto& layer : layers) {
            x = layer->forward(x, encoder_output, mask);
        }
        return x;
    }

    Matrix backward(Matrix grad) {
        Matrix g = grad;
        for (auto it = layers.rbegin(); it != layers.rend(); ++it) {
            g = (*it)->backward(g);
        }
        return g;
    }
};
```

---

## Comparison with EncoderBlock

### Structural Differences

|Feature|EncoderBlock|DecoderBlock|
|---------|--------------|--------------|
|**Self-Attention**|Bidirectional|**Causal (masked)**|
|**Cross-Attention**|❌ None|✅ **Attends to encoder**|
|**Sub-layers**|2 (self-attn + FFN)|**3** (self-attn + cross-attn + FFN)|
|**Residual Connections**|2|**3**|
|**LayerNorms**|2|**3**|
|**Inputs**|1 (input only)|**2** (input + encoder_output)|
|**Parameters**|~2.1M (d_model=512)|**~4.2M** (2× encoder)|

### Functional Differences

**EncoderBlock**:

- **Purpose**: Build rich contextual representations
- **Attention**: Bidirectional (sees all positions)
- **Use Case**: Understanding input sequence
- **Example**: BERT, encoder in Transformer

**DecoderBlock**:

- **Purpose**: Generate output autoregressively
- **Attention**: Causal (only past) + cross (to encoder)
- **Use Case**: Generating target sequence
- **Example**: GPT, decoder in Transformer

### Code Comparison

**EncoderBlock Forward**:

```cpp
Matrix forward(const Matrix& input, const Matrix* mask) {
    // 1. Self-attention
    auto attn = self_attention->forward(input, mask);
    auto res1 = input + attn;
    auto norm1 = layer_norm1->forward(res1);

    // 2. Feed-forward
    auto ff = feed_forward->forward(norm1);
    auto res2 = norm1 + ff;
    auto output = layer_norm2->forward(res2);

    return output;
}
```

**DecoderBlock Forward**:

```cpp
Matrix forward(const Matrix& input,
               const Matrix& encoder_output,
               const Matrix& self_mask,
               const Matrix* cross_mask) {
    // 1. MASKED self-attention
    auto attn = self_attention->forward(input, &self_mask);
    auto res1 = input + attn;
    auto norm1 = layer_norm1->forward(res1);

    // 2. CROSS-attention (NEW!)
    auto cross = cross_attention->forward(norm1, encoder_output, cross_mask);
    auto res2 = norm1 + cross;
    auto norm2 = layer_norm2->forward(res2);

    // 3. Feed-forward
    auto ff = feed_forward->forward(norm2);
    auto res3 = norm2 + ff;
    auto output = layer_norm3->forward(res3);

    return output;
}
```

---

## Use Cases

### 1. Machine Translation (Encoder-Decoder)

**Architecture**: Encoder processes source language, decoder generates target language

```cpp
// Encoder: English → hidden representations
Matrix encoder_output = encoder.forward(english_tokens);

// Decoder: hidden representations → French
Matrix french_embeddings = token_embedding.forward(french_tokens);
Matrix causal_mask = create_causal_mask(french_tokens.size());

Matrix decoder_output = decoder_block.forward(
    french_embeddings,
    encoder_output,      // Cross-attention to English
    causal_mask,
    nullptr
);

Matrix logits = lm_head.forward(decoder_output);
// Sample next French word
```

### 2. Decoder-Only Language Model (GPT-style)

**Architecture**: Stack of decoder blocks without encoder (cross-attention skipped or removed)

```cpp
// For pure decoder (GPT), would need modified DecoderBlock without cross-attention
// Or pass dummy encoder_output and ignore it

Matrix input_embeddings = token_embedding.forward(input_tokens);
Matrix causal_mask = create_causal_mask(input_tokens.size());

Matrix hidden = decoder_block.forward(
    input_embeddings,
    input_embeddings,  // Self-attention only (no separate encoder)
    causal_mask,
    nullptr
);
```

### 3. Summarization (Encoder-Decoder)

```cpp
// Encoder: Long document → compressed representation
Matrix document_encoding = encoder.forward(document_tokens);

// Decoder: Generate summary autoregressively
std::vector<int> summary_tokens = {START_TOKEN};

for (int step = 0; step < max_summary_len; ++step) {
    Matrix summary_embeddings = embed(summary_tokens);
    Matrix causal_mask = create_causal_mask(summary_tokens.size());

    Matrix decoder_output = decoder_block.forward(
        summary_embeddings,
        document_encoding,  // Attend to document
        causal_mask,
        nullptr
    );

    Matrix logits = lm_head.forward(decoder_output);
    int next_token = sample(logits[logits.rows - 1]);

    if (next_token == END_TOKEN) break;
    summary_tokens.push_back(next_token);
}
```

---

## Performance Characteristics

### Computational Complexity

**Forward Pass**:

- Self-Attention: O(tgt_len² × d_model)
- Cross-Attention: O(tgt_len × src_len × d_model)
- Feed-Forward: O(tgt_len × d_model × d_ff)
- **Total**: O(tgt_len² × d_model + tgt_len × src_len × d_model + tgt_len × d_model × d_ff)

**Backward Pass**: Similar complexity (2-3× forward pass time)

### Bottlenecks

1. **Cross-Attention with Long Source**:
   - If src_len >> tgt_len, cross-attention dominates
   - Example: Summarizing 1000-token document → 50-token summary
   - O(50 × 1000 × d_model) vs O(50² × d_model)

2. **Feed-Forward Network**:
   - For d_ff = 4 × d_model, FFN is typically the largest component
   - O(tgt_len × d_model × 4d_model) = O(4 × tgt_len × d_model²)

3. **Multiple Layers**:
   - Typical decoders: 6-24 layers
   - Total complexity multiplied by num_layers

### Memory Usage

**For batch_size=1, tgt_len=100, src_len=200, d_model=512, d_ff=2048**:

|Component|Memory|
|-----------|--------|
|Parameters|~16.8 MB|
|Gradients|~16.8 MB|
|Self-attention scores|100×100×4 = 40 KB|
|Cross-attention scores|100×200×4 = 80 KB|
|Intermediate activations|~1-2 MB|
|**Total per layer**|**~35 MB**|

**Scaling**:

- 12 layers: ~420 MB
- 24 layers (GPT-2): ~840 MB

---

## Testing and Validation

### Unit Tests

#### Test 1: Forward Pass Dimensions

```cpp
DecoderBlock layer(512, 8, 2048);

Matrix decoder_input(10, 512);
Matrix encoder_output(20, 512);
Matrix causal_mask = create_causal_mask(10);

Matrix output = layer.forward(decoder_input, encoder_output, causal_mask);

assert(output.rows == 10);
assert(output.cols == 512);
```

#### Test 2: Causal Masking Enforcement

```cpp
// Verify that position i cannot attend to position j > i
// Check attention weights after forward pass
```

#### Test 3: Cross-Attention Gradient Flow

```cpp
// Verify both grad_query_input and grad_kv_input are computed
Matrix grad = layer.backward(grad_output);
assert(grad.rows == decoder_input.rows);
assert(grad.cols == decoder_input.cols);
```

#### Test 4: Gradient Check

```cpp
// Numerical vs analytical gradients
float epsilon = 1e-5f;
Matrix numerical_grad = compute_numerical_gradient(layer, input, epsilon);
Matrix analytical_grad = layer.backward(grad_output);

float max_diff = max_absolute_difference(numerical_grad, analytical_grad);
assert(max_diff < 1e-3f);
```

#### Test 5: Save/Load Consistency

```cpp
layer.save("test_decoder_layer.bin");

DecoderBlock loaded_layer(512, 8, 2048);
loaded_layer.load("test_decoder_layer.bin");

Matrix output1 = layer.forward(input, encoder_output, mask);
Matrix output2 = loaded_layer.forward(input, encoder_output, mask);

assert(matrices_equal(output1, output2, 1e-6f));
```

### Integration Tests

#### Test 6: Multi-Layer Stack

```cpp
// Stack 6 decoder layers
std::vector<DecoderBlock> layers;
for (int i = 0; i < 6; ++i) {
    layers.emplace_back(512, 8, 2048);
}

// Forward through all layers
Matrix x = input;
for (auto& layer : layers) {
    x = layer.forward(x, encoder_output, causal_mask);
}

// Backward through all layers (reverse order)
Matrix grad = loss_gradient;
for (auto it = layers.rbegin(); it != layers.rend(); ++it) {
    grad = it->backward(grad);
}
```

#### Test 7: Training Loop

```cpp
for (int epoch = 0; epoch < num_epochs; ++epoch) {
    layer.zero_grad();

    Matrix output = layer.forward(input, encoder_output, mask);
    float loss = compute_loss(output, target);
    Matrix grad = compute_gradient(output, target);

    layer.backward(grad);
    layer.update_weights();
}
```

---

## Common Pitfalls and Solutions

### Issue 1: Information Leakage from Future Tokens

**Problem**: Incorrect causal mask allows attention to future positions

**Solution**:

```cpp
// CORRECT: Lower triangular mask
Matrix create_causal_mask(int seq_len) {
    Matrix mask(seq_len, seq_len);
    for (int i = 0; i < seq_len; ++i) {
        for (int j = 0; j < seq_len; ++j) {
            mask(i, j) = (j <= i) ? 1.0f : 0.0f;  // Only past and current
        }
    }
    return mask;
}

// WRONG: Allows future attention
mask(i, j) = (j < i) ? 1.0f : 0.0f;  // Excludes current position!
```

### Issue 2: Cross-Attention Gradient Ignored

**Problem**: Forgetting to handle two outputs from cross-attention backward

**Solution**:

```cpp
// CORRECT: Handle both gradients
Matrix grad_decoder, grad_encoder;
cross_attention->backward(grad_output, grad_decoder, grad_encoder);
// Use grad_decoder for decoder path
// grad_encoder typically ignored (encoder already trained)

// WRONG: Single output API (doesn't exist for CrossAttention)
Matrix grad = cross_attention->backward(grad_output);  // Compile error!
```

### Issue 3: Dimension Mismatch (Cross-Attention)

**Problem**: src_len ≠ tgt_len causing shape errors

**Solution**:

```cpp
// CrossAttention handles different lengths correctly
Matrix query(tgt_len, d_model);          // e.g., [10, 512]
Matrix key_value(src_len, d_model);      // e.g., [20, 512]

Matrix output = cross_attention->forward(query, key_value);
// output.shape = [tgt_len, d_model] = [10, 512] ✓
```

### Issue 4: Exploding Gradients in Deep Decoders

**Problem**: 24+ layers cause gradient explosion

**Solutions**:

- **Gradient Clipping**:

```cpp
void clip_gradients(float max_norm) {
    float grad_norm = compute_total_gradient_norm();
    if (grad_norm > max_norm) {
        float scale = max_norm / grad_norm;
        scale_all_gradients(scale);
    }
}
```

- **Layer Normalization** (already included!):

```cpp
// Pre-norm architecture (used here) is more stable
normed = LayerNorm(input)
output = input + Sublayer(normed)

// vs post-norm (less stable)
output = LayerNorm(input + Sublayer(input))
```

- **Learning Rate Warmup**:

```cpp
float get_learning_rate(int step, int warmup_steps) {
    if (step < warmup_steps) {
        return base_lr * (step / warmup_steps);
    }
    return base_lr;
}
```

### Issue 5: Memory Issues with Long Sequences

**Problem**: Attention scores O(seq_len²) don't fit in memory

**Solutions**:

- **Chunked Processing**:

```cpp
for (int start = 0; start < seq_len; start += chunk_size) {
    int end = min(start + chunk_size, seq_len);
    Matrix chunk = input.slice(start, end);
    Matrix chunk_output = layer.forward(chunk, encoder_output, mask);
    // Process chunk...
}
```

- **Sparse Attention** (future enhancement):

  - Strided attention (attend every k-th position)
  - Local attention (attend only nearby positions)
  - Longformer-style patterns

---

## Advanced Features (Future Enhancements)

### 1. Decoder-Only Mode (GPT-style)

**Modification**: Remove cross-attention for pure language modeling

```cpp
class DecoderOnlyBlock : public DecoderBlock {
    Matrix forward(const Matrix& input, const Matrix& self_attn_mask) {
        // Same input for both decoder and "encoder"
        return DecoderBlock::forward(input, input, self_attn_mask, nullptr);
    }
};
```

### 2. Relative Position Embeddings

**Enhancement**: Add position-aware attention bias

```cpp
Matrix compute_relative_position_bias(int tgt_len, int src_len) {
    Matrix bias(tgt_len, src_len);
    for (int i = 0; i < tgt_len; ++i) {
        for (int j = 0; j < src_len; ++j) {
            int relative_pos = j - i;
            bias(i, j) = position_embedding_table[relative_pos];
        }
    }
    return bias;
}
```

### 3. Flash Attention

**Optimization**: Memory-efficient attention computation

```cpp
// Instead of materializing full attention matrix
Matrix scores = Q * K.transpose();  // [tgt_len, src_len] - memory intensive

// Use block-wise computation
Matrix flash_attention(Q, K, V) {
    // Compute attention in blocks, never materialize full matrix
    // Reduces memory from O(n²) to O(n)
}
```

### 4. Mixture of Experts (MoE)

**Enhancement**: Replace FFN with gated experts

```cpp
class MoEFeedForward {
    std::vector<FeedForward> experts;
    Matrix gating_network;

    Matrix forward(Matrix input) {
        Matrix gates = softmax(input * gating_network);  // [seq_len, num_experts]
        Matrix output = zeros(input.rows, input.cols);

        for (int i = 0; i < num_experts; ++i) {
            Matrix expert_out = experts[i].forward(input);
            output += expert_out * gates[:, i];  // Weighted combination
        }
        return output;
    }
};
```

---

## Usage Examples

### Example 1: Single Layer Forward-Backward

```cpp
// Setup
DecoderBlock layer(d_model=512, num_heads=8, d_ff=2048);
layer.set_learning_rate(0.0001f);

// Prepare data
Matrix decoder_input(10, 512);      // 10 target tokens
Matrix encoder_output(20, 512);     // 20 source tokens
Matrix causal_mask = create_causal_mask(10);

// Forward
Matrix output = layer.forward(decoder_input, encoder_output, causal_mask);

// Compute loss (e.g., cross-entropy)
Matrix grad = compute_loss_gradient(output, targets);

// Backward
Matrix input_grad = layer.backward(grad);

// Update
layer.update_weights();
```

### Example 2: Training Loop with Multiple Batches

```cpp
DecoderBlock layer(512, 8, 2048);
layer.set_learning_rate(0.0001f);

for (int epoch = 0; epoch < num_epochs; ++epoch) {
    for (auto& batch : training_data) {
        layer.zero_grad();

        Matrix output = layer.forward(
            batch.decoder_input,
            batch.encoder_output,
            batch.causal_mask,
            batch.padding_mask
        );

        float loss = cross_entropy_loss(output, batch.targets);
        Matrix grad = cross_entropy_gradient(output, batch.targets);

        layer.backward(grad);
        layer.update_weights();

        if (step % 100 == 0) {
            std::cout << "Loss: " << loss << std::endl;
        }
    }
}
```

### Example 3: Save/Load for Inference

```cpp
// Training
DecoderBlock layer(512, 8, 2048);
train_model(layer);
layer.save("trained_decoder_layer.bin");

// Inference (later)
DecoderBlock inference_layer(512, 8, 2048);
inference_layer.load("trained_decoder_layer.bin");

Matrix output = inference_layer.forward(input, encoder_output, mask);
// No backward pass needed for inference
```

---

## See Also

### Core Components

- **[MultiHeadAttention](../attention/multi-head-attention.md)**: Self-attention with KV cache
- **[CrossAttention](../attention/cross-attention.md)**: Encoder-decoder attention with KV cache
- **[FeedForward](../feedforward/feed-forward.md)**: Position-wise transformation
- **[LayerNorm](../normalization/layer-norm.md)**: Layer normalization
- **[EncoderBlock](encoder-block.md)**: Encoder counterpart (bidirectional)

### Related Models

- **[LLMDecoder](decoder.md)**: Full decoder stack with multi-layer caching
- **[EncoderDecoderModel](encoder-decoder-model.md)**: Complete transformer model
- **[LanguageModelHead](../generation/language-model-head.md)**: Output projection

### Optimization & Performance

- **[KVCache API](../../reference/kvcache.md)**: Key-Value caching system for inference
- **[BatchProcessor API](../../reference/batchprocessor.md)**: Batch processing utilities
- **[PerformanceProfiler API](../../reference/performanceprofiler.md)**: Profiling and benchmarking
- **[Inference Optimization Guide](../../guides/inference-optimization.md)**: Complete optimization guide
- **[Inference Quickstart](../../guides/inference-optimization-quickstart.md)**: Quick optimization setup

### Design Documents

- **DECODER_DESIGN.md**: Overall architecture
- **DECODER_IMPLEMENTATION_GUIDE.md**: Coding patterns
- **ENCODER_DECODER_COMPARISON.md**: EncoderBlock differences
- **CROSSATTENTION_CONTEXT.md**: Cross-attention details

## References

### Academic References

- **"Attention Is All You Need"** (Vaswani et al., 2017)
  - Section 3.1: Decoder architecture
  - Section 3.2.3: Masking
  - Section 3.3: Cross-attention in decoder

- **"Language Models are Unsupervised Multitask Learners"** (GPT-2, Radford et al., 2019)
  - Decoder-only architecture
  - Causal self-attention

- **"Exploring the Limits of Transfer Learning"** (T5, Raffel et al., 2020)
  - Encoder-decoder pre-training

---

**Last Updated**: January 25, 2026
**Version**: 1.1
**Dependencies**: `MultiHeadAttention.hpp`, `CrossAttention.hpp`, `FeedForward.hpp`, `LayerNorm.hpp`, `Matrix.hpp`, `KVCache.hpp`

---

## Summary

The **DecoderBlock** is the core building block of transformer decoders, featuring:

✅ **Three Sub-Layers**: Masked self-attention, cross-attention, feed-forward
✅ **Causal Masking**: Prevents future information leakage
✅ **Cross-Attention**: Enables encoder-decoder architectures
✅ **Residual Connections**: Three skip connections for gradient flow
✅ **Layer Normalization**: Stabilizes training
✅ **Full Backpropagation**: Complete gradient computation
✅ **KV Caching**: Dual-cache optimization for inference ✨ NEW

**Key Differences from EncoderBlock**:

- Masked (causal) self-attention vs bidirectional
- Additional cross-attention layer
- Two inputs (decoder + encoder) vs one
- 2× parameters due to extra attention layer
- Dual KV cache support (self + cross attention)

**Performance**:

- Training: O(tgt_len² + tgt_len × src_len) × d_model
- Inference (with cache): O(tgt_len × d_model) per token - ~25x speedup for 50 tokens
- Suitable for sequences up to ~1000 tokens on modern hardware

**Production Ready**: Successfully tested with GPT-2 scale parameters (d_model=768, 50K vocabulary), with KV cache optimization for efficient autoregressive generation.
