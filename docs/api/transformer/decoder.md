# Decoder (LLMDecoder) - Context Documentation

## Overview
The `LLMDecoder` class implements a transformer-based decoder for autoregressive text generation. This component is essential for chatbot applications, machine translation, and any sequence-to-sequence task requiring text generation. It supports both decoder-only architectures (like GPT) and encoder-decoder architectures (like BART, T5).

## Purpose
**Primary Function**: Generate text autoregressively by predicting one token at a time while attending to previously generated tokens and optionally to encoder representations.

**Key Capabilities**:
- Autoregressive text generation with causal masking
- Cross-attention to encoder outputs (encoder-decoder mode)
- Multi-layer transformer architecture
- Token embeddings with learned positional encoding
- Support for training via backpropagation
- Model persistence (save/load)

## Architecture

### Component Stack
```
Input Token IDs [batch_size, seq_length]
    ↓
Token Embedding [seq_length, d_model]
    ↓
Positional Encoding [seq_length, d_model]
    ↓
DecoderBlock 1 (masked self-attention + cross-attention + FFN)
    ↓
DecoderBlock 2
    ↓
...
    ↓
DecoderBlock N
    ↓
Final Layer Normalization [seq_length, d_model]
    ↓
Output [seq_length, d_model]
```

### Internal Components
1. **TokenEmbedding**: Converts token IDs to dense vectors
2. **PositionalEncoding**: Adds position information to embeddings
3. **DecoderBlocks** (×N): Multi-head attention + cross-attention + feed-forward
4. **LayerNorm**: Final normalization layer

### DecoderBlock Structure
Each decoder block contains:
- **Masked Self-Attention**: Attends to previous tokens only (causal masking)
- **Cross-Attention**: Attends to encoder outputs (if provided)
- **Feed-Forward Network**: Non-linear transformation
- **Residual Connections**: Around each sub-layer
- **Layer Normalization**: After each sub-layer

## Class Structure

### File Location
- Header: `/home/rodney/Repos/adai/src/Decoder.hpp`
- Implementation: `/home/rodney/Repos/adai/src/Decoder.cpp`

### Dependencies
```cpp
#include "Matrix.hpp"
#include "LayerNorm.hpp"
#include "PositionalEncoding.hpp"
#include "TokenEmbedding.hpp"
#include "DecoderBlock.hpp"
```

### Class Definition
```cpp
class LLMDecoder {
private:
    // Core components
    std::unique_ptr<TokenEmbedding> token_embedding;
    std::unique_ptr<PositionalEncoding> positional_encoding;
    std::vector<std::unique_ptr<DecoderBlock>> decoder_blocks;
    std::unique_ptr<LayerNorm> final_norm;
    
    // Configuration
    int vocab_size;        // Size of vocabulary
    int d_model;          // Model dimension
    int num_layers;       // Number of decoder layers
    int num_heads;        // Number of attention heads
    int d_ff;             // Feed-forward dimension
    int max_seq_length;   // Maximum sequence length
    
    // Training state
    bool requires_grad;
    float learning_rate;
    
    // Cached values for backward pass
    std::vector<int> cached_token_ids;
    Matrix cached_embeddings;
    Matrix cached_pos_encoded;
    std::vector<Matrix> cached_decoder_outputs;
    Matrix cached_encoder_output;
    
public:
    // Constructor and destructor
    LLMDecoder(int vocab_size, int d_model = 512, int num_layers = 6,
               int num_heads = 8, int d_ff = 2048, int max_seq_length = 512);
    ~LLMDecoder();
    
    // Forward pass methods
    Matrix forward(const std::vector<int>& token_ids);
    Matrix forward_with_encoder(const std::vector<int>& token_ids,
                                const Matrix& encoder_output);
    Matrix forward_with_mask(const std::vector<int>& token_ids,
                            const Matrix& causal_mask,
                            const Matrix* encoder_output = nullptr);
    
    // Training methods
    void backward(const Matrix& grad_output);
    void update_weights(float learning_rate);
    void zero_grad();
    void set_training(bool mode);
    void set_learning_rate(float lr);
    
    // Persistence
    void save_weights(const std::string& filepath) const;
    void load_weights(const std::string& filepath);
    
    // Accessors
    int get_d_model() const;
    int get_vocab_size() const;
    int get_num_layers() const;
    int get_max_seq_length() const;
    TokenEmbedding* get_token_embedding();
    DecoderBlock* get_decoder_block(int layer);
    Matrix get_last_output() const;
    
private:
    Matrix create_causal_mask(int seq_length) const;
};
```

## Key Methods

### Constructor
```cpp
LLMDecoder(int vocab_size, int d_model = 512, int num_layers = 6,
           int num_heads = 8, int d_ff = 2048, int max_seq_length = 512)
```

**Purpose**: Initialize decoder with specified architecture

**Parameters**:
- `vocab_size`: Size of token vocabulary
- `d_model`: Dimension of embeddings and hidden states (default: 512)
- `num_layers`: Number of decoder layers (default: 6)
- `num_heads`: Number of attention heads per layer (default: 8)
- `d_ff`: Dimension of feed-forward network (default: 2048)
- `max_seq_length`: Maximum sequence length (default: 512)

**Initialization**:
1. Creates TokenEmbedding(vocab_size, d_model)
2. Creates PositionalEncoding(d_model, max_seq_length)
3. Creates num_layers DecoderBlocks(d_model, num_heads, d_ff)
4. Creates final LayerNorm(d_model)
5. Sets requires_grad = true, learning_rate = 0.001

### Forward Pass Methods

#### forward()
```cpp
Matrix forward(const std::vector<int>& token_ids)
```

**Purpose**: Decoder-only forward pass (no cross-attention)

**Process**:
1. Converts token IDs to embeddings
2. Adds positional encoding
3. Creates causal mask
4. Passes through decoder blocks (no encoder output)
5. Applies final layer normalization

**Returns**: Matrix [seq_length, d_model]

**Use Case**: GPT-style autoregressive generation

#### forward_with_encoder()
```cpp
Matrix forward_with_encoder(const std::vector<int>& token_ids,
                            const Matrix& encoder_output)
```

**Purpose**: Encoder-decoder forward pass with cross-attention

**Parameters**:
- `token_ids`: Decoder input tokens
- `encoder_output`: Output from encoder [encoder_seq_len, d_model]

**Process**:
1. Converts token IDs to embeddings
2. Adds positional encoding
3. Creates causal mask for autoregressive generation
4. Passes through decoder blocks with cross-attention to encoder
5. Applies final layer normalization

**Returns**: Matrix [seq_length, d_model]

**Use Case**: BART/T5-style sequence-to-sequence models

#### forward_with_mask()
```cpp
Matrix forward_with_mask(const std::vector<int>& token_ids,
                        const Matrix& causal_mask,
                        const Matrix* encoder_output = nullptr)
```

**Purpose**: Forward pass with custom attention mask

**Parameters**:
- `token_ids`: Decoder input tokens
- `causal_mask`: Custom causal mask matrix
- `encoder_output`: Optional encoder output

**Use Case**: Fine-grained control over attention patterns

#### forward_with_cache() ✨ NEW
```cpp
Matrix forward_with_cache(const std::vector<int>& token_ids,
                         DecoderKVCache& kv_cache,
                         const Matrix* encoder_output = nullptr,
                         bool use_cache = true)
```

**Purpose**: Optimized forward pass using KV cache for autoregressive generation

**Parameters**:
- `token_ids`: New token IDs to process [num_new_tokens] (typically 1 during generation)
- `kv_cache`: Multi-layer KV cache structure (DecoderKVCache)
- `encoder_output`: Optional encoder output for cross-attention (can be nullptr)
- `use_cache`: Whether to update cache with new K/V pairs (default: true)

**Returns**: Matrix [num_new_tokens, d_model]

**Performance**: ~2-3x speedup for long sequences by avoiding redundant computation

**How It Works**:

1. **First Call (Empty Cache)**:
   - Processes all tokens in `token_ids`
   - Computes K/V for all positions in self-attention
   - Stores K/V pairs in cache for future use
   - Behavior similar to regular `forward()`

2. **Subsequent Calls (Cache Populated)**:
   - Only processes new tokens (typically 1 token)
   - Reuses cached K/V from previous positions
   - Only computes K/V for new position
   - Much faster than reprocessing entire sequence

**Cache Structure**:
```cpp
DecoderKVCache kv_cache;
kv_cache.initialize(num_layers, max_seq_length, d_model, num_heads);

// Each layer has two caches:
// - Self-attention cache: stores decoder's own K/V pairs
// - Cross-attention cache: stores encoder K/V pairs (computed once)
```

**Typical Usage Pattern**:
```cpp
// Initialize decoder and cache
LLMDecoder decoder(vocab_size=1000, d_model=256, num_layers=4, 
                   num_heads=4, d_ff=1024, max_seq_length=128);
DecoderKVCache kv_cache;
kv_cache.initialize(4, 128, 256, 4);  // num_layers, max_seq_len, d_model, num_heads

// Optional: Encode input for encoder-decoder mode
Matrix encoder_output = encoder.encode(input_text);  // [input_len, 256]

// Generate tokens autoregressively
std::vector<int> generated = {BOS_TOKEN};

for (int i = 0; i < max_gen_length; ++i) {
    // Process only the last token (except first iteration)
    std::vector<int> current_token = {generated.back()};
    
    // Forward with cache (2-3x faster than regular forward)
    Matrix decoder_output = decoder.forward_with_cache(
        current_token, kv_cache, &encoder_output, true
    );
    
    // Get logits and sample next token
    Matrix logits = lm_head.forward(decoder_output);
    int next_token = sample_token(logits);
    
    generated.push_back(next_token);
    if (next_token == EOS_TOKEN) break;
}

// Clear cache for next sequence
kv_cache.clear();
```

**Performance Comparison**:

*Without Cache (Inefficient)*:
```cpp
// Generate 50 tokens
std::vector<int> generated = {BOS_TOKEN};
for (int i = 0; i < 50; ++i) {
    // Reprocesses ALL tokens every iteration (1+2+3+...+50 = 1,275 forward passes)
    Matrix output = decoder.forward_with_encoder(generated, encoder_output);
    int next_token = sample_token(output);
    generated.push_back(next_token);
}
// Total computation: O(n²) where n = sequence length
```

*With Cache (Efficient)*:
```cpp
// Generate 50 tokens
DecoderKVCache kv_cache;
kv_cache.initialize(num_layers, max_seq_length, d_model, num_heads);

std::vector<int> generated = {BOS_TOKEN};
for (int i = 0; i < 50; ++i) {
    // Only processes new token (50 forward passes total)
    std::vector<int> new_token = {generated.back()};
    Matrix output = decoder.forward_with_cache(new_token, kv_cache, &encoder_output);
    int next_token = sample_token(output);
    generated.push_back(next_token);
}
// Total computation: O(n) where n = sequence length
// Speedup: ~25x for 50 tokens (1,275 / 50)
```

**Key Differences from Regular Forward**:

| Aspect | `forward()` | `forward_with_cache()` |
|--------|-------------|------------------------|
| Input | Full sequence | Only new tokens |
| Computation | Recomputes all positions | Only new positions |
| Complexity | O(seq_len²) | O(seq_len) per token |
| Memory | Low | Higher (cache storage) |
| Use Case | Training, first pass | Inference, generation |
| Speedup | Baseline | 2-3x for long sequences |

**Important Notes**:

1. **Positional Encoding**: Automatically adjusts for cache position
   - Uses `current_position = kv_cache.current_length()`
   - Adds correct positional encoding for new tokens

2. **Causal Masking**: Adapts to cache state
   - New tokens can attend to cached + new positions
   - Shape: [num_new_tokens, total_seq_len]

3. **Cross-Attention Cache**: Computed once per sequence
   - Encoder K/V cached on first call
   - Reused for all subsequent decoder steps
   - No recomputation needed

4. **Cache Management**:
   - Call `kv_cache.clear()` between sequences
   - Ensure `max_seq_length` >= generation length
   - Cache grows incrementally during generation

**Integration with DecoderBlock**:
```cpp
// Inside forward_with_cache implementation
for (int layer_idx = 0; layer_idx < num_layers; ++layer_idx) {
    KVCache& self_attn_cache = kv_cache.get_self_attention_cache(layer_idx);
    KVCache& cross_attn_cache = kv_cache.get_cross_attention_cache(layer_idx);
    
    // Each DecoderBlock also has forward_with_cache
    x = decoder_blocks[layer_idx]->forward_with_cache(
        x, encoder_output, causal_mask, 
        &self_attn_cache, &cross_attn_cache, nullptr, use_cache
    );
}
```

**When to Use**:
- ✅ Autoregressive text generation (chatbots, translation)
- ✅ Beam search decoding
- ✅ Sampling-based generation
- ✅ Long sequence generation (>20 tokens)
- ❌ Training (use regular `forward()` or `forward_with_encoder()`)
- ❌ Single-pass inference
- ❌ Batch processing (current implementation is single-sequence)

### Causal Masking

#### create_causal_mask()
```cpp
Matrix create_causal_mask(int seq_length) const
```

**Purpose**: Create lower-triangular mask for autoregressive generation

**Mask Pattern**:
```
Position:  0  1  2  3  4
    0:     1  0  0  0  0    (can only see position 0)
    1:     1  1  0  0  0    (can see positions 0-1)
    2:     1  1  1  0  0    (can see positions 0-2)
    3:     1  1  1  1  0    (can see positions 0-3)
    4:     1  1  1  1  1    (can see positions 0-4)
```

**Rationale**: Prevents positions from attending to future tokens, ensuring autoregressive property

**Implementation**:
```cpp
for (int i = 0; i < seq_length; ++i) {
    for (int j = 0; j < seq_length; ++j) {
        mask.data[i][j] = (j <= i) ? 1.0f : 0.0f;
    }
}
```

### Training Methods

#### backward()
```cpp
void backward(const Matrix& grad_output)
```

**Purpose**: Backpropagate gradients through decoder

**Process** (reverse order):
1. Backward through final layer norm
2. Backward through decoder blocks (N → 1)
3. Skip positional encoding (no learnable parameters)
4. Backward through token embedding

**Note**: Requires cached values from forward pass

#### update_weights()
```cpp
void update_weights(float lr)
```

**Purpose**: Update all learnable parameters

**Updates**:
1. Token embedding weights
2. All decoder block weights (attention + FFN)
3. Final layer norm parameters (gamma, beta)

**Note**: Current implementation doesn't pass lr parameter to components (they use internal learning rates)

#### zero_grad()
```cpp
void zero_grad()
```

**Purpose**: Clear accumulated gradients

**Clears**:
- Token embedding gradients
- All decoder block gradients
- Layer norm gradients (if applicable)

### Persistence

#### save_weights()
```cpp
void save_weights(const std::string& filepath) const
```

**Current Implementation**: Saves configuration only (vocab_size, d_model, num_layers, etc.)

**Warning**: Component weights not saved (TokenEmbedding, DecoderBlock, LayerNorm lack save_weights methods)

**Production Requirement**: Implement save_weights in all components

#### load_weights()
```cpp
void load_weights(const std::string& filepath)
```

**Current Implementation**: Loads and validates configuration

**Warning**: Component weights not loaded (randomly initialized)

**Production Requirement**: Implement load_weights in all components

## Usage Patterns

### Decoder-Only Generation (GPT-style)
```cpp
// Initialize decoder
LLMDecoder decoder(vocab_size=1000, d_model=256, num_layers=4, 
                   num_heads=4, d_ff=1024, max_seq_length=128);

// Generate one step
std::vector<int> tokens = {1, 5, 10};  // Previous tokens
Matrix decoder_output = decoder.forward(tokens);

// decoder_output has shape [3, 256]
// Use last row for next token prediction
```

### Encoder-Decoder Generation (BART-style)
```cpp
// Encode input
Matrix encoder_output = encoder.encode(input_text);  // [input_len, d_model]

// Generate response autoregressively
std::vector<int> generated = {BOS_TOKEN};
for (int i = 0; i < max_length; ++i) {
    Matrix decoder_output = decoder.forward_with_encoder(generated, encoder_output);
    
    // Get logits for next token (requires LanguageModelHead)
    Matrix logits = lm_head.forward(decoder_output);
    int next_token = argmax(logits.data[logits.rows - 1]);
    
    generated.push_back(next_token);
    if (next_token == EOS_TOKEN) break;
}
```

### Training Example
```cpp
// Forward pass
decoder.set_training(true);
Matrix output = decoder.forward_with_encoder(target_tokens, encoder_output);

// Compute loss (external)
Matrix logits = lm_head.forward(output);
float loss = compute_cross_entropy(logits, target_tokens);

// Backward pass
Matrix grad_logits = compute_loss_gradient(logits, target_tokens);
Matrix grad_decoder = lm_head.backward(grad_logits);
decoder.backward(grad_decoder);

// Update weights
float lr = 0.001;
decoder.update_weights(lr);
lm_head.update_weights(lr);
decoder.zero_grad();
lm_head.zero_grad();
```

## Design Decisions

### 1. Decoder-Only vs. Encoder-Decoder
**Decision**: Support both architectures with same class

**Rationale**:
- Decoder-only: Pass empty encoder output (1×d_model matrix)
- Encoder-decoder: Pass actual encoder output
- Unified interface reduces code duplication

**Implementation**:
```cpp
if (encoder_output.rows > 0 && encoder_output.cols > 0) {
    // Use cross-attention
    x = decoder_blocks[i]->forward(x, encoder_output, causal_mask, nullptr);
} else {
    // Skip cross-attention
    Matrix empty_encoder(1, d_model);
    x = decoder_blocks[i]->forward(x, empty_encoder, causal_mask, nullptr);
}
```

### 2. Causal Masking
**Decision**: Generate causal mask automatically in forward pass

**Rationale**:
- Autoregressive generation requires causal masking
- Creating mask per forward pass ensures correct sequence length
- Prevents accidental future information leakage

**Alternative**: Cache and reuse masks (optimization for inference)

### 3. Caching for Backward Pass
**Decision**: Cache all intermediate activations during forward pass

**Cached Values**:
- `cached_token_ids`: Original input tokens
- `cached_embeddings`: Token embeddings
- `cached_pos_encoded`: Positional-encoded embeddings
- `cached_decoder_outputs`: Output from each decoder layer
- `cached_encoder_output`: Encoder output (for cross-attention)

**Rationale**: Gradient computation requires forward pass activations

**Trade-off**: Memory usage vs. recomputation cost

### 4. Component Ownership
**Decision**: Use `std::unique_ptr` for all components

**Rationale**:
- Automatic memory management
- Clear ownership semantics
- Prevents accidental copying

### 5. Partial Save/Load Implementation
**Decision**: Save configuration only, warn about incomplete persistence

**Current State**:
- Saves: vocab_size, d_model, num_layers, num_heads, d_ff, max_seq_length
- Does NOT save: Token embeddings, decoder block weights, layer norm parameters

**Future Work**: Implement save/load in all components

## Integration Points

### Used By
1. **EncoderDecoderModel**: Combines LLMEncoder + LLMDecoder for seq2seq
2. **TextGenerator**: Uses decoder output for autoregressive generation
3. **Training Loop**: Calls forward/backward/update_weights

### Uses
1. **TokenEmbedding**: Converts token IDs to embeddings
2. **PositionalEncoding**: Adds position information
3. **DecoderBlock**: Multi-head attention + FFN layers
4. **LayerNorm**: Final normalization

### Related Components
- **LLMEncoder**: Paired encoder for encoder-decoder models
- **LanguageModelHead**: Projects decoder output to vocabulary logits
- **TextGenerator**: Implements generation strategies (beam search, sampling, etc.)

## Configuration Guidelines

### Small Model (Fast prototyping)
```cpp
LLMDecoder decoder(
    vocab_size = 1000,
    d_model = 128,
    num_layers = 2,
    num_heads = 4,
    d_ff = 512,
    max_seq_length = 64
);
```

### Medium Model (Balanced)
```cpp
LLMDecoder decoder(
    vocab_size = 5000,
    d_model = 256,
    num_layers = 4,
    num_heads = 8,
    d_ff = 1024,
    max_seq_length = 128
);
```

### Large Model (Production-like)
```cpp
LLMDecoder decoder(
    vocab_size = 50000,
    d_model = 512,
    num_layers = 6,
    num_heads = 8,
    d_ff = 2048,
    max_seq_length = 512
);
```

## Performance Considerations

### Memory Usage
**Forward Pass (without cache)**:
- Token embeddings: `seq_length × d_model × sizeof(float)`
- Decoder outputs: `num_layers × seq_length × d_model × sizeof(float)`
- Attention scores: `num_layers × num_heads × seq_length² × sizeof(float)`

**Forward Pass (with cache)**:
- Base memory: Same as above for new tokens only
- Cache memory: `num_layers × 2 × max_seq_length × d_model × sizeof(float)`
  - 2× for self-attention + cross-attention caches
  - Example: 4 layers × 2 × 128 tokens × 256 dims × 4 bytes = 1.05 MB

**Backward Pass**: Approximately 2× forward pass memory (gradients)

### Time Complexity
**Without Cache (Standard Forward)**:
- **Self-Attention**: O(seq_length² × d_model) per layer
- **Cross-Attention**: O(seq_length × encoder_length × d_model) per layer
- **Feed-Forward**: O(seq_length × d_model × d_ff) per layer
- **Total**: O(num_layers × seq_length × (seq_length × d_model + d_model × d_ff))
- **Autoregressive (n tokens)**: O(n² × num_layers × d_model) - quadratic!

**With Cache (forward_with_cache)**:
- **Self-Attention**: O(seq_length × d_model) per layer (linear in cache size)
- **Cross-Attention**: O(1) after first call (encoder K/V cached)
- **Feed-Forward**: O(d_model × d_ff) for new tokens only
- **Total per token**: O(num_layers × seq_length × d_model)
- **Autoregressive (n tokens)**: O(n × num_layers × d_model) - linear!

**Speedup Analysis**:
```
Generation length: n tokens
Without cache: 1 + 2 + 3 + ... + n = n(n+1)/2 computations
With cache: n computations
Speedup: n(n+1)/(2n) ≈ n/2 for large n

Examples:
- 10 tokens: ~5x speedup
- 50 tokens: ~25x speedup
- 100 tokens: ~50x speedup
```

### Optimization Opportunities ✨
1. **KV Caching** ✅ IMPLEMENTED: Use `forward_with_cache()` for 2-3x speedup
2. **Mask Reuse**: Cache causal masks for common sequence lengths
3. **Batch Processing**: Process multiple sequences simultaneously (see BatchProcessor)
4. **Gradient Checkpointing**: Trade computation for memory during training
5. **Performance Profiling**: Use PerformanceProfiler to identify bottlenecks

**Quick Optimization Setup**:
```cpp
// 1. Enable KV cache for generation
DecoderKVCache kv_cache;
kv_cache.initialize(num_layers, max_seq_length, d_model, num_heads);

// 2. Use cached forward pass
for (int i = 0; i < gen_length; ++i) {
    auto output = decoder.forward_with_cache(new_tokens, kv_cache, &enc_out);
    // ... generate next token
}

// 3. Profile performance
PerformanceProfiler profiler;
profiler.start_profiling();
// ... run generation
profiler.stop_profiling();
auto stats = profiler.get_stats();
std::cout << "Avg token time: " << stats.mean_time_ms << "ms" << std::endl;
```

See [Inference Optimization Guide](../../guides/inference-optimization.md) for complete optimization strategies.

## Limitations

### Current Implementation
1. ~~**No KV Caching**~~ ✅ **RESOLVED**: `forward_with_cache()` implemented for 2-3x speedup
2. **Incomplete Persistence**: save/load doesn't include component weights
3. **No Batch Support**: Processes one sequence at a time (single-sequence cache)
4. **Fixed Architecture**: Cannot dynamically change num_layers, d_model, etc.
5. **Learning Rate Propagation**: update_weights() doesn't pass lr to components
6. **Cache Limitations**: Current cache doesn't support batched generation

### Known Issues
1. **LayerNorm Update**: Final layer norm update_weights commented out (method may not exist)
2. **Component Save/Load**: TokenEmbedding, DecoderBlock lack persistence methods
3. **Gradient Flow**: No gradient clipping or normalization

## Testing Strategy

### Unit Tests Needed
1. **Constructor**: Verify component initialization
2. **Causal Masking**: Validate mask shape and values
3. **Forward Pass**: Test decoder-only and encoder-decoder modes
4. **Backward Pass**: Verify gradient computation
5. **Update Weights**: Check parameter updates
6. **Save/Load**: Test configuration persistence
7. **Edge Cases**: Empty sequences, single tokens, max length

### Integration Tests
1. **With LLMEncoder**: Full encoder-decoder pipeline
2. **With LanguageModelHead**: Decoder → logits → tokens
3. **With TextGenerator**: Autoregressive generation
4. **Training Loop**: Forward + backward + update over multiple steps

## Example Usage

### Complete Generation Pipeline
```cpp
#include "Decoder.hpp"
#include "encoder.hpp"
#include "LanguageModelHead.hpp"
#include "TextGenerator.hpp"

// Initialize components
int vocab_size = 1000;
int d_model = 256;
LLMEncoder encoder(vocab_size, d_model, 2, 4, 1024, 128);
LLMDecoder decoder(vocab_size, d_model, 2, 4, 1024, 128);
LanguageModelHead lm_head(d_model, vocab_size);
TextGenerator generator(vocab_size);

// Encode input
std::string input = "Hello, how are you?";
Matrix encoder_output = encoder.encode(input);

// Generate response
std::vector<int> generated = {BOS_TOKEN};
int max_length = 20;

for (int i = 0; i < max_length; ++i) {
    // Decode
    Matrix decoder_output = decoder.forward_with_encoder(generated, encoder_output);
    
    // Project to vocabulary
    Matrix logits = lm_head.forward(decoder_output);
    
    // Sample next token
    std::vector<float> last_logits(logits.cols);
    for (int j = 0; j < logits.cols; ++j) {
        last_logits[j] = logits.data[logits.rows - 1][j];
    }
    
    int next_token = argmax(last_logits);
    generated.push_back(next_token);
    
    if (next_token == EOS_TOKEN) break;
}

// Decode tokens to text
std::string response = decoder.get_token_embedding()->decode(generated);
```

## Future Enhancements

### Short-term
1. Implement complete save/load for all components
2. Add batch processing support (batched KV cache)
3. ~~Implement KV caching for efficient inference~~ ✅ COMPLETED (v1.1)
4. Add gradient clipping to prevent exploding gradients
5. Add batch support to `forward_with_cache()`

### Medium-term
1. Support dynamic architecture modification
2. Add mixed precision training (FP16/BF16)
3. Implement distributed training support
4. Add attention visualization tools

### Long-term
1. Flash Attention integration for faster attention
2. Sparse attention patterns (Longformer, BigBird)
3. Adaptive computation (early exit, layer dropping)
4. Model quantization for deployment

## See Also

### Core Components
- **[DecoderBlock](decoder-block.md)** - Individual decoder layer with self/cross-attention
- **[MultiHeadAttention](../attention/multi-head-attention.md)** - Self-attention mechanism with KV cache
- **[CrossAttention](../attention/cross-attention.md)** - Encoder-decoder attention with KV cache
- **[TokenEmbedding](../embeddings/token-embedding.md)** - Token to vector conversion
- **[PositionalEncoding](../embeddings/positional-encoding.md)** - Position information
- **[LayerNorm](../normalization/layer-norm.md)** - Layer normalization

### Related Models
- **[LLMEncoder](encoder.md)** - Paired encoder for encoder-decoder models
- **[EncoderDecoderModel](encoder-decoder-model.md)** - Complete transformer model
- **[LanguageModelHead](../generation/language-model-head.md)** - Output projection to vocabulary

### Optimization & Generation
- **[KVCache API](../../reference/kvcache.md)** - Key-Value caching system for inference
- **[BatchProcessor API](../../reference/batchprocessor.md)** - Batch processing utilities
- **[PerformanceProfiler API](../../reference/performanceprofiler.md)** - Profiling and benchmarking
- **[TextGenerator](../generation/text-generator.md)** - Generation strategies (beam search, sampling)
- **[Inference Optimization Guide](../../guides/inference-optimization.md)** - Complete optimization guide
- **[Inference Quickstart](../../guides/inference-optimization-quickstart.md)** - Quick optimization setup

### Architecture Documentation
- **[Decoder Architecture](../../architecture/decoder-architecture.md)** - Design patterns
- **[Decoder Design](../../architecture/decoder-design.md)** - Implementation details

## Related Documentation
- **DecoderBlock**: `DECODERBLOCK_CONTEXT.md`
- **CrossAttention**: `CROSSATTENTION_CONTEXT.md`
- **TokenEmbedding**: `TOKENEMBEDDING_CONTEXT.md`
- **PositionalEncoding**: `POSITIONALENCODING_CONTEXT.md`
- **LLMEncoder**: `ENCODER_CONTEXT.md`
- **TextGenerator**: `TEXTGENERATOR_CONTEXT.md`

## Version History
- **v1.1** (2026-01-25): Added forward_with_cache() for inference optimization
- **v1.0** (2026-01-18): Initial implementation with decoder-only and encoder-decoder support

---

**Last Updated**: January 25, 2026  
**Version**: 1.1  
**Dependencies**: `Matrix.hpp`, `LayerNorm.hpp`, `PositionalEncoding.hpp`, `TokenEmbedding.hpp`, `DecoderBlock.hpp`, `KVCache.hpp`
