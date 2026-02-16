# Decoder Design Document

## Overview

This document outlines the design for a **Transformer Decoder** component that complements the existing LLMEncoder implementation. The design follows the established architectural patterns, reuses existing components, and integrates seamlessly with the current codebase.

---

## Design Constraints & Patterns (from existing codebase)

### 1. **Memory Management**

- Use `std::unique_ptr` for owned components
- Smart pointer ownership model (seen in `LLMEncoder`, `EncoderBlock`)
- No raw pointers for managed objects

### 2. **Architecture Pattern**

- Composition over inheritance
- Component-based design with clear separation of concerns
- Each component handles its own forward/backward passes
- Caching mechanism for backward pass efficiency

### 3. **Interface Consistency**

- `forward()` method for forward pass with caching
- `backward()` method for gradient computation
- `update_weights()` for parameter updates (when applicable)
- `zero_grad()` for gradient reset (when applicable)
- Public `learning_rate` member variable

### 4. **Initialization Standards**

- Xavier/He initialization for weight matrices
- Zero initialization for biases
- Constructor prints initialization summary
- Default hyperparameters with reasonable values

### 5. **Matrix Operations**

- Use `Matrix` class for all tensor operations
- Shape: `[sequence_length, d_model]` or `[seq_len, seq_len]` for masks
- Element-wise operations done manually with nested loops
- Use existing `Matrix` operators: `*`, `+`, `-`, `transpose()`, `hadamard()`

### 6. **Gradient Flow**

- Manual gradient computation using chain rule
- Gradients accumulated across backward pass
- Residual connections split gradients
- Layer normalization handles its own gradients

---

## Component Architecture

### Component Hierarchy

```text
LLMDecoder
├── BPETokenizer (shared with encoder)
├── TokenEmbedding (decoder's own embeddings)
├── PositionalEncoding (shared or separate instance)
├── DecoderBlocks (N layers)
│   ├── Masked Multi-Head Self-Attention (causal)
│   ├── LayerNorm
│   ├── Cross-Attention to Encoder Output
│   ├── LayerNorm
│   ├── FeedForward
│   └── LayerNorm
├── Final LayerNorm
└── LanguageModelHead (output projection)
```

### Information Flow

```text
Target Text (during training) / Generated Tokens (during inference)
    ↓
BPE Tokenization → [token_1, token_2, ..., token_n]
    ↓
Token Embedding → [emb_1, emb_2, ..., emb_n]  (shape: [seq_len, d_model])
    ↓
Positional Encoding → [emb_1 + pos_1, emb_2 + pos_2, ...]
    ↓
DecoderBlock 1 (Self-Attn + Cross-Attn + FFN) + Encoder Output
    ↓
DecoderBlock 2 (Self-Attn + Cross-Attn + FFN) + Encoder Output
    ↓
    ...
    ↓
DecoderBlock N (Self-Attn + Cross-Attn + FFN) + Encoder Output
    ↓
Final Layer Normalization
    ↓
Language Model Head → [seq_len, vocab_size] (logits)
    ↓
Softmax → Token Probabilities
    ↓
Sampling/Greedy Selection → Next Token
```

---

## Component Specifications

### 1. DecoderBlock Class

**Purpose:** Single transformer decoder layer with self-attention, cross-attention, and feed-forward.

**File:** `DecoderBlock.hpp`, `DecoderBlock.cpp`

**Interface:**

```cpp
class DecoderBlock {
private:
    // Core components
    std::unique_ptr<MultiHeadAttention> self_attention;      // Masked self-attention
    std::unique_ptr<MultiHeadAttention> cross_attention;     // Attend to encoder
    std::unique_ptr<FeedForward> feed_forward;
    std::unique_ptr<LayerNorm> norm1;  // After self-attention
    std::unique_ptr<LayerNorm> norm2;  // After cross-attention
    std::unique_ptr<LayerNorm> norm3;  // After feed-forward

    // Hyperparameters
    int d_model;
    int num_heads;
    int d_ff;
    float dropout_rate;

    // Cached values for backward pass
    Matrix cached_input;
    Matrix cached_self_attn_output;
    Matrix cached_residual1;
    Matrix cached_normed1;
    Matrix cached_cross_attn_output;
    Matrix cached_residual2;
    Matrix cached_normed2;
    Matrix cached_ff_output;
    Matrix cached_residual3;
    Matrix cached_encoder_output;  // Cached encoder output for cross-attention

public:
    float learning_rate;

    /**
     * Constructor
     *
     * @param d_model Model dimension
     * @param num_heads Number of attention heads
     * @param d_ff Feed-forward hidden dimension
     * @param dropout Dropout rate (default: 0.1)
     */
    DecoderBlock(int d_model, int num_heads, int d_ff, float dropout = 0.1f);

    /**
     * Forward pass through decoder block
     *
     * @param input Decoder input [seq_len, d_model]
     * @param encoder_output Encoder output for cross-attention [enc_seq_len, d_model]
     * @param self_attn_mask Causal mask for self-attention [seq_len, seq_len]
     * @param cross_attn_mask Optional padding mask for encoder [seq_len, enc_seq_len]
     * @return Output [seq_len, d_model]
     */
    Matrix forward(const Matrix& input,
                   const Matrix& encoder_output,
                   const Matrix& self_attn_mask,
                   const Matrix* cross_attn_mask = nullptr);

    /**
     * Backward pass through decoder block
     *
     * @param grad_output Gradient from next layer [seq_len, d_model]
     * @return Gradient w.r.t. input [seq_len, d_model]
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
     * Set learning rate for all sub-components
     */
    void set_learning_rate(float lr);

    /**
     * Save decoder block parameters to file
     */
    void save(const std::string& filepath);

    /**
     * Load decoder block parameters from file
     */
    void load(const std::string& filepath);
};
```

**Key Design Decisions:**

1. **Reuses `MultiHeadAttention`**: No need to create new attention class - existing one handles both self and cross-attention
2. **Three LayerNorm instances**: Follows transformer decoder architecture (norm after each sub-layer)
3. **Causal masking**: Self-attention mask prevents attending to future positions
4. **Cross-attention**: Second attention layer attends to encoder output
5. **Residual connections**: Three residual paths (self-attn, cross-attn, FFN)

**Forward Pass Steps:**

```text
1. Self-Attention:     attn1 = self_attention(input, input, input, causal_mask)
2. Add & Norm:         norm1_out = norm1(input + attn1)
3. Cross-Attention:    attn2 = cross_attention(norm1_out, encoder_out, encoder_out, padding_mask)
4. Add & Norm:         norm2_out = norm2(norm1_out + attn2)
5. Feed-Forward:       ff_out = feed_forward(norm2_out)
6. Add & Norm:         output = norm3(norm2_out + ff_out)
```

---

### 2. LanguageModelHead Class

**Purpose:** Project decoder output to vocabulary logits for next-token prediction.

**File:** `LanguageModelHead.hpp`, `LanguageModelHead.cpp`

**Interface:**

```cpp
class LanguageModelHead {
private:
    int d_model;      // Input dimension
    int vocab_size;   // Output vocabulary size

    // Learnable parameters
    Matrix W_output;  // [d_model, vocab_size]
    Matrix bias;      // [1, vocab_size]

    // Gradients
    Matrix W_output_grad;
    Matrix bias_grad;

    // Cached for backward pass
    Matrix cached_input;

public:
    float learning_rate;

    /**
     * Constructor
     *
     * @param d_model Model dimension
     * @param vocab_size Vocabulary size
     */
    LanguageModelHead(int d_model, int vocab_size);

    /**
     * Forward pass: Project to vocabulary logits
     *
     * @param input Decoder output [seq_len, d_model]
     * @return Logits [seq_len, vocab_size]
     */
    Matrix forward(const Matrix& input);

    /**
     * Get probability distribution for next token prediction
     *
     * @param logits Output logits [vocab_size] (single position)
     * @return Probability distribution [vocab_size]
     */
    std::vector<float> get_probabilities(const std::vector<float>& logits);

    /**
     * Backward pass: Compute gradients
     *
     * @param grad_output Gradient from loss [seq_len, vocab_size]
     * @return Gradient w.r.t. input [seq_len, d_model]
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

**Key Design Decisions:**

1. **Simple linear projection**: `output = input * W + b`
2. **Softmax separate**: Applied during inference, not in forward pass
3. **Weight tying option**: Could optionally share weights with token embedding (common practice)
4. **Follows existing patterns**: Same structure as `FeedForward` class

---

### 3. LLMDecoder Class

**Purpose:** Complete transformer decoder for sequence generation.

**File:** `Decoder.hpp`, `Decoder.cpp`

**Interface:**

```cpp
class LLMDecoder {
private:
    std::unique_ptr<BPETokenizer> tokenizer;
    std::unique_ptr<TokenEmbedding> token_embedding;
    std::unique_ptr<PositionalEncoding> positional_encoding;
    std::vector<std::unique_ptr<DecoderBlock>> decoder_blocks;
    std::unique_ptr<LayerNorm> final_norm;
    std::unique_ptr<LanguageModelHead> lm_head;

    int vocab_size;
    int d_model;
    int num_layers;
    int num_heads;
    int d_ff;
    int max_seq_length;

    // Training state
    bool requires_grad;
    float learning_rate;

    // Cached for backward pass
    std::vector<int> cached_token_ids;
    Matrix cached_embeddings;
    Matrix cached_pos_encoded;
    std::vector<Matrix> cached_decoder_outputs;
    Matrix cached_encoder_output;

    /**
     * Create causal attention mask (lower triangular)
     *
     * @param seq_len Sequence length
     * @return Causal mask [seq_len, seq_len]
     */
    Matrix create_causal_mask(int seq_len);

public:
    /**
     * Constructor
     *
     * @param vocab_size Vocabulary size
     * @param d_model Model dimension (default: 512)
     * @param num_layers Number of decoder layers (default: 6)
     * @param num_heads Number of attention heads (default: 8)
     * @param d_ff Feed-forward dimension (default: 2048)
     * @param max_seq_length Maximum sequence length (default: 512)
     */
    LLMDecoder(int vocab_size, int d_model = 512, int num_layers = 6,
               int num_heads = 8, int d_ff = 2048, int max_seq_length = 512);

    /**
     * Forward pass through decoder
     *
     * @param token_ids Input token IDs
     * @param encoder_output Output from encoder [enc_seq_len, d_model]
     * @return Logits [seq_len, vocab_size]
     */
    Matrix forward(const std::vector<int>& token_ids, const Matrix& encoder_output);

    /**
     * Forward pass with custom masks
     *
     * @param token_ids Input token IDs
     * @param encoder_output Encoder output
     * @param cross_attn_mask Optional encoder padding mask
     * @return Logits [seq_len, vocab_size]
     */
    Matrix forward_with_mask(const std::vector<int>& token_ids,
                            const Matrix& encoder_output,
                            const Matrix* cross_attn_mask = nullptr);

    /**
     * Generate next token logits (for autoregressive generation)
     *
     * @param token_ids Generated tokens so far
     * @param encoder_output Encoder output for cross-attention
     * @return Next token logits [vocab_size]
     */
    std::vector<float> generate_next_token_logits(const std::vector<int>& token_ids,
                                                   const Matrix& encoder_output);

    /**
     * Backward pass through decoder
     *
     * @param grad_output Gradient from loss [seq_len, vocab_size]
     * @return Gradient w.r.t. encoder output [enc_seq_len, d_model]
     */
    Matrix backward(const Matrix& grad_output);

    /**
     * Update all decoder parameters
     */
    void update_weights();

    /**
     * Zero all gradients
     */
    void zero_grad();

    /**
     * Set learning rate for all components
     */
    void set_learning_rate(float lr);

    /**
     * Enable/disable gradient computation
     */
    void set_requires_grad(bool requires_grad);

    /**
     * Save decoder parameters to directory
     */
    void save(const std::string& directory);

    /**
     * Load decoder parameters from directory
     */
    void load(const std::string& directory);

    /**
     * Get tokenizer reference (for text generation)
     */
    BPETokenizer* get_tokenizer() { return tokenizer.get(); }
};
```

**Key Design Decisions:**

1. **Mirrors LLMEncoder structure**: Same initialization pattern, same component organization
2. **Causal masking built-in**: `create_causal_mask()` helper for autoregressive generation
3. **Encoder integration**: Takes encoder output as input to forward pass
4. **Generation-ready**: `generate_next_token_logits()` for incremental generation
5. **Gradient flow**: Returns gradient w.r.t. encoder output for end-to-end training

---

### 4. TextGenerator Class

**Purpose:** High-level interface for text generation with different decoding strategies.

**File:** `TextGenerator.hpp`, `TextGenerator.cpp`

**Interface:**

```cpp
class TextGenerator {
private:
    LLMEncoder* encoder;
    LLMDecoder* decoder;

    int max_length;
    int bos_token_id;  // Beginning of sequence token
    int eos_token_id;  // End of sequence token
    int pad_token_id;  // Padding token

    /**
     * Sample token from probability distribution
     *
     * @param probs Probability distribution
     * @param temperature Sampling temperature (default: 1.0)
     * @return Sampled token ID
     */
    int sample_token(const std::vector<float>& probs, float temperature = 1.0f);

    /**
     * Apply temperature to logits
     *
     * @param logits Raw logits
     * @param temperature Temperature value
     * @return Temperature-scaled logits
     */
    std::vector<float> apply_temperature(const std::vector<float>& logits,
                                         float temperature);

    /**
     * Top-k filtering
     *
     * @param probs Probability distribution
     * @param k Number of top tokens to keep
     * @return Filtered probabilities
     */
    std::vector<float> top_k_filtering(const std::vector<float>& probs, int k);

    /**
     * Nucleus (top-p) sampling
     *
     * @param probs Probability distribution
     * @param p Cumulative probability threshold
     * @return Filtered probabilities
     */
    std::vector<float> nucleus_filtering(const std::vector<float>& probs, float p);

public:
    /**
     * Constructor
     *
     * @param encoder Pointer to encoder
     * @param decoder Pointer to decoder
     * @param max_length Maximum generation length (default: 100)
     */
    TextGenerator(LLMEncoder* encoder, LLMDecoder* decoder, int max_length = 100);

    /**
     * Set special token IDs
     *
     * @param bos Beginning of sequence token ID
     * @param eos End of sequence token ID
     * @param pad Padding token ID
     */
    void set_special_tokens(int bos, int eos, int pad);

    /**
     * Greedy decoding: Always select highest probability token
     *
     * @param prompt Input text prompt
     * @param max_gen_length Maximum tokens to generate (default: uses constructor value)
     * @return Generated text
     */
    std::string generate_greedy(const std::string& prompt, int max_gen_length = -1);

    /**
     * Sampling with temperature
     *
     * @param prompt Input text prompt
     * @param temperature Sampling temperature (higher = more random)
     * @param max_gen_length Maximum tokens to generate
     * @return Generated text
     */
    std::string generate_sampling(const std::string& prompt,
                                  float temperature = 1.0f,
                                  int max_gen_length = -1);

    /**
     * Top-k sampling
     *
     * @param prompt Input text prompt
     * @param k Number of top tokens to sample from
     * @param temperature Sampling temperature
     * @param max_gen_length Maximum tokens to generate
     * @return Generated text
     */
    std::string generate_top_k(const std::string& prompt,
                               int k = 50,
                               float temperature = 1.0f,
                               int max_gen_length = -1);

    /**
     * Nucleus (top-p) sampling
     *
     * @param prompt Input text prompt
     * @param p Cumulative probability threshold
     * @param temperature Sampling temperature
     * @param max_gen_length Maximum tokens to generate
     * @return Generated text
     */
    std::string generate_nucleus(const std::string& prompt,
                                 float p = 0.9f,
                                 float temperature = 1.0f,
                                 int max_gen_length = -1);

    /**
     * Beam search decoding
     *
     * @param prompt Input text prompt
     * @param beam_width Number of beams to maintain
     * @param max_gen_length Maximum tokens to generate
     * @return Generated text (highest scoring beam)
     */
    std::string generate_beam_search(const std::string& prompt,
                                     int beam_width = 5,
                                     int max_gen_length = -1);
};
```

**Key Design Decisions:**

1. **Separation of concerns**: Generation logic separate from model architecture
2. **Multiple strategies**: Greedy, sampling, top-k, nucleus, beam search
3. **Temperature control**: Standard technique for controlling randomness
4. **Special tokens**: BOS/EOS/PAD handling for proper sequence generation
5. **Non-owning pointers**: Takes encoder/decoder as dependencies (doesn't own them)

---

### 5. EncoderDecoderModel Class

**Purpose:** Integrate encoder and decoder into complete seq2seq model.

**File:** `EncoderDecoderModel.hpp`, `EncoderDecoderModel.cpp`

**Interface:**

```cpp
class EncoderDecoderModel {
private:
    std::unique_ptr<LLMEncoder> encoder;
    std::unique_ptr<LLMDecoder> decoder;
    std::unique_ptr<TextGenerator> generator;

    int vocab_size;
    int d_model;
    float learning_rate;

    /**
     * Compute cross-entropy loss
     *
     * @param logits Model output [seq_len, vocab_size]
     * @param targets Target token IDs
     * @return Average loss value
     */
    float compute_loss(const Matrix& logits, const std::vector<int>& targets);

    /**
     * Compute loss gradient
     *
     * @param logits Model output [seq_len, vocab_size]
     * @param targets Target token IDs
     * @return Gradient w.r.t. logits [seq_len, vocab_size]
     */
    Matrix compute_loss_gradient(const Matrix& logits, const std::vector<int>& targets);

public:
    /**
     * Constructor
     *
     * @param vocab_size Vocabulary size
     * @param d_model Model dimension (default: 512)
     * @param num_layers Number of encoder/decoder layers (default: 6)
     * @param num_heads Number of attention heads (default: 8)
     * @param d_ff Feed-forward dimension (default: 2048)
     * @param max_seq_length Maximum sequence length (default: 512)
     */
    EncoderDecoderModel(int vocab_size, int d_model = 512, int num_layers = 6,
                       int num_heads = 8, int d_ff = 2048, int max_seq_length = 512);

    /**
     * Generate response to input text
     *
     * @param input_text Input prompt/question
     * @param strategy Generation strategy: "greedy", "sampling", "top_k", "nucleus", "beam"
     * @param max_length Maximum response length
     * @param temperature Sampling temperature (for sampling strategies)
     * @return Generated response text
     */
    std::string generate_response(const std::string& input_text,
                                  const std::string& strategy = "greedy",
                                  int max_length = 100,
                                  float temperature = 1.0f);

    /**
     * Training step on single (input, target) pair
     *
     * @param input_text Input text
     * @param target_text Target output text
     * @return Loss value
     */
    float train_step(const std::string& input_text, const std::string& target_text);

    /**
     * Training on batch of examples
     *
     * @param input_texts Vector of input texts
     * @param target_texts Vector of target texts
     * @return Average loss over batch
     */
    float train_batch(const std::vector<std::string>& input_texts,
                     const std::vector<std::string>& target_texts);

    /**
     * Set learning rate for all components
     */
    void set_learning_rate(float lr);

    /**
     * Save complete model to directory
     */
    void save(const std::string& directory);

    /**
     * Load complete model from directory
     */
    void load(const std::string& directory);

    /**
     * Get encoder reference (for fine-tuning or feature extraction)
     */
    LLMEncoder* get_encoder() { return encoder.get(); }

    /**
     * Get decoder reference (for fine-tuning)
     */
    LLMDecoder* get_decoder() { return decoder.get(); }

    /**
     * Get generator reference (for custom generation)
     */
    TextGenerator* get_generator() { return generator.get(); }
};
```

**Key Design Decisions:**

1. **High-level interface**: Abstracts encoder+decoder complexity
2. **Training support**: Single-step and batch training methods
3. **Flexible generation**: Multiple generation strategies via single interface
4. **Cross-entropy loss**: Standard for language modeling
5. **Model persistence**: Complete save/load functionality

---

## Code Reuse Strategy

### Existing Components to Reuse

| Component | Purpose | Usage in Decoder |
| ----------- | --------- | ----------------- |
| `MultiHeadAttention` | Attention mechanism | Self-attention & cross-attention in DecoderBlock |
| `FeedForward` | Position-wise FFN | Feed-forward layer in DecoderBlock |
| `LayerNorm` | Normalization | 3 instances per DecoderBlock |
| `TokenEmbedding` | Token→vector mapping | Decoder input embeddings |
| `PositionalEncoding` | Position information | Add to decoder embeddings |
| `BPETokenizer` | Text tokenization | Shared with encoder |
| `Activation` | Softmax/GELU | Output probabilities |
| `Matrix` | Tensor operations | All mathematical operations |

### New Components Required

1. **DecoderBlock** - New class (combines existing components)
2. **LanguageModelHead** - New class (simple linear projection)
3. **LLMDecoder** - New class (orchestrates decoder blocks)
4. **TextGenerator** - New class (generation strategies)
5. **EncoderDecoderModel** - New class (integrates encoder+decoder)

**Code Reuse Percentage:** ~70% reused, ~30% new

---

## Masking Strategy

### Self-Attention Mask (Causal)

**Purpose:** Prevent decoder from attending to future positions

**Implementation:**

```cpp
Matrix LLMDecoder::create_causal_mask(int seq_len) {
    Matrix mask(seq_len, seq_len);

    // Lower triangular matrix: mask(i,j) = 1 if j <= i, else -inf
    for (int i = 0; i < seq_len; ++i) {
        for (int j = 0; j < seq_len; ++j) {
            if (j <= i) {
                mask(i, j) = 0.0f;  // Allow attention
            } else {
                mask(i, j) = -1e9f;  // Block attention (effectively -inf)
            }
        }
    }

    return mask;
}
```

**Example (seq_len = 4):**

```text
Position:  0    1    2    3
    0   [ 0   -∞   -∞   -∞ ]  Token 0 only sees itself
    1   [ 0    0   -∞   -∞ ]  Token 1 sees 0,1
    2   [ 0    0    0   -∞ ]  Token 2 sees 0,1,2
    3   [ 0    0    0    0 ]  Token 3 sees all
```

### Cross-Attention Mask (Padding)

**Purpose:** Prevent decoder from attending to encoder padding tokens

**Implementation:**

```cpp
Matrix create_padding_mask(const std::vector<int>& token_ids, int pad_token_id) {
    int seq_len = token_ids.size();
    Matrix mask(1, seq_len);

    for (int i = 0; i < seq_len; ++i) {
        if (token_ids[i] == pad_token_id) {
            mask(0, i) = -1e9f;  // Block attention to padding
        } else {
            mask(0, i) = 0.0f;   // Allow attention
        }
    }

    return mask;
}
```

---

## Training Flow

### Teacher Forcing (Standard Training)

```text
1. Encode input:
   encoder_output = encoder.encode(input_text)

2. Prepare decoder input:
   decoder_input = [<BOS>] + target_tokens[:-1]  // Shift right
   decoder_target = target_tokens

3. Forward pass:
   logits = decoder.forward(decoder_input, encoder_output)

4. Compute loss:
   loss = cross_entropy(logits, decoder_target)

5. Backward pass:
   grad_encoder = decoder.backward(loss_gradient)
   encoder.backward(grad_encoder)

6. Update weights:
   encoder.update_weights()
   decoder.update_weights()
```

### Inference (Autoregressive Generation)

```text
1. Encode input:
   encoder_output = encoder.encode(input_text)

2. Initialize decoder input:
   generated_tokens = [<BOS>]

3. Generate tokens one by one:
   while len(generated_tokens) < max_length:
       a. Get next token logits:
          logits = decoder.generate_next_token_logits(generated_tokens, encoder_output)

       b. Sample/select token:
          next_token = sample(logits, temperature)

       c. Append to sequence:
          generated_tokens.append(next_token)

       d. Stop if EOS:
          if next_token == <EOS>: break

4. Decode to text:
   output_text = tokenizer.decode(generated_tokens)
```

---

## File Structure

### New Files to Create

```text
src/
├── DecoderBlock.hpp              # DecoderBlock class declaration
├── DecoderBlock.cpp              # DecoderBlock implementation
├── LanguageModelHead.hpp         # LM head class declaration
├── LanguageModelHead.cpp         # LM head implementation
├── Decoder.hpp                   # LLMDecoder class declaration
├── Decoder.cpp                   # LLMDecoder implementation
├── TextGenerator.hpp             # Text generation class declaration
├── TextGenerator.cpp             # Text generation implementation
├── EncoderDecoderModel.hpp       # Full model class declaration
├── EncoderDecoderModel.cpp       # Full model implementation
├── DecoderBlockExample.cpp       # Example usage (optional)
├── DecoderExample.cpp            # Example usage (optional)
└── EncoderDecoderExample.cpp     # Example usage (optional)

tests/
├── decoderblock_test.cpp         # DecoderBlock unit tests
├── decoder_test.cpp              # LLMDecoder unit tests
├── textgenerator_test.cpp        # TextGenerator unit tests
└── encoderdecoder_test.cpp       # Integration tests

Context Documentation/
├── DECODERBLOCK_CONTEXT.md       # DecoderBlock documentation
├── DECODER_CONTEXT.md            # LLMDecoder documentation
├── LANGUAGEMODELHEAD_CONTEXT.md  # LM head documentation
├── TEXTGENERATOR_CONTEXT.md      # Text generation documentation
└── ENCODERDECODER_CONTEXT.md     # Full model documentation
```

---

## Implementation Phases

### Phase 1: Core Components (Foundation)

1. Implement `LanguageModelHead` class
2. Implement `DecoderBlock` class
3. Write unit tests for both components
4. Create context documentation

**Deliverables:**

- `LanguageModelHead.hpp/cpp`
- `DecoderBlock.hpp/cpp`
- Unit tests
- Documentation

### Phase 2: Decoder Architecture

1. Implement `LLMDecoder` class
2. Implement causal masking
3. Test decoder forward/backward passes
4. Create example programs

**Deliverables:**

- `Decoder.hpp/cpp`
- `DecoderExample.cpp`
- Integration tests
- Documentation

### Phase 3: Text Generation

1. Implement `TextGenerator` class
2. Implement generation strategies (greedy, sampling, beam search)
3. Test generation quality
4. Create generation examples

**Deliverables:**

- `TextGenerator.hpp/cpp`
- Generation strategy tests
- Documentation

### Phase 4: Integration

1. Implement `EncoderDecoderModel` class
2. Implement training loop
3. Create end-to-end chatbot example
4. Performance testing

**Deliverables:**

- `EncoderDecoderModel.hpp/cpp`
- `EncoderDecoderExample.cpp`
- Complete chatbot application
- Documentation

---

## Testing Strategy

### Unit Tests

1. **DecoderBlock Tests:**
   - Forward pass shape verification
   - Backward pass gradient checking
   - Component integration
   - Masking correctness

2. **LanguageModelHead Tests:**
   - Output projection correctness
   - Gradient computation
   - Probability distribution validity

3. **LLMDecoder Tests:**
   - Causal masking generation
   - Multi-layer stacking
   - Gradient flow through layers

4. **TextGenerator Tests:**
   - Greedy decoding determinism
   - Sampling randomness
   - Special token handling
   - Generation termination

### Integration Tests

1. **Encoder-Decoder Integration:**
   - End-to-end forward pass
   - End-to-end backward pass
   - Training step correctness

2. **Generation Quality:**
   - Coherence tests
   - Length control
   - Repetition detection

---

## Performance Considerations

### Memory Optimization

- Use cached values efficiently
- Clear caches when not needed
- Consider batch processing for multiple sequences

### Computational Efficiency

- Reuse attention masks across batches
- Cache encoder output during generation
- Optimize matrix operations for specific sizes

### Future Optimizations

- KV-cache for autoregressive generation
- Parallel beam search
- Mixed precision training (float16/float32)

---

## Example Usage

### Simple Chatbot

```cpp
#include "EncoderDecoderModel.hpp"

int main() {
    // Create model
    EncoderDecoderModel chatbot(
        vocab_size=10000,
        d_model=256,
        num_layers=4,
        num_heads=8,
        d_ff=1024,
        max_seq_length=256
    );

    // Training examples
    std::vector<std::string> inputs = {
        "Hello, how are you?",
        "What is your name?",
        "Tell me a joke."
    };

    std::vector<std::string> targets = {
        "I'm doing great, thanks for asking!",
        "I'm an AI assistant created to help you.",
        "Why did the chicken cross the road? To get to the other side!"
    };

    // Train
    for (int epoch = 0; epoch < 10; ++epoch) {
        float loss = chatbot.train_batch(inputs, targets);
        std::cout << "Epoch " << epoch << ", Loss: " << loss << std::endl;
    }

    // Save model
    chatbot.save("./chatbot_model");

    // Generate response
    std::string user_input = "How are you today?";
    std::string response = chatbot.generate_response(
        user_input,
        "nucleus",  // Use nucleus sampling
        max_length=50,
        temperature=0.8
    );

    std::cout << "User: " << user_input << std::endl;
    std::cout << "Bot: " << response << std::endl;

    return 0;
}
```

---

## Compatibility with Existing Code

### No Breaking Changes

- All existing encoder code remains unchanged
- Existing components work independently
- Decoder can be added without modifying encoder

### Integration Points

- Shared `BPETokenizer` instance (optional)
- Encoder output fed to decoder cross-attention
- Shared vocabulary and token embeddings (optional weight tying)

### CMakeLists.txt Updates

```cmake
# Add decoder source files
add_library(decoder
    src/DecoderBlock.cpp
    src/LanguageModelHead.cpp
    src/Decoder.cpp
    src/TextGenerator.cpp
    src/EncoderDecoderModel.cpp
)

# Link decoder with existing components
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
```

---

## Summary

This design provides a **complete transformer decoder** that:

✅ **Follows existing patterns**: Uses same design principles as encoder
✅ **Maximizes code reuse**: 70% of components already exist
✅ **Integrates seamlessly**: No breaking changes to existing code
✅ **Production-ready**: Includes training, inference, and persistence
✅ **Well-documented**: Clear interfaces and usage examples
✅ **Testable**: Comprehensive unit and integration tests
✅ **Flexible**: Multiple generation strategies supported
✅ **Complete**: Enables full chatbot functionality

**Next Steps:**

1. Review and approve design
2. Begin Phase 1 implementation (LanguageModelHead + DecoderBlock)
3. Iterate through phases with testing
4. Create end-to-end chatbot application
