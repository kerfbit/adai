# Encoder vs Decoder - Detailed Comparison

Comprehensive comparison between the existing LLMEncoder and the new LLMDecoder design.

---

## High-Level Comparison

| Aspect | LLMEncoder | LLMDecoder |
| -------- | ----------- | ----------- |
| **Purpose** | Encode input text to contextualized representations | Generate output text autoregressively |
| **Attention Type** | Bidirectional self-attention | Causal self-attention + cross-attention |
| **Input** | Source text (e.g., question) | Target sequence prefix (during training) or generated tokens (during inference) |
| **Output** | Fixed-size embeddings [seq_len × d_model] | Vocabulary logits [seq_len × vocab_size] |
| **Masking** | Optional padding mask only | Mandatory causal mask + optional padding mask |
| **Sub-layers per block** | 2 (self-attention, feed-forward) | 3 (self-attention, cross-attention, feed-forward) |
| **Dependencies** | Independent (only input text) | Depends on encoder output |
| **Use Case** | Feature extraction, embeddings | Text generation, sequence-to-sequence |

---

## Component-Level Comparison

### Shared Components (No Changes Needed)

| Component | Used in Encoder | Used in Decoder | Notes |
| ----------- | ---------------- | ---------------- | ------- |
| `BPETokenizer` | ✅ Yes | ✅ Yes | Can be shared instance or separate |
| `TokenEmbedding` | ✅ Yes | ✅ Yes | Decoder has its own instance |
| `PositionalEncoding` | ✅ Yes | ✅ Yes | Can be shared or separate |
| `MultiHeadAttention` | ✅ Yes (1× per block) | ✅ Yes (2× per block) | Used for both self & cross-attention |
| `FeedForward` | ✅ Yes | ✅ Yes | Same position-wise FFN |
| `LayerNorm` | ✅ Yes (2× per block) | ✅ Yes (3× per block) | More instances in decoder |
| `Activation` | ✅ Yes | ✅ Yes | GELU, Softmax |
| `Matrix` | ✅ Yes | ✅ Yes | All operations |

### New Components (Decoder Only)

| Component | Purpose | Equivalent in Encoder |
| ----------- | --------- | ---------------------- |
| `DecoderBlock` | Single decoder layer | `EncoderBlock` |
| `LanguageModelHead` | Project to vocabulary | None (encoder outputs embeddings) |
| `TextGenerator` | Generation strategies | None |
| `EncoderDecoderModel` | Integration | None (encoder standalone) |

---

## Block-Level Comparison

### EncoderBlock Architecture

```text
Input [seq_len × d_model]
    ↓
┌─────────────────────────┐
│ Self-Attention          │  ← Bidirectional (sees all positions)
│ (Q = K = V = input)     │
└─────────────────────────┘
    ↓
Add & Norm (LayerNorm 1)
    ↓
┌─────────────────────────┐
│ Feed-Forward Network    │
│ (2 linear layers)       │
└─────────────────────────┘
    ↓
Add & Norm (LayerNorm 2)
    ↓
Output [seq_len × d_model]
```

**Total Sub-layers:** 2
**Attention Layers:** 1 (self-attention)
**LayerNorm Layers:** 2

### DecoderBlock Architecture

```text
Input [seq_len × d_model]      Encoder Output [enc_len × d_model]
    ↓                                      │
┌─────────────────────────┐               │
│ Masked Self-Attention   │  ← Causal     │
│ (Q = K = V = input)     │  (only past)  │
└─────────────────────────┘               │
    ↓                                      │
Add & Norm (LayerNorm 1)                   │
    ↓                                      │
┌──────────────────────────────────────────┘
│ Cross-Attention          │
│ Q = decoder              │
│ K = V = encoder_output   │
└─────────────────────────┘
    ↓
Add & Norm (LayerNorm 2)
    ↓
┌─────────────────────────┐
│ Feed-Forward Network    │
│ (2 linear layers)       │
└─────────────────────────┘
    ↓
Add & Norm (LayerNorm 3)
    ↓
Output [seq_len × d_model]
```

**Total Sub-layers:** 3
**Attention Layers:** 2 (self + cross)
**LayerNorm Layers:** 3

---

## Attention Mechanism Comparison

### Encoder Self-Attention (Bidirectional)

```cpp
// All positions can attend to all positions
Matrix attention_output = self_attention->forward(
    input,     // Q
    input,     // K
    input,     // V
    mask       // Optional padding mask only
);

// Example attention pattern (4 tokens):
//      To: [T0]  [T1]  [T2]  [T3]
// From T0: [✓]   [✓]   [✓]   [✓]   ← Sees all
//      T1: [✓]   [✓]   [✓]   [✓]   ← Sees all
//      T2: [✓]   [✓]   [✓]   [✓]   ← Sees all
//      T3: [✓]   [✓]   [✓]   [✓]   ← Sees all
```

**Characteristics:**

- ✓ Full context (bidirectional)
- ✓ Rich representations
- ✗ Cannot be used for generation (sees future)

### Decoder Self-Attention (Causal)

```cpp
// Each position can only attend to past positions
Matrix causal_mask = create_causal_mask(seq_len);
Matrix attention_output = self_attention->forward(
    input,        // Q
    input,        // K
    input,        // V
    &causal_mask  // Mandatory causal mask
);

// Example attention pattern (4 tokens):
//      To: [T0]  [T1]  [T2]  [T3]
// From T0: [✓]   [✗]   [✗]   [✗]   ← Sees only T0
//      T1: [✓]   [✓]   [✗]   [✗]   ← Sees T0, T1
//      T2: [✓]   [✓]   [✓]   [✗]   ← Sees T0, T1, T2
//      T3: [✓]   [✓]   [✓]   [✓]   ← Sees all
```

**Characteristics:**

- ✓ Autoregressive (can generate)
- ✓ No future information leakage
- ✗ Less context per position

### Decoder Cross-Attention (Encoder-Decoder)

```cpp
// Decoder queries encoder's output
Matrix cross_attention_output = cross_attention->forward(
    decoder_state,   // Q (from decoder)
    encoder_output,  // K (from encoder)
    encoder_output,  // V (from encoder)
    padding_mask     // Optional encoder padding mask
);

// Example: Decoder attending to encoder
// Encoder: ["What", "is", "the", "weather"]
// Decoder position "sunny" can attend to all encoder tokens:
//
//   "sunny" → [What: 0.3, is: 0.2, the: 0.1, weather: 0.4]
//                                              ↑
//                                   Most relevant for "sunny"
```

**Characteristics:**

- ✓ Conditions on input
- ✓ Full encoder context available
- ✓ Enables seq2seq tasks

---

## Forward Pass Comparison

### LLMEncoder Forward Pass

```cpp
Matrix LLMEncoder::encode(const std::string& text) {
    // 1. Tokenize
    std::vector<int> token_ids = tokenizer->encode(text, true);

    // 2. Embed
    Matrix embeddings = token_embedding->forward(token_ids);

    // 3. Add positional encoding
    Matrix encoded = positional_encoding->forward(embeddings);

    // 4. Pass through encoder blocks
    for (int i = 0; i < num_layers; ++i) {
        encoded = encoder_blocks[i]->forward(encoded);
        // No encoder output needed (self-contained)
    }

    // 5. Final layer norm
    encoded = final_norm->forward(encoded);

    // 6. Return embeddings [seq_len × d_model]
    return encoded;
}
```

**Input:** Text string
**Output:** Embeddings matrix
**Dependencies:** None (self-contained)

### LLMDecoder Forward Pass

```cpp
Matrix LLMDecoder::forward(const std::vector<int>& token_ids,
                          const Matrix& encoder_output) {
    // 1. Embed (tokens already from tokenizer)
    Matrix embeddings = token_embedding->forward(token_ids);

    // 2. Add positional encoding
    Matrix decoded = positional_encoding->forward(embeddings);

    // 3. Create causal mask
    Matrix causal_mask = create_causal_mask(token_ids.size());

    // 4. Pass through decoder blocks
    for (int i = 0; i < num_layers; ++i) {
        decoded = decoder_blocks[i]->forward(
            decoded,         // Decoder state
            encoder_output,  // From encoder (cross-attention)
            causal_mask      // Causal mask
        );
    }

    // 5. Final layer norm
    decoded = final_norm->forward(decoded);

    // 6. Project to vocabulary
    Matrix logits = lm_head->forward(decoded);

    // 7. Return logits [seq_len × vocab_size]
    return logits;
}
```

**Input:** Token IDs + Encoder output
**Output:** Vocabulary logits
**Dependencies:** Requires encoder output

---

## Training Comparison

### Encoder Training (Feature Extraction)

```cpp
// Typically trained as part of encoder-decoder
// or with task-specific head

// Example: Sentence classification
Matrix encoder_output = encoder.encode(text);
Matrix sentence_emb = mean_pool(encoder_output);  // [d_model]
Matrix logits = classifier_head->forward(sentence_emb);
float loss = cross_entropy(logits, label);

// Backward
Matrix grad = compute_gradient(loss);
encoder.backward(grad);
encoder.update_weights();
```

**Training Mode:** Supervised (with task-specific objective)
**Gradient Source:** Classification/regression loss
**Update Frequency:** Per batch

### Decoder Training (Language Modeling)

```cpp
// Teacher forcing: use ground truth tokens
std::vector<int> target_ids = tokenizer->encode(target_text);
std::vector<int> decoder_input = shift_right(target_ids);  // Add <BOS>

// Forward
Matrix encoder_output = encoder.encode(input_text);
Matrix logits = decoder.forward(decoder_input, encoder_output);

// Loss (cross-entropy)
float loss = cross_entropy(logits, target_ids);

// Backward
Matrix grad_logits = compute_gradient(loss, target_ids);
Matrix grad_encoder = decoder.backward(grad_logits);
encoder.backward(grad_encoder);

// Update both
decoder.update_weights();
encoder.update_weights();
```

**Training Mode:** Sequence-to-sequence (teacher forcing)
**Gradient Source:** Next-token prediction loss
**Update Frequency:** Per batch (both encoder & decoder)

---

## Inference Comparison

### Encoder Inference (One-Shot)

```cpp
// Single forward pass produces all embeddings
Matrix embeddings = encoder.encode(input_text);

// Use embeddings for downstream tasks:
// - Sentence similarity: cosine(emb1, emb2)
// - Classification: classifier(emb)
// - Retrieval: search(emb, database)
```

**Characteristics:**

- ✓ Single forward pass
- ✓ Parallel processing
- ✓ Fast (O(n) in sequence length)
- ✓ Deterministic output

### Decoder Inference (Autoregressive)

```cpp
// Encode input once
Matrix encoder_output = encoder.encode(input_text);

// Initialize with <BOS>
std::vector<int> generated = {bos_token_id};

// Generate token by token
for (int step = 0; step < max_length; ++step) {
    // Forward pass (gets longer each step)
    Matrix logits = decoder.forward(generated, encoder_output);

    // Sample/select next token
    int next_token = sample(logits);
    generated.push_back(next_token);

    if (next_token == eos_token_id) break;
}

// Decode to text
std::string output = tokenizer->decode(generated);
```

**Characteristics:**

- ✗ Multiple forward passes (one per token)
- ✗ Sequential processing
- ✗ Slower (O(n²) due to repeated decoding)
- ⚠ Non-deterministic (with sampling)

---

## Parameter Count Comparison

### LLMEncoder Parameters

For d_model=512, num_layers=6, num_heads=8, d_ff=2048, vocab_size=10000:

| Component | Parameters | Notes |
| ----------- | ----------- | ------- |
| Token Embedding | 10000 × 512 = 5.12M | Vocab → d_model |
| Positional Encoding | 0 (pre-computed) | Not learnable |
| **Per EncoderBlock:** |  |  |
| - MultiHeadAttention | 4 × (512 × 512) = 1.05M | W_q, W_k, W_v, W_o |
| - FeedForward | 2 × (512 × 2048) = 2.10M | W1, W2 |
| - LayerNorm (×2) | 2 × (2 × 512) = 2.05K | gamma, beta |
| **Total per block** | ~3.15M |  |
| **All 6 blocks** | 18.9M |  |
| Final LayerNorm | 1.02K |  |
| **TOTAL ENCODER** | **~24M parameters** |  |

### LLMDecoder Parameters

Same hyperparameters as encoder:

| Component | Parameters | Notes |
| ----------- | ----------- | ------- |
| Token Embedding | 10000 × 512 = 5.12M | Separate from encoder |
| Positional Encoding | 0 (pre-computed) |  |
| **Per DecoderBlock:** |  |  |
| - Self-Attention | 4 × (512 × 512) = 1.05M |  |
| - Cross-Attention | 4 × (512 × 512) = 1.05M | **Extra** |
| - FeedForward | 2 × (512 × 2048) = 2.10M |  |
| - LayerNorm (×3) | 3 × (2 × 512) = 3.07K | **Extra norm** |
| **Total per block** | ~4.20M | ~33% more than encoder |
| **All 6 blocks** | 25.2M |  |
| Final LayerNorm | 1.02K |  |
| Language Model Head | 512 × 10000 = 5.12M | **New component** |
| **TOTAL DECODER** | **~35M parameters** |  |

### Complete Encoder-Decoder Model

| Component | Parameters |
| ----------- | ----------- |
| Encoder | 24M |
| Decoder | 35M |
| **TOTAL** | **~59M parameters** |

**Note:** Parameters can be reduced by:

- Weight tying (share token embeddings): -5M
- Shared positional encoding: 0 (already shared)
- Smaller d_ff: proportional reduction

---

## Memory Usage Comparison

### Encoder Memory (Inference)

| Component | Size (seq_len=256, d_model=512) |
| ----------- | -------------------------------- |
| Input embeddings | 256 × 512 × 4B = 512KB |
| Per block cache | ~5 matrices × 512KB = 2.5MB |
| 6 blocks total | 6 × 2.5MB = 15MB |
| **Total activations** | **~16MB** |

### Decoder Memory (Inference, autoregressive)

| Component | Size (max_gen=100, d_model=512) |
| ----------- | -------------------------------- |
| Encoder output (cached) | 256 × 512 × 4B = 512KB |
| Decoder embeddings (growing) | 100 × 512 × 4B = 200KB |
| Per block cache | ~7 matrices × 200KB = 1.4MB |
| 6 blocks total | 6 × 1.4MB = 8.4MB |
| Logits | 100 × 10000 × 4B = 4MB |
| **Total activations** | **~13MB** |

**Total Encoder + Decoder:** ~29MB (activations only, not including parameters)

---

## Use Case Comparison

### When to Use Encoder Only

✅ **Sentence Classification**

```cpp
Matrix emb = encoder.encode("This movie is great!");
int sentiment = classify(emb);  // Positive/Negative
```

✅ **Semantic Search**

```cpp
Matrix query_emb = encoder.encode("machine learning");
std::vector<float> scores = search(query_emb, document_embeddings);
```

✅ **Named Entity Recognition**

```cpp
Matrix token_embs = encoder.encode("John lives in Paris");
std::vector<std::string> entities = ner_tagger(token_embs);
```

### When to Use Decoder Only (GPT-style)

✅ **Text Completion**

```cpp
std::string completed = decoder_only.generate("Once upon a time");
// Output: "Once upon a time, there was a brave knight..."
```

⚠ **Requires redesign** - Current encoder not suitable for decoder-only

### When to Use Encoder-Decoder

✅ **Machine Translation**

```cpp
EncoderDecoderModel translator;
std::string french = translator.generate_response(
    "Hello, how are you?",  // English input
    "greedy"
);
// Output: "Bonjour, comment allez-vous?"
```

✅ **Chatbot / Question Answering**

```cpp
EncoderDecoderModel chatbot;
std::string answer = chatbot.generate_response(
    "What is the capital of France?",
    "nucleus"
);
// Output: "The capital of France is Paris."
```

✅ **Summarization**

```cpp
EncoderDecoderModel summarizer;
std::string summary = summarizer.generate_response(
    long_document,
    "beam_search",
    max_length=100
);
```

✅ **Dialogue Generation**

```cpp
EncoderDecoderModel dialogue;
std::string reply = dialogue.generate_response(
    "How's the weather today?",
    "sampling",
    temperature=0.7
);
// Output: "It's quite sunny and pleasant!"
```

---

## Code Reuse Summary

### Components Fully Reused (No Changes)

1. ✅ `Matrix` - All tensor operations
2. ✅ `MultiHeadAttention` - Used for both self & cross-attention
3. ✅ `FeedForward` - Position-wise FFN
4. ✅ `LayerNorm` - Normalization layers
5. ✅ `TokenEmbedding` - Token → vector mapping
6. ✅ `PositionalEncoding` - Position information
7. ✅ `BPETokenizer` - Text tokenization
8. ✅ `Activation` - GELU, Softmax

**Reuse Rate:** 8/13 components = **~62%**

### New Components Required

1. 🆕 `DecoderBlock` - Combines existing components with new pattern
2. 🆕 `LanguageModelHead` - Simple linear projection (follows existing patterns)
3. 🆕 `LLMDecoder` - Orchestrates decoder blocks (mirrors LLMEncoder)
4. 🆕 `TextGenerator` - Generation logic (new functionality)
5. 🆕 `EncoderDecoderModel` - Integration layer

**New Code:** 5 components = **~38%**

---

## Performance Comparison

### Encoder Performance

| Metric | Value (CPU, single-threaded) |
| -------- | ------------------------------ |
| Throughput | ~5000 tokens/sec |
| Latency | O(n) with sequence length |
| Memory | O(n × d_model) |
| Parallelization | Easy (batch processing) |

### Decoder Performance

| Metric | Value (CPU, single-threaded) |
| -------- | ------------------------------ |
| Throughput | ~50-100 tokens/sec (generation) |
| Latency | O(n²) with sequence length |
| Memory | O(n × d_model) per step |
| Parallelization | Hard (sequential generation) |

**Why Slower?**

- Sequential generation (can't parallelize across tokens)
- Repeated forward passes (one per generated token)
- Growing sequence length

---

## Migration Path

### Current State (Encoder Only)

```text
Input Text → Encoder → Embeddings → Classifier → Output Label
```

### After Adding Decoder

```text
Option 1: Keep encoder standalone
Input Text → Encoder → Embeddings → Classifier → Output Label

Option 2: Use encoder-decoder
Input Text → Encoder → Decoder → Generated Text
```

**No Breaking Changes:**

- Existing encoder code unchanged
- Encoder can still be used independently
- Decoder is additive functionality

---

## Summary Table

| Feature | Encoder | Decoder |
| --------- | --------- | --------- |
| **Architecture** | Transformer Encoder | Transformer Decoder |
| **Blocks** | EncoderBlock | DecoderBlock |
| **Attention Layers/Block** | 1 (self) | 2 (self + cross) |
| **Attention Type** | Bidirectional | Causal + Cross |
| **LayerNorm/Block** | 2 | 3 |
| **Parameters/Block** | ~3.15M | ~4.20M |
| **Output** | Embeddings | Logits |
| **Output Head** | None (optional task-specific) | Language Model Head |
| **Masking** | Optional (padding) | Mandatory (causal) |
| **Dependencies** | None | Encoder output |
| **Inference** | One-shot | Autoregressive |
| **Speed** | Fast (parallel) | Slower (sequential) |
| **Use Cases** | Classification, embeddings | Generation, seq2seq |
| **Code Reuse** | Foundation | Builds on encoder |

---

**Conclusion:** The decoder design maximizes code reuse while adding the critical generation capabilities needed for chatbot functionality. Both components can coexist and be used independently or together.

---

**Document Version:** 1.0
**Last Updated:** January 18, 2026
**Related:** DECODER_DESIGN.md, DECODER_DESIGN_SUMMARY.md
