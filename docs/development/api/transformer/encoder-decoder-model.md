# EncoderDecoderModel - Context Documentation

## Overview

The `EncoderDecoderModel` class implements a complete sequence-to-sequence transformer architecture that combines an encoder and decoder for advanced NLP tasks. This is the highest-level component in the chatbot system, integrating all previously built components into a unified model.

## Purpose

**Primary Function**: Enable sequence-to-sequence transformations where the output depends on understanding and transforming an input sequence.

**Key Applications**:

- **Chatbot Conversations**: Generate contextual responses to user inputs
- **Machine Translation**: Translate text from one language to another
- **Text Summarization**: Generate concise summaries of long documents
- **Question Answering**: Produce answers based on context
- **Dialog Systems**: Multi-turn conversation management

## Architecture

### High-Level Pipeline

```text
Input Text
    ↓
[Tokenization]
    ↓
Token IDs → [Encoder] → Context Representation
                              ↓
                        (Cross-Attention)
                              ↓
<BOS> → [Decoder] → [LM Head] → Token Probabilities → Token₁
           ↓
Token₁ → [Decoder] → [LM Head] → Token Probabilities → Token₂
           ↓
Token₂ → [Decoder] → [LM Head] → Token Probabilities → Token₃
           ↓
         ...
           ↓
Token_n → [Decoder] → [LM Head] → <EOS> → STOP
    ↓
[Detokenization]
    ↓
Output Text
```

### Component Integration

```text
┌─────────────────────────────────────────────────────┐
│           EncoderDecoderModel                       │
├─────────────────────────────────────────────────────┤
│                                                     │
│  ┌──────────────┐                                   │
│  │ BPETokenizer │ (Shared vocabulary)               │
│  └──────────────┘                                   │
│         ↓                                           │
│  ┌──────────────┐      ┌──────────────┐             │
│  │  LLMEncoder  │ ───→ │  LLMDecoder  │             │
│  │              │      │ (cross-attn) │             │
│  └──────────────┘      └──────────────┘             │
│                               ↓                     │
│                    ┌──────────────────┐             │
│                    │ LanguageModelHead│             │
│                    └──────────────────┘             │
│                               ↓                     │
│                    ┌──────────────────┐             │
│                    │  TextGenerator   │             │
│                    └──────────────────┘             │
│                                                     │
└─────────────────────────────────────────────────────┘
```

### Internal Components

#### 1. BPETokenizer

- **Role**: Convert text ↔ token IDs
- **Shared**: Same vocabulary for encoder and decoder
- **Special Tokens**: `<pad>`, `<bos>`, `<eos>`

#### 2. LLMEncoder

- **Role**: Encode input text into context representations
- **Architecture**: Multi-layer transformer encoder
- **Output**: Context matrix [input_length, d_model]
- **Used**: Once per generation/training step

#### 3. LLMDecoder

- **Role**: Generate output autoregressively
- **Architecture**: Multi-layer transformer decoder with cross-attention
- **Input**: Previous tokens + encoder context
- **Output**: Hidden states [output_length, d_model]
- **Used**: Once per token during generation

#### 4. LanguageModelHead

- **Role**: Project decoder output to vocabulary probabilities
- **Architecture**: Linear layer [d_model → vocab_size]
- **Output**: Logits [output_length, vocab_size]

#### 5. TextGenerator

- **Role**: Implement generation strategies
- **Strategies**: Greedy, beam search, sampling, top-k, nucleus
- **Features**: Temperature, repetition penalty, length control

## Class Structure

### File Location

- Header: `/home/rodney/Repos/adai/src/EncoderDecoderModel.hpp`
- Implementation: `/home/rodney/Repos/adai/src/EncoderDecoderModel.cpp`
- Example: `/home/rodney/Repos/adai/src/EncoderDecoderModelExample.cpp`

### Dependencies

```cpp
#include "encoder.hpp"          // LLMEncoder
#include "Decoder.hpp"          // LLMDecoder
#include "LanguageModelHead.hpp"
#include "TextGenerator.hpp"
#include "BPETokenizer.hpp"
#include "Matrix.hpp"
```

### Class Definition

```cpp
class EncoderDecoderModel {
private:
    // Core components
    std::unique_ptr<BPETokenizer> tokenizer;
    std::unique_ptr<LLMEncoder> encoder;
    std::unique_ptr<LLMDecoder> decoder;
    std::unique_ptr<LanguageModelHead> lm_head;
    std::unique_ptr<TextGenerator> generator;

    // Architecture configuration
    int vocab_size;
    int d_model;
    int encoder_layers;
    int decoder_layers;
    int num_heads;
    int d_ff;
    int max_seq_length;

    // Special tokens
    int bos_token_id;  // Beginning of sequence
    int eos_token_id;  // End of sequence
    int pad_token_id;  // Padding

    // Training state
    bool requires_grad;
    float learning_rate;

    // Cached values
    Matrix cached_encoder_output;
    Matrix cached_decoder_output;
    std::vector<int> cached_input_tokens;
    std::vector<int> cached_target_tokens;

public:
    // Constructor & destructor
    EncoderDecoderModel(int vocab_size, int d_model = 512,
                        int encoder_layers = 6, int decoder_layers = 6,
                        int num_heads = 8, int d_ff = 2048,
                        int max_seq_length = 512);
    ~EncoderDecoderModel();

    // Inference methods
    std::string generate_response(const std::string& input_text, int max_length = 100);
    std::string generate_response_with_strategy(...);

    // Training methods
    float train_step(const std::string& input_text, const std::string& target_text);
    float train_step_tokenized(const std::vector<int>& input_tokens,
                               const std::vector<int>& target_tokens);
    float evaluate(const std::string& input_text, const std::string& target_text);
    float compute_perplexity(const std::vector<std::string>& inputs,
                            const std::vector<std::string>& targets);

    // Configuration
    void set_training(bool mode);
    void set_learning_rate(float lr);
    void set_generation_config(const TextGenerator::GenerationConfig& config);
    void set_tokenizer(BPETokenizer* tokenizer_ptr);

    // Weight management
    void update_weights();
    void zero_grad();
    void save_model(const std::string& filepath) const;
    void load_model(const std::string& filepath);

    // Advanced usage
    Matrix forward(const std::vector<int>& input_tokens,
                  const std::vector<int>& target_tokens);
    void backward(const Matrix& grad_output);

    // Component access
    LLMEncoder* get_encoder();
    LLMDecoder* get_decoder();
    LanguageModelHead* get_lm_head();
    TextGenerator* get_generator();
    BPETokenizer* get_tokenizer();

private:
    float compute_loss(const Matrix& logits, const std::vector<int>& target_tokens);
    Matrix compute_loss_gradient(const Matrix& logits,
                                 const std::vector<int>& target_tokens);
};
```

## Key Methods

### Constructor

```cpp
EncoderDecoderModel(int vocab_size, int d_model = 512,
                    int encoder_layers = 6, int decoder_layers = 6,
                    int num_heads = 8, int d_ff = 2048,
                    int max_seq_length = 512)
```

**Purpose**: Initialize complete encoder-decoder model

**Parameters**:

- `vocab_size`: Size of shared vocabulary
- `d_model`: Model dimension (default: 512)
- `encoder_layers`: Number of encoder layers (default: 6)
- `decoder_layers`: Number of decoder layers (default: 6)
- `num_heads`: Number of attention heads (default: 8)
- `d_ff`: Feed-forward dimension (default: 2048)
- `max_seq_length`: Maximum sequence length (default: 512)

**Initialization**:

1. Creates BPETokenizer (empty, needs vocab building)
2. Creates LLMEncoder with specified configuration
3. Creates LLMDecoder with specified configuration
4. Creates LanguageModelHead(d_model, vocab_size)
5. Creates TextGenerator with default config
6. Sets special tokens: `bos=1`, `eos=2`, `pad=0`
7. Sets `requires_grad=true`, `learning_rate=0.001`

### Inference Methods

#### generate_response()

```cpp
std::string generate_response(const std::string& input_text, int max_length = 100)
```

**Purpose**: Generate response using default/configured strategy

**Process**:

1. Tokenize input text
2. Encode with LLMEncoder → context representation
3. Create model_fn lambda capturing encoder output
4. Call TextGenerator.generate() starting with `<bos>`
5. Decode output tokens to text
6. Return response string

**Model Function**:

```cpp
auto model_fn = [this](const std::vector<int>& tokens) -> Matrix {
    Matrix decoder_out = decoder->forward_with_encoder(tokens, encoder_output);
    Matrix logits = lm_head->forward(decoder_out);
    return logits;
};
```

**Use Case**: Simple inference with configured generation settings

#### generate_response_with_strategy()

```cpp
std::string generate_response_with_strategy(
    const std::string& input_text,
    int max_length = 100,
    const std::string& strategy = "greedy",
    float temperature = 1.0f,
    int top_k = 50,
    float top_p = 0.9f,
    int num_beams = 4
)
```

**Purpose**: Generate with explicit strategy selection

**Strategies**:

- `"greedy"`: Deterministic, pick highest probability token
- `"beam"`: Beam search with `num_beams` beams
- `"sampling"`: Sample from distribution with `temperature`
- `"topk"`: Sample from top-k tokens
- `"nucleus"`: Sample from nucleus (top-p)

**Process**:

1. Encode input (same as generate_response)
2. Create model_fn lambda
3. Call strategy-specific TextGenerator method
4. Decode and return

**Use Case**: Override default strategy for specific generation needs

### Training Methods

#### train_step()

```cpp
float train_step(const std::string& input_text, const std::string& target_text)
```

**Purpose**: Single training iteration on text pair

**Process**:

1. Tokenize input and target texts
2. Call train_step_tokenized()

**Returns**: Cross-entropy loss value

#### train_step_tokenized()

```cpp
float train_step_tokenized(const std::vector<int>& input_tokens,
                           const std::vector<int>& target_tokens)
```

**Purpose**: Training step on pre-tokenized sequences

**Process**:

1. **Check mode**: Verify `requires_grad=true`
2. **Zero gradients**: Clear previous gradients
3. **Forward pass**:
   - Encode input → context
   - Decode with teacher forcing (use target tokens)
   - Project to vocabulary
4. **Compute loss**: Cross-entropy on logits vs targets
5. **Backward pass**: Compute and propagate gradients
6. **Update weights**: Apply gradient descent

**Teacher Forcing**:

```cpp
decoder_input = [<bos>, target[0], target[1], ..., target[n-1]]
decoder_output = decoder.forward_with_encoder(decoder_input, encoder_output)
```

**Returns**: Loss value for this batch

**Use Case**: Training loop iteration

#### evaluate()

```cpp
float evaluate(const std::string& input_text, const std::string& target_text)
```

**Purpose**: Compute loss without updating weights

**Process**:

1. Temporarily disable training mode
2. Forward pass only
3. Compute and return loss
4. Restore original training mode

**Use Case**: Validation during training

#### compute_perplexity()

```cpp
float compute_perplexity(const std::vector<std::string>& input_texts,
                        const std::vector<std::string>& target_texts)
```

**Purpose**: Evaluate model quality on dataset

**Formula**: `perplexity = exp(avg_cross_entropy_loss)`

**Process**:

1. Disable training mode
2. Evaluate each (input, target) pair
3. Compute average loss
4. Return exp(avg_loss)

**Interpretation**:

- Lower perplexity = better model
- Perplexity ≈ "how many tokens model is confused between"

**Use Case**: Model evaluation and comparison

### Loss Computation

#### compute_loss()

```cpp
float compute_loss(const Matrix& logits, const std::vector<int>& target_tokens)
```

**Purpose**: Compute cross-entropy loss

**Formula**:

```text
loss = -mean(log(P(target_token | context)))
     = -mean(log(softmax(logits)[target]))
```

**Implementation**:

1. For each timestep t:
   - Compute softmax: `P = exp(logits - max) / sum(exp(logits - max))`
   - Cross-entropy: `-log(P[target[t]] + epsilon)`
2. Average over sequence length

**Numerical Stability**:

- Subtract max logit before exp (prevents overflow)
- Add epsilon=1e-10 before log (prevents log(0))

#### compute_loss_gradient()

```cpp
Matrix compute_loss_gradient(const Matrix& logits,
                             const std::vector<int>& target_tokens)
```

**Purpose**: Compute gradient of cross-entropy loss

**Formula**: `grad = softmax(logits) - one_hot(target)`

**Implementation**:

1. Compute softmax probabilities
2. Subtract 1.0 at target position
3. Scale by 1/sequence_length

**Use Case**: Backpropagation starting point

### Configuration Methods

#### set_training()

```cpp
void set_training(bool mode)
```

**Purpose**: Switch between training and inference modes

**Effects**:

- Sets `requires_grad` flag
- Propagates to decoder (encoder method not available)
- Affects gradient computation and caching

#### set_learning_rate()

```cpp
void set_learning_rate(float lr)
```

**Purpose**: Update learning rate for gradient descent

**Propagates to**:

- Decoder (encoder method not available)
- Internal learning_rate variable

#### set_generation_config()

```cpp
void set_generation_config(const TextGenerator::GenerationConfig& config)
```

**Purpose**: Configure text generation behavior

**Config Parameters**:

```cpp
struct GenerationConfig {
    int max_length;           // Maximum output length
    int min_length;           // Minimum output length
    int num_beams;            // Beam search width
    float temperature;        // Sampling temperature
    int top_k;                // Top-k filtering
    float top_p;              // Nucleus sampling threshold
    float repetition_penalty; // Penalty for repeated tokens
    int bos_token_id;         // Beginning token
    int eos_token_id;         // End token
    int pad_token_id;         // Padding token
};
```

**Updates**: TextGenerator config + special token IDs

#### set_tokenizer()

```cpp
void set_tokenizer(BPETokenizer* tokenizer_ptr)
```

**Purpose**: Replace default tokenizer with custom one

**Use Case**:

- Load pre-trained tokenizer
- Use domain-specific vocabulary

### Weight Management

#### register_parameters()

```cpp
void register_parameters(Optimizer& optimizer)
```

**Purpose**: Register all model parameters with an external optimizer

**Process**:

1. Register encoder parameters recursively (all layers)
2. Register decoder parameters recursively (all layers)
3. Register language model head parameters

**Registered Components**:

- **Encoder**: Token embeddings, encoder blocks (attention, feed-forward, layer norms), final layer norm
- **Decoder**: Token embeddings, decoder blocks (self-attention, cross-attention, feed-forward, layer norms), final layer norm
- **LM Head**: Output projection weights and biases

**Benefits**:

- Enables advanced optimization (Adam, AdamW, etc.)
- Centralized gradient management
- Gradient clipping and weight decay
- Automatic parameter tracking

**Example**:

```cpp
Optimizer optimizer(OptimizerType::ADAMW, 0.001f);
optimizer.set_betas(0.9f, 0.999f);
optimizer.set_weight_decay(0.01f);
optimizer.set_max_grad_norm(1.0f);

model.register_parameters(optimizer);

// Training loop
for (auto& batch : dataset) {
    optimizer.zero_grad();
    float loss = model.train_step(batch.input, batch.target);
    optimizer.step();
}
```

**Note**: Replaces the simplified `update_weights()` method with full optimizer integration

#### update_weights()

```cpp
void update_weights()
```

**Purpose**: Apply gradient descent to all parameters (legacy method)

**Updates**:

- Decoder weights
- LanguageModelHead weights

**Note**: Uses internal learning_rate. **Deprecated** - use `register_parameters()` with external `Optimizer` instead for production training.

#### zero_grad()

```cpp
void zero_grad()
```

**Purpose**: Clear accumulated gradients

**Clears**:

- Encoder gradients
- Decoder gradients
- LanguageModelHead gradients

**Use Case**: Called before each training iteration

### Persistence

#### save_model()

```cpp
void save_model(const std::string& filepath) const
```

**Purpose**: Save complete model to disk

**Files Created**:

1. `{filepath}.config` - Architecture configuration (binary)
2. `{filepath}.vocab` - Tokenizer vocabulary
3. `{filepath}.encoder` - Encoder weights (partial)
4. `{filepath}.decoder` - Decoder weights (partial)

**Saved in Config**:

- vocab_size, d_model, encoder_layers, decoder_layers
- num_heads, d_ff, max_seq_length
- bos_token_id, eos_token_id, pad_token_id

**Limitations**:

- LanguageModelHead lacks save_weights method (commented out)
- Component save methods incomplete (only config saved)

#### load_model()

```cpp
void load_model(const std::string& filepath)
```

**Purpose**: Load complete model from disk

**Process**:

1. Load and verify architecture config
2. Load tokenizer vocabulary
3. Load encoder weights (partial)
4. Load decoder weights (partial)
5. Update special token IDs

**Validation**: Throws `std::runtime_error` if architecture mismatch

**Limitations**: Same as save_model (incomplete component loading)

### Advanced Methods

#### forward()

```cpp
Matrix forward(const std::vector<int>& input_tokens,
              const std::vector<int>& target_tokens)
```

**Purpose**: Complete forward pass for training

**Process**:

1. **Cache inputs**: Store for backward pass
2. **Encode**:

   ```cpp
   encoder_output = encoder->encode_with_mask(input_tokens, no_mask)
   ```

3. **Prepare decoder input** (teacher forcing):

   ```cpp
   decoder_input = [<bos>, target[0], target[1], ..., target[n-1]]
   ```

4. **Decode**:

   ```cpp
   decoder_output = decoder->forward_with_encoder(decoder_input, encoder_output)
   ```

5. **Project to vocabulary**:

   ```cpp
   logits = lm_head->forward(decoder_output)
   ```

**Returns**: Logits [seq_length, vocab_size]

**Teacher Forcing**: Uses ground truth target tokens as decoder input during training (prevents error accumulation)

#### backward()

```cpp
void backward(const Matrix& grad_output)
```

**Purpose**: Backpropagate gradients through model

**Process**:

1. Check `requires_grad` flag
2. Backward through LM head:

   ```cpp
   grad_decoder = lm_head->backward(grad_output)
   ```

3. Backward through decoder, capturing the summed encoder-side gradient:

   ```cpp
   Matrix grad_encoder_output;
   decoder->backward(grad_decoder, grad_encoder_output)
   ```

4. Backward through encoder:

   ```cpp
   encoder->backward(grad_encoder_output)
   ```

**Note**: The gradient w.r.t. encoder output — computed by `CrossAttention::backward`
as `grad_kv_input`, summed across every decoder block's cross-attention by
`LLMDecoder::backward` — is explicitly propagated into the encoder end-to-end.

## Usage Patterns

### Simple Chatbot

```cpp
// Initialize
EncoderDecoderModel model(
    vocab_size=1000,
    d_model=128,
    encoder_layers=2,
    decoder_layers=2
);

// Build tokenizer
std::vector<std::string> corpus = {"hello", "how are you", ...};
model.get_tokenizer()->build_vocab(corpus, vocab_size);

// Generate response
std::string input = "Hello!";
std::string response = model.generate_response(input, max_length=20);
```

### Training Loop

```cpp
// Prepare training data
std::vector<std::pair<std::string, std::string>> data = {
    {"How are you?", "I'm doing well, thank you!"},
    {"What's your name?", "I'm an AI assistant."},
    // ...
};

// Training
model.set_training(true);
model.set_learning_rate(0.001f);

for (int epoch = 0; epoch < 10; ++epoch) {
    float total_loss = 0.0f;

    for (const auto& [input, target] : data) {
        float loss = model.train_step(input, target);
        total_loss += loss;
    }

    std::cout << "Epoch " << epoch << " Loss: "
              << total_loss / data.size() << std::endl;
}
```

### Multiple Generation Strategies

```cpp
std::string input = "Tell me a story";

// Greedy (deterministic)
std::string greedy = model.generate_response_with_strategy(
    input, 100, "greedy"
);

// Beam search (explores alternatives)
std::string beam = model.generate_response_with_strategy(
    input, 100, "beam", 1.0f, 50, 0.9f, 4
);

// Sampling (creative)
std::string creative = model.generate_response_with_strategy(
    input, 100, "sampling", 1.2f  // Higher temperature = more random
);

// Top-k (balanced)
std::string balanced = model.generate_response_with_strategy(
    input, 100, "topk", 1.0f, 40
);
```

### Model Persistence

```cpp
// After training
model.save_model("chatbot_model");

// Later, load for inference
EncoderDecoderModel loaded_model(vocab_size, d_model, ...);
loaded_model.load_model("chatbot_model");
loaded_model.set_training(false);

std::string response = loaded_model.generate_response("Hello!");
```

### Custom Training Loop

```cpp
// For advanced users who need fine control
model.set_training(true);
model.zero_grad();

// Manual forward
Matrix logits = model.forward(input_tokens, target_tokens);

// Custom loss computation
float custom_loss = my_loss_function(logits, target_tokens);
Matrix custom_grad = my_gradient_function(logits, target_tokens);

// Manual backward
model.backward(custom_grad);

// Manual weight update with custom optimizer
model.update_weights();
```

## Design Decisions

### 1. Shared Vocabulary

**Decision**: Encoder and decoder use same tokenizer/vocabulary

**Rationale**:

- Simplifies architecture (single vocab)
- Enables weight sharing between embeddings
- Sufficient for most seq2seq tasks

**Trade-off**: Cannot use different vocabularies for input/output

### 2. Teacher Forcing in Training

**Decision**: Use target tokens as decoder input during training

**Rationale**:

- Prevents error accumulation during training
- Faster convergence
- Standard practice in seq2seq models

**Alternative**: Scheduled sampling (gradually mix predicted tokens)

### 3. Component Ownership

**Decision**: Use `std::unique_ptr` for all components

**Rationale**:

- Automatic memory management
- Clear ownership semantics
- Prevents accidental copying

### 4. Separate Generation and Training Paths

**Decision**: Distinct methods for inference vs training

**Rationale**:

- Different input preparation (teacher forcing vs autoregressive)
- Different optimization needs
- Clearer API for users

### 5. Multiple Generation Strategies

**Decision**: Expose multiple generation methods

**Rationale**:

- Different use cases need different strategies
- Easy experimentation
- Flexibility for production deployment

### 6. Incomplete Persistence

**Decision**: Save configuration even though component weights not fully saved

**Current State**: Only saves configuration + tokenizer vocab

**Rationale**:

- Component classes lack save/load methods
- Partial functionality better than none
- Clear warning messages

**Future**: Implement full persistence in all components

## Integration Points

### Components Used

1. **BPETokenizer**: Text ↔ tokens conversion
2. **LLMEncoder**: Input encoding
3. **LLMDecoder**: Autoregressive decoding
4. **LanguageModelHead**: Vocabulary projection
5. **TextGenerator**: Generation strategies
6. **Matrix**: Data structures

### Used By

- Application code (chatbot, translation, etc.)
- Training scripts
- Evaluation scripts
- Deployment systems

## Configuration Guidelines

### Minimal Model (Prototyping)

```cpp
EncoderDecoderModel model(
    vocab_size = 500,
    d_model = 64,
    encoder_layers = 1,
    decoder_layers = 1,
    num_heads = 2,
    d_ff = 256,
    max_seq_length = 32
);
```

**Use**: Quick experiments, debugging

### Small Model (Development)

```cpp
EncoderDecoderModel model(
    vocab_size = 1000,
    d_model = 128,
    encoder_layers = 2,
    decoder_layers = 2,
    num_heads = 4,
    d_ff = 512,
    max_seq_length = 64
);
```

**Use**: Local development, unit testing

### Medium Model (Demo/Small Production)

```cpp
EncoderDecoderModel model(
    vocab_size = 5000,
    d_model = 256,
    encoder_layers = 4,
    decoder_layers = 4,
    num_heads = 8,
    d_ff = 1024,
    max_seq_length = 128
);
```

**Use**: Demos, small-scale deployment

### Large Model (Production)

```cpp
EncoderDecoderModel model(
    vocab_size = 30000,
    d_model = 512,
    encoder_layers = 6,
    decoder_layers = 6,
    num_heads = 8,
    d_ff = 2048,
    max_seq_length = 512
);
```

**Use**: Production chatbots, translation systems

## Performance Considerations

### Memory Usage

**Training**:

- Encoder: `encoder_layers × input_len × d_model`
- Decoder: `decoder_layers × output_len × d_model`
- Attention: `num_heads × (input_len² + output_len²)`
- Gradients: ~2× forward pass memory

**Inference**:

- Encoder: Once per conversation turn
- Decoder: Once per generated token
- Can cache encoder output across tokens

### Time Complexity

**Per Training Step**:

- Encoder: `O(encoder_layers × input_len² × d_model)`
- Decoder: `O(decoder_layers × output_len² × d_model)`
- Cross-attention: `O(decoder_layers × input_len × output_len × d_model)`

**Per Inference Token**:

- Encoder: `O(encoder_layers × input_len² × d_model)` (once)
- Decoder: `O(decoder_layers × output_len² × d_model)` (per token)

### Optimization Opportunities

1. **KV Caching**: Cache decoder key/value tensors
2. **Encoder Reuse**: Cache encoder output for multi-turn conversations
3. **Batch Processing**: Process multiple inputs simultaneously
4. **Mixed Precision**: Use FP16 for faster computation
5. **Model Distillation**: Train smaller student model

## Limitations

### Current Implementation

1. **No Batch Support**: Processes one sequence at a time
2. **Incomplete Persistence**: Component weights not fully saved
3. **No Attention Masking**: Encoder uses full attention (no padding masks)
4. **Teacher Forcing Only**: No scheduled sampling or other curricula

### Known Issues

1. **LLMEncoder Methods**: Missing `update_weights()` — `set_requires_grad()`/`set_learning_rate()` exist and are wired up, but `EncoderDecoderModel::update_weights()` has no encoder counterpart to call, so the standalone `train_step()`/`train_step_tokenized()` convenience API computes correct encoder gradients but doesn't apply them. The production training path (external `Optimizer` via `register_parameters_with_optimizer`) is unaffected.
2. **LanguageModelHead Methods**: Missing save_weights(), load_weights()
3. **Memory Inefficiency**: No KV caching, regenerates attention each token
4. **No Validation**: Doesn't validate token IDs in vocabulary range

## Testing Strategy

### Unit Tests Needed

1. **Constructor**: Verify component initialization
2. **Forward Pass**: Test encoder → decoder → lm_head pipeline
3. **Generation**: Test all generation strategies
4. **Training**: Test train_step, loss computation, gradients
5. **Persistence**: Test save/load with validation
6. **Configuration**: Test set_training, set_learning_rate, set_generation_config
7. **Edge Cases**: Empty inputs, long sequences, special tokens

### Integration Tests

1. **End-to-End Generation**: Input text → output text
2. **Training Loop**: Multiple epochs, convergence
3. **Multi-turn Conversation**: Sequential generation
4. **Custom Tokenizer**: External vocabulary
5. **Model Loading**: Train → save → load → generate

## Future Enhancements

### Short-term

1. Implement complete save/load for all components
2. Add batch processing support
3. Add attention masking for padding tokens
4. Implement KV caching for efficient inference
5. Add gradient clipping

### Medium-term

1. Support different encoder/decoder vocabularies
2. Add scheduled sampling for training
3. Implement label smoothing
4. Add multi-GPU training support
5. Implement beam search length penalty

### Long-term

1. Pre-training support (masked language modeling)
2. Transfer learning utilities
3. Model compression (pruning, quantization)
4. Efficient attention mechanisms (Flash Attention)
5. Reinforcement learning from human feedback (RLHF)

## Related Documentation

- **LLMEncoder**: `ENCODER_CONTEXT.md`
- **LLMDecoder**: `DECODER_CONTEXT.md`
- **LanguageModelHead**: `LANGUAGEMODELHEAD_CONTEXT.md`
- **TextGenerator**: `TEXTGENERATOR_CONTEXT.md`
- **BPETokenizer**: `BPE_TOKENIZER_CONTEXT.md`

## Version History

- **v1.0** (2026-01-18): Initial implementation with encoder-decoder integration, multiple generation strategies, and training support
