# TokenEmbedding Class - Technical Context Documentation

**Version:** 1.1
**Last Updated:** January 24, 2026

## Overview

The `TokenEmbedding` class is a foundational component for neural language models, converting discrete token IDs into continuous dense vector representations. It implements a learnable embedding layer that maps vocabulary indices to high-dimensional vectors, serving as the first layer in transformer-based architectures and other sequence models.

Files:

- `src/TokenEmbedding.hpp` - Header file with class declaration and interface
- `src/TokenEmbedding.cpp` - Implementation file with all method definitions

Dependencies:

- `Matrix.hpp` - Matrix operations and storage
- `Optimizer.hpp` - Advanced optimization algorithms (optional)

**Purpose:** Provide a trainable lookup table that transforms discrete token identifiers into continuous vector embeddings, enabling neural networks to process textual data while learning semantic relationships between tokens.

---

## Class Structure

### Private Members

```cpp
Matrix embedding_matrix;              // [vocab_size, d_model] - learnable embeddings
Matrix embedding_grad;                // [vocab_size, d_model] - accumulated gradients
int vocab_size;                       // Size of vocabulary
int d_model;                          // Embedding dimension
std::vector<int> cached_token_ids;   // Cache for backward pass
Optimizer* optimizer;                 // Optional optimizer (nullptr = simple gradient descent)
```

### Public Members

```cpp
float learning_rate;  // Learning rate for gradient updates (used when optimizer is nullptr)
```

### Memory Layout

- **Embedding Matrix**: `vocab_size × d_model` matrix where each row represents one token's embedding
- **Gradient Matrix**: Same dimensions as embedding matrix, accumulates gradients during backpropagation
- **Row-major storage**: Token ID directly indexes into embedding matrix rows

---

## Mathematical Foundation

### Forward Pass Operation

For a single token ID `t`:

```text
embedding(t) = embedding_matrix[t, :]
```

For a sequence of tokens `[t₁, t₂, ..., tₙ]`:

```text
output = [embedding_matrix[t₁, :],
          embedding_matrix[t₂, :],
          ...,
          embedding_matrix[tₙ, :]]
```

Dimensions:

- Input: `[sequence_length]` - vector of token IDs
- Output: `[sequence_length, d_model]` - matrix of embeddings

### Backward Pass (Gradient Computation)

For each position `i` in the sequence with token ID `t`:

```text
embedding_grad[t, :] += grad_output[i, :]
```

Key Properties:

- Gradients accumulate for tokens appearing multiple times in sequence
- Only embeddings for tokens in the sequence receive gradient updates
- Other embedding vectors remain unchanged during this backward pass

### Applying Gradients

Gradient descent update rule:

```text
embedding_matrix -= learning_rate × embedding_grad
```

After update, gradients are zeroed for next iteration.

---

## Initialization

### Xavier/Glorot Initialization

The embedding matrix is initialized with random values scaled by:

```text
scale = √(1 / d_model)
```

**Purpose:** Ensures gradients have appropriate magnitude during early training stages, preventing vanishing or exploding gradients.

**Distribution:** Random values from uniform or normal distribution, scaled by the factor above.

---

## Constructors

### Primary Constructor

```cpp
TokenEmbedding(int vocab_size, int d_model)
```

Parameters:

- `vocab_size` - Total number of unique tokens in vocabulary
- `d_model` - Dimension of embedding vectors (typically 128, 256, 512, or 1024)

Initialization Steps:

1. Creates `vocab_size × d_model` embedding matrix
2. Creates gradient matrix with same dimensions
3. Applies Xavier initialization to embeddings
4. Zeros out gradient matrix
5. Sets default learning rate to 0.001

Example:

```cpp
// For a vocabulary of 10,000 tokens with 512-dimensional embeddings
TokenEmbedding embeddings(10000, 512);
embeddings.learning_rate = 0.0001f;  // Adjust learning rate
```

---

## Core Methods

### Forward Pass

```cpp
Matrix forward(const std::vector<int>& token_ids)
```

**Purpose:** Convert sequence of token IDs to embeddings

Parameters:

- `token_ids` - Vector of token indices `[sequence_length]`

**Returns:** Matrix of embeddings `[sequence_length, d_model]`

Process:

1. Validates all token IDs are within bounds `[0, vocab_size)`
2. Caches token IDs for backward pass
3. Looks up embedding for each token ID
4. Constructs output matrix with embeddings as rows

Exceptions:

- `std::out_of_range` - If any token ID is negative or >= vocab_size

**Complexity:** O(sequence_length × d_model)

Example:

```cpp
std::vector<int> tokens = {5, 12, 3, 5};  // "Hello world Hello"
Matrix embeddings = layer.forward(tokens);
// embeddings shape: [4, d_model]
// embeddings[0, :] and embeddings[3, :] are identical (same token ID 5)
```

### Backward Pass

```cpp
void backward(const std::vector<int>& token_ids, const Matrix& grad_output)
```

**Purpose:** Accumulate gradients for embedding parameters

Parameters:

- `token_ids` - Vector of token indices (must match forward pass)
- `grad_output` - Gradient from upstream layer `[sequence_length, d_model]`

Process:

1. Validates dimensions match (token_ids.size() == grad_output.rows)
2. For each position in sequence, accumulates gradient to corresponding embedding
3. Handles repeated tokens by summing their gradients

Gradient Accumulation:

```cpp
for (i = 0; i < sequence_length; ++i) {
    token_id = token_ids[i];
    embedding_grad[token_id, :] += grad_output[i, :];
}
```

**Important:** If token ID 5 appears 3 times, its embedding receives sum of 3 gradients.

Exceptions:

- `std::invalid_argument` - Dimension mismatch
- `std::out_of_range` - Invalid token ID

**Complexity:** O(sequence_length × d_model)

### Weight Update

```cpp
void update_weights()
```

**Purpose:** Apply accumulated gradients to embedding matrix

Process:

1. If optimizer is set, uses `optimizer->step()` for advanced optimization (Adam, AdamW, etc.)
2. Otherwise, falls back to simple gradient descent: `embedding_matrix -= learning_rate × embedding_grad`
3. Calls `zero_grad()` to reset gradients

**When to Call:** After one or more backward passes, before next forward pass

Example Training Loop:

```cpp
for (int epoch = 0; epoch < num_epochs; ++epoch) {
    embeddings.zero_grad();  // Clear previous gradients

    // Forward pass
    Matrix output = embeddings.forward(token_ids);

    // ... compute loss and gradients ...

    // Backward pass
    embeddings.backward(token_ids, grad_from_loss);

    // Update weights
    embeddings.update_weights();
}
```

### Optimizer Integration

```cpp
void set_optimizer(Optimizer* opt)
```

**Purpose:** Set optimizer for advanced optimization algorithms

Parameters:

- `opt` - Pointer to optimizer (nullptr to use simple gradient descent)

Process:

1. Stores optimizer pointer
2. If optimizer is not nullptr, automatically calls `register_parameters()`

Example:

```cpp
Optimizer adam(OptimizerType::ADAM, 0.001f);
adam.set_betas(0.9f, 0.999f);
embeddings.set_optimizer(&adam);
```

```cpp
void register_parameters()
```

**Purpose:** Register embedding parameters with optimizer

Process:

- Registers `embedding_matrix` and `embedding_grad` with the optimizer
- Called automatically by `set_optimizer()`
- No-op if optimizer is nullptr

**Note:** Typically called internally; manual calls rarely needed.

### Zero Gradients

```cpp
void zero_grad()
```

**Purpose:** Reset all gradients to zero

When to Call:

- Before starting a new training iteration
- After calling `update_weights()` (automatically called)
- When resetting training state

**Complexity:** O(vocab_size × d_model)

---

## Accessor Methods

### Get Single Token Embedding

```cpp
std::vector<float> get_token_embedding(int token_id) const
```

**Returns:** Embedding vector for specified token `[d_model]`

Use Cases:

- Inspecting learned embeddings
- Visualization (e.g., t-SNE, PCA)
- Computing token similarities

Example:

```cpp
std::vector<float> hello_embedding = embeddings.get_token_embedding(5);
// hello_embedding.size() == d_model
```

### Get Embedding Matrix

```cpp
const Matrix& get_embeddings() const
```

**Returns:** Read-only reference to entire embedding matrix `[vocab_size, d_model]`

Use Cases:

- Saving embeddings
- Analysis and visualization
- Transfer learning

### Get Dimensions

```cpp
int get_vocab_size() const    // Returns vocab_size
int get_embedding_dim() const // Returns d_model
```

Use Cases:

- Dimension verification
- Dynamic architecture construction
- Debugging

---

## Persistence and Loading

### Save Embeddings

```cpp
void save_embeddings(const std::string& filename) const
```

**Purpose:** Persist learned embeddings to binary file

File Format:

```text
[vocab_size: int]           // 4 bytes
[d_model: int]              // 4 bytes
[embedding data: float[]]   // vocab_size × d_model × 4 bytes
```

**Storage Order:** Row-major (all elements of token 0, then token 1, etc.)

Example:

```cpp
embeddings.save_embeddings("embeddings_epoch_100.bin");
```

### Load Pre-trained Embeddings

```cpp
void load_pretrained(const std::string& filename)
```

**Purpose:** Load embeddings from file (e.g., Word2Vec, GloVe format converted to binary)

Validation:

- Checks vocab_size and d_model match current instance
- Throws exception if dimensions mismatch

Use Cases:

- Transfer learning
- Fine-tuning pre-trained models
- Resuming training

Example:

```cpp
TokenEmbedding embeddings(10000, 300);
embeddings.load_pretrained("glove.6B.300d.bin");
```

Exceptions:

- `std::runtime_error` - File cannot be opened or dimension mismatch

---

## Initialization Methods

### Initialize with Constant

```cpp
void initialize_constant(float value)
```

**Purpose:** Set all embeddings to a specific value

Use Cases:

- Testing
- Debugging gradient flow
- Baseline comparisons

Example:

```cpp
embeddings.initialize_constant(0.0f);  // Zero initialization
embeddings.initialize_constant(1.0f);  // Ones initialization
```

### Reinitialize with Xavier

```cpp
void reinitialize()
```

**Purpose:** Reset embeddings to new random values using Xavier initialization

Use Cases:

- Restarting training
- Testing initialization impact
- Multiple training runs with different seeds

---

## Training Utilities

### Get Gradient Norm

```cpp
float get_gradient_norm() const
```

**Purpose:** Compute L2 norm of gradient matrix

Formula:

```text
norm = √(Σᵢⱼ (embedding_grad[i,j])²)
```

Use Cases:

- Monitoring gradient magnitudes
- Detecting vanishing/exploding gradients
- Adaptive learning rate strategies

Example:

```cpp
float grad_norm = embeddings.get_gradient_norm();
if (grad_norm > 10.0f) {
    std::cout << "Warning: Large gradients detected!" << std::endl;
}
```

### Clip Gradients

```cpp
void clip_gradients(float max_norm)
```

**Purpose:** Prevent exploding gradients by scaling down if norm exceeds threshold

Algorithm:

```text
norm = get_gradient_norm()
if (norm > max_norm):
    scale = max_norm / norm
    embedding_grad *= scale
```

Use Cases:

- Stabilizing training
- Handling extreme gradient values
- Training recurrent models

Example:

```cpp
// Clip gradients before weight update
embeddings.clip_gradients(5.0f);
embeddings.update_weights();
```

---

## Debugging and Monitoring

### Print Configuration

```cpp
void print_config(const std::string& name = "TokenEmbedding") const
```

**Purpose:** Display layer configuration and statistics

Output Example:

```text
TokenEmbedding Configuration:
  Vocabulary Size: 10000
  Embedding Dimension: 512
  Total Parameters: 5120000
  Memory Usage: 19.53 MB
  Learning Rate: 0.001
```

Use Cases:

- Debugging network architecture
- Memory profiling
- Documentation and logging

---

## Usage Patterns

### Basic Training Example (with Optimizer)

```cpp
// Initialize
TokenEmbedding embeddings(vocab_size, 512);

// Set up Adam optimizer
Optimizer optimizer(OptimizerType::ADAM, 0.001f);
optimizer.set_betas(0.9f, 0.999f);
embeddings.set_optimizer(&optimizer);

// Training loop
for (int epoch = 0; epoch < num_epochs; ++epoch) {
    for (const auto& batch : training_data) {
        // Forward pass
        Matrix embedded = embeddings.forward(batch.token_ids);

        // ... pass through rest of model ...
        // ... compute loss ...
        // ... backpropagate to get grad_embedded ...

        // Backward pass
        embeddings.backward(batch.token_ids, grad_embedded);

        // Update (uses Adam optimizer)
        embeddings.update_weights();
    }
}
```

### Basic Training Example (without Optimizer - Legacy)

```cpp
// Initialize
TokenEmbedding embeddings(vocab_size, 512);
embeddings.learning_rate = 0.0001f;  // Simple gradient descent

// Training loop
for (int epoch = 0; epoch < num_epochs; ++epoch) {
    for (const auto& batch : training_data) {
        // Forward pass
        Matrix embedded = embeddings.forward(batch.token_ids);

        // ... pass through rest of model ...
        // ... compute loss ...
        // ... backpropagate to get grad_embedded ...

        // Backward pass
        embeddings.backward(batch.token_ids, grad_embedded);

        // Update (uses simple gradient descent)
        embeddings.update_weights();
    }
}
```

### Transfer Learning Example

```cpp
// Load pre-trained embeddings
TokenEmbedding embeddings(50000, 300);
embeddings.load_pretrained("word2vec.bin");
embeddings.learning_rate = 0.00001f;  // Lower LR for fine-tuning

// Fine-tune on specific task
for (int epoch = 0; epoch < fine_tune_epochs; ++epoch) {
    // ... training loop ...
}

// Save fine-tuned embeddings
embeddings.save_embeddings("fine_tuned_embeddings.bin");
```

### Gradient Monitoring Example

```cpp
// Training with gradient monitoring
for (int step = 0; step < max_steps; ++step) {
    embeddings.zero_grad();

    Matrix output = embeddings.forward(tokens);
    // ... forward through model ...

    embeddings.backward(tokens, grad_output);

    // Monitor gradients
    float grad_norm = embeddings.get_gradient_norm();
    if (grad_norm > 10.0f) {
        embeddings.clip_gradients(5.0f);
        std::cout << "Clipped gradients from " << grad_norm << " to 5.0" << std::endl;
    }

    embeddings.update_weights();
}
```

---

## Integration with Other Components

### With Positional Encoding

```cpp
TokenEmbedding token_emb(vocab_size, d_model);
PositionalEncoding pos_enc(max_seq_len, d_model);

// Combine embeddings and positional encoding
Matrix token_vectors = token_emb.forward(token_ids);
Matrix encoded = pos_enc.forward(token_vectors);  // Add positional information
```

### With Transformer Layers

```cpp
// Typical transformer encoder input preparation
TokenEmbedding embeddings(vocab_size, d_model);
PositionalEncoding pos_encoding(max_len, d_model);

Matrix embedded = embeddings.forward(input_tokens);
Matrix with_position = pos_encoding.forward(embedded);

// Now feed to transformer layers...
```

### With BPE Tokenizer

```cpp
BPETokenizer tokenizer("vocab.txt", "merges.txt");
TokenEmbedding embeddings(tokenizer.vocab_size(), d_model);

// Encode text to tokens
std::string text = "Hello, world!";
std::vector<int> tokens = tokenizer.encode(text);

// Convert tokens to embeddings
Matrix embedded = embeddings.forward(tokens);
```

---

## Performance Considerations

### Memory Usage

For `vocab_size = V` and `d_model = D`:

- **Embedding Matrix**: V × D × 4 bytes (float)
- **Gradient Matrix**: V × D × 4 bytes (float)
- **Total**: 2 × V × D × 4 bytes

Example Sizes:

- V=10,000, D=512: ~39 MB
- V=50,000, D=512: ~195 MB
- V=50,000, D=1024: ~390 MB

### Computational Complexity

- **Forward Pass**: O(S × D) where S = sequence_length
- **Backward Pass**: O(S × D)
- **Update Weights**: O(V × D)
- **Zero Gradients**: O(V × D)

**Optimization Note:** Forward and backward are linear in sequence length, making them efficient for typical sequences (up to thousands of tokens).

### Cache Efficiency

The lookup operation has good cache locality since:

- Each token lookup reads one contiguous row
- Sequential token lookups may benefit from prefetching
- Gradient accumulation writes to rows in random order (depends on token frequency)

---

## Common Patterns and Best Practices

### 1. Learning Rate Scheduling

```cpp
embeddings.learning_rate = initial_lr;

for (int epoch = 0; epoch < num_epochs; ++epoch) {
    // Decay learning rate
    embeddings.learning_rate = initial_lr * std::pow(0.95f, epoch);

    // Training loop...
}
```

### 2. Gradient Clipping

```cpp
// Always clip before update to prevent instability
embeddings.backward(tokens, grad);
embeddings.clip_gradients(5.0f);
embeddings.update_weights();
```

### 3. Checkpoint Saving

```cpp
// Save embeddings periodically
if (step % save_interval == 0) {
    std::string filename = "embeddings_step_" + std::to_string(step) + ".bin";
    embeddings.save_embeddings(filename);
}
```

### 4. Embedding Freezing (Transfer Learning)

```cpp
// Don't update embeddings for some layers
Matrix embedded = embeddings.forward(tokens);
// ... forward pass through model ...

// Skip backward and update for embeddings
// embeddings.backward(...);  // Commented out - frozen embeddings
// embeddings.update_weights();

// Only update other layers
```

### 5. Warm-up Period

```cpp
// Gradual learning rate increase
int warmup_steps = 1000;
for (int step = 0; step < warmup_steps; ++step) {
    float warmup_lr = initial_lr * (step + 1) / warmup_steps;
    embeddings.learning_rate = warmup_lr;
    // ... training step ...
}
```

---

## Error Handling

### Common Exceptions

1. **std::out_of_range**
   - Cause: Token ID < 0 or >= vocab_size
   - Solution: Validate tokens before forward/backward

2. **std::invalid_argument**
   - Cause: Dimension mismatch in backward pass
   - Solution: Ensure grad_output dimensions match forward output

3. **std::runtime_error**
   - Cause: File I/O errors or dimension mismatch in load_pretrained
   - Solution: Check file paths and embedding dimensions

### Validation Example

```cpp
// Validate token IDs before use
bool validate_tokens(const std::vector<int>& tokens, int vocab_size) {
    for (int token : tokens) {
        if (token < 0 || token >= vocab_size) {
            return false;
        }
    }
    return true;
}

// Use validation
if (validate_tokens(token_ids, embeddings.get_vocab_size())) {
    Matrix output = embeddings.forward(token_ids);
} else {
    std::cerr << "Invalid token IDs detected!" << std::endl;
}
```

---

## Testing and Validation

### Unit Test Considerations

1. **Forward Pass**
   - Test correct shape of output
   - Verify embedding lookup correctness
   - Test bounds checking

2. **Backward Pass**
   - Verify gradient accumulation
   - Test repeated token handling
   - Check gradient dimensions

3. **Weight Update**
   - Numerical gradient checking
   - Verify learning_rate effect
   - Test gradient zeroing

4. **Persistence**
   - Save and load round-trip
   - Verify embedding preservation
   - Test error handling for invalid files

### Numerical Gradient Check

```cpp
// Finite difference approximation for gradient verification
float epsilon = 1e-4f;
std::vector<int> tokens = {5};

// Perturb one embedding element
float original = embeddings.get_embeddings()(5, 0);

embeddings.get_embeddings()(5, 0) = original + epsilon;
Matrix out_plus = embeddings.forward(tokens);
float loss_plus = compute_loss(out_plus);

embeddings.get_embeddings()(5, 0) = original - epsilon;
Matrix out_minus = embeddings.forward(tokens);
float loss_minus = compute_loss(out_minus);

float numerical_grad = (loss_plus - loss_minus) / (2 * epsilon);

// Compare with backprop gradient
embeddings.backward(tokens, analytical_grad);
float analytical = embeddings.embedding_grad(5, 0);

// Should be very close
assert(std::abs(numerical_grad - analytical) < 1e-3);
```

---

## Theoretical Background

### Why Embeddings?

**Problem:** Neural networks operate on continuous values, but text consists of discrete tokens.

**Solution:** Learn a mapping from discrete token space to continuous vector space where:

- Similar tokens have similar embeddings (cosine similarity)
- Semantic relationships are captured (e.g., king - man + woman ≈ queen)
- Embedding space enables mathematical operations on words

### Embedding Dimension Selection

Trade-offs:

- **Larger d_model**:
  - More capacity to capture semantic nuances
  - Higher memory and computation cost
  - Risk of overfitting with small datasets

- **Smaller d_model**:
  - Faster training and inference
  - Less memory usage
  - May miss subtle semantic relationships

Common Choices:

- Small models: 128-256
- Medium models: 512
- Large models: 768-1024
- Very large models: 1536-2048

### Initialization Importance

Xavier initialization ensures:

```text
Var(embedding[i,j]) ≈ 1/d_model
```

This maintains variance of activations through layers, critical for:

- Stable gradient flow
- Faster convergence
- Avoiding saturation in activation functions

---

## Future Extensions

Potential enhancements to consider:

1. **Adaptive Embeddings**
   - Reduce embedding dimensions for rare tokens
   - Save memory for large vocabularies

2. **Subword Embeddings**
   - Character-level components
   - Morphological awareness

3. **Contextualized Embeddings**
   - Different embeddings for same token in different contexts
   - Integration with attention mechanisms

4. **Sparse Embeddings**
   - Reduce memory for extremely large vocabularies
   - Use sparse matrix representations

5. **Embedding Regularization**
   - L2 regularization on embeddings
   - Dropout on embedding outputs

---

## Optimizer Usage Patterns

### Using Adam Optimizer

```cpp
TokenEmbedding embeddings(vocab_size, d_model);

// Configure Adam optimizer
Optimizer adam(OptimizerType::ADAM, 0.001f);
adam.set_betas(0.9f, 0.999f);
embeddings.set_optimizer(&adam);

// Training loop
for (int step = 0; step < training_steps; ++step) {
    Matrix output = embeddings.forward(token_ids);
    // ... compute loss and gradients ...
    embeddings.backward(token_ids, grad_output);
    embeddings.update_weights();  // Uses Adam
}
```

### Using AdamW with Weight Decay

```cpp
Optimizer adamw(OptimizerType::ADAMW, 0.001f);
adamw.set_betas(0.9f, 0.999f);
adamw.set_weight_decay(0.01f);
embeddings.set_optimizer(&adamw);
```

### Switching Optimizers During Training

```cpp
// Start with SGD with momentum
Optimizer sgd(OptimizerType::SGD, 0.1f);
sgd.set_momentum(0.9f);
embeddings.set_optimizer(&sgd);

// Train for warmup period
for (int step = 0; step < warmup_steps; ++step) {
    // ... training ...
}

// Switch to Adam for fine-tuning
Optimizer adam(OptimizerType::ADAM, 0.001f);
adam.set_betas(0.9f, 0.999f);
embeddings.set_optimizer(&adam);

// Continue training
for (int step = 0; step < fine_tune_steps; ++step) {
    // ... training ...
}
```

### Learning Rate Scheduling with Optimizer

```cpp
Optimizer optimizer(OptimizerType::ADAM, 0.001f);
embeddings.set_optimizer(&optimizer);

for (int epoch = 0; epoch < num_epochs; ++epoch) {
    // Decay learning rate
    float new_lr = 0.001f * std::pow(0.95f, epoch);
    optimizer.set_learning_rate(new_lr);

    // Training epoch
    for (const auto& batch : training_data) {
        // ... training ...
    }
}
```

### Backward Compatibility - No Optimizer

```cpp
// Old code continues to work without modification
TokenEmbedding embeddings(vocab_size, d_model);
embeddings.learning_rate = 0.001f;

// No optimizer set - uses simple gradient descent
embeddings.forward(tokens);
embeddings.backward(tokens, grad);
embeddings.update_weights();  // Simple: param -= lr * grad
```

---

## Recent Updates

### Version 1.1 (January 24, 2026)

Optimizer Integration:

- Added optional `Optimizer` support for advanced optimization algorithms
- New method: `set_optimizer(Optimizer* opt)` - configure optimizer
- New method: `register_parameters()` - register with optimizer (called automatically)
- Modified `update_weights()` to use optimizer when available, fallback to simple gradient descent
- Fully backward compatible - existing code without optimizer continues to work

Benefits:

- Access to Adam, AdamW, SGD with momentum, and other advanced optimizers
- Better convergence on complex tasks
- Per-parameter adaptive learning rates
- Weight decay and other regularization techniques
- Learning rate scheduling at optimizer level

Migration Guide:

Old code (still works):

```cpp
TokenEmbedding embeddings(vocab_size, d_model);
embeddings.learning_rate = 0.001f;
embeddings.forward(tokens);
embeddings.backward(tokens, grad);
embeddings.update_weights();
```

New code (recommended):

```cpp
TokenEmbedding embeddings(vocab_size, d_model);
Optimizer adam(OptimizerType::ADAM, 0.001f);
adam.set_betas(0.9f, 0.999f);
embeddings.set_optimizer(&adam);
embeddings.forward(tokens);
embeddings.backward(tokens, grad);
embeddings.update_weights();  // Now uses Adam
```

---

## Summary

The `TokenEmbedding` class provides a robust, efficient implementation of learnable token embeddings with:

- **Simplicity**: Clean interface for forward/backward passes
- **Efficiency**: Direct matrix lookup, O(S×D) complexity
- **Flexibility**: Configurable dimensions, learning rates, optional advanced optimizers
- **Modern Optimization**: Support for Adam, AdamW, SGD with momentum via Optimizer class
- **Backward Compatibility**: Works with or without optimizer - existing code unchanged
- **Persistence**: Save/load functionality for transfer learning
- **Safety**: Comprehensive bounds checking and validation
- **Debuggability**: Gradient monitoring and configuration printing

It serves as the foundation for transformer-based models and other neural architectures processing sequential discrete data, enabling the network to learn rich representations of tokens that capture semantic and syntactic relationships. The optional optimizer integration enables state-of-the-art training techniques while maintaining a simple, easy-to-use interface.
