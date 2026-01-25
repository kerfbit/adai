# End-to-End Training Example with Gradient Flow

This document demonstrates how to use the newly implemented gradient flow capabilities in the LLMEncoder for end-to-end training with a downstream classifier.

## Overview

The encoder now supports full backpropagation through all components, enabling fine-tuning on specific tasks. The gradient flow implementation includes:

- ✅ Backward passes for all components (Matrix, Activation, LayerNorm, TokenEmbedding, MultiHeadAttention, FeedForward, EncoderBlock)
- ✅ Gradient accumulation and storage
- ✅ Weight updates with configurable learning rate
- ✅ Training mode control via `requires_grad` flag
- ✅ Sentence embedding support with backpropagation

## Training Workflow

### 1. Basic Setup

```cpp
#include "encoder.hpp"

// Initialize encoder
LLMEncoder encoder(vocab_size, d_model=256, num_layers=4, 
                   num_heads=8, d_ff=1024, max_seq_length=512);

// Load or build tokenizer
encoder.load_tokenizer_vocab("vocab.txt");

// Enable training mode
encoder.set_requires_grad(true);
encoder.set_learning_rate(0.001f);
```

### 2. Training Loop with NeuralNetwork Classifier

```cpp
// Pseudo-code for training with a classifier
for (int epoch = 0; epoch < num_epochs; epoch++) {
    for (auto& [text, label] : training_data) {
        // Zero gradients
        encoder.zero_grad();
        classifier.zero_grad();
        
        // Forward pass through encoder
        Matrix sentence_emb = encoder.get_sentence_embedding_trainable(text);
        
        // Forward pass through classifier
        std::vector<float> logits = classifier.forward(sentence_emb);
        
        // Compute loss (cross-entropy)
        float loss = compute_cross_entropy_loss(logits, label);
        
        // Backward pass through classifier
        Matrix grad_sentence_emb = classifier.backward(loss_gradient);
        
        // Backward pass through encoder
        encoder.backward_sentence_embedding(grad_sentence_emb);
        
        // Weights are updated automatically during backward passes
    }
}
```

### 3. Token-Level Training (e.g., Named Entity Recognition)

```cpp
// Train on token-level predictions
for (int epoch = 0; epoch < num_epochs; epoch++) {
    for (auto& [text, token_labels] : training_data) {
        // Zero gradients
        encoder.zero_grad();
        token_classifier.zero_grad();
        
        // Forward pass through encoder (returns seq_len x d_model)
        Matrix token_embeddings = encoder.encode(text);
        
        // Forward pass through token classifier
        Matrix logits = token_classifier.forward(token_embeddings);
        
        // Compute loss
        Matrix loss_grad = compute_token_loss_gradient(logits, token_labels);
        
        // Backward pass through classifier
        Matrix grad_embeddings = token_classifier.backward(loss_grad);
        
        // Backward pass through encoder
        encoder.backward(grad_embeddings);
    }
}
```

## Component-Level Gradient Flow

### Matrix Operations
```cpp
Matrix A, B, grad_output;

// Forward
Matrix C = A * B;

// Backward (automatic during component backward passes)
Matrix grad_A = grad_output * B.transpose();
Matrix grad_B = A.transpose() * grad_output;
```

### Layer Normalization
```cpp
LayerNorm norm(d_model);

// Forward
Matrix normalized = norm.forward(input);

// Backward
Matrix grad_input = norm.backward(grad_output);
// Automatically updates gamma and beta weights
```

### Multi-Head Attention
```cpp
MultiHeadAttention attn(d_model, num_heads);

// Forward
Matrix attn_output = attn.forward(input, mask);

// Backward
Matrix grad_input = attn.backward(grad_output);
// Automatically updates W_q, W_k, W_v, W_o weights
```

### Feed-Forward Network
```cpp
FeedForward ff(d_model, d_ff);

// Forward
Matrix ff_output = ff.forward(input);

// Backward
Matrix grad_input = ff.backward(grad_output);
// Automatically updates W1, W2, b1, b2 weights
```

## Key Features

### 1. Activation Caching
Forward passes automatically cache intermediate values needed for backpropagation:
- Input values
- Normalized outputs
- Attention weights
- Activated values (before/after GELU)

### 2. Gradient Accumulation
Each component maintains gradient matrices:
- `W_*_grad` for weight matrices
- `b_*_grad` for biases
- `embedding_grad` for token embeddings

### 3. Automatic Weight Updates
During `backward()`, weights are updated using:
```cpp
W = W - learning_rate * grad_W
```

### 4. Residual Connection Gradients
EncoderBlock properly handles gradient flow through residual connections:
```cpp
// Forward: output = x + f(x)
// Backward: grad_x = grad_output + grad_from_f
```

## Learning Rate Management

Set learning rates at different levels:

```cpp
// Global learning rate for encoder
encoder.set_learning_rate(0.001f);

// Individual component learning rates (if needed)
// Components inherit from encoder by default
```

## Gradient Checking (Recommended)

Verify gradient correctness using numerical gradients:

```cpp
float epsilon = 1e-5f;
Matrix numerical_grad(rows, cols);

for (int i = 0; i < rows; i++) {
    for (int j = 0; j < cols; j++) {
        // Perturb weight
        W(i, j) += epsilon;
        float loss_plus = forward_and_compute_loss();
        
        W(i, j) -= 2 * epsilon;
        float loss_minus = forward_and_compute_loss();
        
        W(i, j) += epsilon; // Restore
        
        numerical_grad(i, j) = (loss_plus - loss_minus) / (2 * epsilon);
    }
}

// Compare with analytical gradient
float grad_diff = compare_gradients(numerical_grad, W_grad);
```

## Performance Considerations

1. **Memory Usage**: Caching intermediate values increases memory consumption during training
2. **Computation**: Backward passes roughly double computation time compared to forward-only
3. **Batch Processing**: Consider implementing batched operations for efficiency
4. **Gradient Clipping**: Add gradient clipping to prevent exploding gradients

```cpp
// Gradient clipping example
void clip_gradients(float max_norm) {
    float total_norm = compute_gradient_norm();
    if (total_norm > max_norm) {
        float scale = max_norm / total_norm;
        // Scale all gradients
    }
}
```

## Common Training Patterns

### Fine-tuning for Sentiment Analysis
```cpp
encoder.set_requires_grad(true);
encoder.set_learning_rate(0.0001f); // Lower LR for fine-tuning

for (auto& [text, sentiment] : sentiment_dataset) {
    encoder.zero_grad();
    Matrix emb = encoder.get_sentence_embedding_trainable(text);
    // Train classifier on emb
}
```

### Feature Extraction (Frozen Encoder)
```cpp
encoder.set_requires_grad(false); // Freeze encoder

for (auto& [text, label] : dataset) {
    Matrix emb = encoder.get_sentence_embedding_trainable(text);
    // Train only classifier, encoder stays frozen
}
```

### Gradual Unfreezing
```cpp
// Epoch 1-5: Train only classifier
encoder.set_requires_grad(false);

// Epoch 6-10: Fine-tune top layers
encoder.set_requires_grad(true);
// Optionally: only update top N layers

// Epoch 11+: Fine-tune all layers
encoder.set_learning_rate(0.00001f);
```

## Integration with NeuralNetwork Class

Example integration following the recommended pattern from ENCODER_CONTEXT.md:

```cpp
class SentimentClassifier {
private:
    LLMEncoder encoder;
    NeuralNetwork classifier;
    
public:
    SentimentClassifier(/* params */) {
        encoder = LLMEncoder(vocab_size, d_model, ...);
        
        // Build classifier on top of encoder
        std::vector<int> layers = {d_model, 128, 64, 3}; // 3 classes
        classifier = NeuralNetwork(layers);
        
        encoder.set_requires_grad(true);
    }
    
    void train(const std::vector<std::pair<std::string, int>>& data) {
        for (auto& [text, label] : data) {
            // Forward
            Matrix emb = encoder.get_sentence_embedding_trainable(text);
            std::vector<float> emb_vec = matrix_to_vector(emb);
            std::vector<float> logits = classifier.forward(emb_vec);
            
            // Compute loss
            auto loss_info = compute_loss(logits, label);
            
            // Backward through classifier
            classifier.backward(loss_info.gradient);
            
            // Backward through encoder
            Matrix grad_emb = vector_to_matrix(classifier.get_input_gradient());
            encoder.backward_sentence_embedding(grad_emb);
        }
    }
};
```

## Mathematical Foundations

### Attention Gradient Flow
```
∂L/∂Q = ∂L/∂Attention × softmax'(QK^T/√d_k) × K
∂L/∂K = ∂L/∂Attention × softmax'(QK^T/√d_k)^T × Q
∂L/∂V = ∂L/∂Attention × Attention_weights^T
```

### Layer Norm Gradients
```
∂L/∂x = γ/σ × (∂L/∂y - mean(∂L/∂y) - (x-μ)/σ² × mean(∂L/∂y × (x-μ)))
∂L/∂γ = Σ(∂L/∂y × (x-μ)/σ)
∂L/∂β = Σ(∂L/∂y)
```

### GELU Derivative
```
GELU(x) = x × Φ(x)  where Φ is standard normal CDF
GELU'(x) = Φ(x) + x × φ(x)  where φ is standard normal PDF

Approximation used:
GELU(x) ≈ 0.5 × x × (1 + tanh(√(2/π) × (x + 0.044715 × x³)))
```

## Next Steps

1. Implement gradient clipping for stability
2. Add learning rate scheduling
3. Implement optimizer variants (Adam, AdamW)
4. Add batch processing support
5. Create validation/testing utilities
6. Add gradient checkpointing for memory efficiency

## References

- Original implementation: `encoder.cpp` and `encoder.hpp`
- Context document: `ENCODER_CONTEXT.md`
- Transformer paper: "Attention Is All You Need" (Vaswani et al., 2017)
