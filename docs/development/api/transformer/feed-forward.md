# FeedForward Class - Technical Context Documentation

**Version:** 1.1
**Last Updated:** January 24, 2026
**Status:** Active - Optimizer Integration Complete

## Overview

The `FeedForward` class implements a position-wise feed-forward neural network, a critical component in transformer architectures. It applies two linear transformations with a GELU activation function in between, processing each position in the sequence independently and identically.

Files:

- `src/FeedForward.hpp` - Header file with class declaration and interface
- `src/FeedForward.cpp` - Implementation file with all method definitions
- `src/FeedForwardExample.cpp` - Standalone example demonstrating usage
- `tests/feedforward_test.cpp` - Unit tests (51/51 tests passing)

Dependencies:

- `Matrix.hpp` - Matrix operations and linear algebra
- `Activation.hpp` - GELU activation function
- `Optimizer.hpp` - Advanced optimization algorithms (Adam, AdamW, SGD)

**Purpose:** Provide a two-layer feed-forward network that adds non-linear transformation capacity to transformer models, enabling the network to learn complex representations beyond what attention mechanisms alone can capture.

---

## Table of Contents

1. [Mathematical Foundation](#mathematical-foundation)
2. [Class Architecture](#class-architecture)
3. [Implementation Details](#implementation-details)
4. [Forward Pass](#forward-pass)
5. [Backward Pass](#backward-pass)
6. [Weight Initialization](#weight-initialization)
7. [Gradient Management](#gradient-management)
8. [Optimizer Integration](#optimizer-integration)
9. [Usage Patterns](#usage-patterns)
10. [Performance Considerations](#performance-considerations)
11. [Integration with Transformers](#integration-with-transformers)

---

## Mathematical Foundation

### Core Architecture

The feed-forward network consists of two linear transformations with a GELU (Gaussian Error Linear Unit) activation:

```text
FFN(x) = GELU(xW₁ + b₁)W₂ + b₂
```

Components:

- **W₁**: First layer weights `[d_model × d_ff]`
- **b₁**: First layer bias `[d_ff]`
- **W₂**: Second layer weights `[d_ff × d_model]`
- **b₂**: Second layer bias `[d_model]`
- **GELU**: Gaussian Error Linear Unit activation function

### GELU Activation

The GELU activation function is defined as:

```text
GELU(x) = x · Φ(x)
```

Where Φ(x) is the cumulative distribution function of the standard normal distribution.

Approximation used:

```text
GELU(x) ≈ 0.5 · x · (1 + tanh(√(2/π) · (x + 0.044715 · x³)))
```

Properties:

- Smooth, non-monotonic function
- Differentiable everywhere
- Gives small negative values for negative inputs (unlike ReLU which zeroes them)
- Better gradient flow than ReLU in deep networks

### Dimensions and Expansion Ratio

Typical Configuration:

- Input/Output dimension: `d_model` (e.g., 512, 768)
- Hidden dimension: `d_ff` (typically 4 × d_model)
- Expansion ratio: `d_ff / d_model = 4`

Example:

```text
d_model = 512  →  d_ff = 2048
Input: [seq_len, 512] → Hidden: [seq_len, 2048] → Output: [seq_len, 512]
```

### Position-wise Application

The network is applied identically and independently to each position:

```text
For each position i in sequence:
    output[i] = FFN(input[i])
```

This means:

- Same weights applied to all positions
- No interaction between different positions
- Fully parallelizable across sequence

### Complexity Analysis

Time Complexity:

- First linear layer: O(seq_len × d_model × d_ff)
- GELU activation: O(seq_len × d_ff)
- Second linear layer: O(seq_len × d_ff × d_model)
- Total: **O(seq_len × d_model × d_ff)**

Space Complexity:

- Weights: O(d_model × d_ff + d_ff × d_model) = O(2 × d_model × d_ff)
- Biases: O(d_ff + d_model)
- Activations (cached): O(seq_len × d_ff + seq_len × d_model)
- Total: **O(d_model × d_ff + seq_len × d_ff)**

Parameter Count:

```text
Total parameters = (d_model × d_ff) + d_ff + (d_ff × d_model) + d_model
                 = 2 × d_model × d_ff + d_ff + d_model
```

For d_model=512, d_ff=2048:

```text
Total = 2 × 512 × 2048 + 2048 + 512 = 2,099,712 parameters
```

---

## Class Architecture

### Private Members

```cpp
// Model dimensions
int d_model;  // Input/output dimension (e.g., 512)
int d_ff;     // Hidden layer dimension (e.g., 2048)

// Weights and biases
Matrix W1;    // First layer weights [d_model × d_ff]
Matrix W2;    // Second layer weights [d_ff × d_model]
Matrix b1;    // First layer bias [1 × d_ff]
Matrix b2;    // Second layer bias [1 × d_model]

// Gradients (accumulated during backward pass)
Matrix W1_grad;  // Gradient of W1
Matrix W2_grad;  // Gradient of W2
Matrix b1_grad;  // Gradient of b1
Matrix b2_grad;  // Gradient of b2

// Cached values for backward pass
Matrix cached_input;             // Input to forward pass
Matrix cached_hidden;            // Hidden layer before GELU
Matrix cached_hidden_activated;  // Hidden layer after GELU

// Optimizer integration (optional)
Optimizer* optimizer;  // Pointer to optimizer (nullptr = simple gradient descent)
```

### Public Members

```cpp
float learning_rate;  // Learning rate for gradient descent (default: 0.001)
                      // Used when optimizer is nullptr (backward compatibility)
```

### Memory Layout

Weight Matrices:

- `W1`: Maps input dimension to expanded hidden dimension
- `W2`: Maps hidden dimension back to output dimension
- Both use row-major storage from Matrix class

Bias Vectors:

- Stored as 1×n matrices for broadcasting during addition
- `b1`: Broadcast across all positions during first transformation
- `b2`: Broadcast across all positions during second transformation

Gradient Accumulation:

- Gradients accumulate across backward passes
- Must call `zero_grad()` or `update_weights()` to reset

---

## Implementation Details

### Constructor

```cpp
FeedForward::FeedForward(int d_model, int d_ff)
```

Initialization Steps:

1. Set dimensions (d_model, d_ff)
2. Allocate weight matrices (W1, W2)
3. Allocate bias vectors (b1, b2)
4. Initialize weights using Xavier/He initialization
5. Zero-initialize biases
6. Zero-initialize gradients

Weight Initialization:

```cpp
float scale = √(2.0 / d_model);  // He initialization for GELU
W1 ~ Normal(0, scale)
W2 ~ Normal(0, √(2.0 / d_ff))
b1 = 0
b2 = 0
```

### Method: `forward(const Matrix& input)`

**Purpose:** Compute feed-forward transformation on input

Algorithm:

```text
1. Cache input for backward pass
2. Compute hidden = input × W1
3. Add bias: hidden += b1 (broadcast)
4. Cache pre-activation hidden
5. Apply GELU: hidden_activated = GELU(hidden)
6. Cache activated hidden
7. Compute output = hidden_activated × W2
8. Add bias: output += b2 (broadcast)
9. Return output
```

**Input:** Matrix of shape `[seq_len, d_model]`
**Output:** Matrix of shape `[seq_len, d_model]`

**Time Complexity:** O(seq_len × d_model × d_ff)

Example:

```cpp
FeedForward ff(512, 2048);
Matrix input(10, 512);  // 10 tokens, 512 dimensions
Matrix output = ff.forward(input);  // [10, 512]
```

### Method: `backward(const Matrix& grad_output)`

**Purpose:** Compute gradients for all parameters and return gradient w.r.t. input

Algorithm:

```text
1. Validate gradient dimensions
2. Compute b2_grad = sum(grad_output) over batch dimension
3. Compute W2_grad = cached_hidden_activated^T × grad_output
4. Compute grad_hidden_activated = grad_output × W2^T
5. Compute GELU gradient
6. Compute grad_hidden = grad_hidden_activated ⊙ GELU'(cached_hidden)
7. Compute b1_grad = sum(grad_hidden) over batch dimension
8. Compute W1_grad = cached_input^T × grad_hidden
9. Compute grad_input = grad_hidden × W1^T
10. Return grad_input
```

**Input:** Gradient matrix `[seq_len, d_model]`
**Output:** Gradient matrix `[seq_len, d_model]`

Gradient Equations:

For the second layer:

```text
∂L/∂b2 = Σᵢ ∂L/∂output[i,:]
∂L/∂W2 = hidden_activated^T × ∂L/∂output
∂L/∂hidden_activated = ∂L/∂output × W2^T
```

Through GELU activation:

```text
∂L/∂hidden = ∂L/∂hidden_activated ⊙ GELU'(hidden)
```

For the first layer:

```text
∂L/∂b1 = Σᵢ ∂L/∂hidden[i,:]
∂L/∂W1 = input^T × ∂L/∂hidden
∂L/∂input = ∂L/∂hidden × W1^T
```

GELU Derivative:

```cpp
GELU'(x) = Φ(x) + x · φ(x)
```

Where φ(x) is the probability density function of the standard normal distribution.

### Method: `update_weights()`

**Purpose:** Apply accumulated gradients to weights and biases

Algorithm:

```text
1. W1 -= learning_rate × W1_grad  (using Matrix::apply_gradients)
2. W2 -= learning_rate × W2_grad  (using Matrix::apply_gradients)
3. b1 -= learning_rate × b1_grad
4. b2 -= learning_rate × b2_grad
5. Call zero_grad() to reset gradients
```

**Note:** Automatically zeros gradients after update, preparing for next iteration.

### Method: `zero_grad()`

**Purpose:** Reset all gradient accumulators to zero

Algorithm:

```text
For all elements in W1_grad, W2_grad, b1_grad, b2_grad:
    gradient = 0.0
```

**Usage:** Call before each new backward pass when manually accumulating gradients.

### Method: `get_gradient_norm()`

**Purpose:** Compute L2 norm of all gradients combined

Algorithm:

```text
norm = √(||W1_grad||² +||W2_grad||² +||b1_grad||² +||b2_grad||²)
```

**Returns:** Float value representing gradient magnitude

Use Cases:

- Monitor gradient flow during training
- Detect vanishing/exploding gradients
- Decide when to apply gradient clipping

### Method: `clip_gradients(float max_norm)`

**Purpose:** Prevent exploding gradients by scaling

Algorithm:

```text
1. current_norm = get_gradient_norm()
2. If current_norm > max_norm:
       scale = max_norm / current_norm
       W1_grad *= scale
       W2_grad *= scale
       b1_grad *= scale
       b2_grad *= scale
```

Example:

```cpp
ff.forward(input);
ff.backward(grad_output);

float norm = ff.get_gradient_norm();
if (norm > 5.0) {
    ff.clip_gradients(5.0);  // Clip to max norm of 5.0
}

ff.update_weights();
```

### Method: `save_weights(const std::string& filename)`

**Purpose:** Persist weights and biases to binary file

File Format:

```text
[int32] d_model
[int32] d_ff
[float32 × d_model × d_ff] W1 elements (row-major)
[float32 × d_ff × d_model] W2 elements (row-major)
[float32 × d_ff] b1 elements
[float32 × d_model] b2 elements
```

**Total File Size:** `8 + 4×(2×d_model×d_ff + d_ff + d_model)` bytes

Example:

```cpp
ff.save_weights("feedforward_layer3.bin");
```

### Method: `load_weights(const std::string& filename)`

**Purpose:** Load weights and biases from binary file

Validation:

- Checks that saved dimensions match current instance
- Throws `std::runtime_error` on mismatch or file not found

Example:

```cpp
FeedForward ff(512, 2048);
ff.load_weights("feedforward_layer3.bin");  // Must match dimensions
```

### Method: `print_config(const std::string& name)`

**Purpose:** Display network configuration

Output Example:

```text
MyFFN Configuration:
  Model Dimension (d_model): 512
  Feed-Forward Dimension (d_ff): 2048
  Expansion Ratio: 4x
  Total Parameters: 2099712
  W1 Parameters: 1048576
  W2 Parameters: 1048576
  Bias Parameters: 2560
  Memory Usage: 8.00 MB
  Learning Rate: 0.001
```

---

## Forward Pass

### Detailed Computation Flow

#### Step 1: First Linear Transformation

```cpp
hidden = input × W1
// Shape: [seq_len, d_model] × [d_model, d_ff] = [seq_len, d_ff]
```

#### Step 2: Add First Bias

```cpp
For each row i in hidden:
    For each column j:
        hidden[i,j] += b1[0,j]
// Broadcasting b1 across all sequence positions
```

#### Step 3: GELU Activation

```cpp
hidden_activated = GELU(hidden)
// Applied element-wise to entire matrix
```

#### Step 4: Second Linear Transformation

```cpp
output = hidden_activated × W2
// Shape: [seq_len, d_ff] × [d_ff, d_model] = [seq_len, d_model]
```

#### Step 5: Add Second Bias

```cpp
For each row i in output:
    For each column j:
        output[i,j] += b2[0,j]
// Broadcasting b2 across all sequence positions
```

### Caching for Backpropagation

During forward pass, three matrices are cached:

1. **cached_input**: Original input (needed for W1 gradient)
2. **cached_hidden**: Pre-activation hidden layer (needed for GELU gradient)
3. **cached_hidden_activated**: Post-activation hidden (needed for W2 gradient)

**Memory Cost:** O(seq_len × (d_model + 2×d_ff))

---

## Backward Pass

### Gradient Flow Diagram

```text
grad_output [seq_len, d_model]
    |
    ├─→ ∂L/∂b2 (sum over batch)
    |
    ├─→ ∂L/∂W2 = hidden_activated^T × grad_output
    |
    └─→ grad_hidden_activated = grad_output × W2^T
            |
            └─→ grad_hidden = grad_hidden_activated ⊙ GELU'(hidden)
                    |
                    ├─→ ∂L/∂b1 (sum over batch)
                    |
                    ├─→ ∂L/∂W1 = input^T × grad_hidden
                    |
                    └─→ grad_input = grad_hidden × W1^T
```

### Computational Steps

#### 1. Gradient w.r.t. Second Bias (b2)

```cpp
b2_grad[0,j] = Σᵢ grad_output[i,j]
// Sum gradient contributions from all positions
```

#### 2. Gradient w.r.t. Second Weight (W2)

```cpp
W2_grad = cached_hidden_activated^T × grad_output
// Shape: [d_ff, seq_len] × [seq_len, d_model] = [d_ff, d_model]
```

#### 3. Gradient w.r.t. Activated Hidden Layer

```cpp
grad_hidden_activated = grad_output × W2^T
// Shape: [seq_len, d_model] × [d_model, d_ff] = [seq_len, d_ff]
```

#### 4. Gradient Through GELU

```cpp
gelu_derivative = GELU'(cached_hidden)
grad_hidden = grad_hidden_activated ⊙ gelu_derivative
// Element-wise multiplication (Hadamard product)
```

#### 5. Gradient w.r.t. First Bias (b1)

```cpp
b1_grad[0,j] = Σᵢ grad_hidden[i,j]
// Sum gradient contributions from all positions
```

#### 6. Gradient w.r.t. First Weight (W1)

```cpp
W1_grad = cached_input^T × grad_hidden
// Shape: [d_model, seq_len] × [seq_len, d_ff] = [d_model, d_ff]
```

#### 7. Gradient w.r.t. Input

```cpp
grad_input = grad_hidden × W1^T
// Shape: [seq_len, d_ff] × [d_ff, d_model] = [seq_len, d_model]
```

### GELU Derivative Computation

The derivative is computed using the `Activation::gelu_derivative()` method:

```cpp
GELU'(x) ≈ 0.5 · tanh(z) +
           (0.0535161 · x³ + 0.398942 · x) · sech²(z) + 0.5

Where: z = 0.797885 · (x + 0.044715 · x³)
```

This approximation provides efficient gradient computation while maintaining numerical stability.

---

## Weight Initialization

### Xavier/He Initialization Strategy

First Layer (W1):

```cpp
scale1 = √(2.0 / d_model)
W1[i,j] ~ Normal(0, scale1)
```

Rationale:

- Factor of 2 compensates for GELU's non-linearity (He initialization)
- Scaled by input dimension (d_model)
- Prevents gradient vanishing in deep networks

Second Layer (W2):

```cpp
scale2 = √(2.0 / d_ff)
W2[i,j] ~ Normal(0, scale2)
```

Rationale:

- Scaled by hidden dimension (d_ff)
- Ensures appropriate gradient magnitude at output

Biases:

```cpp
b1[j] = 0
b2[j] = 0
```

Rationale:

- Zero initialization is standard for biases
- Allows weights to determine initial behavior
- Symmetry breaking provided by weight initialization

### Initialization Impact

Good Initialization:

- Gradients have appropriate magnitude
- Training converges smoothly
- Avoids saturation in early epochs

Poor Initialization:

- Vanishing gradients (too small)
- Exploding gradients (too large)
- Slow or unstable training

---

## Gradient Management

### Gradient Accumulation

Gradients accumulate across multiple backward passes:

```cpp
ff.forward(batch1);
ff.backward(grad1);  // Gradients stored in W1_grad, W2_grad, b1_grad, b2_grad

ff.forward(batch2);
ff.backward(grad2);  // Gradients ADDED to existing gradients

// Total gradient = grad1 + grad2
ff.update_weights();  // Apply accumulated gradients
```

**Use Case:** Mini-batch gradient descent with gradient accumulation

### Gradient Clipping

**Purpose:** Prevent exploding gradients in deep networks

Strategies:

1. **Norm-based clipping:**

```cpp
float max_norm = 5.0f;
ff.clip_gradients(max_norm);
```

1. **Monitoring before clipping:**

```cpp
float norm = ff.get_gradient_norm();
if (norm > threshold) {
    std::cout << "Large gradient detected: " << norm << std::endl;
    ff.clip_gradients(max_norm);
}
```

### Gradient Monitoring

Diagnostic Checks:

```cpp
float norm = ff.get_gradient_norm();

if (norm < 1e-6) {
    std::cout << "Warning: Vanishing gradients!" << std::endl;
}

if (norm > 100.0) {
    std::cout << "Warning: Exploding gradients!" << std::endl;
}
```

Typical Gradient Norms:

- Small models (d_model=64): 1-10
- Medium models (d_model=512): 10-100
- Large models (d_model=1024): 100-1000

---

## Optimizer Integration

### Integration Overview

As of Version 1.1, `FeedForward` supports integration with the `Optimizer` class, enabling advanced optimization algorithms (Adam, AdamW, SGD with momentum) while maintaining full backward compatibility with simple gradient descent.

Design Philosophy:

- **Optional Integration:** Optimizer pointer defaults to `nullptr`
- **Backward Compatible:** Existing code works unchanged
- **Fallback Behavior:** When optimizer is not set, uses simple gradient descent with `learning_rate`
- **Automatic Management:** `update_weights()` automatically calls `zero_grad()`

### New Methods

```cpp
// Set optimizer and register parameters
void set_optimizer(Optimizer* opt);

// Explicitly register parameters with optimizer
void register_parameters();

// Updated weight update (now uses optimizer if available)
void update_weights();
```

### Parameter Registration

When `set_optimizer()` is called, FeedForward registers 4 parameter groups:

```cpp
ff.set_optimizer(&optimizer);
// Internally registers:
//   - W1 and W1_grad (d_model × d_ff parameters)
//   - W2 and W2_grad (d_ff × d_model parameters)
//   - b1 and b1_grad (d_ff parameters)
//   - b2 and b2_grad (d_model parameters)
```

### Weight Update Behavior

With Optimizer (New Approach):

```cpp
void FeedForward::update_weights() {
    if (optimizer) {
        optimizer->step();  // Uses Adam/AdamW/SGD with momentum
    } else {
        // Fallback to simple gradient descent
        W1.apply_gradients(W1_grad, learning_rate);
        W2.apply_gradients(W2_grad, learning_rate);
        for (int i = 0; i < d_ff; ++i) {
            b1(0, i) -= learning_rate * b1_grad(0, i);
        }
        for (int i = 0; i < d_model; ++i) {
            b2(0, i) -= learning_rate * b2_grad(0, i);
        }
    }
    zero_grad();  // Always called automatically
}
```

### Optimizer Usage Patterns

#### Pattern 1: Training with Adam Optimizer

```cpp
FeedForward ff(512, 2048);
Optimizer adam_optimizer(OptimizerType::ADAM, 0.001f);
adam_optimizer.set_betas(0.9f, 0.999f);

ff.set_optimizer(&adam_optimizer);

for (int epoch = 0; epoch < epochs; ++epoch) {
    for (auto& batch : data) {
        Matrix output = ff.forward(batch.input);
        Matrix grad = compute_loss_gradient(output, batch.target);
        ff.backward(grad);
        ff.update_weights();  // Uses Adam
    }
}
```

#### Pattern 2: Learning Rate Scheduling

```cpp
FeedForward ff(512, 2048);
Optimizer optimizer(OptimizerType::ADAM, 0.001f);
ff.set_optimizer(&optimizer);

for (int epoch = 0; epoch < epochs; ++epoch) {
    // Decay learning rate
    if (epoch > 0 && epoch % 10 == 0) {
        float new_lr = optimizer.learning_rate * 0.9f;
        optimizer.set_learning_rate(new_lr);
        std::cout << "LR decreased to: " << new_lr << std::endl;
    }

    // Training loop
    for (auto& batch : data) {
        ff.forward(batch.input);
        ff.backward(grad);
        ff.update_weights();
    }
}
```

#### Pattern 3: Switching Optimizers

```cpp
FeedForward ff(512, 2048);

// Start with Adam for fast convergence
Optimizer adam(OptimizerType::ADAM, 0.001f);
ff.set_optimizer(&adam);
train_epochs(ff, 50);

// Switch to SGD for fine-tuning
Optimizer sgd(OptimizerType::SGD, 0.0001f);
ff.set_optimizer(&sgd);
train_epochs(ff, 10);
```

#### Pattern 4: Backward Compatible (No Optimizer)

```cpp
FeedForward ff(512, 2048);
ff.learning_rate = 0.001f;

// Works exactly as before - no optimizer needed
for (auto& batch : data) {
    Matrix output = ff.forward(batch.input);
    ff.backward(grad);
    ff.update_weights();  // Uses simple gradient descent
}
```

#### Pattern 5: Multi-Component Training

```cpp
// Train entire transformer block with same optimizer
Optimizer shared_optimizer(OptimizerType::ADAMW, 0.0001f);
shared_optimizer.set_weight_decay(0.01f);

MultiHeadAttention attention(512, 8);
FeedForward feedforward(512, 2048);
LayerNorm norm1(512), norm2(512);

attention.set_optimizer(&shared_optimizer);
feedforward.set_optimizer(&shared_optimizer);
norm1.set_optimizer(&shared_optimizer);
norm2.set_optimizer(&shared_optimizer);

// All components now share same optimization strategy
for (auto& batch : data) {
    // Forward
    Matrix attn_out = attention.forward(batch);
    Matrix ff_out = feedforward.forward(attn_out);
    Matrix final_out = norm2.forward(ff_out);

    // Backward
    Matrix grad = compute_loss_gradient(final_out, batch.target);
    Matrix grad_ff = norm2.backward(grad);
    Matrix grad_attn = feedforward.backward(grad_ff);
    attention.backward(grad_attn);

    // Update all with shared optimizer
    attention.update_weights();
    feedforward.update_weights();
    norm1.update_weights();
    norm2.update_weights();
}
```

### Migration Guide

Old Code (Version 1.0):

```cpp
FeedForward ff(512, 2048);
ff.learning_rate = 0.001f;

ff.forward(input);
ff.backward(grad);
ff.update_weights();
```

New Code (Version 1.1 - with Optimizer):

```cpp
FeedForward ff(512, 2048);
Optimizer opt(OptimizerType::ADAM, 0.001f);
ff.set_optimizer(&opt);

ff.forward(input);
ff.backward(grad);
ff.update_weights();  // Now uses Adam
```

Important Notes:

- Old code continues to work without modification
- `learning_rate` is still respected when optimizer is nullptr
- `update_weights()` now automatically calls `zero_grad()`
- No need to manually zero gradients anymore

---

## Usage Patterns

### Basic Training Loop

```cpp
FeedForward ff(512, 2048);
ff.learning_rate = 0.001f;

for (int epoch = 0; epoch < num_epochs; ++epoch) {
    for (auto& batch : training_data) {
        // Forward pass
        Matrix output = ff.forward(batch.input);

        // Compute loss gradient (from loss function)
        Matrix grad_output = compute_loss_gradient(output, batch.target);

        // Backward pass
        Matrix grad_input = ff.backward(grad_output);

        // Update weights
        ff.update_weights();
    }
}
```

### Integration with Transformer Block

```cpp
class TransformerBlock {
    MultiHeadAttention attention;
    FeedForward feedforward;
    LayerNorm norm1, norm2;

    Matrix forward(const Matrix& input) {
        // Attention sublayer with residual
        Matrix attn_out = attention.forward(input);
        Matrix residual1 = input + attn_out;
        Matrix normed1 = norm1.forward(residual1);

        // Feed-forward sublayer with residual
        Matrix ff_out = feedforward.forward(normed1);
        Matrix residual2 = normed1 + ff_out;
        Matrix normed2 = norm2.forward(residual2);

        return normed2;
    }
};
```

### Gradient Accumulation for Large Batches

```cpp
int accumulation_steps = 4;
ff.zero_grad();

for (int i = 0; i < accumulation_steps; ++i) {
    Matrix output = ff.forward(mini_batches[i]);
    Matrix grad = compute_gradient(output);
    ff.backward(grad);
    // Gradients accumulate, no weight update yet
}

// Update with accumulated gradients
ff.update_weights();  // Equivalent to larger batch
```

### Weight Persistence

```cpp
// Training phase
FeedForward ff(512, 2048);
// ... training code ...
ff.save_weights("checkpoint_epoch_100.bin");

// Inference phase
FeedForward ff_inference(512, 2048);
ff_inference.load_weights("checkpoint_epoch_100.bin");

// Use for prediction without training
Matrix prediction = ff_inference.forward(test_input);
```

### Different Configurations

```cpp
// Small model (mobile/edge)
FeedForward ff_small(256, 1024);  // 2x expansion

// Standard transformer
FeedForward ff_standard(512, 2048);  // 4x expansion

// Large model
FeedForward ff_large(1024, 4096);  // 4x expansion

// Very large model
FeedForward ff_xl(2048, 8192);  // 4x expansion
```

---

## Performance Considerations

### Computational Bottlenecks

Matrix Multiplications:

- First layer: O(seq_len × d_model × d_ff)
- Second layer: O(seq_len × d_ff × d_model)
- Dominant operations in feed-forward network

Optimization Strategies:

1. Use optimized BLAS libraries (if available)
2. Batch process multiple sequences together
3. Utilize GPU acceleration for matrix operations
4. Consider mixed-precision training (FP16/FP32)

### Memory Optimization

Activation Checkpointing:

```cpp
// Standard approach: Cache all activations
Matrix output = ff.forward(input);  // Caches 3 matrices

// Memory-efficient: Recompute activations during backward
// (Not implemented by default, requires modification)
```

Gradient Accumulation:

```cpp
// Instead of processing large batch at once:
for (int i = 0; i < num_mini_batches; ++i) {
    ff.forward(mini_batch[i]);
    ff.backward(grad[i]);
    // Don't update weights yet
}
ff.update_weights();  // Single update for all mini-batches
```

### Batch Size Impact

Small Batches (seq_len = 1-10):

- Low memory usage
- Less efficient matrix operations
- Good for inference

Medium Batches (seq_len = 10-100):

- Balanced efficiency
- Good for training

Large Batches (seq_len = 100+):

- High memory usage
- More efficient matrix operations
- May require gradient accumulation

### Numerical Stability

Bias Addition:

- Broadcasting ensures consistent behavior
- No accumulation errors

GELU Computation:

- Approximation formula is numerically stable
- No special handling needed for extreme values

Gradient Flow:

- GELU provides better gradient flow than ReLU
- Less prone to dead neurons
- Smoother optimization landscape

---

## Integration with Transformers

### Position in Transformer Architecture

```text
Input Embedding
    ↓
Positional Encoding
    ↓
┌─────────────────────┐
│ Encoder Block 1     │
│  ├─ Multi-Head Attn │
│  ├─ Add & Norm      │
│  ├─ FeedForward  ←──│ This class
│  └─ Add & Norm      │
└─────────────────────┘
    ↓
┌─────────────────────┐
│ Encoder Block 2     │
│  ├─ Multi-Head Attn │
│  ├─ Add & Norm      │
│  ├─ FeedForward  ←──│ This class
│  └─ Add & Norm      │
└─────────────────────┘
    ↓
   ...
    ↓
Final Layer Norm
    ↓
Output
```

### Residual Connection Pattern

```cpp
// Typical usage in transformer block
Matrix ff_input = normed_attention_output;
Matrix ff_output = feedforward.forward(ff_input);
Matrix residual = ff_input + ff_output;  // Residual connection
Matrix final_output = layer_norm.forward(residual);
```

Benefits of Residual Connections:

- Gradient flow through skip connections
- Easier to train deep networks
- Model can learn identity function if needed

### Pre-Norm vs Post-Norm

Post-Norm (Original Transformer):

```cpp
x = LayerNorm(x + Attention(x))
x = LayerNorm(x + FeedForward(x))
```

Pre-Norm (Modern Transformers):

```cpp
x = x + Attention(LayerNorm(x))
x = x + FeedForward(LayerNorm(x))
```

Trade-offs:

- Post-Norm: Better final performance, harder to train
- Pre-Norm: Easier to train, slightly worse performance

### Typical Hyperparameters

GPT-2 Small:

- d_model: 768
- d_ff: 3072 (4x expansion)

BERT Base:

- d_model: 768
- d_ff: 3072 (4x expansion)

GPT-3:

- d_model: 12288
- d_ff: 49152 (4x expansion)

T5:

- d_model: 512
- d_ff: 2048 (4x expansion)

---

## Common Issues and Solutions

### Issue: Exploding Gradients

Symptoms:

- Gradient norm > 100
- NaN values in weights
- Training divergence

Solutions:

```cpp
// 1. Gradient clipping
ff.clip_gradients(5.0f);

// 2. Lower learning rate
ff.learning_rate = 0.0001f;

// 3. Better initialization
// (Already implemented with He initialization)
```

### Issue: Vanishing Gradients

Symptoms:

- Gradient norm < 1e-6
- No learning progress
- Weights barely change

Solutions:

```cpp
// 1. Check activation function (GELU is good)
// 2. Verify residual connections
// 3. Increase learning rate
ff.learning_rate = 0.01f;

// 4. Check weight initialization
// (Should be automatic with He init)
```

### Issue: Slow Training

Symptoms:

- High wall-clock time per iteration
- Low GPU/CPU utilization

Solutions:

```cpp
// 1. Increase batch size (if memory allows)
Matrix large_batch(128, 512);  // Instead of (16, 512)

// 2. Use gradient accumulation
for (int i = 0; i < 8; ++i) {
    ff.forward(mini_batch[i]);
    ff.backward(grad[i]);
}
ff.update_weights();

// 3. Profile code to find bottlenecks
```

### Issue: Memory Overflow

Symptoms:

- Out of memory errors
- System slowdown

Solutions:

```cpp
// 1. Reduce batch size
Matrix smaller_batch(8, 512);  // Instead of (128, 512)

// 2. Reduce model size
FeedForward ff_smaller(512, 1024);  // Instead of (512, 2048)

// 3. Use gradient accumulation
// (Process smaller batches, accumulate gradients)
```

---

## Debugging and Monitoring

### Monitoring Gradients

```cpp
// During training loop
ff.forward(input);
ff.backward(grad_output);

float norm = ff.get_gradient_norm();
std::cout << "Gradient norm: " << norm << std::endl;

if (std::isnan(norm) || std::isinf(norm)) {
    std::cerr << "Invalid gradient detected!" << std::endl;
    // Take corrective action
}
```

### Weight Statistics

```cpp
// Pseudo-code for monitoring weight distribution
void print_weight_stats(const FeedForward& ff) {
    // Access to internal matrices would require friend function
    // or getter methods

    std::cout << "W1 mean: " << compute_mean(W1) << std::endl;
    std::cout << "W1 std: " << compute_std(W1) << std::endl;
    std::cout << "W2 mean: " << compute_mean(W2) << std::endl;
    std::cout << "W2 std: " << compute_std(W2) << std::endl;
}
```

### Output Validation

```cpp
Matrix output = ff.forward(input);

// Check for invalid values
for (int i = 0; i < output.rows; ++i) {
    for (int j = 0; j < output.cols; ++j) {
        if (std::isnan(output(i,j)) || std::isinf(output(i,j))) {
            std::cerr << "Invalid output at [" << i << "," << j << "]"
                      << std::endl;
        }
    }
}
```

### Configuration Verification

```cpp
ff.print_config("Layer 3 FeedForward");
// Output:
// Layer 3 FeedForward Configuration:
//   Model Dimension (d_model): 512
//   Feed-Forward Dimension (d_ff): 2048
//   Expansion Ratio: 4x
//   Total Parameters: 2099712
//   ...
```

---

## Best Practices

### 1. Initialization

✅ **Do:**

- Use provided He initialization (automatic)
- Verify dimensions match your model architecture

❌ **Don't:**

- Manually initialize with arbitrary values
- Use same initialization for all layers

### 2. Learning Rate

✅ **Do:**

- Start with default (0.001)
- Use learning rate scheduling
- Monitor gradient norms

❌ **Don't:**

- Use very large learning rates (>0.1)
- Keep learning rate constant throughout training
- Ignore gradient behavior

### 3. Gradient Management

✅ **Do:**

- Call `zero_grad()` before each training iteration (or use `update_weights()`)
- Monitor gradient norms regularly
- Apply gradient clipping when needed

❌ **Don't:**

- Forget to zero gradients between iterations
- Ignore exploding/vanishing gradient warnings
- Apply extreme gradient clipping (<0.1)

### 4. Memory Management

✅ **Do:**

- Process appropriate batch sizes
- Use gradient accumulation for large effective batches
- Monitor memory usage

❌ **Don't:**

- Process entire dataset in one forward pass
- Ignore out-of-memory warnings
- Create unnecessary copies of large matrices

### 5. Model Persistence

✅ **Do:**

- Save checkpoints regularly
- Verify dimensions when loading
- Test loaded weights before deployment

❌ **Don't:**

- Overwrite previous checkpoints without backup
- Load weights without dimension validation
- Deploy without testing loaded model

---

## Advanced Topics

### Custom Activation Functions

The current implementation uses GELU. To use a different activation:

```cpp
// Modify forward pass:
// hidden = Activation::relu(hidden);  // Instead of GELU

// Modify backward pass:
// Matrix activation_grad = Activation::relu_derivative(cached_hidden);
```

Supported Alternatives:

- ReLU: `Activation::relu()` and `Activation::relu_derivative()`
- Tanh: `Activation::tanh()` and `Activation::tanh_derivative()`
- Sigmoid: `Activation::sigmoid()` and `Activation::sigmoid_derivative()`

### Dropout Implementation

To add dropout (not currently implemented):

```cpp
// After first activation:
if (training_mode) {
    hidden_activated = apply_dropout(hidden_activated, dropout_rate);
}

// Dropout should be disabled during inference
```

### Layer Freezing

To freeze weights (prevent updates):

```cpp
// Set learning rate to 0 for this layer
ff.learning_rate = 0.0f;

// Or skip backward/update for frozen layers
if (!layer_frozen) {
    ff.backward(grad_output);
    ff.update_weights();
}
```

### Mixed Precision Training

For memory efficiency and speed (requires external support):

```cpp
// Convert weights to FP16 for forward pass
// Keep gradients in FP32 for stability
// (Requires modification to Matrix class)
```

---

## API Reference Summary

### Constructors

- `FeedForward(int d_model, int d_ff)` - Initialize with dimensions

### Core Methods

- `Matrix forward(const Matrix& input)` - Forward propagation
- `Matrix backward(const Matrix& grad_output)` - Backward propagation
- `void update_weights()` - Apply gradients and zero them
- `void zero_grad()` - Reset gradient accumulators

### Gradient Management Methods

- `float get_gradient_norm() const` - Get L2 norm of all gradients
- `void clip_gradients(float max_norm)` - Scale gradients to max norm

### Persistence

- `void save_weights(const std::string& filename) const` - Save to file
- `void load_weights(const std::string& filename)` - Load from file

### Utilities

- `void print_config(const std::string& name) const` - Display configuration
- `int get_d_model() const` - Get input/output dimension
- `int get_d_ff() const` - Get hidden dimension

### Public Data Members

- `float learning_rate` - Learning rate (default: 0.001)

---

## Example Use Cases

### 1. Standard Transformer Encoder

```cpp
FeedForward ff(512, 2048);
ff.learning_rate = 0.0001f;

// Within encoder block:
Matrix attn_out = attention.forward(input);
Matrix residual1 = input + attn_out;
Matrix normed1 = layer_norm1.forward(residual1);

Matrix ff_out = ff.forward(normed1);
Matrix residual2 = normed1 + ff_out;
Matrix output = layer_norm2.forward(residual2);
```

### 2. Language Model Training

```cpp
std::vector<FeedForward> ff_layers;
for (int i = 0; i < num_layers; ++i) {
    ff_layers.push_back(FeedForward(768, 3072));  // GPT-2 size
}

for (int epoch = 0; epoch < epochs; ++epoch) {
    for (auto& batch : data) {
        // Forward through all layers
        Matrix x = embeddings;
        for (auto& ff : ff_layers) {
            x = ff.forward(x);
        }

        // Backward through all layers (reverse order)
        Matrix grad = compute_loss_gradient(x, targets);
        for (int i = num_layers - 1; i >= 0; --i) {
            grad = ff_layers[i].backward(grad);
            ff_layers[i].update_weights();
        }
    }
}
```

### 3. Transfer Learning

```cpp
// Load pre-trained weights
FeedForward ff(512, 2048);
ff.load_weights("pretrained_model.bin");

// Fine-tune with lower learning rate
ff.learning_rate = 0.00001f;

for (auto& batch : fine_tuning_data) {
    Matrix output = ff.forward(batch.input);
    Matrix grad = compute_gradient(output, batch.target);
    ff.backward(grad);
    ff.update_weights();
}
```

---

## Performance Benchmarks

### Typical Performance (CPU, single-threaded)

Configuration: d_model=512, d_ff=2048, seq_len=128

|Operation|Time (ms)|Memory (MB)|
|-----------|-----------|-------------|
|Constructor|5-10|16.5|
|Forward pass|20-30|+2.0|
|Backward pass|30-50|+2.0|
|Update weights|5-10|0|

Scaling with Sequence Length:

|seq_len|Forward (ms)|Backward (ms)|
|---------|--------------|---------------|
|16|3-5|5-8|
|64|12-18|20-30|
|128|20-30|30-50|
|512|80-120|120-200|

Parameter Counts for Common Configurations:

|d_model|d_ff|Parameters|Memory|
|---------|------|------------|--------|
|256|1024|525,568|2.0 MB|
|512|2048|2,099,712|8.0 MB|
|768|3072|4,722,432|18.0 MB|
|1024|4096|8,392,704|32.0 MB|

---

## Summary

The `FeedForward` class provides a production-ready implementation of the position-wise feed-forward network used in transformer architectures. Key features include:

✅ **Robust Implementation:**

- Xavier/He weight initialization
- GELU activation for better gradient flow
- Proper gradient computation and accumulation

✅ **Training Support:**

- Gradient clipping to prevent exploding gradients
- Gradient monitoring for debugging
- Weight persistence for checkpointing
- Advanced optimizer integration (Adam, AdamW, SGD)

✅ **Flexibility:**

- Configurable dimensions (d_model, d_ff)
- Adjustable learning rate
- Integration-ready for transformer blocks
- Optional optimizer for advanced optimization strategies

✅ **Production Ready:**

- Comprehensive error handling
- Memory-efficient caching
- Well-documented API
- Full backward compatibility

The implementation follows modern best practices for deep learning components and is suitable for both research and production use cases.

---

## Recent Updates

### Version 1.1 (January 24, 2026)

#### Major Feature: Optimizer Integration

Added support for advanced optimization algorithms through the `Optimizer` class:

New Functionality:

- `set_optimizer(Optimizer* opt)` - Register optimizer and parameters
- `register_parameters()` - Explicitly register parameter groups
- Enhanced `update_weights()` - Uses optimizer when available, falls back to gradient descent

Parameter Management:

- Registers 4 parameter groups: W1, W2, b1, b2
- Each group linked with corresponding gradient matrix
- Supports Adam, AdamW, SGD with momentum

Backward Compatibility:

- Optimizer pointer defaults to `nullptr`
- Existing code works without modification
- `learning_rate` still used when optimizer not set
- Simple gradient descent fallback maintained

Benefits:

- Better convergence with adaptive learning rates
- Built-in weight decay and momentum support
- Learning rate scheduling capability
- Unified optimization across model components

Test Coverage:

- Added 12 comprehensive optimizer integration tests
- Total test suite: 51/51 tests passing
- Tests cover all optimizer types and edge cases

Migration:

```cpp
// Old code (still works):
FeedForward ff(512, 2048);
ff.learning_rate = 0.001f;
ff.update_weights();

// New code (with optimizer):
FeedForward ff(512, 2048);
Optimizer opt(OptimizerType::ADAM, 0.001f);
ff.set_optimizer(&opt);
ff.update_weights();  // Uses Adam
```

Documentation Updates:

- Added Optimizer Integration section
- Updated Class Architecture with optimizer member
- Added 5 optimizer usage patterns
- Included migration guide for Version 1.0 → 1.1

---

End of Documentation
