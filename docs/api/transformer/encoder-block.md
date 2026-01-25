# EncoderBlock Class - Technical Context Documentation

## Overview

The `EncoderBlock` class implements a single layer of the transformer encoder architecture, combining multi-head self-attention and position-wise feed-forward networks with residual connections and layer normalization. This is the fundamental building block of transformer-based models.

**Files:**
- `src/EncoderBlock.hpp` - Header file with class declaration and interface
- `src/EncoderBlock.cpp` - Implementation file with all method definitions
- `src/EncoderBlockExample.cpp` - Standalone example demonstrating usage

**Purpose:** Provide a complete transformer encoder layer that processes sequences through self-attention and feed-forward transformations, enabling the model to capture complex contextual relationships while maintaining gradient flow through residual connections.

---

## Table of Contents
1. [Mathematical Foundation](#mathematical-foundation)
2. [Class Architecture](#class-architecture)
3. [Implementation Details](#implementation-details)
4. [Forward Pass](#forward-pass)
5. [Backward Pass](#backward-pass)
6. [Residual Connections](#residual-connections)
7. [Layer Normalization](#layer-normalization)
8. [Gradient Management](#gradient-management)
9. [Usage Patterns](#usage-patterns)
10. [Integration with Transformers](#integration-with-transformers)

---

## Mathematical Foundation

### Core Architecture

The encoder block applies two sub-layers with residual connections and layer normalization:

```
EncoderBlock(x) = LayerNorm(x + FFN(LayerNorm(x + Attention(x))))
```

**Expanded Step-by-Step:**
```
Step 1: attn_output = MultiHeadAttention(x, x, x, mask)
Step 2: residual1 = x + attn_output
Step 3: normed1 = LayerNorm(residual1)
Step 4: ff_output = FeedForward(normed1)
Step 5: residual2 = normed1 + ff_output
Step 6: output = LayerNorm(residual2)
```

### Sub-Components

#### 1. Multi-Head Self-Attention

Computes weighted relationships between all positions in the sequence:

```
Attention(Q, K, V) = softmax(QK^T / √d_k) V
MultiHead(x) = Concat(head₁, ..., head_h) W_o
```

**Properties:**
- Captures long-range dependencies
- Learns different relationship types through multiple heads
- Complexity: O(seq_len² × d_model)

#### 2. Position-wise Feed-Forward Network

Applies non-linear transformations independently at each position:

```
FFN(x) = GELU(xW₁ + b₁)W₂ + b₂
```

**Properties:**
- Adds non-linear transformation capacity
- Applied identically at each position
- Expands to d_ff (typically 4 × d_model) then projects back
- Complexity: O(seq_len × d_model × d_ff)

#### 3. Residual Connections

Adds input directly to output of each sub-layer:

```
output = x + SubLayer(x)
```

**Benefits:**
- Enables gradient flow in deep networks
- Prevents vanishing gradients
- Allows learning of identity function when needed
- Stabilizes training

#### 4. Layer Normalization

Normalizes activations across features for each sample:

```
LayerNorm(x) = γ · (x - μ) / √(σ² + ε) + β
```

**Benefits:**
- Stabilizes training
- Reduces internal covariate shift
- Enables higher learning rates
- Works well with variable batch sizes

### Complexity Analysis

**Time Complexity:**
- Multi-head attention: O(seq_len² × d_model)
- Feed-forward network: O(seq_len × d_model × d_ff)
- Layer normalization (2×): O(2 × seq_len × d_model)
- Residual additions (2×): O(2 × seq_len × d_model)
- **Total: O(seq_len² × d_model + seq_len × d_model × d_ff)**

For typical configuration (d_ff = 4 × d_model):
- If seq_len < d_model: Feed-forward dominates
- If seq_len > d_model: Attention dominates

**Space Complexity:**
- Attention weights: O(d_model²) - projection matrices
- Feed-forward weights: O(2 × d_model × d_ff)
- Layer norm parameters: O(2 × d_model)
- Cached activations: O(seq_len × d_model)
- **Total: O(d_model² + d_model × d_ff + seq_len × d_model)**

**Parameter Count:**
```
Attention: 4 × d_model² (Q, K, V, output projections)
Feed-forward: 2 × d_model × d_ff + d_ff + d_model
Layer norm: 2 × 2 × d_model (gamma, beta for both layers)

Total = 4d_model² + 2d_model·d_ff + d_ff + 5d_model
```

Example (d_model=512, num_heads=8, d_ff=2048):
```
Total = 4(512²) + 2(512)(2048) + 2048 + 5(512)
      = 1,048,576 + 2,097,152 + 2,048 + 2,560
      = 3,150,336 parameters
```

---

## Class Architecture

### Private Members

```cpp
// Core components
std::unique_ptr<MultiHeadAttention> attention;
std::unique_ptr<FeedForward> feed_forward;
std::unique_ptr<LayerNorm> norm1;
std::unique_ptr<LayerNorm> norm2;

// Hyperparameters
int d_model;         // Model dimension (e.g., 512)
int num_heads;       // Number of attention heads (e.g., 8)
int d_ff;            // Feed-forward dimension (e.g., 2048)
float dropout_rate;  // Dropout probability (e.g., 0.1)

// Cached values for backward pass
Matrix cached_input;             // Original input
Matrix cached_attn_output;       // Attention output
Matrix cached_residual1;         // After first residual
Matrix cached_normed1;           // After first layer norm
Matrix cached_ff_output;         // Feed-forward output
Matrix cached_residual2;         // After second residual
```

### Public Members

```cpp
float learning_rate;  // Learning rate for gradient updates (default: 0.001)
```

### Component Responsibilities

**MultiHeadAttention:**
- Computes self-attention across sequence positions
- Learns contextual relationships
- Manages Q, K, V projections and output projection
- Handles attention masking

**FeedForward:**
- Applies position-wise transformations
- Provides non-linear modeling capacity
- Expands to higher dimension then projects back
- Uses GELU activation

**LayerNorm (×2):**
- Normalizes activations for training stability
- One after attention sub-layer
- One after feed-forward sub-layer
- Learnable scale (gamma) and shift (beta) parameters

### Memory Layout

**Cached Matrices (for backpropagation):**
- `cached_input`: [seq_len, d_model] - needed for residual gradient
- `cached_attn_output`: [seq_len, d_model] - needed for residual gradient
- `cached_residual1`: [seq_len, d_model] - needed for norm1 backward
- `cached_normed1`: [seq_len, d_model] - needed for ff backward
- `cached_ff_output`: [seq_len, d_model] - needed for residual gradient
- `cached_residual2`: [seq_len, d_model] - needed for norm2 backward

**Total Cache Size:** 6 × seq_len × d_model floating-point values

---

## Implementation Details

### Constructor

```cpp
EncoderBlock::EncoderBlock(int d_model, int num_heads, int d_ff, float dropout)
```

**Initialization Steps:**
1. Store hyperparameters (d_model, num_heads, d_ff, dropout_rate)
2. Set default learning rate (0.001)
3. Create MultiHeadAttention instance
4. Create FeedForward instance
5. Create two LayerNorm instances
6. Propagate learning rate to all components

**Validation:**
- d_model must be divisible by num_heads (enforced by MultiHeadAttention)
- All dimensions must be positive

**Example:**
```cpp
EncoderBlock encoder(512, 8, 2048, 0.1f);
// Creates block with:
//   - 512-dim embeddings
//   - 8 attention heads (64 dims per head)
//   - 2048-dim feed-forward hidden layer
//   - 10% dropout rate
```

### Method: `forward(const Matrix& input, const Matrix* mask)`

**Purpose:** Process input through encoder block

**Algorithm:**
```
1. Cache input
2. Compute multi-head attention: attn_output = attention->forward(input, mask)
3. Cache attention output
4. First residual: residual1 = input + attn_output
5. Cache residual1
6. First layer norm: normed1 = norm1->forward(residual1)
7. Cache normed1
8. Feed-forward: ff_output = feed_forward->forward(normed1)
9. Cache ff_output
10. Second residual: residual2 = normed1 + ff_output
11. Cache residual2
12. Second layer norm: normed2 = norm2->forward(residual2)
13. Return normed2
```

**Residual Addition Implementation:**
```cpp
// Element-wise addition
for (int i = 0; i < rows; ++i) {
    for (int j = 0; j < cols; ++j) {
        residual(i, j) = input(i, j) + layer_output(i, j);
    }
}
```

**Input:** 
- `input`: [seq_len, d_model]
- `mask`: [seq_len, seq_len] or nullptr

**Output:** [seq_len, d_model]

**Time Complexity:** O(seq_len² × d_model + seq_len × d_model × d_ff)

### Method: `backward(const Matrix& grad_output)`

**Purpose:** Compute gradients for all parameters and return gradient w.r.t. input

**Algorithm:**
```
1. Backward through second LayerNorm
   grad_residual2 = norm2->backward(grad_output)

2. Split gradient at second residual connection
   grad_normed1 = grad_residual2  (copy)
   grad_ff_output = grad_residual2  (copy)

3. Backward through FeedForward
   grad_normed1_from_ff = feed_forward->backward(grad_ff_output)

4. Accumulate gradients from both paths
   grad_normed1 += grad_normed1_from_ff

5. Backward through first LayerNorm
   grad_residual1 = norm1->backward(grad_normed1)

6. Split gradient at first residual connection
   grad_input = grad_residual1  (copy)
   grad_attn_output = grad_residual1  (copy)

7. Backward through MultiHeadAttention
   grad_input_from_attn = attention->backward(grad_attn_output)

8. Accumulate gradients from both paths
   grad_input += grad_input_from_attn

9. Return grad_input
```

**Gradient Flow Diagram:**
```
grad_output
    ↓
[LayerNorm2 backward]
    ↓
grad_residual2
    ├─→ grad_normed1 ────────────────┐
    │                                 │
    └─→ grad_ff_output                │
            ↓                         │
    [FeedForward backward]            │
            ↓                         │
    grad_normed1_from_ff ────→ [Accumulate]
                                      ↓
                              grad_normed1 (total)
                                      ↓
                          [LayerNorm1 backward]
                                      ↓
                              grad_residual1
                                 ├─→ grad_input ──────────────┐
                                 │                             │
                                 └─→ grad_attn_output          │
                                         ↓                     │
                                [Attention backward]           │
                                         ↓                     │
                                grad_input_from_attn ──→ [Accumulate]
                                                               ↓
                                                      grad_input (total)
                                                               ↓
                                                            Return
```

**Residual Gradient Splitting:**
```cpp
// Gradient flows through both paths of residual connection
Matrix grad_path1(grad_residual.rows, grad_residual.cols);
Matrix grad_path2(grad_residual.rows, grad_residual.cols);

// Copy gradient to both paths
for (int i = 0; i < grad_residual.rows; ++i) {
    for (int j = 0; j < grad_residual.cols; ++j) {
        grad_path1(i, j) = grad_residual(i, j);
        grad_path2(i, j) = grad_residual(i, j);
    }
}
```

**Input:** `grad_output` [seq_len, d_model]  
**Output:** `grad_input` [seq_len, d_model]

**Side Effects:** Accumulates gradients in all component layers

### Method: `update_weights()`

**Purpose:** Apply gradient descent to all learnable parameters

**Algorithm:**
```
1. Propagate current learning rate to all components
2. attention->update_weights()  (updates Q, K, V, output projections)
3. feed_forward->update_weights()  (updates W1, W2, b1, b2)
4. LayerNorm updates handled internally during backward
```

**Weight Update Formula:**
```
W = W - learning_rate × ∇W
```

**Note:** LayerNorm updates gamma and beta internally during backward pass, so no explicit update call needed.

### Method: `zero_grad()`

**Purpose:** Reset all accumulated gradients to zero

**Algorithm:**
```
1. attention->zero_grad()
2. feed_forward->zero_grad()
3. norm1->zero_grad()
4. norm2->zero_grad()
```

**Usage:** Call before each new backward pass to prevent gradient accumulation.

### Method: `get_gradient_norm()`

**Purpose:** Compute L2 norm of all gradients combined

**Algorithm:**
```
1. attn_norm = attention->get_gradient_norm()
2. ff_norm = feed_forward->get_gradient_norm()
3. norm_sq = attn_norm² + ff_norm²
4. return √norm_sq
```

**Note:** LayerNorm gradients not included (API limitation), but they are typically small relative to attention and feed-forward.

**Returns:** Float value representing total gradient magnitude

**Use Cases:**
- Monitor gradient flow during training
- Detect vanishing/exploding gradients
- Determine when to apply gradient clipping

### Method: `clip_gradients(float max_norm)`

**Purpose:** Prevent exploding gradients by scaling

**Algorithm:**
```
1. current_norm = get_gradient_norm()
2. if current_norm > max_norm:
       scale = max_norm / current_norm
       attention->clip_gradients(attention_norm × scale)
       feed_forward->clip_gradients(ff_norm × scale)
```

**Example:**
```cpp
encoder_block.forward(input);
encoder_block.backward(grad_output);

float norm = encoder_block.get_gradient_norm();
if (norm > 5.0f) {
    encoder_block.clip_gradients(5.0f);
}

encoder_block.update_weights();
```

### Method: `save_weights(const std::string& filename)`

**Purpose:** Persist all learnable parameters to disk

**File Format:**
```
[Header - 16 bytes]
int32: d_model
int32: num_heads
int32: d_ff
float32: dropout_rate

[LayerNorm1 Parameters - 2 × d_model × 4 bytes]
float32[d_model]: gamma1
float32[d_model]: beta1

[LayerNorm2 Parameters - 2 × d_model × 4 bytes]
float32[d_model]: gamma2
float32[d_model]: beta2

[Component Files]
<filename>_attention.bin (MultiHeadAttention weights)
<filename>_feedforward.bin (FeedForward weights)
```

**Total Size:**
- Header: 16 bytes
- LayerNorm params: 4 × d_model × 4 bytes
- Attention file: ~4 × d_model² × 4 bytes
- Feed-forward file: ~(2 × d_model × d_ff + d_ff + d_model) × 4 bytes

**Example:**
```cpp
encoder_block.save_weights("encoder_layer_3.bin");
// Creates:
//   encoder_layer_3.bin
//   encoder_layer_3_attention.bin
//   encoder_layer_3_feedforward.bin
```

### Method: `load_weights(const std::string& filename)`

**Purpose:** Load previously saved weights from disk

**Validation:**
- Checks dimension compatibility (d_model, num_heads, d_ff)
- Throws `std::runtime_error` on mismatch
- Throws `std::runtime_error` if file not found

**Algorithm:**
```
1. Open and read header
2. Validate dimensions match current instance
3. Read LayerNorm parameters (gamma, beta for both layers)
4. Set LayerNorm parameters via setters
5. Load component weights from separate files
```

**Example:**
```cpp
EncoderBlock encoder_block(512, 8, 2048);
encoder_block.load_weights("encoder_layer_3.bin");
// Loads all weights from saved files
```

### Method: `print_config(const std::string& name)`

**Purpose:** Display encoder block configuration and parameter count

**Output Example:**
```
EncoderBlock_0 Configuration:
  Model Dimension (d_model): 512
  Number of Heads: 8
  Feed-Forward Dimension (d_ff): 2048
  Dropout Rate: 0.1
  Learning Rate: 0.001
  Total Parameters: 3,150,336
    - Attention: 1,048,576
    - Feed-Forward: 2,099,712
    - Layer Norm: 2,048
```

**Parameter Calculation:**
```cpp
attn_params = d_model² × 4  // Q, K, V, output
ff_params = 2 × d_model × d_ff + d_ff + d_model  // W1, W2, b1, b2
norm_params = 4 × d_model  // gamma, beta for both norms
total = attn_params + ff_params + norm_params
```

---

## Forward Pass

### Detailed Computation Flow

**Step 1: Multi-Head Self-Attention**
```cpp
Matrix attn_output = attention->forward(input, mask);
// Computes: Attention(Q, K, V) = softmax(QK^T / √d_k) V
// For all heads, then concatenates and projects
```

**Step 2: First Residual Connection**
```cpp
Matrix residual1(input.rows, input.cols);
for (int i = 0; i < input.rows; ++i) {
    for (int j = 0; j < input.cols; ++j) {
        residual1(i, j) = input(i, j) + attn_output(i, j);
    }
}
```

**Why Residual?**
- Allows gradient to flow directly through skip connection
- Network can learn identity function if needed
- Prevents degradation in deep networks

**Step 3: First Layer Normalization**
```cpp
Matrix normed1 = norm1->forward(residual1);
// Normalizes each position independently across features
// output = gamma * (input - mean) / sqrt(var + eps) + beta
```

**Step 4: Feed-Forward Network**
```cpp
Matrix ff_output = feed_forward->forward(normed1);
// Applies: GELU(x·W1 + b1)·W2 + b2
// Expands to d_ff, then projects back to d_model
```

**Step 5: Second Residual Connection**
```cpp
Matrix residual2(normed1.rows, normed1.cols);
for (int i = 0; i < normed1.rows; ++i) {
    for (int j = 0; j < normed1.cols; ++j) {
        residual2(i, j) = normed1(i, j) + ff_output(i, j);
    }
}
```

**Step 6: Second Layer Normalization**
```cpp
Matrix normed2 = norm2->forward(residual2);
// Final normalization before output
```

### Caching Strategy

All intermediate activations are cached for efficient backpropagation:

| Cached Variable | Shape | Purpose |
|-----------------|-------|---------|
| `cached_input` | [seq_len, d_model] | Needed for first residual gradient |
| `cached_attn_output` | [seq_len, d_model] | Needed for attention gradient path |
| `cached_residual1` | [seq_len, d_model] | Needed for norm1 backward |
| `cached_normed1` | [seq_len, d_model] | Needed for feed-forward backward |
| `cached_ff_output` | [seq_len, d_model] | Needed for second residual gradient |
| `cached_residual2` | [seq_len, d_model] | Needed for norm2 backward |

**Memory Cost:** 6 × seq_len × d_model floating-point values

### Pre-Norm vs Post-Norm Architecture

**This Implementation (Post-Norm):**
```
x → Attention → Add → Norm → FeedForward → Add → Norm → output
    ↑___________|            ↑__________________|
```

**Alternative (Pre-Norm):**
```
x → Norm → Attention → Add → Norm → FeedForward → Add → output
           ↑_______________|        ↑___________________|
```

**Trade-offs:**
- **Post-Norm (implemented)**: Better final performance, harder to train
- **Pre-Norm**: Easier to train deep networks, slightly worse performance

---

## Backward Pass

### Gradient Flow Through Residual Connections

**Key Insight:** Residual connections split gradients into two paths:

```
output = input + layer(input)

∂Loss/∂input = ∂Loss/∂output × (1 + ∂layer/∂input)
             = ∂Loss/∂output + ∂Loss/∂output × ∂layer/∂input
```

This means gradient flows through BOTH:
1. **Direct path** (identity/skip connection)
2. **Indirect path** (through the layer)

### Step-by-Step Gradient Computation

**Step 1: Gradient Through Second LayerNorm**
```cpp
Matrix grad_residual2 = norm2->backward(grad_output);
// Computes: ∂Loss/∂residual2
```

**Step 2: Split at Second Residual**
```cpp
// residual2 = normed1 + ff_output
// Therefore:
grad_normed1 = grad_residual2;     // Path to normed1
grad_ff_output = grad_residual2;   // Path to ff_output
```

**Step 3: Backpropagate Through Feed-Forward**
```cpp
Matrix grad_normed1_from_ff = feed_forward->backward(grad_ff_output);
// Computes: ∂Loss/∂normed1 via feed-forward path
```

**Step 4: Accumulate Gradients**
```cpp
// Total gradient to normed1 comes from TWO sources:
// 1. Direct from residual connection
// 2. Through feed-forward layer
for (int i = 0; i < grad_normed1.rows; ++i) {
    for (int j = 0; j < grad_normed1.cols; ++j) {
        grad_normed1(i, j) += grad_normed1_from_ff(i, j);
    }
}
```

**Step 5: Gradient Through First LayerNorm**
```cpp
Matrix grad_residual1 = norm1->backward(grad_normed1);
// Computes: ∂Loss/∂residual1
```

**Step 6: Split at First Residual**
```cpp
// residual1 = input + attn_output
// Therefore:
grad_input = grad_residual1;           // Path to input
grad_attn_output = grad_residual1;     // Path to attention
```

**Step 7: Backpropagate Through Attention**
```cpp
Matrix grad_input_from_attn = attention->backward(grad_attn_output);
// Computes: ∂Loss/∂input via attention path
```

**Step 8: Final Accumulation**
```cpp
// Total gradient to input comes from TWO sources:
// 1. Direct from residual connection
// 2. Through attention layer
for (int i = 0; i < grad_input.rows; ++i) {
    for (int j = 0; j < grad_input.cols; ++j) {
        grad_input(i, j) += grad_input_from_attn(i, j);
    }
}
```

### Gradient Amplification Through Residuals

**Without Residual:**
```
∂Loss/∂input = ∂Loss/∂output × ∂output/∂input
```

**With Residual:**
```
∂Loss/∂input = ∂Loss/∂output × (1 + ∂layer/∂input)
             ≥ ∂Loss/∂output  (assuming ∂layer/∂input ≥ -1)
```

This ensures gradients don't vanish even in very deep networks!

### Gradient Equations

For each component:

**LayerNorm:**
```
∂Loss/∂x = ∂Loss/∂y × ∂LayerNorm/∂x
```

**FeedForward:**
```
∂Loss/∂x = ∂Loss/∂y × ∂FFN/∂x
         = ∂Loss/∂y × W₂^T × GELU'(hidden) × W₁^T
```

**Attention:**
```
∂Loss/∂x = ∂Loss/∂y × ∂Attention/∂x
         = complex chain rule through softmax and projections
```

**Residual:**
```
∂Loss/∂x = ∂Loss/∂(x + layer(x))
         = ∂Loss/∂output × (I + ∂layer/∂x)
```

---

## Residual Connections

### Mathematical Formulation

**Standard Layer:**
```
y = f(x)
```

**With Residual Connection:**
```
y = x + f(x)
```

### Benefits

**1. Gradient Flow**
```
∂y/∂x = I + ∂f/∂x

Even if ∂f/∂x → 0, we still have ∂y/∂x = I
```

This prevents vanishing gradients in deep networks!

**2. Identity Learning**

If optimal transformation is identity, residual connection allows:
```
f(x) = 0  →  y = x
```

Much easier to learn than forcing f(x) = x directly.

**3. Ensemble Behavior**

Residual networks can be viewed as ensembles of paths:
```
Output = Σ (paths through different combinations of layers)
```

Different gradient paths provide redundancy and robustness.

### Implementation Pattern

**This Encoder Block:**
```cpp
// First residual
attn_output = attention(input);
residual1 = input + attn_output;
normed1 = norm(residual1);

// Second residual
ff_output = feedforward(normed1);
residual2 = normed1 + ff_output;
output = norm(residual2);
```

**Alternative (Pre-Activation):**
```cpp
normed = norm(input);
attn_output = attention(normed);
residual = input + attn_output;
```

---

## Layer Normalization

### Purpose in Encoder Block

**Placement:**
- After each residual connection
- Before next sub-layer (in pre-norm variants)

**Function:**
```
LayerNorm(x) = γ · (x - μ) / √(σ² + ε) + β
```

Where:
- μ: mean across features for each sample
- σ²: variance across features for each sample
- γ (gamma): learned scale parameter
- β (beta): learned shift parameter
- ε (epsilon): small constant for numerical stability (1e-5)

### Why Layer Norm?

**1. Training Stability**
- Prevents activation explosion/vanishing
- Reduces internal covariate shift
- Enables higher learning rates

**2. Batch Independence**
- Normalizes across features, not batch
- Works with batch size = 1
- Essential for RNNs and variable sequences

**3. Representation Quality**
- Forces network to use standardized representations
- Prevents single features from dominating
- Improves generalization

### Interaction with Residuals

**Order Matters:**

**Post-Norm (this implementation):**
```
output = LayerNorm(input + SubLayer(input))
```
- Normalization after residual addition
- Better final performance
- Can be harder to optimize

**Pre-Norm:**
```
output = input + SubLayer(LayerNorm(input))
```
- Normalization before sub-layer
- Easier to train (especially deep networks)
- Slightly lower final performance

### Parameters

**For Each LayerNorm Layer:**
- Gamma (γ): [d_model] scale parameters
- Beta (β): [d_model] shift parameters
- **Total per layer:** 2 × d_model parameters

**For Entire Encoder Block:**
- Two LayerNorm layers
- **Total:** 4 × d_model parameters

Example (d_model=512): 2,048 LayerNorm parameters

---

## Gradient Management

### Gradient Norm Computation

**Formula:**
```
total_norm = √(||∇attention||² + ||∇feedforward||²)
```

**Note:** LayerNorm gradients not included due to API limitations, but they're typically small relative to other components.

**Typical Gradient Norms:**
- Small models (d_model=64): 0.1 - 10
- Medium models (d_model=512): 1 - 100
- Large models (d_model=1024): 10 - 1000

### Gradient Clipping

**Purpose:** Prevent exploding gradients

**Algorithm:**
```python
current_norm = get_gradient_norm()
if current_norm > max_norm:
    scale = max_norm / current_norm
    gradients *= scale
```

**Recommended Thresholds:**
- Conservative: max_norm = 1.0
- Standard: max_norm = 5.0
- Aggressive: max_norm = 10.0

**Example Usage:**
```cpp
encoder_block.forward(input);
encoder_block.backward(grad_output);

float norm = encoder_block.get_gradient_norm();
if (norm > 5.0f) {
    std::cout << "Clipping gradients: " << norm << " → 5.0" << std::endl;
    encoder_block.clip_gradients(5.0f);
}

encoder_block.update_weights();
```

### Gradient Monitoring

**Warning Signs:**

**Vanishing Gradients:**
```cpp
if (norm < 1e-6f) {
    std::cerr << "Warning: Vanishing gradients!" << std::endl;
    // Consider:
    // - Checking learning rate
    // - Verifying data normalization
    // - Inspecting attention patterns
}
```

**Exploding Gradients:**
```cpp
if (norm > 100.0f || std::isnan(norm) || std::isinf(norm)) {
    std::cerr << "Warning: Exploding gradients!" << std::endl;
    // Consider:
    // - Applying gradient clipping
    // - Reducing learning rate
    // - Checking for numerical instability
}
```

### Gradient Accumulation

**Motivation:** Simulate larger batch sizes with limited memory

**Pattern:**
```cpp
int accumulation_steps = 4;
encoder_block.zero_grad();

for (int i = 0; i < accumulation_steps; ++i) {
    Matrix output = encoder_block.forward(mini_batches[i]);
    Matrix grad = compute_loss_gradient(output, targets[i]);
    encoder_block.backward(grad);
    // Gradients accumulate, no weight update yet
}

// Update with accumulated gradients
encoder_block.update_weights();  // Equivalent to 4x batch size
```

---

## Usage Patterns

### Basic Training Loop

```cpp
EncoderBlock encoder(512, 8, 2048);
encoder.learning_rate = 0.001f;

for (int epoch = 0; epoch < num_epochs; ++epoch) {
    for (auto& batch : training_data) {
        // Forward pass
        Matrix output = encoder.forward(batch.input);
        
        // Compute loss gradient
        Matrix grad = compute_loss_gradient(output, batch.target);
        
        // Backward pass
        encoder.backward(grad);
        
        // Gradient clipping
        float norm = encoder.get_gradient_norm();
        if (norm > 5.0f) {
            encoder.clip_gradients(5.0f);
        }
        
        // Update weights
        encoder.update_weights();
    }
}
```

### Stacked Encoder Blocks

```cpp
// Create multiple layers
std::vector<std::unique_ptr<EncoderBlock>> layers;
for (int i = 0; i < 6; ++i) {
    layers.push_back(std::make_unique<EncoderBlock>(512, 8, 2048));
}

// Forward through all layers
Matrix encoded = input;
for (auto& layer : layers) {
    encoded = layer->forward(encoded);
}

// Backward through all layers (reverse order)
Matrix grad = loss_gradient;
for (int i = layers.size() - 1; i >= 0; --i) {
    grad = layers[i]->backward(grad);
}

// Update all layers
for (auto& layer : layers) {
    layer->update_weights();
}
```

### With Attention Masking

```cpp
// Create causal mask for autoregressive modeling
int seq_len = 10;
Matrix causal_mask(seq_len, seq_len);
for (int i = 0; i < seq_len; ++i) {
    for (int j = 0; j < seq_len; ++j) {
        // Allow attention to current and previous positions only
        causal_mask(i, j) = (j <= i) ? 0.0f : -1e9f;
    }
}

// Forward with mask
Matrix output = encoder_block.forward(input, &causal_mask);
```

### Weight Persistence

```cpp
// Training phase
EncoderBlock encoder(512, 8, 2048);
// ... training ...
encoder.save_weights("checkpoint_epoch_100.bin");

// Later: Inference phase
EncoderBlock encoder_inference(512, 8, 2048);
encoder_inference.load_weights("checkpoint_epoch_100.bin");

// Use for prediction
Matrix prediction = encoder_inference.forward(test_input);
```

### Learning Rate Scheduling

```cpp
EncoderBlock encoder(512, 8, 2048);

// Warmup schedule
for (int step = 0; step < warmup_steps; ++step) {
    float lr = base_lr * (step + 1) / warmup_steps;
    encoder.learning_rate = lr;
    // ... training step ...
}

// Cosine decay
for (int step = warmup_steps; step < total_steps; ++step) {
    float progress = (step - warmup_steps) / (total_steps - warmup_steps);
    float lr = base_lr * 0.5f * (1.0f + std::cos(M_PI * progress));
    encoder.learning_rate = lr;
    // ... training step ...
}
```

---

## Integration with Transformers

### Position in Transformer Architecture

```
Input Tokens
    ↓
Token Embedding
    ↓
Positional Encoding
    ↓
┌─────────────────────┐
│ EncoderBlock 1      │ ← This class
│  ├─ Attention       │
│  ├─ Add & Norm      │
│  ├─ FeedForward     │
│  └─ Add & Norm      │
└─────────────────────┘
    ↓
┌─────────────────────┐
│ EncoderBlock 2      │ ← This class (repeated)
│  ├─ Attention       │
│  ├─ Add & Norm      │
│  ├─ FeedForward     │
│  └─ Add & Norm      │
└─────────────────────┘
    ↓
    ... (more layers)
    ↓
Final Layer Norm
    ↓
Output / Task Head
```

### Typical Configurations

**BERT Base:**
- d_model: 768
- num_heads: 12
- d_ff: 3072 (4x expansion)
- num_layers: 12
- dropout: 0.1

**GPT-2 Small:**
- d_model: 768
- num_heads: 12
- d_ff: 3072 (4x expansion)
- num_layers: 12
- dropout: 0.1

**Transformer Base (Original Paper):**
- d_model: 512
- num_heads: 8
- d_ff: 2048 (4x expansion)
- num_layers: 6
- dropout: 0.1

**Custom (Example in Tests):**
- d_model: 64
- num_heads: 4
- d_ff: 256 (4x expansion)
- num_layers: 4
- dropout: 0.1

### Integration Example

```cpp
class TransformerEncoder {
    std::unique_ptr<TokenEmbedding> embedding;
    std::unique_ptr<PositionalEncoding> pos_encoding;
    std::vector<std::unique_ptr<EncoderBlock>> encoder_blocks;
    std::unique_ptr<LayerNorm> final_norm;
    
public:
    TransformerEncoder(int vocab_size, int d_model, int num_layers,
                       int num_heads, int d_ff) {
        embedding = std::make_unique<TokenEmbedding>(vocab_size, d_model);
        pos_encoding = std::make_unique<PositionalEncoding>(max_len, d_model);
        
        for (int i = 0; i < num_layers; ++i) {
            encoder_blocks.push_back(
                std::make_unique<EncoderBlock>(d_model, num_heads, d_ff)
            );
        }
        
        final_norm = std::make_unique<LayerNorm>(d_model);
    }
    
    Matrix encode(const std::vector<int>& token_ids) {
        Matrix embedded = embedding->forward(token_ids);
        Matrix encoded = pos_encoding->forward(embedded);
        
        for (auto& block : encoder_blocks) {
            encoded = block->forward(encoded);
        }
        
        return final_norm->forward(encoded);
    }
};
```

### Self-Attention in Encoder

**Key Difference from Decoder:**
- **Encoder**: Bidirectional attention (can attend to all positions)
- **Decoder**: Causal attention (can only attend to previous positions)

**Encoder Attention Pattern:**
```cpp
// No mask needed for bidirectional attention
Matrix output = encoder_block.forward(input);  // mask = nullptr
```

**Decoder Attention Pattern:**
```cpp
// Causal mask prevents looking ahead
Matrix causal_mask = create_causal_mask(seq_len);
Matrix output = encoder_block.forward(input, &causal_mask);
```

---

## Performance Considerations

### Computational Bottlenecks

**For Long Sequences (seq_len >> d_model):**
- Attention dominates: O(seq_len² × d_model)
- Strategies:
  - Sparse attention patterns
  - Local attention windows
  - Linear attention variants

**For Wide Models (d_model >> seq_len):**
- Feed-forward dominates: O(seq_len × d_model × d_ff)
- Strategies:
  - Reduce d_ff expansion ratio
  - Use parameter sharing
  - Apply layer pruning

### Memory Optimization

**Cached Activations:**
- Current: 6 × seq_len × d_model values
- Optimization: Recompute activations during backward (trading compute for memory)

**Gradient Checkpointing:**
```cpp
// Standard (high memory):
Matrix output = encoder_block.forward(input);
Matrix grad_input = encoder_block.backward(grad_output);

// Checkpointed (low memory, slower):
// Recompute forward during backward instead of caching
```

### Batch Processing

**Benefits:**
- Amortize fixed costs over multiple sequences
- Better GPU/CPU utilization
- More stable gradient estimates

**Challenges:**
- Variable sequence lengths require padding
- Padding increases computation on masked positions
- Attention masking needed to ignore padding

### Parallelization

**Within Encoder Block:**
- Multi-head attention: Heads computed independently
- Position-wise feed-forward: Positions processed independently
- Layer normalization: Samples normalized independently

**Across Layers:**
- Layers must be computed sequentially
- Cannot parallelize across encoder blocks in same model

---

## Debugging and Monitoring

### Common Issues

**1. NaN/Inf in Outputs**
```cpp
Matrix output = encoder_block.forward(input);
for (int i = 0; i < output.rows; ++i) {
    for (int j = 0; j < output.cols; ++j) {
        if (!std::isfinite(output(i, j))) {
            std::cerr << "NaN/Inf at position [" << i << "," << j << "]" << std::endl;
            // Check:
            // - Input normalization
            // - Learning rate too high
            // - Gradient clipping
        }
    }
}
```

**2. Vanishing Gradients**
```cpp
float norm = encoder_block.get_gradient_norm();
if (norm < 1e-6f) {
    std::cerr << "Vanishing gradients detected!" << std::endl;
    // Solutions:
    // - Increase learning rate
    // - Check layer normalization
    // - Verify residual connections working
}
```

**3. Exploding Gradients**
```cpp
float norm = encoder_block.get_gradient_norm();
if (norm > 100.0f) {
    std::cerr << "Exploding gradients: " << norm << std::endl;
    // Solutions:
    // - Apply gradient clipping
    // - Reduce learning rate
    // - Check initialization
}
```

**4. Attention Collapse**
```cpp
// All attention weights focused on one position
// Symptoms: Output becomes uniform across positions
// Solutions:
// - Check attention mask
// - Verify positional encoding
// - Inspect temperature scaling (1/√d_k)
```

### Monitoring Metrics

**Training:**
```cpp
// Log every N steps
if (step % 100 == 0) {
    std::cout << "Step " << step << std::endl;
    std::cout << "  Gradient norm: " << encoder_block.get_gradient_norm() << std::endl;
    std::cout << "  Learning rate: " << encoder_block.learning_rate << std::endl;
    
    // Print configuration
    encoder_block.print_config("Layer_" + std::to_string(layer_idx));
}
```

**Gradient Flow:**
```cpp
// Monitor gradient norm across layers
for (int i = 0; i < num_layers; ++i) {
    float norm = encoder_blocks[i]->get_gradient_norm();
    std::cout << "Layer " << i << " gradient norm: " << norm << std::endl;
}
```

---

## Advanced Topics

### Dropout in Encoder Blocks

**Note:** Current implementation stores dropout_rate but doesn't apply it in forward pass. For production:

```cpp
// After attention (during training)
if (training && dropout_rate > 0) {
    attn_output = apply_dropout(attn_output, dropout_rate);
}

// After feed-forward (during training)
if (training && dropout_rate > 0) {
    ff_output = apply_dropout(ff_output, dropout_rate);
}
```

**Dropout Locations:**
1. After attention output projection
2. After feed-forward network
3. (Optional) Within attention (on attention weights)
4. (Optional) Within feed-forward (on activation)

### Weight Initialization

**Current Strategy:**
- MultiHeadAttention: Xavier/Glorot initialization
- FeedForward: He initialization (for GELU)
- LayerNorm: gamma=1, beta=0

**Alternative Strategies:**
```cpp
// Scaled initialization for deep networks
float scale = 1.0f / std::sqrt(2.0f * num_layers);
// Apply scale to all weight initializations
```

### Layer Freezing

**Freeze for Transfer Learning:**
```cpp
// Don't update weights
encoder_block.learning_rate = 0.0f;

// Still compute gradients for downstream layers
Matrix grad_input = encoder_block.backward(grad_output);
```

### Mixed Precision Training

**Benefits:**
- Faster computation
- Reduced memory usage
- Maintain accuracy with loss scaling

**Consideration:**
- Current implementation uses float32
- Converting to float16 requires careful handling of:
  - Gradient scaling
  - Numerical stability in softmax
  - Layer normalization precision

---

## API Reference

### Constructor

```cpp
EncoderBlock(int d_model, int num_heads, int d_ff, float dropout = 0.1f)
```
**Parameters:**
- `d_model`: Model dimension (must be divisible by num_heads)
- `num_heads`: Number of attention heads
- `d_ff`: Feed-forward hidden dimension
- `dropout`: Dropout rate (default: 0.1)

**Throws:** `std::invalid_argument` if d_model not divisible by num_heads

---

### Forward Pass

```cpp
Matrix forward(const Matrix& input, const Matrix* mask = nullptr)
```
**Parameters:**
- `input`: Input matrix [seq_len, d_model]
- `mask`: Optional attention mask [seq_len, seq_len]

**Returns:** Output matrix [seq_len, d_model]

**Complexity:** O(seq_len² × d_model + seq_len × d_model × d_ff)

---

### Backward Pass

```cpp
Matrix backward(const Matrix& grad_output)
```
**Parameters:**
- `grad_output`: Gradient from next layer [seq_len, d_model]

**Returns:** Gradient w.r.t. input [seq_len, d_model]

**Side Effects:** Accumulates gradients in all parameters

---

### Weight Management

```cpp
void update_weights()
```
**Description:** Apply gradient descent to all parameters, then zero gradients

---

```cpp
void zero_grad()
```
**Description:** Reset all gradient accumulators to zero

---

### Gradient Monitoring

```cpp
float get_gradient_norm() const
```
**Returns:** L2 norm of all gradients combined

---

```cpp
void clip_gradients(float max_norm)
```
**Parameters:**
- `max_norm`: Maximum allowed gradient norm

**Description:** Scale gradients if norm exceeds threshold

---

### Persistence

```cpp
void save_weights(const std::string& filename) const
```
**Parameters:**
- `filename`: Path to output file

**Creates:**
- `filename`: Main file with dimensions and LayerNorm params
- `filename_attention.bin`: Attention weights
- `filename_feedforward.bin`: Feed-forward weights

**Throws:** `std::runtime_error` if file cannot be created

---

```cpp
void load_weights(const std::string& filename)
```
**Parameters:**
- `filename`: Path to input file

**Throws:**
- `std::runtime_error` if file not found
- `std::runtime_error` if dimension mismatch

---

### Configuration

```cpp
void print_config(const std::string& name = "EncoderBlock") const
```
**Parameters:**
- `name`: Name to display in output

**Output:** Configuration details and parameter count

---

### Accessors

```cpp
int get_d_model() const
int get_num_heads() const
int get_d_ff() const
float get_dropout_rate() const
```
**Returns:** Respective configuration values

---

## Example Use Cases

### 1. Standard Transformer Encoder

```cpp
// Configuration
int d_model = 512;
int num_heads = 8;
int d_ff = 2048;
int num_layers = 6;

// Create encoder stack
std::vector<std::unique_ptr<EncoderBlock>> encoder;
for (int i = 0; i < num_layers; ++i) {
    encoder.push_back(std::make_unique<EncoderBlock>(d_model, num_heads, d_ff));
}

// Forward pass through all layers
Matrix encoded = input_embeddings;
for (auto& layer : encoder) {
    encoded = layer->forward(encoded);
}
```

### 2. Training with Gradient Accumulation

```cpp
EncoderBlock encoder(512, 8, 2048);
encoder.learning_rate = 0.001f;

int gradient_accumulation_steps = 4;
encoder.zero_grad();

for (int step = 0; step < gradient_accumulation_steps; ++step) {
    Matrix output = encoder.forward(batches[step]);
    Matrix grad = compute_loss_gradient(output, targets[step]);
    encoder.backward(grad);
}

// Update with accumulated gradients
encoder.clip_gradients(5.0f);
encoder.update_weights();
```

### 3. Transfer Learning

```cpp
// Load pretrained weights
EncoderBlock pretrained_encoder(512, 8, 2048);
pretrained_encoder.load_weights("pretrained_bert_layer_5.bin");

// Freeze for feature extraction
pretrained_encoder.learning_rate = 0.0f;

// Use for inference
Matrix features = pretrained_encoder.forward(input);

// Or fine-tune with small learning rate
pretrained_encoder.learning_rate = 0.00001f;
```

---

## Conclusion

The EncoderBlock class provides a production-ready implementation of the transformer encoder layer with:

✅ **Complete Functionality**
- Multi-head self-attention
- Position-wise feed-forward network
- Residual connections
- Layer normalization

✅ **Training Support**
- Full backpropagation
- Gradient management (norm, clipping, zeroing)
- Weight updates with configurable learning rate

✅ **Robustness**
- Gradient flow through residual connections
- Numerical stability via layer normalization
- Attention masking support

✅ **Flexibility**
- Configurable dimensions and heads
- Adjustable dropout rate
- Save/load functionality

✅ **Integration**
- Drop-in component for transformer architectures
- Compatible with existing Matrix, MultiHeadAttention, FeedForward, and LayerNorm classes
- Tested with various sequence lengths and configurations

The implementation follows modern best practices from transformer research, providing a solid foundation for building NLP models, chatbots, and other sequence processing applications.
