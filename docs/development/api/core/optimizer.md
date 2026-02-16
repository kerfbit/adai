# Optimizer Context Documentation

## Purpose and Role

The `Optimizer` class provides centralized gradient-based optimization for neural network training. It implements industry-standard optimization algorithms and essential training features like gradient clipping and weight decay, serving as the central component for parameter updates during training.

## Core Concept

In neural network training, the optimizer is responsible for updating model parameters based on computed gradients to minimize the loss function. The `Optimizer` class abstracts this process, providing multiple optimization strategies and training stability features in a unified interface.

### Mathematical Foundation

All optimizers follow the general update rule:

```text
θ_{t+1} = θ_t - Update(∇L(θ_t))
```

Where:

- `θ` = model parameters (weights)
- `∇L` = gradient of loss with respect to parameters
- `Update()` = optimizer-specific update function

## Architecture Overview

### Class Structure

```cpp
class Optimizer {
private:
    OptimizerType type;                           // Algorithm selection
    float learning_rate;                          // Base learning rate
    std::vector<ParameterGroup> parameter_groups; // Registered parameters

    // Hyperparameters (algorithm-specific)
    float momentum_beta, beta1, beta2;
    float epsilon, weight_decay;
    float max_grad_norm;

    // Update methods for each algorithm
    void step_sgd(ParameterGroup& param);
    void step_sgd_momentum(ParameterGroup& param);
    void step_adam(ParameterGroup& param);
    void step_adamw(ParameterGroup& param);

public:
    // Configuration
    void add_parameter_group(Matrix* weights, Matrix* gradients);
    void set_learning_rate(float lr);
    void set_betas(float beta1, float beta2);
    void set_weight_decay(float wd);
    void set_max_grad_norm(float max_norm);

    // Training operations
    void zero_grad();
    float clip_gradients();
    void step();

    // Monitoring
    float get_gradient_norm() const;
    size_t total_parameters() const;
};
```

### Parameter Group Structure

```cpp
struct ParameterGroup {
    Matrix* weights;      // Pointer to weight matrix
    Matrix* gradients;    // Pointer to gradient matrix
    Matrix momentum;      // First moment (velocity)
    Matrix velocity;      // Second moment (adaptive LR)
    int step;            // Number of updates
};
```

## Optimization Algorithms

### 1. SGD (Stochastic Gradient Descent)

**Algorithm:**

```text
w = w - lr * (grad + weight_decay * w)
```

**Characteristics:**

- Simplest optimizer
- No adaptive learning rates
- No momentum
- Memory efficient

**Use Cases:**

- Simple baselines
- Memory-constrained environments
- Small models

**Limitations:**

- Slow convergence
- Sensitive to learning rate
- Poor performance on transformers

### 2. SGD with Momentum

**Algorithm:**

```text
m = beta * m + grad
w = w - lr * m
```

**Characteristics:**

- Accumulates velocity in gradient direction
- Smooths out oscillations
- Accelerates convergence in consistent directions

**Hyperparameters:**

- `momentum_beta`: Typically 0.9 (90% of previous velocity retained)

**Use Cases:**

- When SGD is too slow
- Navigating ravines in loss landscape
- CNNs and traditional architectures

**Advantages over SGD:**

- Faster convergence
- Smoother optimization path
- Better handling of noisy gradients

### 3. Adam (Adaptive Moment Estimation)

**Algorithm:**

```text
m = beta1 * m + (1 - beta1) * grad        // First moment
v = beta2 * v + (1 - beta2) * grad²       // Second moment
m_hat = m / (1 - beta1^t)                  // Bias correction
v_hat = v / (1 - beta2^t)                  // Bias correction
w = w - lr * m_hat / (sqrt(v_hat) + epsilon)
```

**Characteristics:**

- Adaptive per-parameter learning rates
- Combines momentum and RMSprop
- Bias correction for initialization
- Industry standard for deep learning

**Hyperparameters:**

- `beta1`: 0.9 (first moment decay)
- `beta2`: 0.999 (second moment decay)
- `epsilon`: 1e-8 (numerical stability)

**Use Cases:**

- Default choice for most deep learning
- Transformer models
- When you need fast convergence
- When different parameters need different learning rates

**Advantages:**

- Fast convergence
- Works well out-of-the-box
- Adaptive to parameter scale
- Handles sparse gradients well

**Disadvantages:**

- Weight decay is coupled with gradients (incorrect regularization)
- Can overfit more than SGD
- Requires 2x memory (for m and v)

### 4. AdamW (Adam with Decoupled Weight Decay)

**Algorithm:**

```text
m = beta1 * m + (1 - beta1) * grad
v = beta2 * v + (1 - beta2) * grad²
m_hat = m / (1 - beta1^t)
v_hat = v / (1 - beta2^t)
w = w - lr * (m_hat / (sqrt(v_hat) + epsilon) + weight_decay * w)
```

**Key Difference from Adam:**
Weight decay is applied directly to weights, not added to gradients.

**Characteristics:**

- Proper L2 regularization
- Better generalization than Adam
- Recommended for transformer training
- Same computational cost as Adam

**Hyperparameters:**

- Same as Adam, plus:
- `weight_decay`: 0.01 - 0.1 (regularization strength)

**Use Cases:**

- **Recommended for transformers** (BERT, GPT, etc.)
- Large models requiring regularization
- When better generalization is needed

**Advantages over Adam:**

- Correct weight decay implementation
- Better generalization
- Standard for transformer training

## Key Features

### Gradient Clipping

**Purpose:** Prevent exploding gradients by limiting gradient magnitude.

**Algorithm:**

```cpp
float total_norm = sqrt(sum(grad_i²))  // L2 norm of all gradients
if (total_norm > max_norm) {
    scale = max_norm / total_norm
    for each gradient:
        grad = grad * scale
}
```

**When to Use:**

- ✅ Always for transformer training
- ✅ RNNs and LSTMs
- ✅ Deep networks (>6 layers)
- ✅ When seeing NaN losses
- ✅ When gradients explode during training

**Typical Values:**

- Transformers: 1.0
- RNNs: 5.0
- Small networks: 10.0 or disabled

**Implementation:**

```cpp
optimizer->set_max_grad_norm(1.0f);
float norm = optimizer->clip_gradients();  // Returns pre-clip norm
```

### Weight Decay (L2 Regularization)

**Purpose:** Prevent overfitting by penalizing large weights.

**Adam vs AdamW:**

- **Adam:** `grad += weight_decay * w` (incorrect - affects adaptive learning)
- **AdamW:** `w -= lr * weight_decay * w` (correct - independent of gradients)

**Effect:**

- Pushes weights toward zero
- Reduces model complexity
- Improves generalization

**Typical Values:**

- No regularization: 0.0
- Light regularization: 0.001 - 0.01
- Standard: 0.01
- Heavy regularization: 0.1

**Trade-offs:**

- Too low: Overfitting
- Too high: Underfitting, slow convergence

### Gradient Norm Monitoring

**Purpose:** Track gradient magnitude for training diagnostics.

**What it tells you:**

- **Norm 0.1 - 10:** Healthy training
- **Norm > 10:** Potential exploding gradients
- **Norm < 0.01:** Vanishing gradients or model converged
- **Norm = NaN/Inf:** Training has diverged

**Usage:**

```cpp
float norm = optimizer->get_gradient_norm();
std::cout << "Gradient norm: " << norm << "\n";
```

## Integration with Training Pipeline

### Typical Training Loop

```cpp
// 1. Setup
Optimizer optimizer(OptimizerType::ADAMW, 0.0001f);
optimizer.set_weight_decay(0.01f);
optimizer.set_max_grad_norm(1.0f);
optimizer.set_betas(0.9f, 0.999f);

// Register model parameters
optimizer.add_parameter_group(&encoder_weights, &encoder_grads);
optimizer.add_parameter_group(&decoder_weights, &decoder_grads);

// 2. Training iteration
for (int epoch = 0; epoch < num_epochs; epoch++) {
    for (auto& batch : training_data) {
        // Zero gradients
        optimizer.zero_grad();

        // Forward pass
        auto output = model.forward(batch.input);
        float loss = compute_loss(output, batch.target);

        // Backward pass (compute gradients)
        auto grad_loss = compute_loss_gradient(output, batch.target);
        model.backward(grad_loss);

        // Monitor gradient health
        float grad_norm = optimizer.get_gradient_norm();
        if (grad_norm > 10.0f) {
            std::cerr << "Warning: Large gradients!\n";
        }

        // Clip gradients
        optimizer.clip_gradients();

        // Update weights
        optimizer.step();
    }

    // Adjust learning rate
    float new_lr = learning_rate_schedule(epoch);
    optimizer.set_learning_rate(new_lr);
}
```

### Integration with ChatbotTrainer

The optimizer is integrated into the training pipeline:

1. **Initialization** (`initialize_model()`):

   ```cpp
   optimizer = new Optimizer(config.optimizer_type, config.learning_rate);
   optimizer->set_weight_decay(config.weight_decay);
   optimizer->set_max_grad_norm(config.gradient_clip_norm);
   ```

2. **Training Step** (`train_epoch()`):

   ```cpp
   optimizer->zero_grad();
   // ... forward and backward passes ...
   float grad_norm = optimizer->get_gradient_norm();
   optimizer->clip_gradients();
   optimizer->step();  // (Currently uses model->update_weights())
   ```

3. **Learning Rate Scheduling** (`update_learning_rate()`):

   ```cpp
   float new_lr = calculate_learning_rate(global_step);
   optimizer->set_learning_rate(new_lr);
   ```

## Memory and Performance

### Memory Overhead

Per-parameter memory requirements:

- **SGD:** 0 bytes (no state)
- **SGD+Momentum:** 1x (momentum vector)
- **Adam/AdamW:** 2x (momentum + velocity)

**Example:**
Model with 100M parameters (400MB):

- SGD: 400MB total
- SGD+Momentum: 800MB total
- Adam/AdamW: 1200MB total

### Computational Complexity

All optimizers: **O(n)** where n = number of parameters

**Per-parameter operations:**

- SGD: 1-2 operations
- SGD+Momentum: 3-4 operations
- Adam: 8-10 operations
- AdamW: 9-11 operations

**Performance impact:**
Negligible compared to forward/backward passes. Adam's extra computation is worth the convergence speedup.

## Best Practices

### Choosing an Optimizer

**For Transformers (Recommended):**

```cpp
Optimizer opt(OptimizerType::ADAMW, 0.0001f);
opt.set_betas(0.9f, 0.999f);
opt.set_weight_decay(0.01f);
opt.set_max_grad_norm(1.0f);
```

**For CNNs:**

```cpp
Optimizer opt(OptimizerType::ADAM, 0.001f);
opt.set_betas(0.9f, 0.999f);
opt.set_weight_decay(0.0001f);
```

**For Small Models/Debugging:**

```cpp
Optimizer opt(OptimizerType::SGD_MOMENTUM, 0.01f);
opt.set_momentum(0.9f);
```

### Hyperparameter Tuning

**Learning Rate:**

1. Start with recommended defaults (0.001 for Adam, 0.01 for SGD)
2. If loss doesn't decrease: reduce by 10x
3. If training is slow: increase by 3x
4. Use learning rate scheduling for best results

**Weight Decay:**

1. Start with 0.01 for AdamW
2. Increase if overfitting (up to 0.1)
3. Decrease if underfitting (down to 0.001)

**Gradient Clipping:**

1. Start with 1.0 for transformers
2. Monitor gradient norms
3. Adjust if seeing frequent clipping or NaN losses

### Common Issues and Solutions

#### Issue: Loss becomes NaN

- ✅ Enable gradient clipping
- ✅ Reduce learning rate
- ✅ Check for numerical instabilities in model

#### Issue: Slow convergence

- ✅ Switch from SGD to Adam/AdamW
- ✅ Increase learning rate
- ✅ Add warmup period
- ✅ Check if gradients are too small

#### Issue: Overfitting

- ✅ Increase weight decay
- ✅ Use AdamW instead of Adam
- ✅ Add dropout in model
- ✅ Reduce model size

#### Issue: High gradient norms

- ✅ Reduce gradient clip threshold
- ✅ Reduce learning rate
- ✅ Check model initialization

#### Issue: Vanishing gradients (norm < 0.01)

- ✅ Increase learning rate
- ✅ Check activation functions
- ✅ Reduce network depth
- ✅ Use residual connections

## Configuration Reference

### Constructor (Configuration)

```cpp
Optimizer(OptimizerType type, float lr)
```

- `type`: SGD, SGD_MOMENTUM, ADAM, ADAMW
- `lr`: Initial learning rate

### Methods

**Parameter Management:**

```cpp
void add_parameter_group(Matrix* weights, Matrix* gradients)
```

Register weight/gradient matrices for optimization.

**Hyperparameters:**

```cpp
void set_learning_rate(float lr)          // Update learning rate
void set_momentum(float beta)             // SGD+Momentum: 0.9
void set_betas(float beta1, float beta2)  // Adam: (0.9, 0.999)
void set_weight_decay(float wd)           // L2 reg: 0.01
void set_max_grad_norm(float max_norm)    // Clip: 1.0
```

**Training Operations:**

```cpp
void zero_grad()                    // Zero all gradients
float clip_gradients()              // Clip and return pre-clip norm
void step()                         // Update all parameters
```

**Monitoring:**

```cpp
float get_gradient_norm() const     // Current gradient L2 norm
size_t total_parameters() const     // Total parameter count
const char* get_optimizer_name()    // "SGD", "Adam", etc.
```

**State Management:**

```cpp
void reset_state()  // Clear momentum/velocity, reset step counter
```

## Implementation Details

### Parameter Group Registration

When `add_parameter_group()` is called:

1. Validates weights and gradients have same shape
2. Stores pointers (no copies)
3. Initializes optimizer state matrices (momentum, velocity)
4. Sets step counter to 0

### Gradient Clipping Implementation

Global norm clipping:

```cpp
1. Compute total_norm = sqrt(Σ grad_i²) across all parameters
2. if total_norm > max_norm:
       scale = max_norm / total_norm
       multiply all gradients by scale
3. Return original total_norm
```

### Update Methods

**SGD:**

```cpp
for each parameter:
    grad += weight_decay * weight  // L2 regularization
    weight -= lr * grad
```

**Adam:**

```cpp
for each parameter:
    grad += weight_decay * weight  // Coupled weight decay
    m = beta1 * m + (1-beta1) * grad
    v = beta2 * v + (1-beta2) * grad²
    m_hat = m / (1 - beta1^step)
    v_hat = v / (1 - beta2^step)
    weight -= lr * m_hat / (sqrt(v_hat) + epsilon)
```

**AdamW:**

```cpp
for each parameter:
    m = beta1 * m + (1-beta1) * grad
    v = beta2 * v + (1-beta2) * grad²
    m_hat = m / (1 - beta1^step)
    v_hat = v / (1 - beta2^step)
    // Decoupled weight decay
    weight -= lr * (m_hat / (sqrt(v_hat) + epsilon) + weight_decay * weight)
```

## Future Enhancements

### Potential Additions

1. **Additional Optimizers:**
   - RMSprop
   - Adagrad
   - Lookahead
   - LAMB (for large batch training)

2. **Advanced Features:**
   - Per-parameter learning rates
   - Learning rate warmup (built-in)
   - Gradient accumulation for larger batch sizes
   - Mixed precision support

3. **Full Parameter Exposure:**
   - Complete integration with model components
   - Automatic parameter discovery
   - Parameter grouping with different settings

4. **State Persistence:**
   - Save/load optimizer state
   - Resume training with momentum

## Related Components

### Dependencies

- `Matrix.hpp` - Parameter and gradient storage
- `EncoderDecoderModel` - Model being optimized
- `ChatbotTrainer` - Training orchestration

### Used By

- `ChatbotTrainer` - Main training loop
- Potentially other training scripts

### Related Files

- `src/Optimizer.hpp` - Class declaration
- `src/Optimizer.cpp` - Implementation
- `src/OptimizerExample.cpp` - Usage examples
- `tests/optimizer_test.cpp` - Comprehensive unit tests (45 test cases)
- `OPTIMIZER_README.md` - User documentation
- `OPTIMIZER_INTEGRATION_SUMMARY.md` - Integration details

### Testing

Comprehensive unit tests are available in `tests/optimizer_test.cpp` with 45 test cases covering:

**Constructor Tests (5 tests):**

- Default, SGD, SGD+Momentum, Adam, AdamW constructors

**Parameter Management Tests (5 tests):**

- Adding single/multiple parameter groups
- Null pointer validation
- Shape mismatch detection

**Hyperparameter Tests (5 tests):**

- Learning rate, momentum, betas, weight decay, gradient norm settings

**Gradient Operations Tests (9 tests):**

- Zero gradients (single and multiple groups)
- Gradient norm computation
- Gradient clipping (threshold, custom norm, disabled)

**Optimization Algorithm Tests (8 tests):**

- SGD basic update, weight decay, multiple steps
- SGD+Momentum accumulation
- Adam bias correction, weight decay
- AdamW decoupled weight decay

**State Management Tests (2 tests):**

- Reset state
- Zero momentum after reset

**Integration Tests (3 tests):**

- Complete training loop
- Learning rate scheduling
- Multi-parameter group updates

**Edge Case Tests (4 tests):**

- Zero gradients
- Very small/large values
- Empty optimizer

**Performance Tests (2 tests):**

- Large parameter matrices (100x100)
- Many parameter groups (50 groups)

Run tests with:

```bash
cd build
make optimizerTests
./tests/optimizerTests
```

## References

### Academic Papers

1. **SGD with Momentum:** Qian (1999) "On the momentum term in gradient descent"
2. **Adam:** Kingma & Ba (2015) "Adam: A Method for Stochastic Optimization"
3. **AdamW:** Loshchilov & Hutter (2019) "Decoupled Weight Decay Regularization"

### Implementation References

- PyTorch Optimizers: torch.optim
- TensorFlow Optimizers: tf.keras.optimizers
- Hugging Face Transformers: transformers.optimization

## Version History

**v1.0** (January 2026)

- Initial implementation
- SGD, SGD+Momentum, Adam, AdamW
- Gradient clipping
- Weight decay
- Gradient norm monitoring
- Integration with ChatbotTrainer

## Example Code

### Complete Training Example

```cpp
#include "Optimizer.hpp"
#include "EncoderDecoderModel.hpp"

int main() {
    // Create model
    EncoderDecoderModel model(vocab_size, d_model, ...);

    // Create optimizer
    Optimizer optimizer(OptimizerType::ADAMW, 0.0001f);
    optimizer.set_weight_decay(0.01f);
    optimizer.set_max_grad_norm(1.0f);
    optimizer.set_betas(0.9f, 0.999f);

    // Register parameters (when fully implemented)
    model.register_parameters(optimizer);

    // Training loop
    for (int epoch = 0; epoch < 10; epoch++) {
        optimizer.zero_grad();

        float loss = model.train_step(input, target);
        float grad_norm = optimizer.get_gradient_norm();

        std::cout << "Loss: " << loss
                  << ", GradNorm: " << grad_norm << "\n";

        optimizer.clip_gradients();
        optimizer.step();
    }

    return 0;
}
```

This context document provides comprehensive information about the Optimizer class implementation, usage, and integration within the chatbot training system.

---

## Quick Start and API Reference

## Optimizer - Centralized Gradient Optimization (Reference)

### Overview

The `Optimizer` class provides centralized gradient processing and parameter updates for neural network training. It implements multiple state-of-the-art optimization algorithms and essential features like gradient clipping and weight decay.

## Features

### ✅ Optimization Algorithms

- **SGD** - Stochastic Gradient Descent
- **SGD+Momentum** - SGD with momentum for smoother updates
- **Adam** - Adaptive Moment Estimation (industry standard)
- **AdamW** - Adam with decoupled weight decay (recommended for transformers)

### ✅ Training Stability

- **Gradient Clipping** - Prevents exploding gradients
- **Weight Decay** - L2 regularization to prevent overfitting
- **Bias Correction** - Proper handling of Adam moments

### ✅ Monitoring & Control

- **Gradient Norm Tracking** - Monitor training stability
- **Learning Rate Scheduling** - Dynamic LR adjustment
- **State Reset** - Clear optimizer state when needed

## Quick Start

### Basic Usage

```cpp
#include "Optimizer.hpp"

// Create optimizer
Optimizer optimizer(OptimizerType::ADAM, 0.001f);

// Add parameter groups (weights and their gradients)
optimizer.add_parameter_group(&weights, &gradients);

// Training loop
for (int epoch = 0; epoch < num_epochs; epoch++) {
    optimizer.zero_grad();           // Clear gradients

    // ... forward pass ...
    // ... loss computation ...
    // ... backward pass (computes gradients) ...

    optimizer.clip_gradients(1.0f);  // Optional: clip gradients
    optimizer.step();                 // Update weights
}
```

### Adam with Gradient Clipping (Recommended for Transformers)

```cpp
Optimizer optimizer(OptimizerType::ADAM, 0.001f);
optimizer.add_parameter_group(&encoder_weights, &encoder_grads);
optimizer.add_parameter_group(&decoder_weights, &decoder_grads);
optimizer.set_max_grad_norm(1.0f);    // Clip to norm of 1.0
optimizer.set_weight_decay(0.01f);     // L2 regularization

// Training loop
optimizer.zero_grad();
// ... forward/backward ...
optimizer.clip_gradients();  // Uses max_grad_norm
optimizer.step();
```

### AdamW with Custom Settings

```cpp
Optimizer optimizer(OptimizerType::ADAMW, 0.0001f);
optimizer.add_parameter_group(&weights, &gradients);
optimizer.set_betas(0.9f, 0.999f);     // Adam beta parameters
optimizer.set_weight_decay(0.01f);      // Decoupled weight decay
optimizer.set_max_grad_norm(1.0f);      // Gradient clipping

// Training step
optimizer.zero_grad();
// ... training ...
optimizer.step();
```

## API Reference

### Constructor (API)

```cpp
Optimizer(OptimizerType type = OptimizerType::ADAM, float lr = 0.001f)
```

- `type` - Optimization algorithm (SGD, SGD_MOMENTUM, ADAM, ADAMW)
- `lr` - Initial learning rate

### Adding Parameters

```cpp
void add_parameter_group(Matrix* weights, Matrix* gradients)
```

Registers a weight matrix and its corresponding gradient matrix for optimization.

**Requirements:**

- `weights` and `gradients` must have the same shape
- Should be called during model initialization
- Can be called multiple times for different parameter groups

### Configuration

```cpp
void set_learning_rate(float lr)
```

Update learning rate (for LR scheduling).

```cpp
void set_momentum(float beta)
```

Set momentum coefficient for SGD+Momentum (default: 0.9).

```cpp
void set_betas(float beta1, float beta2)
```

Set Adam beta parameters:

- `beta1` - First moment decay (default: 0.9)
- `beta2` - Second moment decay (default: 0.999)

```cpp
void set_weight_decay(float wd)
```

Set weight decay coefficient for L2 regularization (default: 0.0).

```cpp
void set_max_grad_norm(float max_norm)
```

Set maximum gradient norm for clipping (default: 0.0 = disabled).

### Training Operations

```cpp
void zero_grad()
```

Zero all gradients. Call at the start of each training iteration.

```cpp
float clip_gradients()
float clip_gradients(float max_norm)
```

Clip gradients by global norm. Returns gradient norm before clipping.

```cpp
void step()
```

Perform optimization step - updates all parameters based on gradients.

### Monitoring

```cpp
float get_gradient_norm() const
```

Get L2 norm of all gradients (useful for monitoring training).

```cpp
size_t total_parameters() const
```

Get total number of trainable parameters.

```cpp
const char* get_optimizer_name() const
```

Get name of current optimization algorithm.

### Utilities

```cpp
void reset_state()
```

Reset optimizer state (clears momentum and velocity). Useful when:

- Loading a model
- Changing optimization strategy
- Restarting training

## Optimization Algorithms (Reference)

### SGD (Stochastic Gradient Descent)

**Update rule:**

```text
w = w - lr * (grad + weight_decay * w)
```

**Use when:**

- Simple baseline needed
- Memory constrained
- Small models

**Pros:** Simple, memory efficient
**Cons:** Slow convergence, requires careful LR tuning

### SGD with Momentum

**Update rule:**

```text
m = beta * m + grad
w = w - lr * m
```

**Use when:**

- Need faster convergence than vanilla SGD
- Navigating ravines in loss landscape

**Pros:** Faster than SGD, smoother updates
**Cons:** Still requires LR tuning, not adaptive

### Adam (Adaptive Moment Estimation)

**Update rule:**

```text
m = beta1 * m + (1 - beta1) * grad           // First moment
v = beta2 * v + (1 - beta2) * grad^2         // Second moment
m_hat = m / (1 - beta1^t)                     // Bias correction
v_hat = v / (1 - beta2^t)
w = w - lr * m_hat / (sqrt(v_hat) + epsilon)
```

**Use when:**

- Training transformers (industry standard)
- Need adaptive learning rates
- Different parameters need different LR

**Pros:** Fast convergence, adaptive, works well out-of-the-box
**Cons:** Can overfit, weight decay coupled with gradients

**Recommended settings:**

- `lr`: 0.001 - 0.0001
- `beta1`: 0.9
- `beta2`: 0.999
- `epsilon`: 1e-8

### AdamW (Adam with Decoupled Weight Decay)

**Update rule:**

```text
m = beta1 * m + (1 - beta1) * grad
v = beta2 * v + (1 - beta2) * grad^2
m_hat = m / (1 - beta1^t)
v_hat = v / (1 - beta2^t)
w = w - lr * (m_hat / (sqrt(v_hat) + epsilon) + weight_decay * w)
```

**Use when:**

- Training large transformers (BERT, GPT, etc.)
- Want better generalization than Adam
- Need proper weight decay behavior

**Pros:** Better generalization than Adam, proper regularization
**Cons:** Slightly more complex

**Recommended settings:**

- `lr`: 0.0001 - 0.00001
- `beta1`: 0.9
- `beta2`: 0.999
- `weight_decay`: 0.01 - 0.1
- `epsilon`: 1e-8

## Gradient Clipping (Details)

Gradient clipping prevents exploding gradients by scaling all gradients if their global norm exceeds a threshold.

### When to Use

- ✅ Always for transformer training
- ✅ RNNs and LSTMs
- ✅ When experiencing NaN losses
- ✅ Deep networks (>6 layers)

### Typical Values (Gradient Clipping)

- **Transformers:** 1.0
- **RNNs:** 5.0
- **Small networks:** Not needed or 10.0

### Example (Gradient Clipping)

```cpp
optimizer.set_max_grad_norm(1.0f);
optimizer.clip_gradients();  // Automatically clips if norm > 1.0
```

## Weight Decay

Weight decay (L2 regularization) prevents overfitting by penalizing large weights.

### Adam vs AdamW

- **Adam:** Weight decay coupled with gradients (incorrect)
- **AdamW:** Weight decay decoupled (correct, recommended)

### Typical Values (Weight Decay)

- **Small models:** 0.0 (no regularization)
- **Medium models:** 0.01
- **Large models:** 0.01 - 0.1

### Example (Weight Decay)

```cpp
optimizer.set_weight_decay(0.01f);
```

## Best Practices (Implementation)

### 1. For Transformer Training

```cpp
Optimizer optimizer(OptimizerType::ADAMW, 0.0001f);
optimizer.set_betas(0.9f, 0.999f);
optimizer.set_weight_decay(0.01f);
optimizer.set_max_grad_norm(1.0f);
```

### 2. Learning Rate Scheduling

```cpp
float initial_lr = 0.001f;
Optimizer optimizer(OptimizerType::ADAM, initial_lr);

// In training loop
if (epoch % 10 == 0) {
    float new_lr = optimizer.get_learning_rate() * 0.5f;
    optimizer.set_learning_rate(new_lr);
}
```

### 3. Monitoring Training

```cpp
float grad_norm = optimizer.get_gradient_norm();
if (grad_norm > 10.0f) {
    std::cout << "Warning: Large gradient norm: " << grad_norm << "\n";
}
```

### 4. Debugging Gradient Issues

```cpp
// Check for NaN/Inf gradients
float norm = optimizer.get_gradient_norm();
if (std::isnan(norm) |  | std::isinf(norm)) {
    std::cerr << "Invalid gradients detected!\n";
    optimizer.zero_grad();
    continue;  // Skip this step
}
```

## Common Issues & Solutions (Reference)

### Issue: Loss becomes NaN (Solution)

**Solutions:**

- Enable gradient clipping: `set_max_grad_norm(1.0f)`
- Reduce learning rate
- Check for numerical instabilities in forward pass

### Issue: Slow convergence (Solution)

**Solutions:**

- Switch from SGD to Adam/AdamW
- Increase learning rate
- Add momentum (if using SGD)

### Issue: Overfitting (Solution)

**Solutions:**

- Add weight decay: `set_weight_decay(0.01f)`
- Reduce model capacity
- Add dropout (in model architecture)

### Issue: Training unstable (Solution)

**Solutions:**

- Enable gradient clipping
- Reduce learning rate
- Use AdamW instead of Adam
- Add warmup period

## Performance Considerations

### Memory Usage

- **SGD:** Minimal (no extra state)
- **SGD+Momentum:** 1x parameter memory (for momentum)
- **Adam/AdamW:** 2x parameter memory (momentum + velocity)

### Computation

All optimizers have O(n) time complexity where n is number of parameters.

### Recommendations

- Use Adam/AdamW for most applications (worth the memory)
- Use SGD only if severely memory constrained

## Integration Examples

### With EncoderDecoderModel

```cpp
EncoderDecoderModel model(...);
Optimizer optimizer(OptimizerType::ADAMW, 0.0001f);

// Register all model parameters
// (This would require model to expose parameter groups)
// optimizer.add_parameter_group(&model.encoder_weights, &model.encoder_grads);
// optimizer.add_parameter_group(&model.decoder_weights, &model.decoder_grads);

// Training loop
for (int epoch = 0; epoch < epochs; epoch++) {
    optimizer.zero_grad();
    float loss = model.train_step(input, target);
    optimizer.clip_gradients(1.0f);
    optimizer.step();
}
```

### With Custom Layers

```cpp
Matrix weights(100, 50);
Matrix gradients(100, 50);

Optimizer optimizer(OptimizerType::ADAM, 0.001f);
optimizer.add_parameter_group(&weights, &gradients);

// Training
optimizer.zero_grad();
// ... compute gradients ...
optimizer.step();
```

## See Also

- `OptimizerExample.cpp` - Complete working examples
- `ChatbotTrainer.cpp` - Integration with training pipeline
- `EncoderDecoderModel.hpp` - Model architecture
